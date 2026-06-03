// Grouped MXFP8 transpose quantization (SM100 / Blackwell) — TMA v2.
//
// Quantizes [total_M, K] BF16 → transposed [K, total_M] FP8 E4M3 + E8M0 SF.
//
// Tile: [TILE_M=32, TILE_K=64] BF16.
//   TILE_K=64 chosen so that boxX * sizeof(BF16) = 64*2 = 128 B, satisfying
//   the SWIZZLE_128B constraint.  TMA applies XOR swizzle on write, so column
//   reads (each thread reads in_sh[0..31][t]) are conflict-free.
//
// Thread layout: TILE_K=64 threads per block.
//   - Thread 0 (master) issues the TMA load; all wait on the mbarrier.
//   - Each thread t owns column t of the tile: reads 32 BF16 from shmem,
//     computes per-column E8M0, writes 32 FP8 bytes to [K, total_M].
//
// SF layout (SfKMajorAtom, same formula as forward kernel but roles swapped):
//   mIdx = global_k,  kIdx = m_tile,  numKTiles = ceil(ceil(m_g/32)/4)
//   sf_size_T_g = ceil(K/128) * ceil(ceil(m_g/32)/4) * 512 bytes
//
// Grid: group-aligned CTAs.  For each group g:
//   n_ctas_g = ceil(m_g/TILE_M) * ceil(K/TILE_K)
//
// Shmem:  TILE_M * TILE_K * sizeof(BF16) + 8 (mbarrier) = 4096 + 8 bytes
//   With SWIZZLE_128B the physical layout XORs address bits so that reading
//   column t across rows hits distinct banks → zero bank conflict.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "../../../ops.h"
#include "../../../util/quarter/utils.cuh"
#include "../../../util/transformer_engine/common.h"
#include "../../../util/transformer_engine/ptx.cuh"

// ---------------------------------------------------------------------------
// E8M0 helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint8_t mxfp8t_float_to_e8m0(float amax) {
    if (amax == 0.0f) return 0u;
    uint32_t bits = __float_as_uint(amax);
    int biased = (int)((bits >> 23) & 0xFF);   // IEEE-754 biased exp = unbiased+127 = E8M0 value
    biased = max(0, min(254, biased));
    return static_cast<uint8_t>(biased);
}

__device__ __forceinline__ float mxfp8t_e8m0_to_scale(uint8_t e8m0) {
    return ldexpf(1.0f, (int)e8m0 - 127);
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

static constexpr int TILE_M = 32;
static constexpr int TILE_K = 64;    // boxX*2B = 128B → satisfies SWIZZLE_128B

__global__ void __launch_bounds__(TILE_K)
mxfp8_group_transpose_quantize_kernel(
    uint8_t*           __restrict__ y_T_ptr,           // [K, total_M] FP8 E4M3
    uint8_t*           __restrict__ sf_ptr,             // flat E8M0 SF buffer
    const __grid_constant__ CUtensorMap tensor_map_in,  // [total_M, K] BF16, SWIZZLE_128B
    const int32_t*     __restrict__ group_cta_offsets,  // [num_groups+1]
    const int32_t*     __restrict__ row_offsets,        // [num_groups+1]
    const int64_t*     __restrict__ sf_offsets,         // [num_groups+1]
    int    K,
    int    n_k_tiles,    // ceil(K / TILE_K)
    int    num_groups,
    int    total_M
)
{
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

    // ---- Shared memory layout ----
    // [TILE_M * TILE_K] BF16 input buffer  (4096 B, 128B aligned)
    // + uint64_t mbarrier                  (8 B)
    extern __shared__ char dynamic_shmem[];
    uintptr_t base = reinterpret_cast<uintptr_t>(dynamic_shmem);
    uintptr_t aligned = (base + TMA_SHMEM_ALIGNMENT - 1) & ~(uintptr_t)(TMA_SHMEM_ALIGNMENT - 1);

    nv_bfloat16* in_sh = reinterpret_cast<nv_bfloat16*>(aligned);
    uint64_t*    mbar  = reinterpret_cast<uint64_t*>(aligned + TILE_M * TILE_K * sizeof(nv_bfloat16));

    const int t = static_cast<int>(threadIdx.x);   // 0..TILE_K-1
    const bool is_master = (t == 0);

    // ---- Binary search: find group_id ----
    int group_id;
    {
        int lo = 0, hi = num_groups - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (group_cta_offsets[mid] <= (int)blockIdx.x) lo = mid;
            else hi = mid - 1;
        }
        group_id = lo;
    }

    const int     local_cta  = (int)blockIdx.x - group_cta_offsets[group_id];
    const int     group_row0 = row_offsets[group_id];
    const int     m_g        = row_offsets[group_id + 1] - group_row0;
    const int64_t sf_base    = sf_offsets[group_id];

    const int m_tile   = local_cta / n_k_tiles;
    const int k_tile   = local_cta % n_k_tiles;
    const int m_offset = m_tile * TILE_M;
    const int k_offset = k_tile * TILE_K;

    // ---- Initialize mbarrier (1 thread, count=TILE_K) ----
    if (is_master) {
        mbarrier_init(mbar, TILE_K);
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    // ---- TMA load: [TILE_M, TILE_K] BF16 from global → shmem ----
    // TMA offset_x = k_offset (column start), offset_y = global row start.
    // Out-of-bound rows are zero-filled by TMA (OOB_FILL_NONE → 0).
    const uint32_t tma_off_x = static_cast<uint32_t>(k_offset);
    const uint32_t tma_off_y = static_cast<uint32_t>(group_row0 + m_offset);

    const size_t tma_bytes = TILE_M * TILE_K * sizeof(nv_bfloat16);  // 4096 B
    copy_2d_to_shared(in_sh, &tensor_map_in, tma_off_x, tma_off_y, tma_bytes, mbar, is_master);

    // ---- Wait for TMA to finish ----
    mbarrier_wait_parity(mbar, 0);
    fence_proxy_async_shared_cta();

    // Early exit for threads whose K-column is out of bounds
    const int global_k = k_offset + t;
    if (global_k >= K) goto cleanup;

    {
        // ---- Read column t from shmem (conflict-free due to SWIZZLE_128B) ----
        // With SWIZZLE_128B the TMA XOR's address bits so that
        // in_sh[m * TILE_K + t] and in_sh[(m+1) * TILE_K + t] hit different banks.
        float vals[TILE_M];
        float local_amax = 0.0f;
        #pragma unroll
        for (int m = 0; m < TILE_M; m++) {
            float v = __bfloat162float(in_sh[m * TILE_K + t]);
            vals[m] = v;
            if ((m_offset + m) < m_g)
                local_amax = fmaxf(local_amax, fabsf(v));
        }

        // ---- E8M0 scale ----
        const uint8_t e8m0_val  = mxfp8t_float_to_e8m0(local_amax);
        const float   scale_fp  = (e8m0_val == 0u) ? 1.0f : mxfp8t_e8m0_to_scale(e8m0_val);
        const float   inv_scale = (scale_fp == 0.0f) ? 0.0f : 1.0f / scale_fp;

        // ---- Write 32 FP8 bytes to transposed output [K, total_M] ----
        constexpr float clamp_v = 448.0f;
        const int64_t out_base  = (int64_t)global_k * total_M + (group_row0 + m_offset);
        #pragma unroll
        for (int m = 0; m < TILE_M; m++) {
            if ((m_offset + m) < m_g) {
                float s = vals[m] * inv_scale;
                s = fminf(fmaxf(s, -clamp_v), clamp_v);
                y_T_ptr[out_base + m] = static_cast<uint8_t>(static_cast<__nv_fp8_e4m3>(s));
            }
        }

        // ---- Write E8M0 SF ----
        if (m_offset < m_g) {
            const int m_k_groups = (m_g + 31) >> 5;
            const int numMKTiles = (m_k_groups + 3) >> 2;
            const int64_t sf_off = cvt_quant_to_fp4_get_sf_out_offset(global_k, m_tile, numMKTiles);
            sf_ptr[sf_base + sf_off] = e8m0_val;
        }
    }

cleanup:
    // Invalidate the barrier so shmem can be reused (good practice)
    if (is_master) mbarrier_invalid(mbar);

#else
    // SM < 100: not supported
    __trap();
#endif
}

// ---------------------------------------------------------------------------
// Launch helper
// ---------------------------------------------------------------------------

static void launch_mxfp8_group_transpose_quantize(
    uint8_t*           y_T_ptr,
    uint8_t*           sf_ptr,
    const nv_bfloat16* x_ptr,
    int                num_groups,
    int                K,
    int                total_M,
    const int32_t*     row_offsets_gpu,
    const int64_t*     sf_offsets_gpu,
    const int32_t*     row_offsets_host,
    const at::Tensor&  inp_tensor,       // needed for create_2D_tensor_map
    cudaStream_t       stream)
{
    const int n_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Build group_cta_offsets on host
    std::vector<int32_t> h_cta_offsets(num_groups + 1);
    h_cta_offsets[0] = 0;
    for (int g = 0; g < num_groups; g++) {
        const int m_g       = row_offsets_host[g + 1] - row_offsets_host[g];
        const int n_m_tiles = (m_g + TILE_M - 1) / TILE_M;
        h_cta_offsets[g + 1] = h_cta_offsets[g] + n_m_tiles * n_k_tiles;
    }
    const int total_ctas = h_cta_offsets[num_groups];
    if (total_ctas == 0) return;

    // TMA descriptor for input [total_M, K] BF16, SWIZZLE_128B
    // boxX=TILE_K=64, boxY=TILE_M=32; boxX*2B=128B → SWIZZLE_128B is legal.
    alignas(64) CUtensorMap tensor_map_in{};
    create_2D_tensor_map(
        tensor_map_in, inp_tensor,
        /*globalY=*/static_cast<uint64_t>(total_M),
        /*globalX=*/static_cast<uint64_t>(K),
        /*shmemY=*/TILE_M,
        /*shmemX=*/TILE_K,
        /*stride_elems=*/static_cast<uint32_t>(K),
        /*offset_elems=*/0,
        /*type_num_bits=*/sizeof(nv_bfloat16) * 8,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);

    // Copy group_cta_offsets to device
    int32_t* d_cta_offsets = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_cta_offsets,
        (num_groups + 1) * sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_cta_offsets, h_cta_offsets.data(),
        (num_groups + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    // shmem: 128B-aligned input buffer + 8B mbarrier + 128B alignment padding
    const size_t smem_bytes = TMA_SHMEM_ALIGNMENT
                            + TILE_M * TILE_K * sizeof(nv_bfloat16)
                            + sizeof(uint64_t);

    CUDA_CHECK(cudaFuncSetAttribute(
        mxfp8_group_transpose_quantize_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    mxfp8_group_transpose_quantize_kernel<<<total_ctas, TILE_K, smem_bytes, stream>>>(
        y_T_ptr, sf_ptr, tensor_map_in,
        d_cta_offsets, row_offsets_gpu, sf_offsets_gpu,
        K, n_k_tiles, num_groups, total_M);

    CUDA_CHECK(cudaFreeAsync(d_cta_offsets, stream));
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// PyTorch binding
// ---------------------------------------------------------------------------

void mxfp8_grouped_transpose_quantize_binding(
    at::Tensor&       out_T,        // [K, total_M]  uint8
    at::Tensor&       sf,           // flat E8M0 buffer  uint8
    const at::Tensor& inp,          // [total_M, K]  bfloat16
    const at::Tensor& row_offsets,  // [num_groups+1]  int32  GPU
    const at::Tensor& sf_offsets)   // [num_groups+1]  int64  GPU
{
    TORCH_CHECK(inp.dtype()  == at::kBFloat16, "inp must be bfloat16");
    TORCH_CHECK(inp.is_cuda(),                 "inp must be CUDA");
    TORCH_CHECK(inp.is_contiguous(),           "inp must be contiguous");
    TORCH_CHECK(inp.dim() == 2,                "inp must be 2D [total_M, K]");
    TORCH_CHECK(inp.size(1) % TILE_K == 0,     "K must be divisible by TILE_K(64)");
    TORCH_CHECK(inp.size(0) % TILE_M == 0,     "total_M must be divisible by TILE_M(32)");
    TORCH_CHECK(row_offsets.dtype() == at::kInt,  "row_offsets must be int32");
    TORCH_CHECK(sf_offsets.dtype()  == at::kLong, "sf_offsets must be int64");

    const int num_groups = static_cast<int>(row_offsets.numel()) - 1;
    const int total_M    = static_cast<int>(inp.size(0));
    const int K          = static_cast<int>(inp.size(1));

    TORCH_CHECK(out_T.numel() == (int64_t)K * total_M, "out_T size mismatch");
    TORCH_CHECK(out_T.dtype() == at::kByte, "out_T must be uint8");

    at::Tensor row_offsets_cpu = row_offsets.cpu();
    const int32_t* h_row_off = reinterpret_cast<const int32_t*>(row_offsets_cpu.data_ptr());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(inp.device().index());
    launch_mxfp8_group_transpose_quantize(
        reinterpret_cast<uint8_t*>(out_T.data_ptr()),
        reinterpret_cast<uint8_t*>(sf.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(inp.data_ptr()),
        num_groups, K, total_M,
        reinterpret_cast<const int32_t*>(row_offsets.data_ptr()),
        reinterpret_cast<const int64_t*>(sf_offsets.data_ptr()),
        h_row_off,
        inp,
        stream);
}
