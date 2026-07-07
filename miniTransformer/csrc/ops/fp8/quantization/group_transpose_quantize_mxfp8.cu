// Grouped MXFP8 transpose quantization (SM100 / Blackwell) — V2 coalesced.
//
// Quantizes [total_M, K] BF16 → per-group transposed FP8 E4M3 + E8M0 SF.
// Output layout: each group's [K, m_per_group] block stored contiguously
// (row-major), groups stacked.  Viewable as [ng*K, m_per_group] uint8.
//
// V2 improvements:
//   * TILE_M=128 (4× V1) → 4× fewer CTAs, better SM occupancy.
//   * 256 threads (8 warps): each warp handles 8 K-columns, processing
//     4 SF-groups of 32 M-values each via warp-level amax reduction.
//   * Coalesced writes: 32 warp lanes write consecutive M-addresses
//     (32 bytes per transaction → 4 transactions fill one 128B cache line).
//   * No .cpu() sync: uniform groups → all offsets from scalar params.
//   * No per-call cudaMalloc: group_cta_offsets replaced by integer division.
//
// Assumes uniform groups (caller passes m_per_group as a scalar).
//
// Tile: [TILE_M=128, TILE_K=64] BF16.
//   TILE_K=64 for SWIZZLE_128B (64 bf16 × 2B = 128B).
//
// Shmem: 128*64*2 = 16384 B input + 8B mbarrier + alignment ≈ 16.5 KB.
//
// SF layout (SfKMajorAtom, transpose convention):
//   mIdx = global_k,  kIdx = m_tile_within_group
//   numKTiles = ceil(ceil(m_per_group/32) / 4)

#ifndef MINI_TE_NO_BLACKWELL_OPS

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "../../../ops.h"
#include "../../../util/quarter/utils.cuh"
#include "../../../util/transformer_engine/common.h"
#include "../../../util/transformer_engine/ptx.cuh"

namespace {

__device__ __forceinline__ uint8_t mxfp8t_float_to_e8m0(float amax) {
    if (amax == 0.0f) return 0u;
    constexpr float max_norm_rcp = 1.0f / 448.0f;
    const float val = amax * max_norm_rcp;
    const uint32_t bits     = __float_as_uint(val);
    uint32_t       exponent = (bits >> 23) & 0xFF;
    const uint32_t mantissa = bits & 0x7FFFFFu;
    if ((mantissa > 0u && exponent != 0xFEu) &&
        !(exponent == 0u && mantissa <= 0x400000u)) {
        ++exponent;
    }
    return static_cast<uint8_t>(exponent);
}

__device__ __forceinline__ float mxfp8t_e8m0_to_scale(uint8_t e8m0) {
    return ldexpf(1.0f, (int)e8m0 - 127);
}

static constexpr int TILE_M  = 128;
static constexpr int TILE_K  = 64;
static constexpr int THREADS = 256;   // 8 warps
static constexpr int WARPS   = THREADS / 32;           // 8
static constexpr int K_PER_WARP = TILE_K / WARPS;      // 8

__global__ void __launch_bounds__(THREADS)
mxfp8_group_transpose_quantize_kernel(
    uint8_t*           __restrict__ y_T_ptr,
    uint8_t*           __restrict__ sf_ptr,
    const __grid_constant__ CUtensorMap tensor_map_in,
    int    K,
    int    n_k_tiles,
    int    num_groups,
    int    m_per_group,
    int    ctas_per_group,
    int    sf_per_group,
    int    num_m_k_tiles)
{
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

    extern __shared__ char dynamic_shmem[];
    uintptr_t base_ptr = reinterpret_cast<uintptr_t>(dynamic_shmem);
    uintptr_t aligned  = (base_ptr + TMA_SHMEM_ALIGNMENT - 1)
                       & ~(uintptr_t)(TMA_SHMEM_ALIGNMENT - 1);

    nv_bfloat16* in_sh = reinterpret_cast<nv_bfloat16*>(aligned);
    uint64_t*    mbar  = reinterpret_cast<uint64_t*>(
        aligned + TILE_M * TILE_K * sizeof(nv_bfloat16));

    const int tid       = static_cast<int>(threadIdx.x);
    const int warp_id   = tid >> 5;
    const int lane_id   = tid & 31;
    const bool is_master = (tid == 0);

    // ---- Uniform-group identification (no binary search) ----
    const int group_id   = static_cast<int>(blockIdx.x) / ctas_per_group;
    const int local_cta  = static_cast<int>(blockIdx.x) % ctas_per_group;
    const int m_tile     = local_cta / n_k_tiles;
    const int k_tile     = local_cta % n_k_tiles;
    const int m_offset   = m_tile * TILE_M;
    const int k_offset   = k_tile * TILE_K;
    const int group_row0 = group_id * m_per_group;

    // ---- Initialize mbarrier ----
    if (is_master) {
        mbarrier_init(mbar, THREADS);
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    // ---- TMA load [128, 64] BF16 → shmem (SWIZZLE_128B) ----
    const uint32_t tma_off_x = static_cast<uint32_t>(k_offset);
    const uint32_t tma_off_y = static_cast<uint32_t>(group_row0 + m_offset);
    const size_t   tma_bytes = TILE_M * TILE_K * sizeof(nv_bfloat16);
    copy_2d_to_shared(in_sh, &tensor_map_in, tma_off_x, tma_off_y,
                      tma_bytes, mbar, is_master);
    mbarrier_wait_parity(mbar, 0);
    fence_proxy_async_shared_cta();

    // ---- Output bases (all from scalars, no GPU array reads) ----
    const int64_t data_base = (int64_t)group_id * m_per_group * K;
    const int64_t sf_base   = (int64_t)group_id * sf_per_group;

    // ---- Each warp processes K_PER_WARP K-columns ----
    const int k_start = warp_id * K_PER_WARP;

    #pragma unroll 1
    for (int kw = 0; kw < K_PER_WARP; kw++) {
        const int local_k  = k_start + kw;
        const int global_k = k_offset + local_k;
        if (global_k >= K) break;

        const int64_t row_base = data_base + (int64_t)global_k * m_per_group;

        // 4 SF-groups of 32 M-values each cover the full TILE_M=128
        #pragma unroll
        for (int sg = 0; sg < 4; sg++) {
            const int m_in_tile  = sg * 32 + lane_id;
            const int m_in_group = m_offset + m_in_tile;
            if (m_in_group >= m_per_group) continue;

            // Read BF16 from shmem with SWIZZLE_128B inverse:
            //   physical_col = logical_col ^ ((row & 7) << 3)
            const int swizzled_k = local_k ^ ((m_in_tile & 7) << 3);
            const float val = __bfloat162float(
                in_sh[m_in_tile * TILE_K + swizzled_k]);

            // Warp-level amax reduction (full 32 lanes)
            float amax = fabsf(val);
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 4));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 8));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 16));

            // E8M0 + quantize
            const uint8_t e8m0_val = mxfp8t_float_to_e8m0(amax);
            const float   scale_fp = (e8m0_val == 0u)
                                       ? 1.0f : mxfp8t_e8m0_to_scale(e8m0_val);
            const float inv_scale  = (scale_fp == 0.0f) ? 0.0f : 1.0f / scale_fp;

            float scaled = val * inv_scale;
            scaled = fminf(fmaxf(scaled, -448.0f), 448.0f);
            const __nv_fp8_e4m3 q = static_cast<__nv_fp8_e4m3>(scaled);

            // COALESCED write: 32 lanes → 32 consecutive bytes
            y_T_ptr[row_base + m_in_group] = q.__x;

            // SF (one per 32-element group, lane 0 writes)
            if (lane_id == 0) {
                const int m_tile_idx = m_offset / 32 + sg;
                const int64_t sf_off = cvt_quant_to_fp4_get_sf_out_offset(
                    global_k, m_tile_idx, num_m_k_tiles);
                sf_ptr[sf_base + sf_off] = e8m0_val;
            }
        }
    }

    if (is_master) mbarrier_invalid(mbar);

#else
    __trap();
#endif
}

static void launch_mxfp8_group_transpose_quantize(
    uint8_t*           y_T_ptr,
    uint8_t*           sf_ptr,
    int                num_groups,
    int                K,
    int                total_M,
    int                m_per_group,
    const at::Tensor&  inp_tensor,
    cudaStream_t       stream)
{
    const int n_k_tiles = (K + TILE_K - 1) / TILE_K;
    const int n_m_tiles = (m_per_group + TILE_M - 1) / TILE_M;
    const int ctas_per_group = n_m_tiles * n_k_tiles;
    const int total_ctas = num_groups * ctas_per_group;
    if (total_ctas == 0) return;

    const int m_k_groups    = (m_per_group + 31) / 32;
    const int num_m_k_tiles = (m_k_groups + 3) / 4;
    const int sf_per_group  = ((K + 127) / 128) * num_m_k_tiles * 512;

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

    const size_t smem_bytes = TMA_SHMEM_ALIGNMENT
                            + TILE_M * TILE_K * sizeof(nv_bfloat16)
                            + sizeof(uint64_t);

    CUDA_CHECK(cudaFuncSetAttribute(
        mxfp8_group_transpose_quantize_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    mxfp8_group_transpose_quantize_kernel<<<total_ctas, THREADS, smem_bytes, stream>>>(
        y_T_ptr, sf_ptr, tensor_map_in,
        K, n_k_tiles, num_groups, m_per_group, ctas_per_group,
        sf_per_group, num_m_k_tiles);

    CUDA_CHECK(cudaGetLastError());
}

} // anonymous namespace

void mxfp8_grouped_transpose_quantize_binding(
    at::Tensor&       out_T,
    at::Tensor&       sf,
    const at::Tensor& inp,
    int               m_per_group,
    int               num_groups)
{
    TORCH_CHECK(inp.dtype()  == at::kBFloat16, "inp must be bfloat16");
    TORCH_CHECK(inp.is_cuda(),                 "inp must be CUDA");
    TORCH_CHECK(inp.is_contiguous(),           "inp must be contiguous");
    TORCH_CHECK(inp.dim() == 2,                "inp must be 2D [total_M, K]");
    TORCH_CHECK(inp.size(1) % TILE_K == 0,     "K must be divisible by 64");
    TORCH_CHECK(m_per_group % 32 == 0,          "m_per_group must be divisible by 32");
    TORCH_CHECK(inp.size(0) == (int64_t)num_groups * m_per_group,
                "total_M must equal num_groups * m_per_group");

    const int total_M = static_cast<int>(inp.size(0));
    const int K       = static_cast<int>(inp.size(1));

    TORCH_CHECK(out_T.numel() == (int64_t)K * total_M, "out_T size mismatch");
    TORCH_CHECK(out_T.dtype() == at::kByte, "out_T must be uint8");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(inp.device().index());
    launch_mxfp8_group_transpose_quantize(
        reinterpret_cast<uint8_t*>(out_T.data_ptr()),
        reinterpret_cast<uint8_t*>(sf.data_ptr()),
        num_groups, K, total_M, m_per_group,
        inp, stream);
}

#else  // MINI_TE_NO_BLACKWELL_OPS — stubs for non-Blackwell builds

#include "../../../ops.h"
#include <stdexcept>

void mxfp8_grouped_transpose_quantize_binding(
    at::Tensor&       out_T,
    at::Tensor&       sf,
    const at::Tensor& inp,
    int               m_per_group,
    int               num_groups) {
    throw std::runtime_error("mxfp8_grouped_transpose_quantize_binding requires Blackwell+ (SM100). Rebuild on SM100 hardware or set MINI_TE_FORCE_BLACKWELL_OPS=1.");
}

#endif  // MINI_TE_NO_BLACKWELL_OPS
