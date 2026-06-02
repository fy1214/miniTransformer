// Grouped MXFP8 quantization kernel (SM100 / Blackwell).
//
// Quantizes a packed BF16 matrix of shape [total_rows, K] (all groups
// concatenated row-wise) into MXFP8 E4M3 data + E8M0 scale factors.
// The scale factor layout matches the CUTLASS SM100 `SfKMajorAtom`
// expected by `gemm_mxfp8_grouped_cutlass.cu`.
//
// Layout recap (SFVecSize=32, Blk_MN=128, Blk_SF=4):
//   SF[mIdx, kIdx]  →  offset = cvt_quant_to_fp4_get_sf_out_offset(mIdx, kIdx, numKTiles)
//   where  kIdx    = element_index / 32        (k-group index)
//          numKTiles = ceil(K/32 / 4)           (number of 4-kgroup tiles)
//
// The same formula is already implemented in utils.cuh for FP4; it is
// directly reusable here (same SfAtom structure, different SFVecSize).
//
// Grid / block:
//   threads_per_row = K / 32   (each thread owns 32 consecutive BF16 → 1 E8M0 SF)
//   warps_per_row   = ceil(threads_per_row / 32)
//   rows_per_block  = max(1, 32 / warps_per_row)
//   blockDim        = rows_per_block * warps_per_row * 32
//   gridDim         = total_ctas (from group_cta_offsets, one CTA per rows_per_block slice)
//
// Shared memory:
//   uint32_t amax_scratch[rows_per_block]  — for cross-warp amax reduction

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

// ---------------------------------------------------------------------------
// E8M0 scale computation
// ---------------------------------------------------------------------------

// Returns (E8M0 biased exponent, float scale value = 2^exp).
// E8M0: unsigned 8-bit, value = 2^(byte - 127).  No NaN/Inf (values 0..254).
// For amax==0 we store 0 (= 2^(-127)) and the quant val is 0 anyway.
__device__ __forceinline__ uint8_t float_to_e8m0(float amax) {
    if (amax == 0.0f) return 0u;
    // E8M0 exponent = floor(log2(amax)) + 127 (bias).
    // Extract IEEE-754 biased exponent from float bit pattern:
    //   float biased exp = bits[30:23], unbiased = biased_exp - 127
    // floor(log2(amax)) = unbiased_exp  (for normal floats)
    uint32_t bits = __float_as_uint(amax);
    int unbiased_exp = (int)((bits >> 23) & 0xFF) - 127;
    // E8M0 bias = 127
    int biased = unbiased_exp + 127;
    biased = max(0, min(254, biased));
    return static_cast<uint8_t>(biased);
}

__device__ __forceinline__ float e8m0_to_scale(uint8_t e8m0) {
    // scale = 2^(e8m0 - 127) = ldexpf(1.0f, (int)e8m0 - 127)
    return ldexpf(1.0f, (int)e8m0 - 127);
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

__global__ void mxfp8_group_quantize_kernel(
    uint8_t*            __restrict__ y_ptr,           // [total_rows, K]  FP8 E4M3
    uint8_t*            __restrict__ sf_ptr,          // flat swizzled E8M0 SF buffer
    const nv_bfloat16*  __restrict__ x_ptr,           // [total_rows, K]  BF16 input
    const int32_t*      __restrict__ group_cta_offsets, // [num_groups+1]  prefix-sum of CTAs per group
    const int32_t*      __restrict__ row_offsets,       // [num_groups+1]  row index per group
    const int64_t*      __restrict__ sf_offsets,        // [num_groups+1]  SF element offset per group
    int    K,                // number of columns
    int    K_groups,         // K / 32
    int    num_groups,
    int    warps_per_row,
    int    rows_per_block,
    int    num_k_tiles)      // ceil(K_groups / 4)  — used in SF offset formula
{
    // ---- Shared memory: one uint32 per row for amax atomic reduce ----
    extern __shared__ uint32_t smem_amax[];   // [rows_per_block]

    const int flat_tid    = static_cast<int>(threadIdx.x);
    const int warp_id     = flat_tid >> 5;
    const int lane_id     = flat_tid & 31;
    const int local_row   = warp_id / warps_per_row;
    const int warp_in_row = warp_id % warps_per_row;
    // thread-in-row index (0 .. warps_per_row*32 - 1)
    const int tir         = warp_in_row * 32 + lane_id;  // = thread-in-row

    // ---- Binary search: find group_id for this CTA ----
    // group_cta_offsets[g] <= blockIdx.x < group_cta_offsets[g+1]
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

    const int local_cta       = (int)blockIdx.x - group_cta_offsets[group_id];
    const int group_row0      = row_offsets[group_id];
    const int m_g             = row_offsets[group_id + 1] - group_row0;
    const int64_t sf_base     = sf_offsets[group_id];

    const int base_row_in_g   = local_cta * rows_per_block;
    const int abs_row_base    = group_row0 + base_row_in_g;

    // ---- Initialize smem amax to 0 ----
    if (flat_tid < rows_per_block)
        smem_amax[flat_tid] = 0u;
    __syncthreads();

    // ---- Per-thread: owns 32 BF16 elements (1 MXFP8 block) ----
    // thread tir in a row covers K elements [tir*32 .. (tir+1)*32)
    // but actually each warp handles 32 contiguous k-groups (32*32=1024 elems).
    // Each thread handles exactly one k-group: k_group = tir (if tir < K_groups).

    const int row_in_g  = base_row_in_g + local_row;
    const int global_row = abs_row_base + local_row;
    const bool valid_row = (row_in_g < m_g);
    const bool valid_kgroup = valid_row && (tir < K_groups);

    // Load 32 BF16 elements → compute abs_max for this k-group (pass 1)
    float local_amax = 0.0f;

    if (valid_kgroup) {
        const nv_bfloat16* row_ptr = x_ptr + (int64_t)global_row * K + tir * 32;
        #pragma unroll 4
        for (int i = 0; i < 32; i++) {
            float av = fabsf(__bfloat162float(row_ptr[i]));
            local_amax = fmaxf(local_amax, av);
        }
    }

    // ---- Warp-level reduce of local_amax across threads of same row ----
    // All warps belonging to the same row reduce together via smem atomicMax.
    // Step 1: warp reduce within each warp (only rows contribute one warp,
    //         but multiple warps per row need cross-warp reduce).
    // warp reduce:
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1)
        local_amax = fmaxf(local_amax, __shfl_xor_sync(0xffffffff, local_amax, mask));
    // lane 0 of each warp has the warp's max → atomicMax to smem
    if (lane_id == 0 && valid_row) {
        atomicMax(&smem_amax[local_row], __float_as_uint(local_amax));
    }
    __syncthreads();

    if (!valid_kgroup) return;

    // ---- Broadcast row amax from smem, compute E8M0 scale ----
    const float row_amax     = __uint_as_float(smem_amax[local_row]);
    const uint8_t e8m0_val   = float_to_e8m0(row_amax);
    const float   scale_fp   = (e8m0_val == 0u) ? 1.0f : e8m0_to_scale(e8m0_val);
    const float   inv_scale  = (scale_fp == 0.0f) ? 0.0f : 1.0f / scale_fp;

    // ---- Quantize 32 BF16 → 32 FP8 E4M3 (pass 2: re-read from global memory) ----
    const int64_t out_base  = (int64_t)global_row * K + tir * 32;
    constexpr float clamp_v = 448.0f;
    {
        const nv_bfloat16* row_ptr = x_ptr + (int64_t)global_row * K + tir * 32;
        #pragma unroll 4
        for (int i = 0; i < 32; i++) {
            float scaled = __bfloat162float(row_ptr[i]) * inv_scale;
            scaled = fminf(fmaxf(scaled, -clamp_v), clamp_v);
            y_ptr[out_base + i] = static_cast<uint8_t>(static_cast<__nv_fp8_e4m3>(scaled));
        }
    }

    // ---- Write E8M0 scale factor to swizzled SF layout ----
    // kIdx = tir (= k_group index within the row)
    // mIdx = row_in_g (local row within group, 0-based)
    // Swizzled offset formula (same as FP4 SfKMajorAtom):
    //   numKTiles = ceil(K_groups / 4)
    const int64_t sf_offset = cvt_quant_to_fp4_get_sf_out_offset(row_in_g, tir, num_k_tiles);
    sf_ptr[sf_base + sf_offset] = e8m0_val;
}

// ---------------------------------------------------------------------------
// Launch helper
// ---------------------------------------------------------------------------

static void launch_mxfp8_group_quantize(
    uint8_t*           y_ptr,
    uint8_t*           sf_ptr,
    const nv_bfloat16* x_ptr,
    int                num_groups,
    int                K,
    const int32_t*     row_offsets_gpu,     // [num_groups+1] on GPU
    const int64_t*     sf_offsets_gpu,      // [num_groups+1] on GPU
    const int32_t*     row_offsets_host,    // [num_groups+1] on host
    cudaStream_t       stream)
{
    const int K_groups       = K / 32;
    const int threads_per_row = K_groups;  // one thread per k-group
    const int warps_per_row   = (threads_per_row + 31) / 32;
    // Target ~32 warps per block for good occupancy; allow up to 1024 threads.
    const int max_warps_per_block = 32;
    const int rows_per_block  = max(1, max_warps_per_block / warps_per_row);
    const int block_dim       = rows_per_block * warps_per_row * 32;

    // Build group_cta_offsets on host
    std::vector<int32_t> h_cta_offsets(num_groups + 1);
    h_cta_offsets[0] = 0;
    for (int g = 0; g < num_groups; g++) {
        const int m_g = row_offsets_host[g + 1] - row_offsets_host[g];
        h_cta_offsets[g + 1] = h_cta_offsets[g] +
            (m_g + rows_per_block - 1) / rows_per_block;
    }
    const int total_ctas = h_cta_offsets[num_groups];
    if (total_ctas == 0) return;

    // Copy group_cta_offsets to device
    int32_t* d_cta_offsets = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_cta_offsets,
        (num_groups + 1) * sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_cta_offsets, h_cta_offsets.data(),
        (num_groups + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    const int num_k_tiles    = (K_groups + 3) / 4;
    const size_t smem_bytes  = (size_t)rows_per_block * sizeof(uint32_t);

    mxfp8_group_quantize_kernel<<<total_ctas, block_dim, smem_bytes, stream>>>(
        y_ptr, sf_ptr, x_ptr,
        d_cta_offsets, row_offsets_gpu, sf_offsets_gpu,
        K, K_groups, num_groups, warps_per_row, rows_per_block,
        num_k_tiles);

    CUDA_CHECK(cudaFreeAsync(d_cta_offsets, stream));
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// PyTorch binding
// ---------------------------------------------------------------------------

void mxfp8_grouped_quantize_binding(
    at::Tensor&       out,           // [total_rows, K]  uint8
    at::Tensor&       sf,            // flat swizzled E8M0 buffer  uint8
    const at::Tensor& inp,           // [total_rows, K]  bfloat16
    const at::Tensor& row_offsets,   // [num_groups+1]  int32  GPU
    const at::Tensor& sf_offsets)    // [num_groups+1]  int64  GPU
{
    TORCH_CHECK(inp.dtype()  == at::kBFloat16, "inp must be bfloat16");
    TORCH_CHECK(inp.is_cuda(),                 "inp must be CUDA");
    TORCH_CHECK(inp.is_contiguous(),           "inp must be contiguous");
    TORCH_CHECK(inp.dim() == 2,                "inp must be 2D [total_rows, K]");
    TORCH_CHECK(inp.size(1) % 32 == 0,         "K must be divisible by 32");
    TORCH_CHECK(row_offsets.dtype()  == at::kInt,  "row_offsets must be int32");
    TORCH_CHECK(sf_offsets.dtype()   == at::kLong, "sf_offsets must be int64");

    const int num_groups = static_cast<int>(row_offsets.numel()) - 1;
    const int total_rows = static_cast<int>(inp.size(0));
    const int K          = static_cast<int>(inp.size(1));

    TORCH_CHECK(out.numel()  == (int64_t)total_rows * K, "out size mismatch");
    TORCH_CHECK(out.dtype()  == at::kByte, "out must be uint8");

    // D2H copy of row_offsets for host-side CTA offset computation (small, num_groups+1 ints).
    at::Tensor row_offsets_cpu = row_offsets.cpu();
    const int32_t* h_row_off = reinterpret_cast<const int32_t*>(row_offsets_cpu.data_ptr());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(inp.device().index());
    launch_mxfp8_group_quantize(
        reinterpret_cast<uint8_t*>(out.data_ptr()),
        reinterpret_cast<uint8_t*>(sf.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(inp.data_ptr()),
        num_groups, K,
        reinterpret_cast<const int32_t*>(row_offsets.data_ptr()),
        reinterpret_cast<const int64_t*>(sf_offsets.data_ptr()),
        h_row_off,
        stream);
}
