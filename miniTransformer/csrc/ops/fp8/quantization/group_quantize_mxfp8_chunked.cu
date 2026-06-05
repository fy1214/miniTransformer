// Grouped MXFP8 quantization kernel (SM100 / Blackwell), per-CTA-chunk variant.
//
// Mirrors the contract of group_quantize_mxfp8.cu but replaces the per-row
// linear scan over row_offsets with TE-style per-CTA-chunk scheduling:
//
//   * Each CTA owns a CHUNK_M=128-row chunk along the M axis.
//   * The chunk is binary-searched into the group table ONCE per chunk
//     (broadcast via shared memory). Inner loop covers all CHUNK_M rows.
//   * Requires every group's M to be a multiple of CHUNK_M (== 128).
//     This matches Megatron's Fp8Padding (TEGroupedMLP.forward), so no
//     chunk ever straddles a group boundary.
//
// Why a new kernel (and not a flag on the existing one):
//   - Different scheduling unit (chunk-of-128 vs row).
//   - Adds a hard 128-row-per-group precondition that the original kernel
//     doesn't require.
//   - Same memory layout / SF layout as the original so it's a drop-in for
//     the CUTLASS SM100 collective, and Python-side wrappers can A/B both.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <algorithm>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "../../../ops.h"
#include "../../../util/quarter/utils.cuh"
#include "../../../util/quarter/vec.cuh"

namespace {

using bf16x8 = GenericVector<nv_bfloat16, 8>;

static constexpr int CHUNK_M    = 128;
static constexpr int MAX_BLOCKS = 1024;

// ---------------------------------------------------------------------------
// E8M0 helpers (kept local — identical math to the per-row kernel, but in an
// anonymous namespace so both translation units coexist without ODR clashes).
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint8_t float_to_e8m0_chunked(float amax) {
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

__device__ __forceinline__ float e8m0_to_scale_chunked(uint8_t e8m0) {
    return ldexpf(1.0f, (int)e8m0 - 127);
}

// ---------------------------------------------------------------------------
// Per-CTA-chunk kernel (software E8M0 + manual clamp).
// ---------------------------------------------------------------------------

__global__ void mxfp8_group_quantize_chunked_kernel(
    uint8_t*            __restrict__ y_ptr,
    uint8_t*            __restrict__ sf_ptr,
    const nv_bfloat16*  __restrict__ x_ptr,
    const int32_t*      __restrict__ row_offsets,   // [num_groups+1] int32
    const int64_t*      __restrict__ sf_offsets,    // [num_groups+1] int64
    int    K,
    int    num_groups,
    int    nvecs_per_row,
    int    warps_per_row,
    int    rows_per_iter,       // rows processed by one block in one inner pass
    int    num_k_tiles,
    int    total_chunks)        // total_rows / CHUNK_M
{
    const int flat_tid    = static_cast<int>(threadIdx.x);
    const int warp_id     = flat_tid >> 5;
    const int lane_id     = flat_tid & 31;
    const int local_row   = warp_id / warps_per_row;
    const int warp_in_row = warp_id % warps_per_row;
    const int tid         = warp_in_row * 32 + lane_id;

    const int n_inner = CHUNK_M / rows_per_iter;

    __shared__ int     s_row_base_in_group;
    __shared__ int64_t s_sf_base;

    for (int chunk_idx = blockIdx.x; chunk_idx < total_chunks; chunk_idx += gridDim.x) {
        const int chunk_row_start = chunk_idx * CHUNK_M;

        // ---- Single-thread binary search per chunk, broadcast via smem ----
        if (flat_tid == 0) {
            // Largest lo such that row_offsets[lo] <= chunk_row_start.
            // CHUNK_M-alignment guarantees the chunk lies wholly inside one group.
            int lo = 0;
            int hi = num_groups - 1;
            while (lo < hi) {
                const int mid = (lo + hi + 1) >> 1;
                if (row_offsets[mid] <= chunk_row_start) lo = mid;
                else                                     hi = mid - 1;
            }
            s_row_base_in_group = chunk_row_start - row_offsets[lo];
            s_sf_base           = sf_offsets[lo];
        }
        __syncthreads();

        const int     row_base_in_group = s_row_base_in_group;
        const int64_t sf_base           = s_sf_base;

        // ---- Inner loop: cover all CHUNK_M rows in this chunk ----
        #pragma unroll 1
        for (int inner = 0; inner < n_inner; ++inner) {
            const int row_in_chunk       = inner * rows_per_iter + local_row;
            const int row                = chunk_row_start + row_in_chunk;
            const int local_row_in_group = row_base_in_group + row_in_chunk;
            const bool valid_vec         = (tid < nvecs_per_row);

            bf16x8 x{};
            float  local_amax = 0.0f;
            if (valid_vec) {
                x = bf16x8::load(x_ptr + 8 * ((int64_t)row * nvecs_per_row + tid));
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    local_amax = fmaxf(local_amax, fabsf(__bfloat162float(x[i])));
            }

            float blk_amax = local_amax;
            blk_amax = fmaxf(blk_amax, __shfl_xor_sync(0xffffffff, blk_amax, 1));
            blk_amax = fmaxf(blk_amax, __shfl_xor_sync(0xffffffff, blk_amax, 2));

            if (!valid_vec) continue;

            const uint8_t e8m0_val  = float_to_e8m0_chunked(blk_amax);
            const float   scale_fp  = (e8m0_val == 0u) ? 1.0f : e8m0_to_scale_chunked(e8m0_val);
            const float   inv_scale = (scale_fp == 0.0f) ? 0.0f : 1.0f / scale_fp;

            constexpr float clamp_v = 448.0f;
            GenericVector<uint8_t, 8> fp8_out;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float scaled = __bfloat162float(x[i]) * inv_scale;
                scaled = fminf(fmaxf(scaled, -clamp_v), clamp_v);
                __nv_fp8_e4m3 q = static_cast<__nv_fp8_e4m3>(scaled);
                fp8_out[i] = q.__x;
            }
            fp8_out.store(y_ptr + (int64_t)row * K + 8 * tid);

            if ((tid & 3) == 0) {
                const int     col    = tid >> 2;
                const int64_t sf_off = cvt_quant_to_fp4_get_sf_out_offset(local_row_in_group, col, num_k_tiles);
                sf_ptr[sf_base + sf_off] = e8m0_val;
            }
        }
        __syncthreads();  // ensure all rows of this chunk are done before next chunk overwrites smem
    }
}

// ---------------------------------------------------------------------------
// Per-CTA-chunk kernel (hardware packed cvts: cvt.rp.ue8m0x2 + cvt.rn.e4m3x2).
// ---------------------------------------------------------------------------

__global__ void mxfp8_group_quantize_chunked_hw_kernel(
    uint8_t*            __restrict__ y_ptr,
    uint8_t*            __restrict__ sf_ptr,
    const nv_bfloat16*  __restrict__ x_ptr,
    const int32_t*      __restrict__ row_offsets,
    const int64_t*      __restrict__ sf_offsets,
    int    K,
    int    num_groups,
    int    nvecs_per_row,
    int    warps_per_row,
    int    rows_per_iter,
    int    num_k_tiles,
    int    total_chunks)
{
    const int flat_tid    = static_cast<int>(threadIdx.x);
    const int warp_id     = flat_tid >> 5;
    const int lane_id     = flat_tid & 31;
    const int local_row   = warp_id / warps_per_row;
    const int warp_in_row = warp_id % warps_per_row;
    const int tid         = warp_in_row * 32 + lane_id;

    const int n_inner = CHUNK_M / rows_per_iter;

    __shared__ int     s_row_base_in_group;
    __shared__ int64_t s_sf_base;

    for (int chunk_idx = blockIdx.x; chunk_idx < total_chunks; chunk_idx += gridDim.x) {
        const int chunk_row_start = chunk_idx * CHUNK_M;

        if (flat_tid == 0) {
            int lo = 0;
            int hi = num_groups - 1;
            while (lo < hi) {
                const int mid = (lo + hi + 1) >> 1;
                if (row_offsets[mid] <= chunk_row_start) lo = mid;
                else                                     hi = mid - 1;
            }
            s_row_base_in_group = chunk_row_start - row_offsets[lo];
            s_sf_base           = sf_offsets[lo];
        }
        __syncthreads();

        const int     row_base_in_group = s_row_base_in_group;
        const int64_t sf_base           = s_sf_base;

        #pragma unroll 1
        for (int inner = 0; inner < n_inner; ++inner) {
            const int row_in_chunk       = inner * rows_per_iter + local_row;
            const int row                = chunk_row_start + row_in_chunk;
            const int local_row_in_group = row_base_in_group + row_in_chunk;
            const bool valid_vec         = (tid < nvecs_per_row);

            bf16x8 x{};
            float  local_amax = 0.0f;
            if (valid_vec) {
                x = bf16x8::load(x_ptr + 8 * ((int64_t)row * nvecs_per_row + tid));
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    local_amax = fmaxf(local_amax, fabsf(__bfloat162float(x[i])));
            }

            float blk_amax = local_amax;
            blk_amax = fmaxf(blk_amax, __shfl_xor_sync(0xffffffff, blk_amax, 1));
            blk_amax = fmaxf(blk_amax, __shfl_xor_sync(0xffffffff, blk_amax, 2));

            if (!valid_vec) continue;

            constexpr float max_norm_rcp = 1.0f / 448.0f;
            const float norm_amax = blk_amax * max_norm_rcp;
            const __nv_fp8x2_storage_t e8_pair =
                __nv_cvt_float2_to_e8m0x2(make_float2(norm_amax, norm_amax),
                                          __NV_SATFINITE, cudaRoundPosInf);
            const uint8_t e8m0_val  = static_cast<uint8_t>(e8_pair & 0xFFu);
            const float   scale_fp  = (e8m0_val == 0u) ? 1.0f : e8m0_to_scale_chunked(e8m0_val);
            const float   inv_scale = (scale_fp == 0.0f) ? 0.0f : 1.0f / scale_fp;

            GenericVector<uint8_t, 8> fp8_out;
            #pragma unroll
            for (int i = 0; i < 8; i += 2) {
                const float2 s2 = make_float2(__bfloat162float(x[i])     * inv_scale,
                                              __bfloat162float(x[i + 1]) * inv_scale);
                const __nv_fp8x2_storage_t p =
                    __nv_cvt_float2_to_fp8x2(s2, __NV_SATFINITE, __NV_E4M3);
                fp8_out[i]     = static_cast<uint8_t>(p & 0xFFu);
                fp8_out[i + 1] = static_cast<uint8_t>(p >> 8);
            }
            fp8_out.store(y_ptr + (int64_t)row * K + 8 * tid);

            if ((tid & 3) == 0) {
                const int     col    = tid >> 2;
                const int64_t sf_off = cvt_quant_to_fp4_get_sf_out_offset(local_row_in_group, col, num_k_tiles);
                sf_ptr[sf_base + sf_off] = e8m0_val;
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Launch planning — pick rows_per_iter that divides CHUNK_M and aims for
// ~1024 threads/block (same target as the per-row kernel).
// ---------------------------------------------------------------------------

struct LaunchPlan {
    int rows_per_iter;
    int block_dim;
    int num_k_tiles;
    int total_chunks;
    int grid_dim;
};

static LaunchPlan plan_launch_chunked(int K, int total_rows) {
    const int K_groups      = K / 32;
    const int nvecs_per_row = K / 8;
    const int warps_per_row = (nvecs_per_row + 31) / 32;
    // Match the per-row kernel's threads/block target (~1024) but constrain
    // rows_per_iter to divide CHUNK_M so the inner loop covers exactly CHUNK_M.
    int rows_per_iter = max(1, 1024 / (warps_per_row * 32));
    while (rows_per_iter > 1 && (CHUNK_M % rows_per_iter) != 0) --rows_per_iter;
    if (rows_per_iter > CHUNK_M) rows_per_iter = CHUNK_M;

    const int block_dim    = rows_per_iter * warps_per_row * 32;
    const int num_k_tiles  = (K_groups + 3) / 4;
    const int total_chunks = total_rows / CHUNK_M;
    const int grid_dim     = min(total_chunks, MAX_BLOCKS);

    LaunchPlan p{rows_per_iter, block_dim, num_k_tiles, total_chunks, grid_dim};
    return p;
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------

static void launch_mxfp8_group_quantize_chunked(
    uint8_t*           y_ptr,
    uint8_t*           sf_ptr,
    const nv_bfloat16* x_ptr,
    int                num_groups,
    int                K,
    int                total_rows,
    const int32_t*     row_offsets_gpu,
    const int64_t*     sf_offsets_gpu,
    cudaStream_t       stream)
{
    if (total_rows == 0) return;
    const LaunchPlan p = plan_launch_chunked(K, total_rows);
    const int nvecs_per_row = K / 8;
    const int warps_per_row = (nvecs_per_row + 31) / 32;

    mxfp8_group_quantize_chunked_kernel<<<p.grid_dim, p.block_dim, 0, stream>>>(
        y_ptr, sf_ptr, x_ptr,
        row_offsets_gpu, sf_offsets_gpu,
        K, num_groups, nvecs_per_row, warps_per_row,
        p.rows_per_iter, p.num_k_tiles, p.total_chunks);

    CUDA_CHECK(cudaGetLastError());
}

static void launch_mxfp8_group_quantize_chunked_hw(
    uint8_t*           y_ptr,
    uint8_t*           sf_ptr,
    const nv_bfloat16* x_ptr,
    int                num_groups,
    int                K,
    int                total_rows,
    const int32_t*     row_offsets_gpu,
    const int64_t*     sf_offsets_gpu,
    cudaStream_t       stream)
{
    if (total_rows == 0) return;
    const LaunchPlan p = plan_launch_chunked(K, total_rows);
    const int nvecs_per_row = K / 8;
    const int warps_per_row = (nvecs_per_row + 31) / 32;

    mxfp8_group_quantize_chunked_hw_kernel<<<p.grid_dim, p.block_dim, 0, stream>>>(
        y_ptr, sf_ptr, x_ptr,
        row_offsets_gpu, sf_offsets_gpu,
        K, num_groups, nvecs_per_row, warps_per_row,
        p.rows_per_iter, p.num_k_tiles, p.total_chunks);

    CUDA_CHECK(cudaGetLastError());
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// PyTorch bindings
// ---------------------------------------------------------------------------

void mxfp8_grouped_quantize_chunked_binding(
    at::Tensor&       out,
    at::Tensor&       sf,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& sf_offsets)
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
    TORCH_CHECK(total_rows % 128 == 0,
                "total_rows must be a multiple of 128 (each group's M must be padded to 128). "
                "Use mxfp8_grouped_quantize (per-row) if your groups aren't 128-aligned.");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(inp.device().index());
    launch_mxfp8_group_quantize_chunked(
        reinterpret_cast<uint8_t*>(out.data_ptr()),
        reinterpret_cast<uint8_t*>(sf.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(inp.data_ptr()),
        num_groups, K, total_rows,
        reinterpret_cast<const int32_t*>(row_offsets.data_ptr()),
        reinterpret_cast<const int64_t*>(sf_offsets.data_ptr()),
        stream);
}

void mxfp8_grouped_quantize_chunked_hw_binding(
    at::Tensor&       out,
    at::Tensor&       sf,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& sf_offsets)
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
    TORCH_CHECK(total_rows % 128 == 0,
                "total_rows must be a multiple of 128 (each group's M must be padded to 128). "
                "Use mxfp8_grouped_quantize_hw (per-row) if your groups aren't 128-aligned.");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(inp.device().index());
    launch_mxfp8_group_quantize_chunked_hw(
        reinterpret_cast<uint8_t*>(out.data_ptr()),
        reinterpret_cast<uint8_t*>(sf.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(inp.data_ptr()),
        num_groups, K, total_rows,
        reinterpret_cast<const int32_t*>(row_offsets.data_ptr()),
        reinterpret_cast<const int64_t*>(sf_offsets.data_ptr()),
        stream);
}
