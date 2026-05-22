// Copyright (c) 2025-2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Per-token NV FP4 quantization — fused amax + quantize kernel.
//
// Key idea: load bf16x8 once into registers, compute per-row abs_max via
// warp shfl reduce, then quantize from registers without a second global read.
//
// Grid/block assignment (auto-tuned to cols at runtime):
//
//   nvecs_per_row  = cols / 8
//   warps_per_row  = ceil(nvecs_per_row / 32)
//   rows_per_block = max(1, 1024 / (warps_per_row * 32))
//   blockDim       = rows_per_block * warps_per_row * 32
//   natural_grid   = ceil(total_rows / rows_per_block)
//   gridDim        = min(natural_grid, MAX_BLOCKS)          ← capped
//
// When rows is small (e.g. MoE experts with few tokens), natural_grid < MAX_BLOCKS
// and there is no difference. When rows is large, the grid-stride outer loop in
// the kernel lets each block process multiple row-batches, keeping all SMs busy.
//
// __syncthreads() safety in the outer loop:
//   Both syncs are called unconditionally every iteration. OOB threads write
//   0.f into smem before the first sync (safe for a max-reduce), then hit
//   `continue` after the second sync — they never touch output buffers.
//
// Shared memory layout: float[rows_per_block * warps_per_row]
//   smem[local_row * warps_per_row + warp_in_row] = warp partial abs_max
//   smem[local_row * warps_per_row + 0]           = final row abs_max (broadcast)
//
// Correctness of __shfl_xor_sync(mask, v, 1) for per-group abs_max:
//   cols % 128 == 0  =>  nvecs_per_row % 16 == 0
//   =>  nvecs_per_row % 32 is either 0 or 16
//   =>  OOB threads in the last warp always come in (even, odd) pairs
//   =>  __shfl_xor_sync(0xffffffff, v, 1) never crosses a row boundary.
//
// Limitation: cols > 8192 (warps_per_row > 32) is unsupported — falls back
// to the two-pass launcher in quantize_pertoken_fp4.cu.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdexcept>
#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "../../../util/quarter/utils.cuh"
#include "../../../util/quarter/vec.cuh"

using bf16x8 = GenericVector<nv_bfloat16, 8>;
using fp4x8  = GenericVector<unsigned char, 4>;

struct QuantResultPT {
    fp4x8 bits;
    float scale;
    __nv_fp8_e4m3 fp8s;
};

__device__ __forceinline__ QuantResultPT quantize_pt(float abs_max, float inv_val_max, float scale, bf16x8& x) {
    float s_group     = abs_max * inv_val_max;
    float inv_scale   = reciprocal_approximate_ftz(scale);
    __nv_fp8_e4m3 s_as_fp8  = static_cast<__nv_fp8_e4m3>(s_group * inv_scale);
    float s_round_fp8 = static_cast<float>(s_as_fp8);
    if (s_round_fp8 == 0.f) s_round_fp8 = 1.f;

    float factor   = reciprocal_approximate_ftz(s_round_fp8 * scale);
    float2 factor2 = {factor, factor};
    fp4x8 result;
    for (int k = 0; k < bf16x8::size; k += 2) {
        float2 src    = make_float2(static_cast<float>(x[k]), static_cast<float>(x[k+1]));
        float2 scaled = __fmul2_rn(src, factor2);
        result[k/2]   = __nv_cvt_float2_to_fp4x2(
            scaled, __nv_fp4_interpretation_t::__NV_E2M1, cudaRoundMode::cudaRoundNearest);
    }
    return QuantResultPT{result, s_round_fp8, s_as_fp8};
}

__forceinline__ __device__ float quant_error_pt(bf16x8 x, const QuantResultPT& q, float scale) {
    const float descale = static_cast<float>(q.fp8s) * scale;
    float2 sum = {0.f, 0.f};
    const float2 dsv = {-descale, -descale};
    for (int i = 0; i < 4; ++i) {
        float2 dq  = __nv_cvt_fp4x2_to_float2(q.bits[i]);
        float2 xv  = {static_cast<float>(x[2*i]), static_cast<float>(x[2*i+1])};
        float2 d   = __ffma2_rn(dq, dsv, xv);
        sum        = __ffma2_rn(d, d, sum);
    }
    float local_error = sum.x + sum.y;
    local_error += __shfl_xor_sync(0xffffffff, local_error, 1);
    return local_error;
}

// ---------------------------------------------------------------------------
// Fused kernel
// ---------------------------------------------------------------------------
// NumCandidates=1 : RTN (max FP4 value = 6)
// NumCandidates=2 : Round-4-or-6 (tries 6 and 4, keeps lower L2 error)
// Swizzle=true    : scale_ptr in tcgen05 swizzled layout
// Swizzle=false   : scale_ptr in plain row-major layout
template<int NumCandidates, bool Swizzle>
__global__ void fp4_pertoken_fused_kernel(
    __nv_fp4x4_e2m1* __restrict__ y_ptr,
    __nv_fp8_e4m3*   __restrict__ scale_ptr,
    float*           __restrict__ global_scale_ptr,
    const nv_bfloat16* __restrict__ x_ptr,
    float scale_override,
    int nvecs_per_row,    // cols / 8
    int cols_div16,       // cols / 16
    int rows_per_block,   // number of rows packed into one block
    int warps_per_row,    // ceil(nvecs_per_row / 32)
    int total_rows,
    int grid_stride)      // step between successive row-batches owned by this block
{
    // smem[local_row * warps_per_row + warp_in_row] = partial / final row abs_max
    extern __shared__ float smem[];

    const int flat_tid    = static_cast<int>(threadIdx.x);
    const int warp_id     = flat_tid / 32;
    const int lane_id     = flat_tid % 32;
    const int local_row   = warp_id / warps_per_row;
    const int warp_in_row = warp_id % warps_per_row;
    const int vec_in_row  = warp_in_row * 32 + lane_id;

    // Outer grid-stride loop: each block handles multiple row-batches.
    // base_row advances by grid_stride (= gridDim.x * rows_per_block) each iteration.
    for (int base_row = blockIdx.x * rows_per_block;
         base_row < total_rows;
         base_row += grid_stride)
    {
        const int global_row = base_row + local_row;
        const bool valid_row = global_row < total_rows;
        const bool valid_vec = valid_row && (vec_in_row < nvecs_per_row);

        // ---- Phase 1: load data into registers ----------------------------
        bf16x8 x{};
        nv_bfloat16 local_abs_max = __float2bfloat16(0.f);
        if (valid_vec) {
            x = bf16x8::load(x_ptr + 8 * (global_row * nvecs_per_row + vec_in_row));
            local_abs_max = vecReduceAbsMax(x);
        }

        // ---- Phase 2: warp-level abs_max reduce ---------------------------
        float warp_max = static_cast<float>(__habs(local_abs_max));
        for (int offset = 16; offset > 0; offset >>= 1)
            warp_max = fmaxf(warp_max, __shfl_down_sync(0xffffffff, warp_max, offset));

        // Lane 0 writes partial max; OOB threads write 0.f (safe for max-reduce).
        if (lane_id == 0)
            smem[local_row * warps_per_row + warp_in_row] = valid_row ? warp_max : 0.f;

        // __syncthreads() must be unconditional — all threads in the block must
        // reach it together regardless of valid_row / valid_vec.
        __syncthreads();

        // ---- Phase 3: inter-warp reduce (first warp of each row) ----------
        if (warp_in_row == 0 && valid_row) {
            float row_max = 0.f;
            for (int w = lane_id; w < warps_per_row; w += 32)
                row_max = fmaxf(row_max, smem[local_row * warps_per_row + w]);
            for (int offset = 16; offset > 0; offset >>= 1)
                row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, offset));
            if (lane_id == 0)
                smem[local_row * warps_per_row] = row_max;
        }
        __syncthreads();

        // OOB threads have nothing left to do this iteration.
        if (!valid_row) continue;

        // ---- Phase 4: compute per-row scale -------------------------------
        float row_abs_max = smem[local_row * warps_per_row];
        constexpr float inv_scales_max = NumCandidates > 1 ? 1.f / 256.f : 1.f / 448.f;
        constexpr float one_over_six   = 1.f / 6.f;
        float scale = row_abs_max == 0.f
                    ? 1.f
                    : row_abs_max * inv_scales_max * (scale_override * one_over_six);

        if (vec_in_row == 0)
            global_scale_ptr[global_row] = scale;

        if (!valid_vec) continue;

        // ---- Phase 5: per-group abs_max + quantize (x still in registers) -
        nv_bfloat16 other_abs_max = __shfl_xor_sync(0xffffffff, local_abs_max, 1);
        float full_abs_max = static_cast<float>(__hmax(__habs(local_abs_max), __habs(other_abs_max)));

        float candidates[] = {6.f, 4.f};
        QuantResultPT results[NumCandidates];
        float scores[NumCandidates];
        float best = INFINITY;
        QuantResultPT res;
        for (int i = 0; i < NumCandidates; ++i) {
            results[i] = quantize_pt(full_abs_max, (1.f / candidates[i]) * scale_override, scale, x);
            scores[i]  = quant_error_pt(x, results[i], scale);
            if (scores[i] < best) { best = scores[i]; res = results[i]; }
        }

        // ---- Phase 6: store outputs ----------------------------------------
        const int global_vec = global_row * nvecs_per_row + vec_in_row;
        res.bits.store(reinterpret_cast<unsigned char*>(y_ptr) + 4 * global_vec);

        if (vec_in_row % 2 == 0) {
            const int col = vec_in_row / 2;
            if constexpr (Swizzle) {
                auto tgt = cvt_quant_to_fp4_get_sf_out_offset(global_row, col, cols_div16 / 4);
                scale_ptr[tgt] = res.fp8s;
            } else {
                scale_ptr[global_row * cols_div16 + col] = res.fp8s;
            }
        }
    } // end grid-stride loop
}

// ---------------------------------------------------------------------------
// Launch helper — auto-tunes block shape to cols, caps grid with MAX_BLOCKS
// ---------------------------------------------------------------------------

// Target: keep all SMs busy even for small `rows`.
// H100 has 132 SMs; with 2 blocks/SM the soft saturation point is ~264.
// We use 1024 as a generous cap that covers most GPU generations.
static constexpr int MAX_BLOCKS = 1024;

static void launch_fp4_pertoken_fused(
    int num_candidates, bool swizzle,
    __nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr,
    const nv_bfloat16* x_ptr, float scale_override, long rows, long cols)
{
    const int nvecs_per_row  = static_cast<int>(cols / 8);
    const int warps_per_row  = (nvecs_per_row + 31) / 32;
    const int rows_per_block = max(1, 1024 / (warps_per_row * 32));
    const int block_dim      = rows_per_block * warps_per_row * 32;
    const int natural_grid   = (static_cast<int>(rows) + rows_per_block - 1) / rows_per_block;
    const int grid_dim       = min(natural_grid, MAX_BLOCKS);
    // grid_stride: how many rows the entire grid advances per outer-loop iteration.
    const int grid_stride    = grid_dim * rows_per_block;
    const size_t smem_bytes  = static_cast<size_t>(rows_per_block) * warps_per_row * sizeof(float);
    const int cdiv16         = static_cast<int>(cols / 16);

    if (num_candidates == 2 && swizzle)
        fp4_pertoken_fused_kernel<2, true> <<<grid_dim, block_dim, smem_bytes>>>(
            y_ptr, scale_ptr, global_scale_ptr, x_ptr,
            scale_override, nvecs_per_row, cdiv16, rows_per_block, warps_per_row, rows, grid_stride);
    else if (num_candidates == 2 && !swizzle)
        fp4_pertoken_fused_kernel<2, false><<<grid_dim, block_dim, smem_bytes>>>(
            y_ptr, scale_ptr, global_scale_ptr, x_ptr,
            scale_override, nvecs_per_row, cdiv16, rows_per_block, warps_per_row, rows, grid_stride);
    else if (num_candidates == 1 && swizzle)
        fp4_pertoken_fused_kernel<1, true> <<<grid_dim, block_dim, smem_bytes>>>(
            y_ptr, scale_ptr, global_scale_ptr, x_ptr,
            scale_override, nvecs_per_row, cdiv16, rows_per_block, warps_per_row, rows, grid_stride);
    else
        fp4_pertoken_fused_kernel<1, false><<<grid_dim, block_dim, smem_bytes>>>(
            y_ptr, scale_ptr, global_scale_ptr, x_ptr,
            scale_override, nvecs_per_row, cdiv16, rows_per_block, warps_per_row, rows, grid_stride);

    CUDA_CHECK(cudaGetLastError());
}

static void check_args_fused(const char* name, long cols) {
    if (cols % 128 != 0)
        throw std::runtime_error(std::string(name) + ": cols must be divisible by 128");
    // warps_per_row <= 32  =>  cols/8 <= 1024  =>  cols <= 8192
    if (cols > 8192)
        throw std::runtime_error(std::string(name) +
            ": cols > 8192 not supported by fused kernel; use the two-pass variant");
}

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

void four_six_fp4_pertoken_fused(
    __nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr,
    const nv_bfloat16* x_ptr, float scale_override, long rows, long cols)
{
    check_args_fused("four_six_fp4_pertoken_fused", cols);
    launch_fp4_pertoken_fused(2, true, y_ptr, scale_ptr, global_scale_ptr, x_ptr, scale_override, rows, cols);
}

void rtn_fp4_pertoken_fused(
    __nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr,
    const nv_bfloat16* x_ptr, float scale_override, long rows, long cols)
{
    check_args_fused("rtn_fp4_pertoken_fused", cols);
    launch_fp4_pertoken_fused(1, true, y_ptr, scale_ptr, global_scale_ptr, x_ptr, scale_override, rows, cols);
}

void four_six_fp4_pertoken_linear_fused(
    __nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr,
    const nv_bfloat16* x_ptr, float scale_override, long rows, long cols)
{
    check_args_fused("four_six_fp4_pertoken_linear_fused", cols);
    launch_fp4_pertoken_fused(2, false, y_ptr, scale_ptr, global_scale_ptr, x_ptr, scale_override, rows, cols);
}

void rtn_fp4_pertoken_linear_fused(
    __nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr,
    const nv_bfloat16* x_ptr, float scale_override, long rows, long cols)
{
    check_args_fused("rtn_fp4_pertoken_linear_fused", cols);
    launch_fp4_pertoken_fused(1, false, y_ptr, scale_ptr, global_scale_ptr, x_ptr, scale_override, rows, cols);
}

// ---------------------------------------------------------------------------
// PyTorch bindings
// ---------------------------------------------------------------------------

static void binding_impl_fused(
    int num_candidates, bool swizzle,
    at::Tensor& out, at::Tensor& scales, at::Tensor& global_scale,
    const at::Tensor& inp, float scale_override)
{
    TORCH_CHECK(inp.dtype() == at::kBFloat16, "Input must be bfloat16");
    TORCH_CHECK(inp.dim() == 2, "Input must be 2-D");
    TORCH_CHECK(out.size(0) == inp.size(0), "Output and input row count must match");
    TORCH_CHECK(global_scale.numel() == inp.size(0), "global_scale must have shape [rows]");
    TORCH_CHECK(inp.size(1) % 128 == 0, "cols must be divisible by 128");
    TORCH_CHECK(inp.size(1) <= 8192, "cols > 8192 not supported by fused kernel");

    launch_fp4_pertoken_fused(
        num_candidates, swizzle,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data_ptr()),
        reinterpret_cast<float*>(global_scale.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(inp.data_ptr()),
        scale_override, inp.size(0), inp.size(1));
}

void four_six_fp4_pertoken_fused_binding(
    at::Tensor& out, at::Tensor& scales, at::Tensor& global_scale,
    const at::Tensor& inp, float scale_override)
{
    binding_impl_fused(2, true, out, scales, global_scale, inp, scale_override);
}

void rtn_fp4_pertoken_fused_binding(
    at::Tensor& out, at::Tensor& scales, at::Tensor& global_scale,
    const at::Tensor& inp, float scale_override)
{
    binding_impl_fused(1, true, out, scales, global_scale, inp, scale_override);
}

void four_six_fp4_pertoken_linear_fused_binding(
    at::Tensor& out, at::Tensor& scales, at::Tensor& global_scale,
    const at::Tensor& inp, float scale_override)
{
    binding_impl_fused(2, false, out, scales, global_scale, inp, scale_override);
}

void rtn_fp4_pertoken_linear_fused_binding(
    at::Tensor& out, at::Tensor& scales, at::Tensor& global_scale,
    const at::Tensor& inp, float scale_override)
{
    binding_impl_fused(1, false, out, scales, global_scale, inp, scale_override);
}
