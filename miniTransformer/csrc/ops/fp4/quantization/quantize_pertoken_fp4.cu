// Copyright (c) 2025-2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Per-token NV FP4 quantization (RTN and Round-4-or-6 variants).
// amax_per_row: per-row abs_max supplied by the caller (shape [rows]).
// Grid: ceil(rows * cols/8 / 256) blocks, blockDim.x = 256 (fixed).
//
// Correctness of __shfl_xor_sync(mask, v, 1) across the flat layout:
//   cols % 128 == 0  =>  nvecs_per_row = cols/8 is divisible by 16
//   => row boundaries always land on even flat indices
//   => every (even, odd) thread pair produced by shfl_xor(1) is within the same row.
//
// Two scale-factor output layouts via the Swizzle template parameter:
//   Swizzle=true  : tcgen05 MMA swizzled layout via cvt_quant_to_fp4_get_sf_out_offset
//   Swizzle=false : plain row-major layout  scale_ptr[row * (cols/16) + group_col]

#ifndef MINI_TE_NO_BLACKWELL_OPS

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

// global_encode_scale = 2688.0f / row_abs_max  (matches golden's formula exactly)
// scale               = scale_override / global_encode_scale
__device__ __forceinline__ QuantResultPT quantize_pt(float abs_max, float val_max, float global_encode_scale, float scale, bf16x8& x) {
    __nv_fp8_e4m3 s_as_fp8  = static_cast<__nv_fp8_e4m3>(fminf(abs_max * (1.0f / val_max) * global_encode_scale, 448.0f));
    float s_round_fp8 = static_cast<float>(s_as_fp8);
    if (s_round_fp8 == 0.f) s_round_fp8 = 1.f;

    float factor   = 1.0f / (s_round_fp8 * scale);
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

// NumCandidates=1 : RTN
// NumCandidates=2 : Round-4-or-6 (tries max=6 and max=4, picks lower L2 error)
// Swizzle=true    : scale_ptr written in tcgen05 swizzled layout
// Swizzle=false   : scale_ptr written in plain row-major layout
template<int NumCandidates, bool Swizzle>
__global__ void fp4_pertoken_kernel(
    __nv_fp4x4_e2m1* __restrict__ y_ptr,
    __nv_fp8_e4m3*   __restrict__ scale_ptr,
    float*           __restrict__ global_scale_ptr,
    const nv_bfloat16* __restrict__ x_ptr,
    const float* __restrict__ amax_per_row,
    float scale_override,
    int nvecs_per_row,  // cols / 8
    int cols_div16,     // cols / 16
    int total_vecs)     // rows * cols / 8
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_vecs) return;

    const int row        = idx / nvecs_per_row;
    const int col_in_row = idx % nvecs_per_row;

    bf16x8 x = bf16x8::load(x_ptr + 8 * idx);

    // Token-level global scale from externally supplied per-row amax.
    float row_abs_max = amax_per_row[row];
    constexpr float SCALES_MAX = NumCandidates > 1 ? 256.0f : 448.0f;
    constexpr float FP4_MAX    = 6.0f;
    float global_encode_scale = row_abs_max == 0.f
                ? 1.f
                : (SCALES_MAX * FP4_MAX) / row_abs_max;
    float scale = row_abs_max == 0.f
                ? scale_override
                : scale_override / global_encode_scale;
    if (col_in_row == 0)
        global_scale_ptr[row] = scale;

    // Per-group abs-max: group = 16 elements = 2 consecutive threads.
    // nvecs_per_row is always divisible by 16, so row boundaries always land
    // on even flat indices — shfl_xor(1) pairs are always within the same row.
    nv_bfloat16 local_abs_max = vecReduceAbsMax(x);
    nv_bfloat16 other_abs_max = __shfl_xor_sync(0xffffffff, local_abs_max, 1);
    float full_abs_max = static_cast<float>(__hmax(__habs(local_abs_max), __habs(other_abs_max)));

    // Quantize
    float candidates[] = {6.f, 4.f};
    QuantResultPT results[NumCandidates];
    float scores[NumCandidates];
    float best = INFINITY;
    QuantResultPT res;
    for (int i = 0; i < NumCandidates; ++i) {
        results[i]    = quantize_pt(full_abs_max, candidates[i], global_encode_scale, scale, x);
        scores[i]     = quant_error_pt(x, results[i], scale);
        if (scores[i] < best) { best = scores[i]; res = results[i]; }
    }

    // Store FP4
    res.bits.store(reinterpret_cast<unsigned char*>(y_ptr) + 4 * idx);

    // Store per-group FP8 scale (even col_in_row only)
    if (col_in_row % 2 == 0) {
        int col = col_in_row / 2;
        if constexpr (Swizzle) {
            auto tgt = cvt_quant_to_fp4_get_sf_out_offset(row, col, cols_div16 / 4);
            scale_ptr[tgt] = res.fp8s;
        } else {
            scale_ptr[row * cols_div16 + col] = res.fp8s;
        }
    }
}

// ---- Launch helper ----

static void launch_fp4_pertoken(
    int num_candidates, bool swizzle,
    __nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr,
    const nv_bfloat16* x_ptr, const float* amax_per_row,
    float scale_override, long rows, long cols, cudaStream_t stream)
{
    int nvecs_per_row = static_cast<int>(cols / 8);
    int total_vecs    = static_cast<int>(rows) * nvecs_per_row;
    int cdiv16        = static_cast<int>(cols / 16);
    constexpr int BLOCK = 256;
    int n_blocks = (total_vecs + BLOCK - 1) / BLOCK;

    if (num_candidates == 2 && swizzle)
        fp4_pertoken_kernel<2, true> <<<n_blocks, BLOCK, 0, stream>>>(y_ptr, scale_ptr, global_scale_ptr, x_ptr, amax_per_row, scale_override, nvecs_per_row, cdiv16, total_vecs);
    else if (num_candidates == 2 && !swizzle)
        fp4_pertoken_kernel<2, false><<<n_blocks, BLOCK, 0, stream>>>(y_ptr, scale_ptr, global_scale_ptr, x_ptr, amax_per_row, scale_override, nvecs_per_row, cdiv16, total_vecs);
    else if (num_candidates == 1 && swizzle)
        fp4_pertoken_kernel<1, true> <<<n_blocks, BLOCK, 0, stream>>>(y_ptr, scale_ptr, global_scale_ptr, x_ptr, amax_per_row, scale_override, nvecs_per_row, cdiv16, total_vecs);
    else
        fp4_pertoken_kernel<1, false><<<n_blocks, BLOCK, 0, stream>>>(y_ptr, scale_ptr, global_scale_ptr, x_ptr, amax_per_row, scale_override, nvecs_per_row, cdiv16, total_vecs);

    CUDA_CHECK(cudaGetLastError());
}

static void check_args(const char* name, long cols) {
    if (cols % 128 != 0)
        throw std::runtime_error(std::string(name) + ": cols must be divisible by 128");
}

// ---- C entry points ----

void four_six_fp4_pertoken(
    __nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr,
    const nv_bfloat16* x_ptr, const float* amax_per_row, float scale_override, long rows, long cols)
{
    check_args("four_six_fp4_pertoken", cols);
    launch_fp4_pertoken(2, true, y_ptr, scale_ptr, global_scale_ptr, x_ptr, amax_per_row, scale_override, rows, cols, 0);
}

void rtn_fp4_pertoken(
    __nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr,
    const nv_bfloat16* x_ptr, const float* amax_per_row, float scale_override, long rows, long cols)
{
    check_args("rtn_fp4_pertoken", cols);
    launch_fp4_pertoken(1, true, y_ptr, scale_ptr, global_scale_ptr, x_ptr, amax_per_row, scale_override, rows, cols, 0);
}

void four_six_fp4_pertoken_linear(
    __nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr,
    const nv_bfloat16* x_ptr, const float* amax_per_row, float scale_override, long rows, long cols)
{
    check_args("four_six_fp4_pertoken_linear", cols);
    launch_fp4_pertoken(2, false, y_ptr, scale_ptr, global_scale_ptr, x_ptr, amax_per_row, scale_override, rows, cols, 0);
}

void rtn_fp4_pertoken_linear(
    __nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr,
    const nv_bfloat16* x_ptr, const float* amax_per_row, float scale_override, long rows, long cols)
{
    check_args("rtn_fp4_pertoken_linear", cols);
    launch_fp4_pertoken(1, false, y_ptr, scale_ptr, global_scale_ptr, x_ptr, amax_per_row, scale_override, rows, cols, 0);
}

// ---- PyTorch bindings ----

static void binding_impl(
    int num_candidates, bool swizzle,
    at::Tensor& out, at::Tensor& scales, at::Tensor& global_scale,
    const at::Tensor& inp, const at::Tensor& amax_per_row, float scale_override)
{
    TORCH_CHECK(out.size(0) == inp.size(0), "Output and input must have the same number of rows");
    TORCH_CHECK(inp.dtype() == at::kBFloat16, "Input must be bfloat16");
    TORCH_CHECK(amax_per_row.numel() == inp.size(0), "amax_per_row must have shape [rows]");
    TORCH_CHECK(amax_per_row.dtype() == at::kFloat, "amax_per_row must be float32");
    TORCH_CHECK(inp.size(1) % 128 == 0, "cols must be divisible by 128");

    launch_fp4_pertoken(
        num_candidates, swizzle,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data_ptr()),
        reinterpret_cast<float*>(global_scale.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(inp.data_ptr()),
        reinterpret_cast<const float*>(amax_per_row.data_ptr()),
        scale_override, inp.size(0), inp.size(1),
        at::cuda::getCurrentCUDAStream());
}

void four_six_fp4_pertoken_binding(
    at::Tensor& out, at::Tensor& scales, at::Tensor& global_scale,
    const at::Tensor& inp, const at::Tensor& amax_per_row, float scale_override)
{
    binding_impl(2, true, out, scales, global_scale, inp, amax_per_row, scale_override);
}

void rtn_fp4_pertoken_binding(
    at::Tensor& out, at::Tensor& scales, at::Tensor& global_scale,
    const at::Tensor& inp, const at::Tensor& amax_per_row, float scale_override)
{
    binding_impl(1, true, out, scales, global_scale, inp, amax_per_row, scale_override);
}

void four_six_fp4_pertoken_linear_binding(
    at::Tensor& out, at::Tensor& scales, at::Tensor& global_scale,
    const at::Tensor& inp, const at::Tensor& amax_per_row, float scale_override)
{
    binding_impl(2, false, out, scales, global_scale, inp, amax_per_row, scale_override);
}

void rtn_fp4_pertoken_linear_binding(
    at::Tensor& out, at::Tensor& scales, at::Tensor& global_scale,
    const at::Tensor& inp, const at::Tensor& amax_per_row, float scale_override)
{
    binding_impl(1, false, out, scales, global_scale, inp, amax_per_row, scale_override);
}

#else  // MINI_TE_NO_BLACKWELL_OPS — stubs for non-Blackwell builds

#include "../../../ops.h"
#include "../../../util/transformer_engine/logging.h"

void four_six_fp4_pertoken_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_per_row,
    float scale_override)
{
    NVTE_ERROR("four_six_fp4_pertoken_binding requires Blackwell+ (SM100). Rebuild on SM100 hardware or set MINI_TE_FORCE_BLACKWELL_OPS=1.");
}

void rtn_fp4_pertoken_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_per_row,
    float scale_override)
{
    NVTE_ERROR("rtn_fp4_pertoken_binding requires Blackwell+ (SM100). Rebuild on SM100 hardware or set MINI_TE_FORCE_BLACKWELL_OPS=1.");
}

void four_six_fp4_pertoken_linear_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_per_row,
    float scale_override)
{
    NVTE_ERROR("four_six_fp4_pertoken_linear_binding requires Blackwell+ (SM100). Rebuild on SM100 hardware or set MINI_TE_FORCE_BLACKWELL_OPS=1.");
}

void rtn_fp4_pertoken_linear_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_per_row,
    float scale_override)
{
    NVTE_ERROR("rtn_fp4_pertoken_linear_binding requires Blackwell+ (SM100). Rebuild on SM100 hardware or set MINI_TE_FORCE_BLACKWELL_OPS=1.");
}

#endif  // MINI_TE_NO_BLACKWELL_OPS
