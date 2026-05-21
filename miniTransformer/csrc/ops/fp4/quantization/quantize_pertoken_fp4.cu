// Copyright (c) 2025-2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Per-token NV FP4 quantization (RTN and Round-4-or-6 variants).
// Computes one per-row global scale (shape [rows]) on-the-fly; no amax_ptr input.
// Grid: one block per row. blockDim.x = cols/8 (requires cols divisible by 128, cols <= 4096).
//
// Two scale-factor output layouts are provided via the Swizzle template parameter:
//   Swizzle=true  : tcgen05 MMA swizzled layout via cvt_quant_to_fp4_get_sf_out_offset
//   Swizzle=false : plain row-major layout  scale_ptr[row * (cols/16) + group_col]

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

// NumCandidates=1 : RTN (max FP4 value = 6 only)
// NumCandidates=2 : Round-4-or-6 (tries max=6 and max=4, picks lower L2 error)
// Swizzle=true    : scale_ptr written in tcgen05 swizzled layout
// Swizzle=false   : scale_ptr written in plain row-major layout
template<int NumCandidates, bool Swizzle>
__global__ void fp4_pertoken_kernel(
    __nv_fp4x4_e2m1* __restrict__ y_ptr,
    __nv_fp8_e4m3*   __restrict__ scale_ptr,
    float*           __restrict__ global_scale_ptr,
    const nv_bfloat16* __restrict__ x_ptr,
    float scale_override,
    int cols_div16)   // cols / 16
{
    extern __shared__ nv_bfloat16 smem[];   // blockDim.x bf16 slots

    const int row           = blockIdx.x;
    const int tid           = threadIdx.x;
    const int nvecs_per_row = blockDim.x;   // cols / 8

    bf16x8 x = bf16x8::load(x_ptr + 8 * (row * nvecs_per_row + tid));

    // ---- Step 1: per-row abs-max via shared-memory tree reduce ----
    nv_bfloat16 local_abs_max = vecReduceAbsMax(x);
    smem[tid] = local_abs_max;
    __syncthreads();

    for (int stride = nvecs_per_row / 2; stride >= 2; stride /= 2) {
        if (tid < stride) {
            nv_bfloat162 a = make_bfloat162(smem[tid],          smem[tid + 1]);
            nv_bfloat162 b = make_bfloat162(smem[tid + stride], smem[tid + stride + 1]);
            nv_bfloat162 m;
            __asm__ volatile(
                "max.xorsign.abs.bf16x2 %0, %1, %2;"
                : "=r"(reinterpret_cast<int&>(m))
                : "r"(reinterpret_cast<int&>(a)), "r"(reinterpret_cast<int&>(b))
            );
            smem[tid]     = m.x;
            smem[tid + 1] = m.y;
        }
        __syncthreads();
    }

    if (tid == 0) {
        nv_bfloat162 last = __habs2(make_bfloat162(smem[0], smem[1]));
        float row_abs_max = static_cast<float>(__hmax(last.x, last.y));

        constexpr float inv_scales_max = NumCandidates > 1 ? 1.f / 256.f : 1.f / 448.f;
        constexpr float one_over_six   = 1.f / 6.f;
        float scale = row_abs_max == 0.f
                    ? 1.f
                    : row_abs_max * inv_scales_max * (scale_override * one_over_six);
        global_scale_ptr[row] = scale;

        *reinterpret_cast<float*>(smem) = scale;
    }
    __syncthreads();
    float scale = *reinterpret_cast<float*>(smem);

    // ---- Step 2: per-group abs-max (16-element group spans 2 adjacent threads) ----
    nv_bfloat16 other_abs_max = __shfl_xor_sync(0xffffffff, local_abs_max, 1);
    float full_abs_max = static_cast<float>(__hmax(__habs(local_abs_max), __habs(other_abs_max)));

    // ---- Step 3: quantize ----
    float candidates[] = {6.f, 4.f};
    QuantResultPT results[NumCandidates];
    float scores[NumCandidates];
    float best = INFINITY;
    QuantResultPT res;
    for (int i = 0; i < NumCandidates; ++i) {
        float inv_val = 1.f / candidates[i];
        results[i]    = quantize_pt(full_abs_max, inv_val * scale_override, scale, x);
        scores[i]     = quant_error_pt(x, results[i], scale);
        if (scores[i] < best) {
            best = scores[i];
            res  = results[i];
        }
    }

    // ---- Step 4: store FP4 values ----
    int global_vec_idx = row * nvecs_per_row + tid;
    res.bits.store(reinterpret_cast<unsigned char*>(y_ptr) + 4 * global_vec_idx);

    // ---- Step 5: store per-group FP8 scale (even threads only) ----
    if (tid % 2 == 0) {
        int col = tid / 2;
        if constexpr (Swizzle) {
            auto tgt = cvt_quant_to_fp4_get_sf_out_offset(row, col, cols_div16 / 4);
            scale_ptr[tgt] = res.fp8s;
        } else {
            scale_ptr[row * cols_div16 + col] = res.fp8s;
        }
    }
}

// ---- Launch helpers ----

static void launch_fp4_pertoken(
    int num_candidates, bool swizzle,
    __nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr,
    const nv_bfloat16* x_ptr, float scale_override, long rows, long cols)
{
    int threads = static_cast<int>(cols / 8);
    size_t smem = static_cast<size_t>(threads) * sizeof(nv_bfloat16);
    int cdiv16  = static_cast<int>(cols / 16);

    if (num_candidates == 2 && swizzle)
        fp4_pertoken_kernel<2, true> <<<rows, threads, smem>>>(y_ptr, scale_ptr, global_scale_ptr, x_ptr, scale_override, cdiv16);
    else if (num_candidates == 2 && !swizzle)
        fp4_pertoken_kernel<2, false><<<rows, threads, smem>>>(y_ptr, scale_ptr, global_scale_ptr, x_ptr, scale_override, cdiv16);
    else if (num_candidates == 1 && swizzle)
        fp4_pertoken_kernel<1, true> <<<rows, threads, smem>>>(y_ptr, scale_ptr, global_scale_ptr, x_ptr, scale_override, cdiv16);
    else
        fp4_pertoken_kernel<1, false><<<rows, threads, smem>>>(y_ptr, scale_ptr, global_scale_ptr, x_ptr, scale_override, cdiv16);

    CUDA_CHECK(cudaGetLastError());
}

static void check_args(const char* name, long cols, long rows_out, long rows_in) {
    if (cols % 128 != 0)
        throw std::runtime_error(std::string(name) + ": cols must be divisible by 128");
    if (cols / 8 > 512)
        throw std::runtime_error(std::string(name) + ": cols must be <= 4096");
}

void four_six_fp4_pertoken(
    __nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr,
    const nv_bfloat16* x_ptr, float scale_override, long rows, long cols)
{
    check_args("four_six_fp4_pertoken", cols, rows, rows);
    launch_fp4_pertoken(2, true, y_ptr, scale_ptr, global_scale_ptr, x_ptr, scale_override, rows, cols);
}

void rtn_fp4_pertoken(
    __nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr,
    const nv_bfloat16* x_ptr, float scale_override, long rows, long cols)
{
    check_args("rtn_fp4_pertoken", cols, rows, rows);
    launch_fp4_pertoken(1, true, y_ptr, scale_ptr, global_scale_ptr, x_ptr, scale_override, rows, cols);
}

void four_six_fp4_pertoken_linear(
    __nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr,
    const nv_bfloat16* x_ptr, float scale_override, long rows, long cols)
{
    check_args("four_six_fp4_pertoken_linear", cols, rows, rows);
    launch_fp4_pertoken(2, false, y_ptr, scale_ptr, global_scale_ptr, x_ptr, scale_override, rows, cols);
}

void rtn_fp4_pertoken_linear(
    __nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr,
    const nv_bfloat16* x_ptr, float scale_override, long rows, long cols)
{
    check_args("rtn_fp4_pertoken_linear", cols, rows, rows);
    launch_fp4_pertoken(1, false, y_ptr, scale_ptr, global_scale_ptr, x_ptr, scale_override, rows, cols);
}

// ---- PyTorch bindings ----

static void binding_impl(
    int num_candidates, bool swizzle,
    at::Tensor& out, at::Tensor& scales, at::Tensor& global_scale,
    const at::Tensor& inp, float scale_override)
{
    TORCH_CHECK(out.size(0) == inp.size(0), "Output and input must have the same number of rows");
    TORCH_CHECK(inp.dtype() == at::kBFloat16, "Input must be bfloat16");
    TORCH_CHECK(global_scale.numel() == inp.size(0), "global_scale must have shape [rows]");

    int threads = static_cast<int>(inp.size(1) / 8);
    if (inp.size(1) % 128 != 0)
        throw std::runtime_error("cols must be divisible by 128");
    if (threads > 512)
        throw std::runtime_error("cols must be <= 4096");

    launch_fp4_pertoken(
        num_candidates, swizzle,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data_ptr()),
        reinterpret_cast<float*>(global_scale.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(inp.data_ptr()),
        scale_override, inp.size(0), inp.size(1));
}

void four_six_fp4_pertoken_binding(
    at::Tensor& out, at::Tensor& scales, at::Tensor& global_scale,
    const at::Tensor& inp, float scale_override)
{
    binding_impl(2, true, out, scales, global_scale, inp, scale_override);
}

void rtn_fp4_pertoken_binding(
    at::Tensor& out, at::Tensor& scales, at::Tensor& global_scale,
    const at::Tensor& inp, float scale_override)
{
    binding_impl(1, true, out, scales, global_scale, inp, scale_override);
}

void four_six_fp4_pertoken_linear_binding(
    at::Tensor& out, at::Tensor& scales, at::Tensor& global_scale,
    const at::Tensor& inp, float scale_override)
{
    binding_impl(2, false, out, scales, global_scale, inp, scale_override);
}

void rtn_fp4_pertoken_linear_binding(
    at::Tensor& out, at::Tensor& scales, at::Tensor& global_scale,
    const at::Tensor& inp, float scale_override)
{
    binding_impl(1, false, out, scales, global_scale, inp, scale_override);
}
