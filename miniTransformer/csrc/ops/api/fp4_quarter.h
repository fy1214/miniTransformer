#pragma once

#include <torch/all.h>

// Bindings used by miniTransformer/ops/nvfp4_quartet_quantize.py (quarter / clover).

// [STABLE]
void four_six_fp4_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_ptr,
    float scale_override
);

// [STABLE]
void rtn_fp4_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_ptr,
    float scale_override
);

// [ALT]
void eden_fp4_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_ptr,
    float scale_override,
    at::Tensor& seed
);

// [ALT]
void group_transform_128_eden_binding(
    at::Tensor& y,
    at::Tensor& group_scales,
    at::Tensor& tensor_scale,
    at::Tensor& scratch,
    at::Tensor& max_scale,
    at::Tensor& h,
    at::Tensor& x,
    at::Tensor& seed,
    float fp4_max, float fp8_max, bool transpose);

// [ALT]
void dequant_tp_had_quant_binding(
    at::Tensor& y,
    at::Tensor& group_scales,
    at::Tensor& tensor_scale,
    at::Tensor& scratch,
    at::Tensor& max_scale,
    at::Tensor& h,
    at::Tensor& x,
    at::Tensor& x_micro_scales,
    at::Tensor& x_global_scale,
    at::Tensor& seed,
    float fp4_max, float fp8_max);

// [ALT]
void rht_128_eden_binding(
    at::Tensor& y,
    at::Tensor& group_scales,
    at::Tensor& tensor_scale,
    at::Tensor& scratch,
    at::Tensor& max_scale,
    at::Tensor& h,
    at::Tensor& x,
    at::Tensor& seed,
    float fp4_max, float fp8_max, bool transpose);

// [ALT]
void rht128_requant_binding(
    at::Tensor& y,
    at::Tensor& group_scales,
    at::Tensor& tensor_scale,
    at::Tensor& scratch,
    at::Tensor& max_scale,
    at::Tensor& h,
    at::Tensor& x,
    at::Tensor& x_micro_scales,
    at::Tensor& x_global_scale,
    at::Tensor& seed,
    float fp4_max, float fp8_max);
