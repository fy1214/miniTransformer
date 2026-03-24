#pragma once

#include <optional>
#include <torch/library.h>
#include <torch/all.h>

void quantize_transpose_nvfp4(
    const at::Tensor &input,
    const at::Tensor &amax_rowwise,
    const at::Tensor &amax_colwise,
    at::Tensor &output,
    at::Tensor &output_scale,
    at::Tensor &output_transpose,
    at::Tensor &output_scale_transpose,
    bool use_2d_quantization,
    bool return_transpose
);

void cublas_gemm( 
  const at::Tensor A,
  const at::Tensor A_scale,
  float amax_A,
  bool transa,
  const at::Tensor B,
  const at::Tensor B_scale,
  float amax_B,
  bool transb,
  float alpha,
  std::optional<float> beta, 
  at::Tensor D,
  bool accumulate,
  const at::Tensor workspace_tensor,
  size_t workspace_size
);

void block_scaled_matmul_nvfp4(
    const at::Tensor A,
    const at::Tensor B,
    at::Tensor output,
    const at::Tensor scale_A,
    const at::Tensor scale_B,
    float amax_A,
    float amax_B,
    float alpha,
    bool accumulate,
    const at::Tensor workspace_tensor,
    size_t workspace_size);

void cutlass_gemm(
  at::Tensor& D,
  const at::Tensor& A,
  const at::Tensor& B,
  const at::Tensor& A_sf,
  const at::Tensor& B_sf,
  const at::Tensor& alpha
);


// quarter quantization
void four_six_fp4_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_ptr,
    float scale_override
);

void rtn_fp4_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_ptr,
    float scale_override
);

void eden_fp4_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_ptr,
    float scale_override,
    at::Tensor& seed
);

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