#pragma once

#include <optional>
#include <torch/library.h>
#include <torch/all.h>

void quantize_transpose_nvfp4(
    const at::Tensor &input, 
    at::Tensor *output,
    at::Tensor *output_scale,
    at::Tensor *output_transpose,
    at::Tensor *output_scale_transpose,
    bool use_2d_quantization,
    bool return_transpose
);

void cublas_gemm( 
  const at::Tensor A,
  const at::Tensor A_scale,
  const float *amax_A,
  bool transa,
  const at::Tensor B,
  const at::Tensor B_scale,
  const float *amax_B,
  bool transb,
  float alpha,
  std::optional<float> beta, 
  at::Tensor D,
  bool accumulate,
  bool use_split_accumulator,
  const at::Tensor workspace_tensor,
  size_t workspace_size,
);