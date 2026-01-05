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