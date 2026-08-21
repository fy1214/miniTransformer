#pragma once

#include <torch/all.h>

// [UTIL] Scale-factor swizzle used by MXFP8 / NVFP4 GEMM consumers.
at::Tensor swizzle_scaling_factors(
    const at::Tensor& input,
    bool rowwise
);
