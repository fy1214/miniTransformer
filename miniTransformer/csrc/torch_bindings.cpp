#include "ops.h"
#include "ops/swizzle.h"

#include <torch/library.h>
#include <torch/version.h>
#include <torch/extension.h>

// A version of the TORCH_LIBRARY macro that expands the NAME, i.e. so NAME
// could be a macro instead of a literal token.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_transpose_nvfp4", &quantize_transpose_nvfp4, "quantize and transpose (CUDA)");
    m.def("swizzle_scaling_factors", &swizzle_scaling_factors, "Swizzle scaling factors (CUDA)");
    m.def("cublas_gemm", &cublas_gemm, "FP4 GEMM with cuBLASLt (CUDA)");
}
