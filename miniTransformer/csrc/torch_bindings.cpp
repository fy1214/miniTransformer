#include "ops.h"
#include "ops/swizzle.h"
#include "bindings/register_all.h"

#include <torch/library.h>
#include <torch/version.h>
#include <torch/extension.h>

// A version of the TORCH_LIBRARY macro that expands the NAME, i.e. so NAME
// could be a macro instead of a literal token.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    register_utils(m);
    register_fp4_quantize(m);
    register_fp4_gemm(m);
    register_fp8_gemm(m);
    register_fp8_quantize(m);
    register_fp4_quarter(m);
}
