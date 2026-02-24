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
    m.def("block_scaled_matmul_nvfp4", &block_scaled_matmul_nvfp4, "Block-scaled matmul for NVFP4 (CUDA)");
    m.def("cutlass_gemm", &cutlass_gemm, "Block-scaled matmul for NVFP4 (CUDA)-cutlass api");


    // quarter quantization
    m.def("four_six_fp4", &four_six_fp4_binding, "Four-Six FP4 quantization (CUDA)");
    m.def("rtn_fp4", &rtn_fp4_binding, "RTN FP4 quantization (CUDA)");
    m.def("eden_fp4", &eden_fp4_binding, "Eden FP4 quantization (CUDA)");
    m.def("group_transform_128_eden", &group_transform_128_eden_binding, "Group Transform 128 Eden (CUDA)");
    m.def("dequant_tp_had_quant", &dequant_tp_had_quant_binding, "Dequant TP Had Quant (CUDA)");
    m.def("rht_128_eden", &rht_128_eden_binding, "rht 128 eden (CUDA)");
    m.def("rht128_requant", &rht128_requant_binding, "rht 128 requant (CUDA)");
}
