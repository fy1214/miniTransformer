#include "ops.h"
#include "ops/swizzle.h"

#include <torch/library.h>
#include <torch/version.h>
#include <torch/extension.h>

// A version of the TORCH_LIBRARY macro that expands the NAME, i.e. so NAME
// could be a macro instead of a literal token.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // py::arg enables keyword arguments from Python (plain &func only accepts positional).
    m.def(
        "quantize_transpose_nvfp4",
        &quantize_transpose_nvfp4,
        pybind11::arg("input"),
        pybind11::arg("amax_rowwise"),
        pybind11::arg("amax_colwise"),
        pybind11::arg("output"),
        pybind11::arg("output_scale"),
        pybind11::arg("output_transpose"),
        pybind11::arg("output_scale_transpose"),
        pybind11::arg("use_2d_quantization"),
        pybind11::arg("return_transpose"),
        "quantize and transpose (CUDA)");
    m.def("swizzle_scaling_factors", &swizzle_scaling_factors, "Swizzle scaling factors (CUDA)");
    m.def("cublas_gemm", &cublas_gemm, "FP4 GEMM with cuBLASLt (CUDA)");
    m.def("block_scaled_matmul_nvfp4", &block_scaled_matmul_nvfp4, "Block-scaled matmul for NVFP4 (CUDA)");
    m.def("cutlass_gemm", &cutlass_gemm, "Block-scaled matmul for NVFP4 (CUDA)-cutlass api");
    m.def("grouped_cutlass_gemm_with_per_token_scale",
          &grouped_cutlass_gemm_with_per_token_scale,
          "FP4 grouped GEMM (CUTLASS SM100) + per-token output scale (CUDA)");
    m.def("grouped_cutlass_gemm_v2_buffer_sizes",
          &grouped_cutlass_gemm_v2_buffer_sizes,
          "Return (ptrs_bytes, strides_bytes, layouts_bytes) for pre-allocation");
    m.def("grouped_cutlass_gemm_v2",
          &grouped_cutlass_gemm_v2,
          "FP4 grouped GEMM v2: zero-overhead, pre-allocated buffers (CUDA)");
    m.def("mxfp8_grouped_gemm_buffer_sizes",
          &mxfp8_grouped_gemm_buffer_sizes,
          "Return (ptrs_bytes, strides_bytes, layouts_bytes) for MXFP8 grouped GEMM pre-allocation");
    m.def("mxfp8_grouped_gemm",
          &mxfp8_grouped_gemm,
          "MXFP8 grouped GEMM (CUTLASS SM100): zero-overhead, 1SM/2SM selectable (CUDA)");
    m.def("mxfp8_grouped_quantize",
          &mxfp8_grouped_quantize_binding,
          pybind11::arg("out"), pybind11::arg("sf"),
          pybind11::arg("inp"), pybind11::arg("row_offsets"), pybind11::arg("sf_offsets"),
          "MXFP8 grouped rowwise quantization (BF16→FP8 E4M3 + E8M0 SF, SM100 swizzled layout)");
    m.def("mxfp8_grouped_transpose_quantize",
          &mxfp8_grouped_transpose_quantize_binding,
          pybind11::arg("out_T"), pybind11::arg("sf"),
          pybind11::arg("inp"), pybind11::arg("row_offsets"), pybind11::arg("sf_offsets"),
          "MXFP8 grouped colwise (transpose) quantization (BF16→[K,M] FP8 E4M3 + E8M0 SF)");


    // quarter quantization
    m.def("four_six_fp4", &four_six_fp4_binding, "Four-Six FP4 quantization (CUDA)");
    m.def("rtn_fp4", &rtn_fp4_binding, "RTN FP4 quantization (CUDA)");
    // two-pass variants (caller supplies amax_per_row)
    m.def("four_six_fp4_pertoken", &four_six_fp4_pertoken_binding, "Four-Six FP4 per-token, swizzled SF, caller-supplied amax (CUDA)");
    m.def("rtn_fp4_pertoken", &rtn_fp4_pertoken_binding, "RTN FP4 per-token, swizzled SF, caller-supplied amax (CUDA)");
    m.def("four_six_fp4_pertoken_linear", &four_six_fp4_pertoken_linear_binding, "Four-Six FP4 per-token, linear SF, caller-supplied amax (CUDA)");
    m.def("rtn_fp4_pertoken_linear", &rtn_fp4_pertoken_linear_binding, "RTN FP4 per-token, linear SF, caller-supplied amax (CUDA)");
    // fused variants (amax computed inside kernel, cols <= 8192)
    m.def("four_six_fp4_pertoken_fused", &four_six_fp4_pertoken_fused_binding, "Four-Six FP4 per-token fused amax+quant, swizzled SF (CUDA)");
    m.def("rtn_fp4_pertoken_fused", &rtn_fp4_pertoken_fused_binding, "RTN FP4 per-token fused amax+quant, swizzled SF (CUDA)");
    m.def("four_six_fp4_pertoken_linear_fused", &four_six_fp4_pertoken_linear_fused_binding, "Four-Six FP4 per-token fused amax+quant, linear SF (CUDA)");
    m.def("rtn_fp4_pertoken_linear_fused", &rtn_fp4_pertoken_linear_fused_binding, "RTN FP4 per-token fused amax+quant, linear SF (CUDA)");
    m.def("rtn_fp4_group_pertoken", &rtn_fp4_group_pertoken_binding,
          pybind11::arg("out"), pybind11::arg("scales"), pybind11::arg("global_scale"),
          pybind11::arg("inp"), pybind11::arg("row_offsets"), pybind11::arg("scale_offsets"),
          pybind11::arg("scale_override"),
          "RTN FP4 grouped per-token quantization with per-group swizzled SF layout (CUDA)");
    m.def("four_six_fp4_group_pertoken", &four_six_fp4_group_pertoken_binding,
          pybind11::arg("out"), pybind11::arg("scales"), pybind11::arg("global_scale"),
          pybind11::arg("inp"), pybind11::arg("row_offsets"), pybind11::arg("scale_offsets"),
          pybind11::arg("scale_override"),
          "Round-4-or-6 FP4 grouped per-token quantization with per-group swizzled SF layout (CUDA)");
    m.def("rtn_fp4_group_tensor_global", &rtn_fp4_group_tensor_global_binding,
          pybind11::arg("out"), pybind11::arg("scales"), pybind11::arg("global_scale"),
          pybind11::arg("inp"), pybind11::arg("row_offsets"), pybind11::arg("scale_offsets"),
          pybind11::arg("amax"), pybind11::arg("scale_override"),
          "RTN FP4 grouped tensor-global quantization with per-group swizzled SF layout (CUDA)");
    m.def("four_six_fp4_group_tensor_global", &four_six_fp4_group_tensor_global_binding,
          pybind11::arg("out"), pybind11::arg("scales"), pybind11::arg("global_scale"),
          pybind11::arg("inp"), pybind11::arg("row_offsets"), pybind11::arg("scale_offsets"),
          pybind11::arg("amax"), pybind11::arg("scale_override"),
          "Round-4-or-6 FP4 grouped tensor-global quantization with per-group swizzled SF layout (CUDA)");
    m.def("eden_fp4", &eden_fp4_binding, "Eden FP4 quantization (CUDA)");
    m.def("group_transform_128_eden", &group_transform_128_eden_binding, "Group Transform 128 Eden (CUDA)");
    m.def("dequant_tp_had_quant", &dequant_tp_had_quant_binding, "Dequant TP Had Quant (CUDA)");
    m.def("rht_128_eden", &rht_128_eden_binding, "rht 128 eden (CUDA)");
    m.def("rht128_requant", &rht128_requant_binding, "rht 128 requant (CUDA)");
}
