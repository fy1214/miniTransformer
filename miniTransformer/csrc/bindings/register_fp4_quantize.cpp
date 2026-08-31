#include "bindings/register_all.h"
#include "ops.h"

// Bindings for miniTransformer/ops/nvfp4_quantize.py (project-owned NVFP4 path).
void register_fp4_quantize(py::module& m) {
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
        pybind11::arg("swizzle_scales") = false,
        "quantize and transpose (CUDA)");
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
    // [BENCH]
    m.def("rtn_fp4_group_pertoken_legacy_per_row", &rtn_fp4_group_pertoken_legacy_per_row_binding,
          pybind11::arg("out"), pybind11::arg("scales"), pybind11::arg("global_scale"),
          pybind11::arg("inp"), pybind11::arg("row_offsets"), pybind11::arg("scale_offsets"),
          pybind11::arg("scale_override"),
          "RTN FP4 grouped per-token (legacy per-row scheduler, bench only)");
    m.def("four_six_fp4_group_pertoken_legacy_per_row", &four_six_fp4_group_pertoken_legacy_per_row_binding,
          pybind11::arg("out"), pybind11::arg("scales"), pybind11::arg("global_scale"),
          pybind11::arg("inp"), pybind11::arg("row_offsets"), pybind11::arg("scale_offsets"),
          pybind11::arg("scale_override"),
          "Round-4-or-6 FP4 grouped per-token (legacy per-row scheduler, bench only)");
    m.def("rtn_fp4_group_pertoken_pad_fuse", &rtn_fp4_group_pertoken_pad_fuse_binding,
          pybind11::arg("out"), pybind11::arg("scales"), pybind11::arg("global_scale"),
          pybind11::arg("inp"),
          pybind11::arg("padded_row_offsets"), pybind11::arg("unpadded_row_offsets"),
          pybind11::arg("scale_offsets"), pybind11::arg("scale_override"),
          "RTN FP4 grouped per-token quantization, per-CTA-chunk + fused pad (requires 128-row padded alignment)");
    m.def("four_six_fp4_group_pertoken_pad_fuse", &four_six_fp4_group_pertoken_pad_fuse_binding,
          pybind11::arg("out"), pybind11::arg("scales"), pybind11::arg("global_scale"),
          pybind11::arg("inp"),
          pybind11::arg("padded_row_offsets"), pybind11::arg("unpadded_row_offsets"),
          pybind11::arg("scale_offsets"), pybind11::arg("scale_override"),
          "Round-4-or-6 FP4 grouped per-token quantization, per-CTA-chunk + fused pad (requires 128-row padded alignment)");
    m.def("rtn_fp4_group_tensor_global", &rtn_fp4_group_tensor_global_binding,
          pybind11::arg("out"), pybind11::arg("scales"), pybind11::arg("global_scale"),
          pybind11::arg("inp"), pybind11::arg("row_offsets"), pybind11::arg("scale_offsets"),
          pybind11::arg("amax"), pybind11::arg("scale_override"),
          pybind11::arg("swizzle") = true,
          "RTN FP4 grouped tensor-global quantization. swizzle=True writes tcgen05-swizzled "
          "SF (default, consumed by CuTeDSL grouped GEMM); swizzle=False writes row-major "
          "SF [rows_padded, cols/16] (used by slime weight-update path).");
    m.def("four_six_fp4_group_tensor_global", &four_six_fp4_group_tensor_global_binding,
          pybind11::arg("out"), pybind11::arg("scales"), pybind11::arg("global_scale"),
          pybind11::arg("inp"), pybind11::arg("row_offsets"), pybind11::arg("scale_offsets"),
          pybind11::arg("amax"), pybind11::arg("scale_override"),
          pybind11::arg("swizzle") = true,
          "Round-4-or-6 FP4 grouped tensor-global quantization. swizzle=True is tcgen05-swizzled "
          "(default); swizzle=False is row-major SF [rows_padded, cols/16].");

    // v2 grouped pertoken (zero-sync streaming, accepts pre-computed amax)
    m.def("rtn_fp4_group_pertoken_split_amax", &rtn_fp4_group_pertoken_split_amax_binding,
          pybind11::arg("out"), pybind11::arg("scales"), pybind11::arg("global_scale"),
          pybind11::arg("inp"), pybind11::arg("amax_per_row"),
          pybind11::arg("row_offsets"), pybind11::arg("scale_offsets"),
          pybind11::arg("scale_override"),
          "RTN FP4 grouped per-token v2: zero-sync streaming, pre-computed amax (CUDA)");
    m.def("four_six_fp4_group_pertoken_split_amax", &four_six_fp4_group_pertoken_split_amax_binding,
          pybind11::arg("out"), pybind11::arg("scales"), pybind11::arg("global_scale"),
          pybind11::arg("inp"), pybind11::arg("amax_per_row"),
          pybind11::arg("row_offsets"), pybind11::arg("scale_offsets"),
          pybind11::arg("scale_override"),
          "Round-4-or-6 FP4 grouped per-token v2: zero-sync streaming, pre-computed amax (CUDA)");
    m.def("fp4_perrow_amax_te", &fp4_perrow_amax_te_binding,
          pybind11::arg("amax_out"), pybind11::arg("inp"),
          "Per-row absolute max (TE-style, grid=num_rows, block=256) (CUDA)");
}
