#include "bindings/register_all.h"
#include "ops.h"

void register_fp8_quantize(py::module& m) {
    // [STABLE]
    m.def("mxfp8_grouped_quantize",
          &mxfp8_grouped_quantize_binding,
          pybind11::arg("out"), pybind11::arg("sf"),
          pybind11::arg("inp"), pybind11::arg("row_offsets"), pybind11::arg("sf_offsets"),
          "MXFP8 grouped rowwise quantization (BF16→FP8 E4M3 + E8M0 SF, SM100 swizzled layout)");
    // [ALT]
    m.def("mxfp8_grouped_quantize_hw",
          &mxfp8_grouped_quantize_hw_binding,
          pybind11::arg("out"), pybind11::arg("sf"),
          pybind11::arg("inp"), pybind11::arg("row_offsets"), pybind11::arg("sf_offsets"),
          "MXFP8 grouped rowwise quantization, full hardware-cvt path (SM100 inference operator)");
    // [STABLE]
    m.def("mxfp8_grouped_transpose_quantize",
          &mxfp8_grouped_transpose_quantize_binding,
          pybind11::arg("out_T"), pybind11::arg("sf"),
          pybind11::arg("inp"), pybind11::arg("m_per_group"), pybind11::arg("num_groups"),
          "MXFP8 grouped colwise (transpose) quantization V2, uniform groups (BF16→[K,M] FP8 E4M3 + E8M0 SF)");
    // [ALT]
    m.def("mxfp8_grouped_quantize_chunked",
          &mxfp8_grouped_quantize_chunked_binding,
          pybind11::arg("out"), pybind11::arg("sf"),
          pybind11::arg("inp"), pybind11::arg("row_offsets"), pybind11::arg("sf_offsets"),
          "MXFP8 grouped rowwise quantization, per-CTA-chunk scheduling (requires 128-row group alignment)");
    // [ALT]
    m.def("mxfp8_grouped_quantize_chunked_hw",
          &mxfp8_grouped_quantize_chunked_hw_binding,
          pybind11::arg("out"), pybind11::arg("sf"),
          pybind11::arg("inp"), pybind11::arg("row_offsets"), pybind11::arg("sf_offsets"),
          "MXFP8 grouped rowwise quantization, per-CTA-chunk + hardware-cvt path (SM100 inference)");
}
