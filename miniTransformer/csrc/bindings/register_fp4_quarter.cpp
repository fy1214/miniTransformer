#include "bindings/register_all.h"
#include "ops.h"

// Bindings for miniTransformer/ops/nvfp4_quartet_quantize.py (quarter / clover).
void register_fp4_quarter(py::module& m) {
    m.def("four_six_fp4", &four_six_fp4_binding, "Four-Six FP4 quantization (CUDA)");
    m.def("rtn_fp4", &rtn_fp4_binding, "RTN FP4 quantization (CUDA)");
    m.def("eden_fp4", &eden_fp4_binding, "Eden FP4 quantization (CUDA)");
    m.def("group_transform_128_eden", &group_transform_128_eden_binding, "Group Transform 128 Eden (CUDA)");
    m.def("dequant_tp_had_quant", &dequant_tp_had_quant_binding, "Dequant TP Had Quant (CUDA)");
    m.def("rht_128_eden", &rht_128_eden_binding, "rht 128 eden (CUDA)");
    m.def("rht128_requant", &rht128_requant_binding, "rht 128 requant (CUDA)");
}
