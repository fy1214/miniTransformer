#include "bindings/register_all.h"
#include "ops.h"

// [STABLE]
void register_fp8_gemm(py::module& m) {
    m.def("mxfp8_grouped_gemm_buffer_sizes",
          &mxfp8_grouped_gemm_buffer_sizes,
          "Return (ptrs_bytes, strides_bytes, layouts_bytes) for MXFP8 grouped GEMM pre-allocation");
    m.def("mxfp8_grouped_gemm",
          &mxfp8_grouped_gemm,
          "MXFP8 grouped GEMM (CUTLASS SM100): zero-overhead, 1SM/2SM selectable (CUDA)");
}
