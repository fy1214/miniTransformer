#include "bindings/register_all.h"
#include "ops.h"

// [STABLE] unless marked otherwise.
void register_fp4_gemm(py::module& m) {
    m.def("cublas_gemm", &cublas_gemm, "FP4 GEMM with cuBLASLt (CUDA)");
    m.def("block_scaled_matmul_nvfp4", &block_scaled_matmul_nvfp4, "Block-scaled matmul for NVFP4 (CUDA)");
    m.def("cutlass_gemm", &cutlass_gemm, "Block-scaled matmul for NVFP4 (CUDA)-cutlass api");
    // [ALT]
    m.def("grouped_cutlass_gemm_with_per_token_scale",
          &grouped_cutlass_gemm_with_per_token_scale,
          "FP4 grouped GEMM (CUTLASS SM100) + per-token output scale (CUDA)");
    m.def("grouped_cutlass_gemm_v2_buffer_sizes",
          &grouped_cutlass_gemm_v2_buffer_sizes,
          "Return (ptrs_bytes, strides_bytes, layouts_bytes) for pre-allocation");
    m.def("grouped_cutlass_gemm_v2",
          &grouped_cutlass_gemm_v2,
          "FP4 grouped GEMM v2: zero-overhead, pre-allocated buffers (CUDA)");
    // [ALT]
    m.def("grouped_cutlass_gemm_v2_evt_buffer_sizes",
          &grouped_cutlass_gemm_v2_evt_buffer_sizes,
          "Return (ptrs_bytes, strides_bytes, layouts_bytes) for EVT-fused v2 pre-allocation (7 ptrs/group)");
    m.def("grouped_cutlass_gemm_v2_evt",
          &grouped_cutlass_gemm_v2_evt,
          "FP4 grouped GEMM v2 with EVT-fused per-token scale (CUDA)");
}
