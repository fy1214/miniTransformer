#pragma once

#include <optional>
#include <vector>
#include <torch/all.h>

// [STABLE]
void cublas_gemm(
  const at::Tensor A,
  const at::Tensor A_scale,
  float amax_A,
  bool transa,
  const at::Tensor B,
  const at::Tensor B_scale,
  float amax_B,
  bool transb,
  float alpha,
  std::optional<float> beta,
  at::Tensor D,
  bool accumulate,
  const at::Tensor workspace_tensor,
  size_t workspace_size
);

// [STABLE]
void block_scaled_matmul_nvfp4(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& output,
    const at::Tensor& scale_A,
    const at::Tensor& scale_B,
    float amax_A,
    float amax_B,
    at::Tensor& alpha,
    bool accumulate,
    const at::Tensor& workspace_tensor,
    size_t workspace_size);

// [STABLE]
void cutlass_gemm(
  at::Tensor& D,
  const at::Tensor& A,
  const at::Tensor& B,
  const at::Tensor& A_sf,
  const at::Tensor& B_sf,
  const at::Tensor& alpha
);

// [ALT] Grouped CUTLASS v1 (per-call allocations).
void grouped_cutlass_gemm_with_per_token_scale(
    std::vector<at::Tensor>& D_list,
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& A_sf_list,
    const std::vector<at::Tensor>& B_sf_list,
    const std::vector<at::Tensor>& per_token_scale_list
);

// [STABLE]
std::vector<int64_t> grouped_cutlass_gemm_v2_buffer_sizes(int max_groups);

// [STABLE] Zero-overhead grouped FP4 GEMM with optional post-GEMM per-token scale.
void grouped_cutlass_gemm_v2(
    at::Tensor&       D,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& A_sf,
    const at::Tensor& B_sf,
    const at::Tensor& a_row_offsets,
    const at::Tensor& b_row_offsets,
    const at::Tensor& a_sf_offsets,
    const at::Tensor& b_sf_offsets,
    int               num_groups,
    at::Tensor&       buf_ptrs,
    at::Tensor&       buf_strides,
    at::Tensor&       buf_layouts,
    at::Tensor&       buf_problem_sizes_dev,
    const at::Tensor& problem_sizes_host,
    at::Tensor&       workspace,
    const at::Tensor& per_token_scale
);

// [ALT] EVT-fused variant of grouped_cutlass_gemm_v2: per-token scale multiply
// folded into the CUTLASS epilogue via Sm90ColBroadcast.  When per_token_scale
// is undefined/empty, the ColBroadcast EnableNullptr path falls back to a unit
// scale (so this entry point also serves the no-scale case).
std::vector<int64_t> grouped_cutlass_gemm_v2_evt_buffer_sizes(int max_groups);

void grouped_cutlass_gemm_v2_evt(
    at::Tensor&       D,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& A_sf,
    const at::Tensor& B_sf,
    const at::Tensor& a_row_offsets,
    const at::Tensor& b_row_offsets,
    const at::Tensor& a_sf_offsets,
    const at::Tensor& b_sf_offsets,
    int               num_groups,
    at::Tensor&       buf_ptrs,
    at::Tensor&       buf_strides,
    at::Tensor&       buf_layouts,
    at::Tensor&       buf_problem_sizes_dev,
    const at::Tensor& problem_sizes_host,
    at::Tensor&       workspace,
    const at::Tensor& per_token_scale
);
