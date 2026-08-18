#pragma once

#include <vector>
#include <torch/all.h>

// [STABLE]
std::vector<int64_t> mxfp8_grouped_gemm_buffer_sizes(int max_groups);

// [STABLE] MXFP8 grouped GEMM (CUTLASS SM100): zero-overhead, 1SM/2SM selectable.
void mxfp8_grouped_gemm(
    at::Tensor&        D,
    const at::Tensor&  A,
    const at::Tensor&  B,
    const at::Tensor&  A_sf,
    const at::Tensor&  B_sf,
    const at::Tensor&  a_row_offsets,
    const at::Tensor&  b_row_offsets,
    const at::Tensor&  a_sf_offsets,
    const at::Tensor&  b_sf_offsets,
    int                num_groups,
    at::Tensor&        buf_ptrs,
    at::Tensor&        buf_strides,
    at::Tensor&        buf_layouts,
    at::Tensor&        buf_problem_sizes_dev,
    const at::Tensor&  problem_sizes_host,
    at::Tensor&        workspace,
    bool               use_2sm
);
