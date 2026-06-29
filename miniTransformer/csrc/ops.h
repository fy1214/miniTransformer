#pragma once

#include <optional>
#include <torch/library.h>
#include <torch/all.h>

void quantize_transpose_nvfp4(
    const at::Tensor &input,
    const at::Tensor &amax_rowwise,
    const at::Tensor &amax_colwise,
    at::Tensor &output,
    at::Tensor &output_scale,
    at::Tensor &output_transpose,
    at::Tensor &output_scale_transpose,
    bool use_2d_quantization,
    bool return_transpose,
    bool swizzle_scales = false
);

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

void cutlass_gemm(
  at::Tensor& D,
  const at::Tensor& A,
  const at::Tensor& B,
  const at::Tensor& A_sf,
  const at::Tensor& B_sf,
  const at::Tensor& alpha
);

void grouped_cutlass_gemm_with_per_token_scale(
    std::vector<at::Tensor>& D_list,
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& A_sf_list,
    const std::vector<at::Tensor>& B_sf_list,
    const std::vector<at::Tensor>& per_token_scale_list
);

std::vector<int64_t> grouped_cutlass_gemm_v2_buffer_sizes(int max_groups);

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

// EVT-fused variant of grouped_cutlass_gemm_v2: per-token scale multiply
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

std::vector<int64_t> mxfp8_grouped_gemm_buffer_sizes(int max_groups);

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


// MXFP8 grouped quantization (rowwise)
void mxfp8_grouped_quantize_binding(
    at::Tensor&       out,
    at::Tensor&       sf,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& sf_offsets
);

// MXFP8 grouped quantization (rowwise) — full hardware-cvt path (SM100):
// E8M0 via cvt.rp.ue8m0x2, FP8 via cvt.rn.satfinite.e4m3x2. Fast inference path.
void mxfp8_grouped_quantize_hw_binding(
    at::Tensor&       out,
    at::Tensor&       sf,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& sf_offsets
);

// MXFP8 grouped transpose quantization (colwise: output is [K, total_M])
// V2: uniform groups only — no GPU offset arrays, all offsets from scalars.
void mxfp8_grouped_transpose_quantize_binding(
    at::Tensor&       out_T,
    at::Tensor&       sf,
    const at::Tensor& inp,
    int               m_per_group,
    int               num_groups
);

// MXFP8 grouped quantization (rowwise) — per-CTA-chunk scheduling.
// Each CTA owns a 128-row chunk; binary-searches the group table once per
// chunk (broadcast via smem). Requires every group's M to be a multiple of
// 128 (Megatron's Fp8Padding satisfies this).
void mxfp8_grouped_quantize_chunked_binding(
    at::Tensor&       out,
    at::Tensor&       sf,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& sf_offsets
);

// MXFP8 grouped quantization (rowwise), chunked + hardware-cvt path (SM100).
void mxfp8_grouped_quantize_chunked_hw_binding(
    at::Tensor&       out,
    at::Tensor&       sf,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& sf_offsets
);

// quarter quantization
void four_six_fp4_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_ptr,
    float scale_override
);

void rtn_fp4_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_ptr,
    float scale_override
);

void four_six_fp4_pertoken_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_per_row,
    float scale_override
);

void rtn_fp4_pertoken_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_per_row,
    float scale_override
);

void four_six_fp4_pertoken_linear_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_per_row,
    float scale_override
);

void rtn_fp4_pertoken_linear_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_per_row,
    float scale_override
);

// Fused variants: compute per-row amax inside the kernel (no external amax needed).
// Supported for cols <= 8192. Falls back to two-pass for larger cols.
void four_six_fp4_pertoken_fused_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    float scale_override
);

void rtn_fp4_pertoken_fused_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    float scale_override
);

void four_six_fp4_pertoken_linear_fused_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    float scale_override
);

void rtn_fp4_pertoken_linear_fused_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    float scale_override
);

void rtn_fp4_group_pertoken_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& scale_offsets,
    float scale_override
);

void four_six_fp4_group_pertoken_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& scale_offsets,
    float scale_override
);

// Legacy per-row scheduler (bench only — production path uses the chunked
// scheduler exposed via the names above).
void rtn_fp4_group_pertoken_legacy_per_row_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& scale_offsets,
    float scale_override
);

void four_six_fp4_group_pertoken_legacy_per_row_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& scale_offsets,
    float scale_override
);

void rtn_fp4_group_pertoken_pad_fuse_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& padded_row_offsets,
    const at::Tensor& unpadded_row_offsets,
    const at::Tensor& scale_offsets,
    float scale_override
);

void four_six_fp4_group_pertoken_pad_fuse_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& padded_row_offsets,
    const at::Tensor& unpadded_row_offsets,
    const at::Tensor& scale_offsets,
    float scale_override
);

void rtn_fp4_group_tensor_global_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& scale_offsets,
    const at::Tensor& amax,
    float scale_override,
    bool swizzle
);

void four_six_fp4_group_tensor_global_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& scale_offsets,
    const at::Tensor& amax,
    float scale_override,
    bool swizzle
);

void eden_fp4_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_ptr,
    float scale_override,
    at::Tensor& seed
);

void group_transform_128_eden_binding(
    at::Tensor& y,
    at::Tensor& group_scales,
    at::Tensor& tensor_scale,
    at::Tensor& scratch,
    at::Tensor& max_scale,
    at::Tensor& h,
    at::Tensor& x,
    at::Tensor& seed,
    float fp4_max, float fp8_max, bool transpose);

void dequant_tp_had_quant_binding(
    at::Tensor& y,
    at::Tensor& group_scales,
    at::Tensor& tensor_scale,
    at::Tensor& scratch,
    at::Tensor& max_scale,
    at::Tensor& h,
    at::Tensor& x,
    at::Tensor& x_micro_scales,
    at::Tensor& x_global_scale,
    at::Tensor& seed,
    float fp4_max, float fp8_max);


void rht_128_eden_binding(
    at::Tensor& y,
    at::Tensor& group_scales,
    at::Tensor& tensor_scale,
    at::Tensor& scratch,
    at::Tensor& max_scale,
    at::Tensor& h,
    at::Tensor& x,
    at::Tensor& seed,
    float fp4_max, float fp8_max, bool transpose);


void rht128_requant_binding(
    at::Tensor& y,
    at::Tensor& group_scales,
    at::Tensor& tensor_scale,
    at::Tensor& scratch,
    at::Tensor& max_scale,
    at::Tensor& h,
    at::Tensor& x,
    at::Tensor& x_micro_scales,
    at::Tensor& x_global_scale,
    at::Tensor& seed,
    float fp4_max, float fp8_max);
// v2 grouped pertoken (zero-sync streaming, accepts pre-computed amax)
void rtn_fp4_group_pertoken_split_amax_binding(
    at::Tensor& out, at::Tensor& scales, at::Tensor& global_scale,
    const at::Tensor& inp, const at::Tensor& amax_per_row,
    const at::Tensor& row_offsets, const at::Tensor& scale_offsets,
    float scale_override);
void four_six_fp4_group_pertoken_split_amax_binding(
    at::Tensor& out, at::Tensor& scales, at::Tensor& global_scale,
    const at::Tensor& inp, const at::Tensor& amax_per_row,
    const at::Tensor& row_offsets, const at::Tensor& scale_offsets,
    float scale_override);
// TE-style per-row absolute max (grid=num_rows, block=256)
void fp4_perrow_amax_te_binding(at::Tensor& amax_out, const at::Tensor& inp);
