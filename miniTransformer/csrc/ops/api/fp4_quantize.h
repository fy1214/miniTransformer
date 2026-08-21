#pragma once

#include <torch/all.h>

// Bindings used by miniTransformer/ops/nvfp4_quantize.py (project-owned NVFP4 path).

// [STABLE] NVFP4 quantize + optional transpose.
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

// [STABLE]
void four_six_fp4_pertoken_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_per_row,
    float scale_override
);

// [STABLE]
void rtn_fp4_pertoken_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_per_row,
    float scale_override
);

// [ALT]
void four_six_fp4_pertoken_linear_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_per_row,
    float scale_override
);

// [ALT]
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
// [ALT]
void four_six_fp4_pertoken_fused_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    float scale_override
);

// [ALT]
void rtn_fp4_pertoken_fused_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    float scale_override
);

// [ALT]
void four_six_fp4_pertoken_linear_fused_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    float scale_override
);

// [ALT]
void rtn_fp4_pertoken_linear_fused_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    float scale_override
);

// [STABLE]
void rtn_fp4_group_pertoken_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& scale_offsets,
    float scale_override
);

// [STABLE]
void four_six_fp4_group_pertoken_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& scale_offsets,
    float scale_override
);

// [BENCH] Legacy per-row scheduler (bench only — production path uses the chunked
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

// [BENCH]
void four_six_fp4_group_pertoken_legacy_per_row_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& scale_offsets,
    float scale_override
);

// [ALT]
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

// [ALT]
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

// [STABLE]
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

// [STABLE]
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

// [ALT] v2 grouped pertoken (zero-sync streaming, accepts pre-computed amax)
void rtn_fp4_group_pertoken_split_amax_binding(
    at::Tensor& out, at::Tensor& scales, at::Tensor& global_scale,
    const at::Tensor& inp, const at::Tensor& amax_per_row,
    const at::Tensor& row_offsets, const at::Tensor& scale_offsets,
    float scale_override);

// [ALT]
void four_six_fp4_group_pertoken_split_amax_binding(
    at::Tensor& out, at::Tensor& scales, at::Tensor& global_scale,
    const at::Tensor& inp, const at::Tensor& amax_per_row,
    const at::Tensor& row_offsets, const at::Tensor& scale_offsets,
    float scale_override);

// [UTIL] TE-style per-row absolute max (grid=num_rows, block=256)
void fp4_perrow_amax_te_binding(at::Tensor& amax_out, const at::Tensor& inp);
