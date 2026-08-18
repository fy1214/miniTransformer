#pragma once

#include <torch/all.h>

// [STABLE] MXFP8 grouped quantization (rowwise)
void mxfp8_grouped_quantize_binding(
    at::Tensor&       out,
    at::Tensor&       sf,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& sf_offsets
);

// [ALT] MXFP8 grouped quantization (rowwise) — full hardware-cvt path (SM100):
// E8M0 via cvt.rp.ue8m0x2, FP8 via cvt.rn.satfinite.e4m3x2. Fast inference path.
void mxfp8_grouped_quantize_hw_binding(
    at::Tensor&       out,
    at::Tensor&       sf,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& sf_offsets
);

// [STABLE] MXFP8 grouped transpose quantization (colwise: output is [K, total_M])
// V2: uniform groups only — no GPU offset arrays, all offsets from scalars.
void mxfp8_grouped_transpose_quantize_binding(
    at::Tensor&       out_T,
    at::Tensor&       sf,
    const at::Tensor& inp,
    int               m_per_group,
    int               num_groups
);

// [ALT] MXFP8 grouped quantization (rowwise) — per-CTA-chunk scheduling.
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

// [ALT] MXFP8 grouped quantization (rowwise), chunked + hardware-cvt path (SM100).
void mxfp8_grouped_quantize_chunked_hw_binding(
    at::Tensor&       out,
    at::Tensor&       sf,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& sf_offsets
);
