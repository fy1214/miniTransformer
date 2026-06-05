"""
CUTLASS SM100 MXFP8 Grouped GEMM (zero-overhead, v2).

Mirrors the structure of nvfp4_grouped_cutlass_gemm.py (v2 portion).

Differences from the FP4 v2:
  - No per-token / per-expert second-level scale (MXFP8 has no secondary scale).
  - Scale factors are E8M0 (uint8), one byte per 32 K-elements.
  - buf_psizes requires groups*3 int32 (groups × (M_g, N_g, K) problem sizes).
  - use_2sm flag selects 1SM or 2SM CUTLASS schedule.

Activations and weights are pre-packed into contiguous flat buffers by
``mxfp8_grouped_quantize``; the GPU micro-kernel fills CUTLASS metadata on-device
so no per-call host-side construction is needed.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import List, Optional

# 1SM 的 M-tile=128，2SM 的 M-tile=256。每组平均 M ≤ 128 时 2SM tile 利用率不足
# 一半，1SM 更快；超过则 2SM 吞吐占优。实测拐点在 128↔160 之间（M=128 时 1SM
# 快 ~18%，M=160 时 2SM 快 ~15%），故以 128 为界自动选择 schedule。
_AUTO_2SM_M_THRESHOLD = 128

import fp4_gemm as _fp4_gemm
from miniTransformer.ops.mxfp8_quantize import mxfp8_grouped_quantize


# ---------------------------------------------------------------------------
# Persistent buffers keyed by (max_groups, device_str).
# ---------------------------------------------------------------------------
_v2_buf_cache: dict = {}


def _get_v2_buffers(max_groups: int, device):
    """Return (or allocate) persistent CUTLASS MXFP8 metadata buffers."""
    key = (max_groups, str(device))
    if key not in _v2_buf_cache:
        ptrs_b, strides_b, layouts_b = _fp4_gemm.mxfp8_grouped_gemm_buffer_sizes(max_groups)
        _v2_buf_cache[key] = (
            torch.empty(ptrs_b,           dtype=torch.uint8, device=device),  # pointer table
            torch.empty(strides_b,        dtype=torch.uint8, device=device),  # stride table
            torch.empty(layouts_b,        dtype=torch.uint8, device=device),  # layout table
            torch.empty(max_groups * 3,   dtype=torch.int32, device=device),  # problem_sizes (M_g, N_g, K) per group
            torch.empty(4 * 1024 * 1024,  dtype=torch.uint8, device=device),  # workspace
        )
    return _v2_buf_cache[key]


# ---------------------------------------------------------------------------
# Input container
# ---------------------------------------------------------------------------

@dataclass
class MxFP8V2GemmInputs:
    """
    Pre-packed inputs for mxfp8_grouped_gemm_v2.

    Built once per quantize call; fields map directly to C++ kernel arguments
    so no per-call tensor construction is needed at GEMM time.

    Unlike the FP4 v2 container, there are no act_ts / wgt_ts fields because
    MXFP8 does not use a second-level (tensor-global) scale.
    """
    # Activation side — produced by mxfp8_grouped_quantize
    act_data:     torch.Tensor   # [total_M, K] uint8  (FP8 E4M3), all groups row-packed
    act_sf:       torch.Tensor   # flat uint8           (E8M0),     all groups concatenated
    a_rows:       torch.Tensor   # [ng+1] int32 device, row offsets into act_data / D
    a_sf_offsets: torch.Tensor   # [ng+1] int64 device, element offsets into act_sf

    # Weight side — produced by mxfp8_grouped_quantize
    wgt_data:     torch.Tensor   # [total_N, K] uint8  (FP8 E4M3), all experts row-packed
    wgt_sf:       torch.Tensor   # flat uint8           (E8M0),     all experts packed
    b_rows:       torch.Tensor   # [ng+1] int32 device, row offsets into wgt_data
    b_sf_offsets: torch.Tensor   # [ng+1] int64 device, element offsets into wgt_sf

    # Problem shape — m_g varies per call; n and k are fixed for a given weight
    prob_host:    torch.Tensor   # [ng*3] int32 CPU, (M_g, N_g, K) per group


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mxfp8_grouped_gemm_v2(
    inputs: MxFP8V2GemmInputs,
    out_dtype: torch.dtype = torch.bfloat16,
    use_2sm: Optional[bool] = None,
) -> torch.Tensor:
    """
    CUTLASS SM100 MXFP8 grouped GEMM (v2, zero-overhead).

    Args:
        inputs:   Pre-packed activation + weight tensors (see MxFP8V2GemmInputs).
        out_dtype: Output dtype, default bfloat16.
        use_2sm:  True → 2SM schedule (higher throughput, larger M);
                  False → 1SM schedule (lower latency, small batch);
                  None → 自动按每组平均 M 选择（avg_M > 128 用 2SM，否则 1SM）。

    Returns:
        D : [total_M, N] out_dtype, where N = wgt_data.size(0) // ng.
    """
    ng      = inputs.a_rows.shape[0] - 1
    device  = inputs.act_data.device
    total_m = inputs.act_data.shape[0]
    n       = inputs.wgt_data.shape[0] // ng

    if use_2sm is None:
        avg_m = total_m / ng
        use_2sm = avg_m > _AUTO_2SM_M_THRESHOLD

    D = torch.empty(total_m, n, dtype=out_dtype, device=device)

    buf_ptrs, buf_strides, buf_layouts, buf_psizes, workspace = _get_v2_buffers(ng, device)

    _fp4_gemm.mxfp8_grouped_gemm(
        D,
        inputs.act_data,  inputs.wgt_data,
        inputs.act_sf,    inputs.wgt_sf,
        inputs.a_rows,    inputs.b_rows,
        inputs.a_sf_offsets, inputs.b_sf_offsets,
        ng,
        buf_ptrs, buf_strides, buf_layouts, buf_psizes,
        inputs.prob_host, workspace,
        use_2sm,
    )
    return D
