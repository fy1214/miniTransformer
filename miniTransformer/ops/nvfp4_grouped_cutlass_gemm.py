"""
CUTLASS SM100 FP4 Grouped GEMM with per-token output scale.

Formula per group i:
    D[i] = (A[i] * SFA[i]) @ (B[i] * SFB[i]).T   (FP32 accumulation)
    D[i][m, :] *= per_token_scale[i][m]            (post-GEMM scale)

Scale factor layout
-------------------
The CUTLASS kernel expects scale factors in the SM100 block-scaled layout,
which is the same 4D swizzled format used by the existing CUTLASS/cuBLAS
single-GEMM backends in this project: (MN//128, K//64, 32, 16), dtype=float8_e4m3fn.

If 2D (MN, K//16) scales are passed instead, set ``sf_is_swizzled=False``
and the conversion is done automatically.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import fp4_gemm
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _swizzle_scale(scale_2d: torch.Tensor, mn: int, k: int) -> torch.Tensor:
    """Convert 2D scale (MN, K//16) → 4D swizzled (MN//128, K//64, 32, 16)."""
    pad = (128 - mn % 128) % 128
    if pad:
        scale_2d = F.pad(scale_2d, (0, 0, 0, pad))
        mn_pad = mn + pad
    else:
        mn_pad = mn
    u = scale_2d.reshape(mn_pad // 128, 4, 32, k // 64, 4)
    return u.permute(0, 3, 2, 1, 4).reshape(mn_pad // 128, k // 64, 32, 16)


def _prepare_sf(sf: torch.Tensor, mn: int, k: int, is_swizzled: bool) -> torch.Tensor:
    """Ensure scale tensor is float8_e4m3fn and in 4D swizzled layout."""
    sf = sf.to(torch.float8_e4m3fn)
    if not is_swizzled:
        sf = _swizzle_scale(sf, mn, k)
    return sf.contiguous()


def _check_tensor(t: torch.Tensor, name: str, expected_dtype: torch.dtype) -> None:
    if not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor.")
    if t.dtype != expected_dtype:
        raise ValueError(f"{name} must be {expected_dtype}, got {t.dtype}.")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_NVFP4_FACTOR = 6.0 * 6.0 * 448.0 * 448.0


def grouped_nvfp4_cutlass_gemm_per_token_scale(
    a_list: List[torch.Tensor],
    b_list: List[torch.Tensor],
    a_sf_list: List[torch.Tensor],
    b_sf_list: List[torch.Tensor],
    per_token_scale_list: List[torch.Tensor],
    out_dtype: torch.dtype = torch.bfloat16,
    out_list: Optional[List[torch.Tensor]] = None,
    sf_is_swizzled: bool = True,
    tensor_scale_a_list: Optional[List[Optional[torch.Tensor]]] = None,
    tensor_scale_b_list: Optional[List[Optional[torch.Tensor]]] = None,
) -> List[torch.Tensor]:
    """CUTLASS SM100 FP4 Grouped GEMM (2SM) with per-token output scale."""
    num_groups = len(a_list)
    if num_groups == 0:
        return []
    if len(b_list) != num_groups:
        raise ValueError(f"b_list length {len(b_list)} != num_groups {num_groups}")
    if len(a_sf_list) != num_groups:
        raise ValueError(f"a_sf_list length {len(a_sf_list)} != num_groups {num_groups}")
    if len(b_sf_list) != num_groups:
        raise ValueError(f"b_sf_list length {len(b_sf_list)} != num_groups {num_groups}")
    if len(per_token_scale_list) != num_groups:
        raise ValueError(f"per_token_scale_list length {len(per_token_scale_list)} != num_groups {num_groups}")
    if out_list is not None and len(out_list) != num_groups:
        raise ValueError(f"out_list length {len(out_list)} != num_groups {num_groups}")

    d_list: List[torch.Tensor] = []
    a_sf_prepared: List[torch.Tensor] = []
    b_sf_prepared: List[torch.Tensor] = []
    scale_prepared: List[torch.Tensor] = []

    for i in range(num_groups):
        a = a_list[i]
        b = b_list[i]
        a_sf = a_sf_list[i]
        b_sf = b_sf_list[i]
        pts = per_token_scale_list[i]

        if a.dim() != 2:
            raise ValueError(f"group {i}: a must be 2D, got {a.dim()}D.")
        if b.dim() != 2:
            raise ValueError(f"group {i}: b must be 2D, got {b.dim()}D.")
        _check_tensor(a, f"a[{i}]", torch.uint8)
        _check_tensor(b, f"b[{i}]", torch.uint8)

        m, k_packed = a.shape
        n = b.shape[0]
        k = k_packed * 2

        if b.shape[1] != k_packed:
            raise ValueError(f"group {i}: K mismatch: a has K_packed={k_packed}, b has {b.shape[1]}.")
        if k % 64 != 0:
            raise ValueError(f"group {i}: K={k} must be divisible by 64.")
        if m % 128 != 0:
            raise ValueError(f"group {i}: M={m} must be divisible by 128.")
        if n % 128 != 0:
            raise ValueError(f"group {i}: N={n} must be divisible by 128.")

        if pts.shape != (m,):
            raise ValueError(f"group {i}: per_token_scale shape must be ({m},), got {tuple(pts.shape)}.")
        if pts.dtype != torch.float32:
            raise ValueError(f"group {i}: per_token_scale must be float32, got {pts.dtype}.")

        ts_a = tensor_scale_a_list[i] if tensor_scale_a_list is not None else None
        ts_b = tensor_scale_b_list[i] if tensor_scale_b_list is not None else None
        if ts_a is not None and ts_b is not None:
            global_alpha = (ts_a.float() * ts_b.float()).squeeze() / _NVFP4_FACTOR
            pts = pts * global_alpha

        if out_list is not None:
            out = out_list[i]
            if out.shape != (m, n) or out.dtype != out_dtype:
                raise ValueError(
                    f"group {i}: out shape/dtype mismatch. "
                    f"Expected ({m},{n}) {out_dtype}, got {tuple(out.shape)} {out.dtype}."
                )
            d_list.append(out)
        else:
            d_list.append(torch.empty((m, n), dtype=out_dtype, device=a.device))

        a_sf_prepared.append(_prepare_sf(a_sf, m, k, sf_is_swizzled))
        b_sf_prepared.append(_prepare_sf(b_sf, n, k, sf_is_swizzled))
        scale_prepared.append(pts.contiguous())

    fp4_gemm.grouped_cutlass_gemm_with_per_token_scale(
        d_list, a_list, b_list, a_sf_prepared, b_sf_prepared, scale_prepared,
    )
    return d_list


# ---------------------------------------------------------------------------
# v2: zero-overhead grouped GEMM with contiguous packed buffers
#
# Differences from v1 (grouped_nvfp4_cutlass_gemm_per_token_scale):
#   - Activations and weights are pre-packed into contiguous flat buffers by
#     group_quantize_nvfp4_pertoken_swizzled / group_quantize_nvfp4_tensor_global_packed.
#   - The GPU micro-kernel fills CUTLASS metadata on-device; no host-side
#     pointer-list construction per call.
#   - Per-token scale is pts_eff[i] = act_ts[i] * wgt_ts[group(i)] / _NVFP4_FACTOR,
#     applied row-wise inside the kernel.
#   - Persistent buffers (pointers, strides, layouts, workspace) are allocated
#     once per (num_groups, device) and reused across calls.
# ---------------------------------------------------------------------------

# Persistent buffers keyed by (max_groups, device).
_v2_buf_cache: dict = {}


def _get_v2_buffers(max_groups: int, device):
    """Return (or allocate) persistent CUTLASS v2 metadata buffers."""
    key = (max_groups, str(device))
    if key not in _v2_buf_cache:
        ptrs_b, strides_b, layouts_b = fp4_gemm.grouped_cutlass_gemm_v2_buffer_sizes(max_groups)
        _v2_buf_cache[key] = (
            torch.empty(ptrs_b,          dtype=torch.uint8, device=device),  # pointer table
            torch.empty(strides_b,       dtype=torch.uint8, device=device),  # stride table
            torch.empty(layouts_b,       dtype=torch.uint8, device=device),  # layout table
            torch.empty(max_groups * 3,  dtype=torch.int32, device=device),  # problem sizes (device)
            torch.empty(4 * 1024 * 1024, dtype=torch.uint8, device=device),  # workspace
        )
    return _v2_buf_cache[key]


@dataclass
class V2GemmInputs:
    """
    Pre-packed inputs for grouped_cutlass_gemm_v2.

    Built once per quantize call; fields map directly to C++ kernel arguments
    so no per-call tensor construction is needed at GEMM time.
    """
    # Activation side — produced by group_quantize_nvfp4_pertoken_swizzled
    act_data:      torch.Tensor   # [total_m, k//2] uint8,  all groups row-packed
    act_sf:        torch.Tensor   # flat fp8,                all groups concatenated
    act_ts:        torch.Tensor   # [total_m] float32,       per-row global scale
    a_rows:        torch.Tensor   # [ng+1] int32 device,     row offsets into act_data
    a_sf_offsets:  torch.Tensor   # [ng+1] int64 device,     element offsets into act_sf

    # Weight side — produced by group_quantize_nvfp4_tensor_global_packed
    wgt_data:      torch.Tensor   # [total_n, k//2] uint8,  all experts row-packed
    wgt_sf:        torch.Tensor   # flat fp8,                all experts packed
    wgt_ts:        torch.Tensor   # [num_experts] float32,   per-expert global scale
    b_rows:        torch.Tensor   # [ng+1] int32 device,     row offsets into wgt_data
    b_sf_offsets:  torch.Tensor   # [ng+1] int64 device,     element offsets into wgt_sf

    # Problem shape — m_g varies per call; n and k are fixed for a given weight
    prob_host:     torch.Tensor   # [ng*3] int32 CPU,        (m_g, n, k) per group
    m_splits:      List[int]


def grouped_cutlass_gemm_v2(inputs: V2GemmInputs, out_dtype: torch.dtype) -> torch.Tensor:
    """
    CUTLASS SM100 grouped FP4 GEMM (v2) with fused per-token output scale.

    Per-token effective scale is computed as:
        pts_eff[i] = act_ts[i] * wgt_ts[group(i)]

    where act_ts is per-row and wgt_ts is per-expert (expanded via repeat_interleave).
    Both act_ts and wgt_ts are already normalized (= amax / (448*6)), so no additional
    _NVFP4_FACTOR division is applied here.
    The kernel applies pts_eff row-wise to the raw GEMM output before writing D.
    """
    ng     = len(inputs.m_splits)
    device = inputs.act_data.device

    # Expand per-expert weight scale to per-row, then fold in activation scale.
    # act_ts = amax_row / (448*6), wgt_ts = amax_expert / (448*6) — both already normalized.
    ts_b    = torch.repeat_interleave(
        inputs.wgt_ts,
        torch.tensor(inputs.m_splits, device=device),
    )
    pts_eff = (inputs.act_ts * ts_b).contiguous()

    total_m = inputs.act_data.shape[0]
    n       = inputs.wgt_data.shape[0] // ng
    D       = torch.empty(total_m, n, dtype=out_dtype, device=device)

    buf_ptrs, buf_strides, buf_layouts, buf_psizes, workspace = _get_v2_buffers(ng, device)

    fp4_gemm.grouped_cutlass_gemm_v2(
        D,
        inputs.act_data, inputs.wgt_data,
        inputs.act_sf,   inputs.wgt_sf,
        inputs.a_rows,   inputs.b_rows,
        inputs.a_sf_offsets, inputs.b_sf_offsets,
        ng,
        buf_ptrs, buf_strides, buf_layouts, buf_psizes,
        inputs.prob_host, workspace, pts_eff,
    )
    return D
