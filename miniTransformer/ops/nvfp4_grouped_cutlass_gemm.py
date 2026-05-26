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
    """CUTLASS SM100 FP4 Grouped GEMM with per-token output scale.

    The final per-element scale applied to output row m of group i is::

        final_scale[m] = per_token_scale[i][m] * global_alpha[i]

    where ``global_alpha[i] = amax_A[i] * amax_B[i] / (6*6*448*448)`` when
    ``tensor_scale_a_list`` and ``tensor_scale_b_list`` are provided, else 1.0.

    Args:
        a_list: Packed FP4 activation tensors, each (M_i, K//2) uint8, row-major.
        b_list: Packed FP4 weight tensors, each (N_i, K//2) uint8, col-major
                (i.e., already transposed so B layout is N×K).
        a_sf_list: Scale factors for A, each float8_e4m3fn.
                   If ``sf_is_swizzled=True``: shape (M_i//128, K//64, 32, 16).
                   If ``sf_is_swizzled=False``: shape (M_i, K//16), auto-converted.
        b_sf_list: Scale factors for B, same convention as a_sf_list but for N_i.
        per_token_scale_list: Per-token output scales, each (M_i,) float32.
                              Applied after GEMM: D[m, :] *= final_scale[m].
        out_dtype: Output dtype. Default: bfloat16.
        out_list: Optional pre-allocated output tensors, each (M_i, N_i) out_dtype.
        sf_is_swizzled: Whether scale factors are already in 4D swizzled layout.
                        Set False to auto-convert from 2D (MN, K//16) layout.
        tensor_scale_a_list: Global amax per group for A (scalar tensors or None).
                             None entries are treated as 1.0.
        tensor_scale_b_list: Global amax per group for B (scalar tensors or None).
                             None entries are treated as 1.0.

    Returns:
        List of output tensors D_i, each (M_i, N_i) in out_dtype.
    """
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

        # Fold global alpha (amax_A * amax_B / factor) into per_token_scale.
        # This avoids an extra scalar-mul kernel and keeps C++ simple (alpha=1 always).
        ts_a = tensor_scale_a_list[i] if tensor_scale_a_list is not None else None
        ts_b = tensor_scale_b_list[i] if tensor_scale_b_list is not None else None
        if ts_a is not None and ts_b is not None:
            global_alpha = (ts_a.float() * ts_b.float()).squeeze() / _NVFP4_FACTOR
            pts = pts * global_alpha

        # Prepare output tensor
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
        d_list,
        a_list,
        b_list,
        a_sf_prepared,
        b_sf_prepared,
        scale_prepared,
    )

    return d_list
