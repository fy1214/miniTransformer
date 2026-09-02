"""Golden / dequant helpers for NVFP4 pytest and eval scripts.

Prefer production dequant APIs from ``miniTransformer.ops.nvfp4_quantize`` in new
tests; this module centralizes low-level nibble/unswizzle helpers duplicated
across legacy test files.
"""
from __future__ import annotations

import torch

from miniTransformer.ops.nvfp4_quantize import (
    _get_fp4_grid,
    _unpack_fp4,
    _unswizzle_scale,
)

__all__ = [
    "FP4_GRID",
    "MMA_TILE_ROWS",
    "effective_bitwidth",
    "fp4_grid",
    "unpack_fp4",
    "unswizzle_scale",
    "dequantize_rowmajor",
    "dequantize_swizzled",
]

MMA_TILE_ROWS = 128

FP4_GRID = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)


def fp4_grid(device: torch.device | str) -> torch.Tensor:
    """Cached per-device FP4 E2M1 lookup table."""
    return _get_fp4_grid(torch.device(device))


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 [m, n//2] → int32 indices [m, n] in range 0–15."""
    return _unpack_fp4(packed)


def unswizzle_scale(scale_flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Undo tcgen05 swizzle: flat buffer → row-major [rows, cols//16]."""
    return _unswizzle_scale(scale_flat, rows, cols)


def dequantize_rowmajor(
    data: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize row-major pertoken quant output back to BF16."""
    grid = fp4_grid(data.device)
    fp4_vals = grid[unpack_fp4(data)].to(torch.float32)
    scale_expanded = scale.to(torch.float32).repeat_interleave(16, dim=1)
    gs = global_scale.to(torch.float32).unsqueeze(1)
    return (fp4_vals * scale_expanded * gs).to(torch.bfloat16)


def dequantize_swizzled(
    data: torch.Tensor,
    scale_flat: torch.Tensor,
    global_scale: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Dequantize swizzled pertoken quant output back to BF16."""
    scale_rowmajor = unswizzle_scale(scale_flat.to(torch.float32), rows, cols)
    grid = fp4_grid(data.device)
    fp4_vals = grid[unpack_fp4(data)].to(torch.float32)
    scale_expanded = scale_rowmajor.repeat_interleave(16, dim=1)
    gs = global_scale.to(torch.float32).unsqueeze(1)
    return (fp4_vals * scale_expanded * gs).to(torch.bfloat16)


def effective_bitwidth(x: torch.Tensor, x_dq: torch.Tensor) -> float:
    """Effective bit-width from mean per-row relative L2 error."""
    x_f = x.to(torch.float32)
    dq_f = x_dq.to(torch.float32)
    row_rel_err = (x_f - dq_f).pow(2).sum(dim=-1) / x_f.pow(2).sum(dim=-1).clamp(min=1e-9)
    return (-torch.log2(row_rel_err.mean()) / 2).item()
