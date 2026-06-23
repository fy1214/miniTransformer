"""Split-amax grouped per-token NVFP4 quantization.

Implements quantization as two independent kernels:
  1. fp4_perrow_amax_te — per-row absolute-max reduction (grid=num_rows, block=256)
  2. rtn_fp4_group_pertoken_split_amax — pure streaming quantize (flat 1-D grid,
     zero __syncthreads, zero shared memory)

Total: 2 kernel launches regardless of number of groups.
No K limit. No M-alignment requirement.
"""
from __future__ import annotations

from typing import List

import torch
import fp4_gemm

from miniTransformer.ops.nvfp4_quantize import (
    GroupQuantizeSwizzledResult,
    nvfp4_output_t,
    nvfp4_scale_t,
    _MMA_TILE_ROWS,
)


def group_quantize_nvfp4_pertoken_split_amax(
    x: torch.Tensor,
    m_splits: List[int],
    *,
    four_six: bool = False,
    scale_override: float = 1.0,
) -> GroupQuantizeSwizzledResult:
    """Grouped per-token NVFP4 quantization via split-amax (2-kernel).

    Drop-in replacement for group_quantize_nvfp4_pertoken_swizzled.
    Returns identical GroupQuantizeSwizzledResult.

    Kernel 1: fp4_perrow_amax_te
        Per-row absolute-max reduction. One CTA per row, 256 threads,
        bf16x2 vectorized loads + warp/block reduce.

    Kernel 2: rtn_fp4_group_pertoken_split_amax
        Pure streaming quantize. Each thread handles 8 bf16 elements,
        O(log G) binary search for owning group, intra-warp __shfl_xor
        for 16-element group scale. Zero __syncthreads, zero shared memory.

    Args:
        x:              [total_m, k] BF16, contiguous, CUDA. k % 128 == 0.
        m_splits:       Per-group token counts. sum(m_splits) == total_m.
                        No alignment requirement (unlike fused kernel's 128 multiple).
        four_six:       Round-4-or-6 if True; RTN otherwise.
        scale_override: Scalar multiplier for the per-row global scale.

    Returns:
        GroupQuantizeSwizzledResult with per-group tcgen05 swizzled scales.
    """
    assert x.is_cuda and x.dtype == torch.bfloat16
    assert x.dim() == 2 and x.is_contiguous()
    assert sum(m_splits) == x.shape[0]

    total_m, k = x.shape
    device = x.device
    T = _MMA_TILE_ROWS

    if total_m == 0:
        n = len(m_splits) + 1
        empty_off = torch.zeros(n, dtype=torch.int32, device=device)
        return GroupQuantizeSwizzledResult(
            data=torch.empty((0, k // 2), dtype=nvfp4_output_t, device=device),
            global_scale=torch.empty(0, dtype=torch.float32, device=device),
            scale_flat=torch.empty(0, dtype=nvfp4_scale_t, device=device),
            row_offsets_gpu=empty_off,
            scale_offsets_gpu=torch.zeros(n, dtype=torch.int64, device=device),
        )

    m_padded = [((m + T - 1) // T) * T for m in m_splits]
    sf_k = k // 16

    scale_flat_size = sum(mp * sf_k for mp in m_padded)
    scale_flat = torch.zeros(scale_flat_size, dtype=nvfp4_scale_t, device=device)

    so = [0]
    for mp in m_padded:
        so.append(so[-1] + mp * sf_k)
    scale_offsets_gpu = (
        torch.tensor(so, dtype=torch.int64).pin_memory().to(device, non_blocking=True)
    )

    ro = [0]
    for m in m_splits:
        ro.append(ro[-1] + m)
    row_offsets_gpu = (
        torch.tensor(ro, dtype=torch.int32).pin_memory().to(device, non_blocking=True)
    )

    data = torch.empty((total_m, k // 2), dtype=nvfp4_output_t, device=device)
    global_scale = torch.empty(total_m, dtype=torch.float32, device=device)
    amax_per_row = torch.empty(total_m, dtype=torch.float32, device=device)

    fp4_gemm.fp4_perrow_amax_te(amax_per_row, x)

    fn = (fp4_gemm.four_six_fp4_group_pertoken_split_amax if four_six
          else fp4_gemm.rtn_fp4_group_pertoken_split_amax)
    fn(data, scale_flat, global_scale, x, amax_per_row,
       row_offsets_gpu, scale_offsets_gpu, scale_override)

    return GroupQuantizeSwizzledResult(
        data=data,
        global_scale=global_scale,
        scale_flat=scale_flat,
        row_offsets_gpu=row_offsets_gpu,
        scale_offsets_gpu=scale_offsets_gpu,
    )
