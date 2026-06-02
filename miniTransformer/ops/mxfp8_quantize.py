"""
MXFP8 grouped rowwise quantization.

Quantizes a BF16 tensor into MXFP8 E4M3 data and E8M0 scale factors
using the CUTLASS SM100 SfKMajorAtom swizzled layout required by
`mxfp8_grouped_gemm`.

The same kernel is used for both activation (A, rowwise) and weight (B,
also stored as [N, K] row-major and quantized rowwise).

SF buffer size per group g with m_g rows and K columns:
    sf_size_g = ceil(m_g / 128) * ceil((K//32) / 4) * 512  [E8M0 bytes]
"""

import math
import torch
from typing import List, Optional, Tuple

import miniTransformer._C as _C


def _sf_size_per_group(m_g: int, K: int) -> int:
    """Number of E8M0 scale factor bytes for one group."""
    k_groups = K // 32
    n_m_tiles = (m_g + 127) // 128
    n_k_tiles = (k_groups + 3) // 4
    return n_m_tiles * n_k_tiles * 512


def mxfp8_grouped_quantize(
    inp: torch.Tensor,          # [total_rows, K] bfloat16, all groups concat
    m_splits: List[int],        # m_g for each group (sum = total_rows)
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a grouped BF16 matrix to MXFP8.

    Returns:
        out         : [total_rows, K]  uint8  (FP8 E4M3)
        sf          : flat uint8 tensor, swizzled E8M0 scale factors
        row_offsets : [G+1] int32 GPU, group row boundaries
        sf_offsets  : [G+1] int64 GPU, group SF element offsets
    """
    assert inp.is_cuda and inp.dtype == torch.bfloat16
    assert inp.is_contiguous()
    total_rows, K = inp.shape
    assert K % 32 == 0, "K must be divisible by 32"
    assert sum(m_splits) == total_rows

    num_groups = len(m_splits)

    # Build row_offsets
    h_row_off = [0] * (num_groups + 1)
    for g, m_g in enumerate(m_splits):
        h_row_off[g + 1] = h_row_off[g] + m_g

    # Build sf_offsets
    h_sf_off = [0] * (num_groups + 1)
    for g, m_g in enumerate(m_splits):
        h_sf_off[g + 1] = h_sf_off[g] + _sf_size_per_group(m_g, K)

    row_offsets = torch.tensor(h_row_off, dtype=torch.int32, device=inp.device)
    sf_offsets  = torch.tensor(h_sf_off,  dtype=torch.int64, device=inp.device)

    total_sf = h_sf_off[num_groups]
    out = torch.empty((total_rows, K), dtype=torch.uint8, device=inp.device)
    sf  = torch.zeros(total_sf, dtype=torch.uint8, device=inp.device)

    _C.mxfp8_grouped_quantize(out, sf, inp, row_offsets, sf_offsets)

    return out, sf, row_offsets, sf_offsets
