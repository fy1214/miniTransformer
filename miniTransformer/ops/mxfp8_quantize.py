import math
import torch
from typing import List, Optional, Tuple

import miniTransformer._C as _C


def _sf_size_per_group(m_g: int, K: int) -> int:
    """SF bytes for rowwise (forward) quantization of group [m_g, K]."""
    k_groups = K // 32
    n_m_tiles = (m_g + 127) // 128
    n_k_tiles = (k_groups + 3) // 4
    return n_m_tiles * n_k_tiles * 512


def _sf_size_T_per_group(m_g: int, K: int) -> int:
    """SF bytes for transpose (colwise) quantization: output view [K, m_g]."""
    m_k_groups = (m_g + 31) // 32          # ceil(m_g / 32)
    n_m_tiles  = (K + 127) // 128          # ceil(K / 128) — K plays the "M" role
    n_k_tiles  = (m_k_groups + 3) // 4     # ceil(m_k_groups / 4)
    return n_m_tiles * n_k_tiles * 512


def mxfp8_grouped_quantize(
    inp: torch.Tensor,
    m_splits,               # List[int] 或 GPU int tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rowwise grouped MXFP8 quantization.

    Returns:
        out         : [total_M, K]  uint8  (FP8 E4M3)
        sf          : flat uint8 tensor, swizzled E8M0 scale factors
        row_offsets : [G+1] int32 GPU, group row boundaries
        sf_offsets  : [G+1] int64 GPU, group SF element offsets
    """
    assert inp.is_cuda and inp.dtype == torch.bfloat16
    assert inp.is_contiguous()
    total_M, K = inp.shape
    assert K % 32 == 0, "K must be divisible by 32"

    if isinstance(m_splits, (list, tuple)):
        m_splits_t = torch.tensor(m_splits, dtype=torch.int64, device=inp.device)
    else:
        m_splits_t = m_splits.to(dtype=torch.int64, device=inp.device)

    num_groups = int(m_splits_t.numel())

    # row_offsets: prefix sum of m_splits
    row_offsets = torch.empty(num_groups + 1, dtype=torch.int32, device=inp.device)
    row_offsets[0] = 0
    row_offsets[1:] = torch.cumsum(m_splits_t.int(), dim=0)

    # sf_offsets: prefix sum of per-group SF sizes (vectorized)
    k_groups  = K // 32
    n_k_tiles = (k_groups + 3) // 4                        # 对所有 group 相同，标量
    n_m_tiles = (m_splits_t + 127) // 128
    sf_sizes  = n_m_tiles * n_k_tiles * 512
    sf_offsets = torch.empty(num_groups + 1, dtype=torch.int64, device=inp.device)
    sf_offsets[0] = 0
    sf_offsets[1:] = torch.cumsum(sf_sizes, dim=0)

    total_sf = int(sf_offsets[-1].item())
    out = torch.empty((total_M, K), dtype=torch.uint8, device=inp.device)
    sf  = torch.zeros(total_sf, dtype=torch.uint8, device=inp.device)

    _C.mxfp8_grouped_quantize(out, sf, inp, row_offsets, sf_offsets)

    return out, sf, row_offsets, sf_offsets


def mxfp8_grouped_transpose_quantize(
    inp: torch.Tensor,
    m_splits,               # List[int] 或 GPU int tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Colwise (transpose) grouped MXFP8 quantization.

    Produces a transposed output [K, total_M] FP8 E4M3 with E8M0 scale
    factors in SfKMajorAtom layout for the [K, m_g] per-group view.

    Returns:
        out_T       : [K, total_M]  uint8  (FP8 E4M3, transposed)
        sf          : flat uint8 tensor, swizzled E8M0 scale factors
        row_offsets : [G+1] int32 GPU, group row boundaries (into total_M axis)
        sf_offsets  : [G+1] int64 GPU, group SF element offsets
    """
    assert inp.is_cuda and inp.dtype == torch.bfloat16
    assert inp.is_contiguous()
    total_M, K = inp.shape
    assert K % 64 == 0, "K must be divisible by 64 (TMA TILE_K)"
    assert total_M % 32 == 0, "total_M must be divisible by 32 (TILE_M)"

    if isinstance(m_splits, (list, tuple)):
        m_splits_t = torch.tensor(m_splits, dtype=torch.int64, device=inp.device)
    else:
        m_splits_t = m_splits.to(dtype=torch.int64, device=inp.device)

    num_groups = int(m_splits_t.numel())

    # row_offsets: prefix sum of m_splits
    row_offsets = torch.empty(num_groups + 1, dtype=torch.int32, device=inp.device)
    row_offsets[0] = 0
    row_offsets[1:] = torch.cumsum(m_splits_t.int(), dim=0)

    # sf_offsets: prefix sum of per-group SF sizes (vectorized, no Python loop)
    m_k_groups = (m_splits_t + 31) // 32
    sf_sizes   = ((K + 127) // 128) * ((m_k_groups + 3) // 4) * 512
    sf_offsets = torch.empty(num_groups + 1, dtype=torch.int64, device=inp.device)
    sf_offsets[0] = 0
    sf_offsets[1:] = torch.cumsum(sf_sizes, dim=0)

    total_sf = int(sf_offsets[-1].item())
    out_T = torch.empty((K, total_M), dtype=torch.uint8, device=inp.device)
    sf    = torch.zeros(total_sf, dtype=torch.uint8, device=inp.device)

    _C.mxfp8_grouped_transpose_quantize(out_T, sf, inp, row_offsets, sf_offsets)

    return out_T, sf, row_offsets, sf_offsets

