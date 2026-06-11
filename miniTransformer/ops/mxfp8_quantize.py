import math
import torch
from typing import List, Optional, Tuple

import fp4_gemm


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


def _compute_offsets_cpu(m_splits, K):
    """Compute row_offsets, sf_offsets, total_sf entirely on CPU from a Python list.

    Avoids ~12 tiny GPU kernel launches (cumsum, elementwise ops) that each
    cost 20-80us of Python dispatch overhead.  Returns plain Python lists;
    caller converts to GPU tensors with a single torch.tensor() each.
    """
    row_off = [0]
    sf_off = [0]
    for m in m_splits:
        row_off.append(row_off[-1] + m)
        sf_off.append(sf_off[-1] + _sf_size_per_group(m, K))
    return row_off, sf_off, sf_off[-1]


def _compute_offsets_gpu(m_splits_t, K):
    """Compute row_offsets, sf_offsets, total_sf on GPU (fallback for tensor m_splits)."""
    num_groups = int(m_splits_t.numel())
    row_offsets = torch.empty(num_groups + 1, dtype=torch.int32, device=m_splits_t.device)
    row_offsets[0] = 0
    row_offsets[1:] = torch.cumsum(m_splits_t.int(), dim=0)

    k_groups  = K // 32
    n_k_tiles = (k_groups + 3) // 4
    n_m_tiles = (m_splits_t + 127) // 128
    sf_sizes  = n_m_tiles * n_k_tiles * 512
    sf_offsets = torch.empty(num_groups + 1, dtype=torch.int64, device=m_splits_t.device)
    sf_offsets[0] = 0
    sf_offsets[1:] = torch.cumsum(sf_sizes, dim=0)
    total_sf = int(sf_offsets[-1].item())
    return row_offsets, sf_offsets, total_sf


def _get_offsets(m_splits, K, device):
    """Dispatch to CPU or GPU offset computation based on m_splits type."""
    if isinstance(m_splits, (list, tuple)):
        row_off, sf_off, total_sf = _compute_offsets_cpu(m_splits, K)
        # pin_memory + non_blocking avoids implicit cudaDeviceSynchronize
        # that pageable H2D copy triggers (was causing 600-800ms stalls
        # when GPU pipeline is backed up with NCCL / checkpoint recompute).
        row_offsets = torch.tensor(row_off, dtype=torch.int32).pin_memory().to(device, non_blocking=True)
        sf_offsets  = torch.tensor(sf_off,  dtype=torch.int64).pin_memory().to(device, non_blocking=True)
        return row_offsets, sf_offsets, total_sf
    else:
        m_splits_t = m_splits.to(dtype=torch.int64, device=device)
        return _compute_offsets_gpu(m_splits_t, K)


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

    row_offsets, sf_offsets, total_sf = _get_offsets(m_splits, K, inp.device)

    out = torch.empty((total_M, K), dtype=torch.uint8, device=inp.device)
    sf  = torch.zeros(total_sf, dtype=torch.uint8, device=inp.device)

    fp4_gemm.mxfp8_grouped_quantize(out, sf, inp, row_offsets, sf_offsets)

    return out, sf, row_offsets, sf_offsets


def mxfp8_grouped_quantize_hw(
    inp: torch.Tensor,
    m_splits,               # List[int] 或 GPU int tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rowwise grouped MXFP8 quantization — full hardware-cvt path (SM100).

    Same layout/contract as mxfp8_grouped_quantize, but the E8M0 scale and FP8
    pack are computed with hardware packed cvts (cvt.rp.ue8m0x2 +
    cvt.rn.satfinite.e4m3x2). Intended as a fast inference operator; SF rounding
    may differ from the software path at boundaries.

    Returns: (out [total_M,K] uint8, sf flat uint8, row_offsets [G+1] int32,
              sf_offsets [G+1] int64).
    """
    assert inp.is_cuda and inp.dtype == torch.bfloat16
    assert inp.is_contiguous()
    total_M, K = inp.shape
    assert K % 32 == 0, "K must be divisible by 32"

    row_offsets, sf_offsets, total_sf = _get_offsets(m_splits, K, inp.device)

    out = torch.empty((total_M, K), dtype=torch.uint8, device=inp.device)
    sf  = torch.zeros(total_sf, dtype=torch.uint8, device=inp.device)

    fp4_gemm.mxfp8_grouped_quantize_hw(out, sf, inp, row_offsets, sf_offsets)

    return out, sf, row_offsets, sf_offsets


def mxfp8_grouped_quantize_chunked(
    inp: torch.Tensor,
    m_splits,               # List[int] 或 GPU int tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rowwise grouped MXFP8 quantization — per-CTA-chunk scheduling (SM100).

    Same layout/contract as mxfp8_grouped_quantize, but each CTA owns one
    128-row chunk and binary-searches the group table only once (broadcast
    via shared memory) instead of every row doing a linear scan.

    Hard precondition: every group's M must be a multiple of 128. Megatron's
    Fp8Padding (TEGroupedMLP.forward) enforces this when fp8/fp4 is on, so
    total_M and every m_splits entry are already 128-aligned in that path.

    Returns: (out [total_M,K] uint8, sf flat uint8, row_offsets [G+1] int32,
              sf_offsets [G+1] int64).
    """
    assert inp.is_cuda and inp.dtype == torch.bfloat16
    assert inp.is_contiguous()
    total_M, K = inp.shape
    assert K % 32 == 0, "K must be divisible by 32"
    assert total_M % 128 == 0, "total_M must be a multiple of 128 (group M must be 128-padded)"

    row_offsets, sf_offsets, total_sf = _get_offsets(m_splits, K, inp.device)

    out = torch.empty((total_M, K), dtype=torch.uint8, device=inp.device)
    sf  = torch.zeros(total_sf, dtype=torch.uint8, device=inp.device)

    fp4_gemm.mxfp8_grouped_quantize_chunked(out, sf, inp, row_offsets, sf_offsets)

    return out, sf, row_offsets, sf_offsets


def mxfp8_grouped_quantize_chunked_hw(
    inp: torch.Tensor,
    m_splits,               # List[int] 或 GPU int tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rowwise grouped MXFP8 quantization — per-CTA-chunk + hardware-cvt path.

    Combines the chunk scheduling of mxfp8_grouped_quantize_chunked with the
    hardware packed cvts of mxfp8_grouped_quantize_hw (cvt.rp.ue8m0x2 +
    cvt.rn.satfinite.e4m3x2). Same 128-row-alignment precondition.
    """
    assert inp.is_cuda and inp.dtype == torch.bfloat16
    assert inp.is_contiguous()
    total_M, K = inp.shape
    assert K % 32 == 0, "K must be divisible by 32"
    assert total_M % 128 == 0, "total_M must be a multiple of 128 (group M must be 128-padded)"

    row_offsets, sf_offsets, total_sf = _get_offsets(m_splits, K, inp.device)

    out = torch.empty((total_M, K), dtype=torch.uint8, device=inp.device)
    sf  = torch.zeros(total_sf, dtype=torch.uint8, device=inp.device)

    fp4_gemm.mxfp8_grouped_quantize_chunked_hw(out, sf, inp, row_offsets, sf_offsets)

    return out, sf, row_offsets, sf_offsets


def mxfp8_grouped_transpose_quantize(
    inp: torch.Tensor,
    n: int,
    num_groups: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Colwise (transpose) grouped MXFP8 quantization for weight tensors.

    Input: [num_groups * n, K] bf16 (all groups stacked, each group has n rows).
    Output: per-group [K, n] stacked contiguously → returned as [ng*K, n] uint8.

    Args:
        inp        : [num_groups * n, K] bf16 contiguous
        n          : per-group row count (= out_features, uniform across groups)
        num_groups : number of groups

    Returns:
        out_T      : [ng*K, n] uint8 (FP8 E4M3), per-group [K, n] stacked
        sf         : flat uint8, swizzled E8M0 scale factors
        sf_offsets : [ng+1] int64 GPU, group SF element offsets
    """
    assert inp.is_cuda and inp.dtype == torch.bfloat16
    assert inp.is_contiguous()
    total_M, K = inp.shape
    assert total_M == num_groups * n
    assert K % 64 == 0, "K must be divisible by 64 (TMA TILE_K)"
    assert n % 32 == 0, "n must be divisible by 32 (TILE_M)"

    # SF offsets computed here for the caller (GEMM needs them), but not
    # passed to the kernel — V2 kernel derives everything from scalars.
    m_k_groups = (n + 31) // 32
    sf_per_group = ((K + 127) // 128) * ((m_k_groups + 3) // 4) * 512
    sf_offsets = torch.arange(
        0, (num_groups + 1) * sf_per_group, sf_per_group, dtype=torch.int64, device=inp.device
    )

    total_sf = num_groups * sf_per_group
    out_T = torch.empty(num_groups * K * n, dtype=torch.uint8, device=inp.device)
    sf    = torch.zeros(total_sf, dtype=torch.uint8, device=inp.device)

    fp4_gemm.mxfp8_grouped_transpose_quantize(out_T, sf, inp, n, num_groups)

    return out_T.view(num_groups * K, n), sf, sf_offsets

