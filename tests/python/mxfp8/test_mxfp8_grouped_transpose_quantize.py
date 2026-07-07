"""
Tests for MXFP8 grouped colwise (transpose) quantization.

Kernel output: [K, total_M] FP8 E4M3 (transposed) + E8M0 scale factors
in SfKMajorAtom layout for the [K, m_g] per-group view.

SF layout (roles swapped vs rowwise):
  mIdx     = global_k        (K dimension, tile-128 outer)
  kIdx     = m_tile          = m_offset // 32
  numKTiles = ceil(ceil(m_g/32) / 4)
"""

import math
import os
import struct
import sys
from typing import List

import pytest
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

pytest.importorskip("fp4_gemm")

from miniTransformer.ops.mxfp8_quantize import mxfp8_grouped_transpose_quantize
import fp4_gemm


def _mxfp8_grouped_transpose_quantize_raw(inp, m_per_group, num_groups):
    """Raw wrapper for V2 uniform-group kernel."""
    total_M, K = inp.shape
    assert total_M == num_groups * m_per_group
    m_k_groups = (m_per_group + 31) // 32
    sf_per_group = ((K + 127) // 128) * ((m_k_groups + 3) // 4) * 512
    total_sf = num_groups * sf_per_group
    out_T = torch.empty(num_groups * K * m_per_group, dtype=torch.uint8, device=inp.device)
    sf = torch.zeros(total_sf, dtype=torch.uint8, device=inp.device)
    fp4_gemm.mxfp8_grouped_transpose_quantize(out_T, sf, inp, m_per_group, num_groups)
    sf_offsets = torch.arange(
        0, (num_groups + 1) * sf_per_group, sf_per_group, dtype=torch.int64, device=inp.device
    )
    return out_T, sf, sf_offsets


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------

def _sf_size_T_per_group(m_g: int, K: int) -> int:
    m_k_groups = (m_g + 31) // 32
    n_m_tiles  = (K + 127) // 128
    n_k_tiles  = (m_k_groups + 3) // 4
    return n_m_tiles * n_k_tiles * 512


def _cvt_sf_offset(mIdx: int, kIdx: int, numKTiles: int) -> int:
    """Python reference for cvt_quant_to_fp4_get_sf_out_offset (reused for transpose)."""
    m_tile_idx  = mIdx >> 7
    outer_m_idx = mIdx & 31
    inner_m_idx = (mIdx >> 5) & 3
    k_tile_idx  = kIdx >> 2
    inner_k_idx = kIdx & 3
    return (((m_tile_idx * numKTiles + k_tile_idx) << 9)
            | (outer_m_idx << 4) | (inner_m_idx << 2) | inner_k_idx)


def _float_to_e8m0(v: float) -> int:
    """OCP MXFP8 / TransformerEngine recipe: normalize by FP8 E4M3 max norm
    (448), then round exponent UP on nonzero mantissa.  Mirrors the CUDA
    transpose kernel mxfp8t_float_to_e8m0 and TE ptx.cuh software fallback."""
    if v == 0.0:
        return 0
    val = abs(v) * (1.0 / 448.0)
    bits = int.from_bytes(struct.pack('>f', val), 'big')
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    if (mantissa > 0 and exponent != 0xFE) and not (exponent == 0 and mantissa <= 0x400000):
        exponent += 1
    return exponent


def _e8m0_to_scale(e8m0: int) -> float:
    return math.ldexp(1.0, e8m0 - 127)


def _reference_transpose_quantize(inp: torch.Tensor, m_splits: List[int]):
    """
    Python reference for grouped transpose MXFP8 quantization.

    For each group g, for each K column k, for each m-tile:
      - one E8M0 scale covers 32 consecutive M elements (m_tile * 32 .. +31)
      - output is stored transposed: out_T[k, m] = FP8(inp[m, k] / scale)

    Returns (fp8_T_uint8, sf_flat_uint8, row_offsets, sf_offsets)
    """
    total_M, K = inp.shape
    num_groups  = len(m_splits)

    h_row_off = [0] * (num_groups + 1)
    for g, m_g in enumerate(m_splits):
        h_row_off[g + 1] = h_row_off[g] + m_g

    h_sf_off = [0] * (num_groups + 1)
    for g, m_g in enumerate(m_splits):
        h_sf_off[g + 1] = h_sf_off[g] + _sf_size_T_per_group(m_g, K)

    inp_f   = inp.float().cpu()
    fp8_T   = torch.zeros(K, total_M, dtype=torch.uint8)
    sf_flat = torch.zeros(h_sf_off[num_groups], dtype=torch.uint8)

    for g in range(num_groups):
        r0, r1   = h_row_off[g], h_row_off[g + 1]
        m_g      = r1 - r0
        sf_base  = h_sf_off[g]
        m_k_groups  = (m_g + 31) // 32
        numMKTiles  = (m_k_groups + 3) // 4   # ceil(m_k_groups / 4)

        for k in range(K):
            m_tile = 0
            m_off  = 0
            while m_off < m_g:
                rows_in_tile = min(32, m_g - m_off)
                block = inp_f[r0 + m_off : r0 + m_off + rows_in_tile, k]
                amax  = block.abs().max().item()
                e8m0  = _float_to_e8m0(amax)
                scale = _e8m0_to_scale(e8m0) if e8m0 > 0 else 1.0
                inv   = 1.0 / scale if scale != 0.0 else 0.0

                for i in range(rows_in_tile):
                    s = float(block[i]) * inv
                    s = max(-448.0, min(448.0, s))
                    fp8_T[k, r0 + m_off + i] = int(
                        torch.tensor(s).to(torch.float8_e4m3fn).view(torch.uint8))

                # SF: mIdx=k, kIdx=m_tile
                sf_off = _cvt_sf_offset(k, m_tile, numMKTiles)
                sf_flat[sf_base + sf_off] = e8m0

                m_tile += 1
                m_off  += 32

    row_offsets = torch.tensor(h_row_off, dtype=torch.int32)
    sf_offsets  = torch.tensor(h_sf_off,  dtype=torch.int64)
    return fp8_T, sf_flat, row_offsets, sf_offsets


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ---------------------------------------------------------------------------
# Helper: run kernel (wrapped interface) + reference
# ---------------------------------------------------------------------------

def _run(m_per_group: int, num_groups: int, K: int, seed: int = 42):
    torch.manual_seed(seed)
    total_M = num_groups * m_per_group
    inp = torch.randn(total_M, K, dtype=torch.bfloat16).cuda()

    out_T, out_sf, _ = _mxfp8_grouped_transpose_quantize_raw(inp, m_per_group, num_groups)
    torch.cuda.synchronize()

    # Kernel stores per-group [K, m_per_group] blocks stacked (group-major):
    #   [g0:[K,m], g1:[K,m], ...] → shape [ng, K, m_per_group]
    # Reference stores [K, total_M] = [K, ng*m_per_group] (K-major).
    # Permute (g,k,m) → (k,g,m) then reshape to [K, total_M].
    out_2d = out_T.cpu().view(num_groups, K, m_per_group).permute(1, 0, 2).reshape(K, total_M)

    m_splits = [m_per_group] * num_groups
    ref_T, ref_sf, _, _ = _reference_transpose_quantize(inp, m_splits)
    return out_2d, out_sf.cpu(), ref_T, ref_sf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMxFP8GroupedTransposeQuantize:

    @pytest.mark.parametrize("K", [128, 256, 512, 7168])
    @pytest.mark.parametrize("m_per_group,num_groups", [(64, 1), (32, 3), (128, 2)])
    def test_fp8_values_match_reference(self, m_per_group, num_groups, K):
        out_T, _, ref_T, _ = _run(m_per_group, num_groups, K)
        assert torch.all(out_T == ref_T), (
            f"FP8 transpose mismatch: m_per_group={m_per_group}, ng={num_groups}, K={K}\n"
            f"max diff: {(out_T.int() - ref_T.int()).abs().max()}"
        )

    @pytest.mark.parametrize("K", [128, 256, 512, 7168])
    @pytest.mark.parametrize("m_per_group,num_groups", [(64, 1), (32, 3)])
    def test_sf_values_match_reference(self, m_per_group, num_groups, K):
        _, out_sf, _, ref_sf = _run(m_per_group, num_groups, K)
        assert torch.all(out_sf == ref_sf), (
            f"SF mismatch: m_per_group={m_per_group}, ng={num_groups}, K={K}\n"
            f"nonzero diff positions: {(out_sf != ref_sf).nonzero()[:10]}"
        )

    def test_output_shape(self):
        m_per_group, num_groups, K = 64, 2, 256
        total_M = num_groups * m_per_group
        out_T, _, ref_T, _ = _run(m_per_group, num_groups, K)
        assert out_T.shape == (K, total_M)

    def test_sf_buffer_size(self):
        m_per_group, num_groups, K = 64, 2, 256
        _, out_sf, _, _ = _run(m_per_group, num_groups, K)
        expected = num_groups * _sf_size_T_per_group(m_per_group, K)
        assert out_sf.numel() == expected

    def test_single_group(self):
        out_T, out_sf, ref_T, ref_sf = _run(128, 1, K=128)
        assert torch.all(out_T == ref_T)
        assert torch.all(out_sf == ref_sf)

    def test_many_groups(self):
        """64 groups (MoE-like)."""
        out_T, _, ref_T, _ = _run(32, 64, K=128)
        assert torch.all(out_T == ref_T)

    def test_transposed_layout(self):
        """Verify out_T[k, m] corresponds to inp[m, k] after scaling."""
        torch.manual_seed(7)
        m_per_group, num_groups, K = 64, 1, 128
        total_M = 64
        inp = torch.randn(total_M, K, dtype=torch.bfloat16).cuda()

        out_T, _, _ = _mxfp8_grouped_transpose_quantize_raw(inp, m_per_group, num_groups)
        torch.cuda.synchronize()

        ref_T, _, _, _ = _reference_transpose_quantize(inp, [m_per_group])
        assert out_T.shape == (K * total_M,)
        assert torch.all(out_T.cpu().view(K, total_M) == ref_T), "Transposed layout mismatch"

    def test_zero_input(self):
        """All-zero input -> all-zero FP8 output, SF=0."""
        m_per_group, num_groups, K = 64, 1, 128
        total_M = 64
        inp = torch.zeros(total_M, K, dtype=torch.bfloat16).cuda()

        out_T, out_sf, _ = _mxfp8_grouped_transpose_quantize_raw(inp, m_per_group, num_groups)
        torch.cuda.synchronize()

        assert torch.all(out_T.cpu() == 0), "Zero input should produce zero FP8"
        assert torch.all(out_sf.cpu() == 0), "Zero input should produce zero SF"
