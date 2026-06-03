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
import struct
import pytest
import torch
from typing import List


def _import_module():
    try:
        import miniTransformer._C as _C
        return _C
    except ImportError:
        return None


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
    if v == 0.0:
        return 0
    bits = int.from_bytes(struct.pack('>f', abs(v)), 'big')
    biased_exp = (bits >> 23) & 0xFF   # IEEE-754 biased exp = unbiased+127 = E8M0
    return max(0, min(254, biased_exp))


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


@pytest.fixture(autouse=True)
def skip_if_no_module():
    if _import_module() is None:
        pytest.skip("miniTransformer._C not available (not compiled)")


# ---------------------------------------------------------------------------
# Helper: run kernel + reference
# ---------------------------------------------------------------------------

def _run(m_splits: List[int], K: int, seed: int = 42):
    _C = _import_module()
    torch.manual_seed(seed)
    total_M = sum(m_splits)
    inp = torch.randn(total_M, K, dtype=torch.bfloat16).cuda()

    num_groups = len(m_splits)
    m_splits_t = torch.tensor(m_splits, dtype=torch.int64, device="cuda")

    row_offsets = torch.empty(num_groups + 1, dtype=torch.int32, device="cuda")
    row_offsets[0] = 0
    row_offsets[1:] = torch.cumsum(m_splits_t.int(), dim=0)

    m_k_groups = (m_splits_t + 31) // 32
    sf_sizes   = ((K + 127) // 128) * ((m_k_groups + 3) // 4) * 512
    sf_offsets = torch.empty(num_groups + 1, dtype=torch.int64, device="cuda")
    sf_offsets[0] = 0
    sf_offsets[1:] = torch.cumsum(sf_sizes, dim=0)

    total_sf = int(sf_offsets[-1].item())
    out_T = torch.empty(K, total_M, dtype=torch.uint8, device="cuda")
    out_sf = torch.zeros(total_sf, dtype=torch.uint8, device="cuda")

    _C.mxfp8_grouped_transpose_quantize(out_T, out_sf, inp, row_offsets, sf_offsets)
    torch.cuda.synchronize()

    ref_T, ref_sf, _, _ = _reference_transpose_quantize(inp, m_splits)
    return out_T.cpu(), out_sf.cpu(), ref_T, ref_sf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMxFP8GroupedTransposeQuantize:

    @pytest.mark.parametrize("K", [128, 256, 512, 7168])
    @pytest.mark.parametrize("m_splits", [[64], [32, 64, 128], [128, 128]])
    def test_fp8_values_match_reference(self, m_splits, K):
        out_T, _, ref_T, _ = _run(m_splits, K)
        assert torch.all(out_T == ref_T), (
            f"FP8 transpose mismatch: m_splits={m_splits}, K={K}\n"
            f"max diff: {(out_T.int() - ref_T.int()).abs().max()}"
        )

    @pytest.mark.parametrize("K", [128, 256, 512, 7168])
    @pytest.mark.parametrize("m_splits", [[64], [32, 64, 128]])
    def test_sf_values_match_reference(self, m_splits, K):
        _, out_sf, _, ref_sf = _run(m_splits, K)
        assert torch.all(out_sf == ref_sf), (
            f"SF mismatch: m_splits={m_splits}, K={K}\n"
            f"nonzero diff positions: {(out_sf != ref_sf).nonzero()[:10]}"
        )

    def test_output_shape(self):
        m_splits = [64, 128]
        K = 256
        total_M = sum(m_splits)
        out_T, out_sf, _, _ = _run(m_splits, K)
        assert out_T.shape == (K, total_M)

    def test_sf_buffer_size(self):
        m_splits = [64, 128]
        K = 256
        _, out_sf, _, _ = _run(m_splits, K)
        expected = sum(_sf_size_T_per_group(m, K) for m in m_splits)
        assert out_sf.numel() == expected

    def test_single_group(self):
        out_T, out_sf, ref_T, ref_sf = _run([128], K=128)
        assert torch.all(out_T == ref_T)
        assert torch.all(out_sf == ref_sf)

    def test_many_groups(self):
        """64 groups (MoE-like)."""
        m_splits = [32] * 64
        K = 128
        out_T, _, ref_T, _ = _run(m_splits, K)
        assert torch.all(out_T == ref_T)

    def test_gpu_tensor_m_splits(self):
        """m_splits passed as a GPU int tensor instead of a Python list."""
        _C = _import_module()
        torch.manual_seed(0)
        m_splits = [64, 96]
        K = 128
        total_M = sum(m_splits)
        inp = torch.randn(total_M, K, dtype=torch.bfloat16).cuda()

        m_splits_gpu = torch.tensor(m_splits, dtype=torch.int64, device="cuda")
        num_groups = 2

        row_offsets = torch.empty(num_groups + 1, dtype=torch.int32, device="cuda")
        row_offsets[0] = 0
        row_offsets[1:] = torch.cumsum(m_splits_gpu.int(), dim=0)

        m_k_groups = (m_splits_gpu + 31) // 32
        sf_sizes   = ((K + 127) // 128) * ((m_k_groups + 3) // 4) * 512
        sf_offsets = torch.empty(num_groups + 1, dtype=torch.int64, device="cuda")
        sf_offsets[0] = 0
        sf_offsets[1:] = torch.cumsum(sf_sizes, dim=0)

        total_sf = int(sf_offsets[-1].item())
        out_T  = torch.empty(K, total_M, dtype=torch.uint8, device="cuda")
        out_sf = torch.zeros(total_sf, dtype=torch.uint8, device="cuda")

        _C.mxfp8_grouped_transpose_quantize(out_T, out_sf, inp, row_offsets, sf_offsets)
        torch.cuda.synchronize()

        ref_T, ref_sf, _, _ = _reference_transpose_quantize(inp, m_splits)
        assert torch.all(out_T.cpu() == ref_T)

    def test_transposed_layout(self):
        """Verify out_T[k, m] corresponds to inp[m, k] after scaling."""
        _C = _import_module()
        torch.manual_seed(7)
        m_splits = [64]
        K = 128
        total_M = 64
        inp = torch.randn(total_M, K, dtype=torch.bfloat16).cuda()

        m_splits_t = torch.tensor(m_splits, dtype=torch.int64, device="cuda")
        row_offsets = torch.tensor([0, total_M], dtype=torch.int32, device="cuda")
        sf_size = _sf_size_T_per_group(total_M, K)
        sf_offsets = torch.tensor([0, sf_size], dtype=torch.int64, device="cuda")

        out_T  = torch.empty(K, total_M, dtype=torch.uint8, device="cuda")
        out_sf = torch.zeros(sf_size, dtype=torch.uint8, device="cuda")
        _C.mxfp8_grouped_transpose_quantize(out_T, out_sf, inp, row_offsets, sf_offsets)
        torch.cuda.synchronize()

        ref_T, _, _, _ = _reference_transpose_quantize(inp, m_splits)
        assert out_T.shape == (K, total_M)
        assert torch.all(out_T.cpu() == ref_T), "Transposed layout mismatch"

    def test_zero_input(self):
        """All-zero input → all-zero FP8 output, SF=0."""
        _C = _import_module()
        m_splits = [64]
        K = 128
        total_M = 64
        inp = torch.zeros(total_M, K, dtype=torch.bfloat16).cuda()

        row_offsets = torch.tensor([0, total_M], dtype=torch.int32, device="cuda")
        sf_size = _sf_size_T_per_group(total_M, K)
        sf_offsets = torch.tensor([0, sf_size], dtype=torch.int64, device="cuda")

        out_T  = torch.empty(K, total_M, dtype=torch.uint8, device="cuda")
        out_sf = torch.zeros(sf_size, dtype=torch.uint8, device="cuda")
        _C.mxfp8_grouped_transpose_quantize(out_T, out_sf, inp, row_offsets, sf_offsets)
        torch.cuda.synchronize()

        assert torch.all(out_T == 0), "Zero input should produce zero FP8"
        assert torch.all(out_sf == 0), "Zero input should produce zero SF"
