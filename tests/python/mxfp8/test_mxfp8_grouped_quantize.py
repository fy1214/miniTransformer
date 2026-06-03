"""
Tests for MXFP8 grouped rowwise quantization.

Reference: quantize BF16 → FP8 E4M3 with E8M0 scale factors.
E8M0: scale = 2^(e8m0 - 127), where e8m0 = floor(log2(row_amax)) + 127.
Each 32 elements share one E8M0 scale (MXFP8 SFVecSize=32).
"""

import math
import struct
import pytest
import torch
from typing import List


def _import_module():
    try:
        import fp4_gemm as _C
        return _C
    except ImportError:
        return None


def _sf_size_per_group(m_g: int, K: int) -> int:
    k_groups = K // 32
    n_m_tiles = (m_g + 127) // 128
    n_k_tiles = (k_groups + 3) // 4
    return n_m_tiles * n_k_tiles * 512


def _cvt_sf_offset(m_idx: int, k_idx: int, num_k_tiles: int) -> int:
    """Python reference for cvt_quant_to_fp4_get_sf_out_offset."""
    m_tile_idx  = m_idx >> 7
    outer_m_idx = m_idx & 31
    inner_m_idx = (m_idx >> 5) & 3
    k_tile_idx  = k_idx >> 2
    inner_k_idx = k_idx & 3
    sf_off = ((m_tile_idx * num_k_tiles + k_tile_idx) << 9
              | (outer_m_idx << 4) | (inner_m_idx << 2) | inner_k_idx)
    return sf_off


def _float_to_e8m0(v: float) -> int:
    """OCP MXFP8 / TransformerEngine recipe: normalize by FP8 E4M3 max norm
    (448), then round exponent UP on nonzero mantissa.  Mirrors the CUDA
    kernel float_to_e8m0 and TE ptx.cuh software fallback."""
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


def _reference_quantize(inp: torch.Tensor, m_splits: List[int]):
    """
    Python reference implementation for grouped MXFP8 quantization.
    Returns (fp8_data_uint8, sf_flat_uint8, row_offsets, sf_offsets)
    """
    total_rows, K = inp.shape
    assert K % 32 == 0
    K_groups = K // 32
    num_k_tiles = (K_groups + 3) // 4

    num_groups = len(m_splits)
    h_row_off = [0] * (num_groups + 1)
    for g, m_g in enumerate(m_splits):
        h_row_off[g + 1] = h_row_off[g] + m_g

    h_sf_off = [0] * (num_groups + 1)
    for g, m_g in enumerate(m_splits):
        h_sf_off[g + 1] = h_sf_off[g] + _sf_size_per_group(m_g, K)

    inp_f = inp.float().cpu()
    fp8_out = torch.zeros(total_rows, K, dtype=torch.uint8)
    sf_flat = torch.zeros(h_sf_off[num_groups], dtype=torch.uint8)

    for g in range(num_groups):
        r0, r1 = h_row_off[g], h_row_off[g + 1]
        sf_base = h_sf_off[g]
        for row in range(r0, r1):
            row_in_g = row - r0
            for kg in range(K_groups):
                block = inp_f[row, kg * 32: (kg + 1) * 32]
                amax = block.abs().max().item()
                e8m0_val = _float_to_e8m0(amax)
                scale = _e8m0_to_scale(e8m0_val) if e8m0_val > 0 else 1.0
                inv_scale = 1.0 / scale if scale != 0 else 0.0
                # Quantize
                scaled = (block * inv_scale).clamp(-448.0, 448.0)
                fp8_block = scaled.to(torch.float8_e4m3fn).view(torch.uint8)
                fp8_out[row, kg * 32: (kg + 1) * 32] = fp8_block
                # Write SF
                sf_off = _cvt_sf_offset(row_in_g, kg, num_k_tiles)
                sf_flat[sf_base + sf_off] = e8m0_val

    row_offsets = torch.tensor(h_row_off, dtype=torch.int32)
    sf_offsets  = torch.tensor(h_sf_off,  dtype=torch.int64)
    return fp8_out, sf_flat, row_offsets, sf_offsets


@pytest.fixture(autouse=True)
def skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture(autouse=True)
def skip_if_no_module():
    if _import_module() is None:
        pytest.skip("fp4_gemm not available (not compiled)")


class TestMxFP8GroupedQuantize:
    def _run(self, m_splits: List[int], K: int, seed: int = 42):
        _C = _import_module()
        torch.manual_seed(seed)
        total_rows = sum(m_splits)
        inp = torch.randn(total_rows, K, dtype=torch.bfloat16).cuda()

        num_groups = len(m_splits)
        h_row_off = [0] * (num_groups + 1)
        for g, m_g in enumerate(m_splits):
            h_row_off[g + 1] = h_row_off[g] + m_g
        h_sf_off = [0] * (num_groups + 1)
        for g, m_g in enumerate(m_splits):
            h_sf_off[g + 1] = h_sf_off[g] + _sf_size_per_group(m_g, K)

        row_offsets = torch.tensor(h_row_off, dtype=torch.int32, device="cuda")
        sf_offsets  = torch.tensor(h_sf_off,  dtype=torch.int64, device="cuda")
        total_sf    = h_sf_off[num_groups]

        out_fp8 = torch.empty(total_rows, K, dtype=torch.uint8, device="cuda")
        out_sf  = torch.zeros(total_sf, dtype=torch.uint8, device="cuda")

        _C.mxfp8_grouped_quantize(out_fp8, out_sf, inp, row_offsets, sf_offsets)
        torch.cuda.synchronize()

        # Reference
        ref_fp8, ref_sf, _, _ = _reference_quantize(inp, m_splits)
        ref_fp8 = ref_fp8.cuda()
        ref_sf  = ref_sf.cuda()

        return out_fp8, out_sf, ref_fp8, ref_sf

    @pytest.mark.parametrize("K", [128, 256, 1024, 7168])
    @pytest.mark.parametrize("m_splits", [[64], [32, 64, 128], [1, 1, 1]])
    def test_fp8_values_match_reference(self, m_splits, K):
        out_fp8, out_sf, ref_fp8, ref_sf = self._run(m_splits, K)
        assert torch.all(out_fp8 == ref_fp8), (
            f"FP8 data mismatch for m_splits={m_splits}, K={K}\n"
            f"max diff: {(out_fp8.int() - ref_fp8.int()).abs().max()}"
        )

    @pytest.mark.parametrize("K", [128, 256, 1024, 7168])
    @pytest.mark.parametrize("m_splits", [[64], [32, 64, 128]])
    def test_sf_values_match_reference(self, m_splits, K):
        out_fp8, out_sf, ref_fp8, ref_sf = self._run(m_splits, K)
        assert torch.all(out_sf == ref_sf), (
            f"SF mismatch for m_splits={m_splits}, K={K}\n"
            f"nonzero diff positions: {(out_sf != ref_sf).nonzero()[:10]}"
        )

    def test_empty_group_is_handled(self):
        """A group with 0 rows should not crash."""
        _C = _import_module()
        K = 128
        m_splits = [64, 0, 32]
        total_rows = sum(m_splits)
        inp = torch.randn(total_rows, K, dtype=torch.bfloat16).cuda()

        num_groups = len(m_splits)
        h_row_off = [0, 64, 64, 96]
        h_sf_off = [0] + [0] * num_groups
        for g in range(num_groups):
            h_sf_off[g + 1] = h_sf_off[g] + _sf_size_per_group(m_splits[g], K)

        row_offsets = torch.tensor(h_row_off, dtype=torch.int32, device="cuda")
        sf_offsets  = torch.tensor(h_sf_off,  dtype=torch.int64, device="cuda")
        out_fp8 = torch.empty(total_rows, K, dtype=torch.uint8, device="cuda")
        out_sf  = torch.zeros(h_sf_off[-1], dtype=torch.uint8, device="cuda")

        _C.mxfp8_grouped_quantize(out_fp8, out_sf, inp, row_offsets, sf_offsets)
        torch.cuda.synchronize()

    @pytest.mark.parametrize("K", [128, 7168])
    def test_output_shapes(self, K):
        m_splits = [32, 64]
        total_rows = sum(m_splits)
        _C = _import_module()
        inp = torch.randn(total_rows, K, dtype=torch.bfloat16).cuda()
        num_groups = len(m_splits)
        h_row_off = [0, 32, 96]
        h_sf_off = [0, _sf_size_per_group(32, K),
                    _sf_size_per_group(32, K) + _sf_size_per_group(64, K)]
        row_offsets = torch.tensor(h_row_off, dtype=torch.int32, device="cuda")
        sf_offsets  = torch.tensor(h_sf_off,  dtype=torch.int64, device="cuda")
        out_fp8 = torch.empty(total_rows, K, dtype=torch.uint8, device="cuda")
        out_sf  = torch.zeros(h_sf_off[-1], dtype=torch.uint8, device="cuda")
        _C.mxfp8_grouped_quantize(out_fp8, out_sf, inp, row_offsets, sf_offsets)
        torch.cuda.synchronize()
        assert out_fp8.shape == (total_rows, K)
        assert out_sf.numel() == h_sf_off[-1]
