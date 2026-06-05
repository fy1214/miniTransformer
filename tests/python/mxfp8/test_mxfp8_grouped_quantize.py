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


# ===========================================================================
# Precision (accuracy) test: quantize -> dequantize -> SNR vs original bf16.
#
# Reconstructs float values from our FP8 E4M3 data + swizzled E8M0 scale
# factors (inverse of cvt_quant_to_fp4_get_sf_out_offset), then measures
# signal-to-noise ratio against the original bf16 input.  Always-on; does
# not require TransformerEngine.
# ===========================================================================


class TestMxFP8GroupedQuantizePrecision:
    @staticmethod
    def _dequantize(out_fp8, sf, m_splits, K):
        """Reconstruct float [total_rows, K] from our FP8 data + swizzled SF."""
        Kg = K // 32
        nkt = (Kg + 3) // 4
        fp8 = out_fp8.detach().cpu().view(torch.float8_e4m3fn).float()
        sf_cpu = sf.detach().cpu()
        deq = torch.empty_like(fp8)
        r0 = 0
        sfb = 0
        for m in m_splits:
            sf_g = sf_cpu[sfb: sfb + _sf_size_per_group(m, K)]
            ri = torch.arange(m).view(m, 1)
            kg = torch.arange(Kg).view(1, Kg)
            # inverse of _cvt_sf_offset, vectorized over (row, k-group)
            off = ((((ri >> 7) * nkt + (kg >> 2)) << 9)
                   | ((ri & 31) << 4) | (((ri >> 5) & 3) << 2) | (kg & 3)).long()
            e8 = sf_g[off].to(torch.int32)                       # [m, Kg]
            scale = torch.where(e8 > 0, torch.exp2((e8 - 127).float()),
                                torch.ones_like(e8, dtype=torch.float32))
            deq[r0:r0 + m] = fp8[r0:r0 + m] * scale.repeat_interleave(32, dim=1)
            r0 += m
            sfb += _sf_size_per_group(m, K)
        return deq

    def _quantize(self, m_splits, K, seed=0):
        _C = _import_module()
        torch.manual_seed(seed)
        total = sum(m_splits)
        inp = torch.randn(total, K, dtype=torch.bfloat16).cuda()
        ng = len(m_splits)
        h_row = [0] * (ng + 1)
        h_sf = [0] * (ng + 1)
        for g, m in enumerate(m_splits):
            h_row[g + 1] = h_row[g] + m
            h_sf[g + 1] = h_sf[g] + _sf_size_per_group(m, K)
        row_off = torch.tensor(h_row, dtype=torch.int32, device="cuda")
        sf_off = torch.tensor(h_sf, dtype=torch.int64, device="cuda")
        out_fp8 = torch.empty(total, K, dtype=torch.uint8, device="cuda")
        out_sf = torch.zeros(h_sf[-1], dtype=torch.uint8, device="cuda")
        _C.mxfp8_grouped_quantize(out_fp8, out_sf, inp, row_off, sf_off)
        torch.cuda.synchronize()
        return inp, out_fp8, out_sf

    @pytest.mark.parametrize("K", [512, 1024, 4096])
    @pytest.mark.parametrize("m_splits", [[128], [32, 64, 128], [64, 64, 64, 64]])
    def test_dequant_snr(self, m_splits, K):
        inp, out_fp8, out_sf = self._quantize(m_splits, K)
        deq = self._dequantize(out_fp8, out_sf, m_splits, K).cuda()
        ref = inp.float()
        assert torch.isfinite(deq).all(), "dequantized output contains NaN/Inf"
        snr = 10 * torch.log10((ref ** 2).mean() / ((ref - deq) ** 2).mean()).item()
        max_rel = ((ref - deq).abs() / ref.abs().clamp_min(1e-6)).max().item()
        print(f"\n[precision] m_splits={m_splits} K={K} -> "
              f"SNR {snr:.2f} dB, max_rel_err {max_rel:.4f}")
        # MXFP8 E4M3 with per-32 block scaling: SNR ~31 dB on N(0,1) data.
        assert snr > 28.0, (
            f"dequant SNR {snr:.2f} dB below 28 dB (m_splits={m_splits}, K={K})")


# ===========================================================================
# Performance benchmark: our fused mxfp8_grouped_quantize  vs TransformerEngine.
#
# TE 2.10 (PyTorch) has NO single `group_quantize` kernel.  Its dedicated
# grouped-quant entry points both launch per-group work:
#   * tex.split_quantize(packed, m_splits, quantizers)   <- packed [total_M,K]
#   * tex.multi_tensor_quantize(tensor_list, quantizers) <- pre-split list
# (`grouped_quantize` exists only in TE's JAX backend.)
# Our kernel quantizes all groups in ONE launch.
#
# Run:  pytest -s tests/python/mxfp8/test_mxfp8_grouped_quantize.py -k Perf
# (-s is required to see the printed timing table)
# ===========================================================================


def _import_te():
    """Return (tex, MXFP8Quantizer, E4M3_dtype) or None if TE unavailable."""
    try:
        import transformer_engine.pytorch as te  # noqa: F401  (triggers ext load)
        import transformer_engine_torch as tex_torch
        import transformer_engine.pytorch.cpp_extensions as tex
        from transformer_engine.pytorch import MXFP8Quantizer
        return tex, MXFP8Quantizer, tex_torch.DType.kFloat8E4M3
    except Exception:
        return None


def _import_wrapper():
    try:
        from miniTransformer.ops.mxfp8_quantize import mxfp8_grouped_quantize
        return mxfp8_grouped_quantize
    except Exception:
        return None


def _cuda_time(fn, warmup: int = 15, iters: int = 100) -> float:
    """Mean wall time per call in microseconds (CUDA events)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # ms -> us


# (num_groups, rows_per_group, K) — representative MoE grouped-GEMM shapes.
_PERF_CONFIGS = [
    (8,   128, 4096),
    (8,   512, 4096),
    (16,  256, 7168),
    (64,  128, 2048),
    (64,  128, 1024),
    (128, 64,  2048),
]


@pytest.mark.skipif(_import_te() is None, reason="TransformerEngine not available")
class TestMxFP8GroupedQuantizePerf:
    """Benchmark our fused kernel against TE split_quantize / multi_tensor_quantize."""

    def _make(self, num_groups, rows_per_group, K, seed=0):
        torch.manual_seed(seed)
        m_splits = [rows_per_group] * num_groups
        total_M = sum(m_splits)
        inp = torch.randn(total_M, K, dtype=torch.bfloat16, device="cuda")
        return inp, m_splits

    def test_fp8_data_matches_te(self):
        """Equivalence guard: our FP8 E4M3 bytes must be bit-identical to BOTH
        TE grouped paths (split_quantize and multi_tensor_quantize).

        SF bytes legitimately differ: TE stores [padM, K_groups] 2D while we
        store the flat swizzled-atom buffer; both are GEMM-correct.  FP8 data
        is layout-free, so bit-equality there proves identical quantization.
        """
        tex, MXFP8Quantizer, E4M3 = _import_te()
        mxfp8_grouped_quantize = _import_wrapper()
        assert mxfp8_grouped_quantize is not None

        for num_groups, rpg, K in [(8, 128, 4096), (16, 256, 7168), (3, 96, 512)]:
            inp, m_splits = self._make(num_groups, rpg, K)
            out, sf, row_off, sf_off = mxfp8_grouped_quantize(inp, m_splits)
            qs = [MXFP8Quantizer(E4M3, rowwise=True, columnwise=False)
                  for _ in range(num_groups)]
            te_split = tex.split_quantize(inp, m_splits, qs)
            qs2 = [MXFP8Quantizer(E4M3, rowwise=True, columnwise=False)
                   for _ in range(num_groups)]
            te_mtq = tex.multi_tensor_quantize(list(torch.split(inp, m_splits)), qs2)
            r0 = 0
            for g, m in enumerate(m_splits):
                ours = out[r0:r0 + m]
                assert torch.equal(ours, te_split[g]._rowwise_data), (
                    f"FP8 mismatch vs split_quantize g={g} G={num_groups} K={K}")
                assert torch.equal(ours, te_mtq[g]._rowwise_data), (
                    f"FP8 mismatch vs multi_tensor_quantize g={g} G={num_groups} K={K}")
                r0 += m

    def test_perf_vs_te(self):
        """Print a timing table; assert kernel beats both TE grouped paths on
        multi-group shapes."""
        tex, MXFP8Quantizer, E4M3 = _import_te()
        import fp4_gemm
        mxfp8_grouped_quantize = _import_wrapper()
        assert mxfp8_grouped_quantize is not None

        def _sf_bytes(m_splits, K):
            return sum(_sf_size_per_group(m, K) for m in m_splits)

        print(f"\n{'='*120}")
        print(f"MXFP8 grouped quantize: ours (fused, 1 launch) vs TE grouped paths  "
              f"[{torch.cuda.get_device_name(0)}]")
        print(f"{'='*120}")
        print(f"{'G':>4} {'M':>7} {'K':>6} | "
              f"{'kernel':>8} {'wrap':>8} {'split_q':>9} {'mtq':>9} (us) | "
              f"{'kern GB/s':>9} | {'k/split':>8} {'k/mtq':>8} {'wrap/split':>10}")
        print(f"{'-'*120}")

        worst_split_speedup = float("inf")
        worst_mtq_speedup = float("inf")
        for num_groups, rpg, K in _PERF_CONFIGS:
            inp, m_splits = self._make(num_groups, rpg, K)
            total_M = sum(m_splits)

            out, sf, row_off, sf_off = mxfp8_grouped_quantize(inp, m_splits)
            t_kern = _cuda_time(
                lambda: fp4_gemm.mxfp8_grouped_quantize(out, sf, inp, row_off, sf_off))
            t_wrap = _cuda_time(lambda: mxfp8_grouped_quantize(inp, m_splits))

            qs = [MXFP8Quantizer(E4M3, rowwise=True, columnwise=False)
                  for _ in range(num_groups)]
            t_split = _cuda_time(lambda: tex.split_quantize(inp, m_splits, qs))
            xs = list(torch.split(inp, m_splits))
            qs2 = [MXFP8Quantizer(E4M3, rowwise=True, columnwise=False)
                   for _ in range(num_groups)]
            t_mtq = _cuda_time(lambda: tex.multi_tensor_quantize(xs, qs2))

            rw_bytes = total_M * K * (2 + 1) + _sf_bytes(m_splits, K)
            kern_gbps = rw_bytes / (t_kern * 1e-6) / 1e9

            print(f"{num_groups:>4} {total_M:>7} {K:>6} | "
                  f"{t_kern:>8.1f} {t_wrap:>8.1f} {t_split:>9.1f} {t_mtq:>9.1f}      | "
                  f"{kern_gbps:>9.0f} | "
                  f"{t_split / t_kern:>7.2f}x {t_mtq / t_kern:>7.2f}x "
                  f"{t_split / t_wrap:>9.2f}x")

            if num_groups >= 16:
                worst_split_speedup = min(worst_split_speedup, t_split / t_kern)
                worst_mtq_speedup = min(worst_mtq_speedup, t_mtq / t_kern)
        print(f"{'='*120}")

        # Fused kernel must beat both TE grouped paths on multi-group shapes
        # (observed: 4-25x vs split_quantize, 7-47x vs multi_tensor_quantize).
        assert worst_split_speedup > 1.0, (
            f"fused kernel slower than TE split_quantize "
            f"(min {worst_split_speedup:.2f}x)")
        assert worst_mtq_speedup > 1.0, (
            f"fused kernel slower than TE multi_tensor_quantize "
            f"(min {worst_mtq_speedup:.2f}x)")
