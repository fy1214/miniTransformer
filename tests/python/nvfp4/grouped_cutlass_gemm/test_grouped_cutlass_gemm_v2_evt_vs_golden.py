"""Precision validation for grouped_cutlass_gemm_v2_evt vs the v2 path and a
Python golden reference.

The EVT path differs from the unfused v2 path only in the post-GEMM scale:

  v2     :  D = bf16(acc) * bf16(scale)        (two bf16 muls, post-GEMM kernel)
  v2_evt :  D = bf16(acc_fp32 * scale_fp32)    (one fp32 mul, fused in epilogue)

Tolerances mirror test_grouped_cutlass_gemm_v2_vs_golden.py — v2_evt should be
*strictly tighter* against the golden than v2 (the EVT path skips the
bf16-domain multiply, so accumulated rounding noise drops by ~36 dB).

Run:
    pytest test_grouped_cutlass_gemm_v2_evt_vs_golden.py -v -s
"""
from __future__ import annotations

import pytest
import torch

# Reuse the v2 test's golden + input builder — they are pure helpers and do not
# call the kernel, so they apply equally to the EVT path.
from tests.python.nvfp4.grouped_cutlass_gemm.test_grouped_cutlass_gemm_v2_vs_golden import (
    _build_v2_inputs,
    _error_metrics,
    _require_sm100,
    grouped_cutlass_gemm_v2_golden,
    V2_CASES,
)

from miniTransformer.ops.nvfp4_grouped_cutlass_gemm import (
    grouped_cutlass_gemm_v2,
    grouped_cutlass_gemm_v2_evt,
)


class TestGroupedCutlassGemmV2EvtVsGolden:
    """Validate v2_evt output and confirm it matches/improves on v2."""

    # --- v2_evt vs golden -------------------------------------------------

    @pytest.mark.parametrize("m_splits, n, k", V2_CASES)
    @pytest.mark.parametrize("seed", [42, 7])
    def test_v2_evt_vs_golden(self, m_splits, n, k, seed):
        _require_sm100()
        device = "cuda"

        inputs, _, _ = _build_v2_inputs(m_splits, n, k, device, seed)

        kernel_out = grouped_cutlass_gemm_v2_evt(inputs, out_dtype=torch.bfloat16)
        golden_out = grouped_cutlass_gemm_v2_golden(inputs, out_dtype=torch.bfloat16)

        assert kernel_out.shape == golden_out.shape
        assert not torch.isnan(kernel_out).any(), "NaN in v2_evt output"
        assert not torch.isinf(kernel_out).any(), "Inf in v2_evt output"

        m = _error_metrics(kernel_out, golden_out)
        print(
            f"\n  m_splits={m_splits} n={n} k={k} seed={seed}\n"
            f"  v2_evt vs golden:"
            f"  max_abs={m['max_abs']:.4f}"
            f"  mean_abs={m['mean_abs']:.4f}"
            f"  norm_rmse={m['norm_rmse']:.4f}"
            f"  cosine={m['cosine']:.6f}"
        )

        # Same loose bounds as the v2 test (EVT should always pass these too)
        assert m["max_abs"]   < 0.15,  f"max_abs={m['max_abs']:.4f} ≥ 0.15"
        assert m["mean_abs"]  < 0.03,  f"mean_abs={m['mean_abs']:.4f} ≥ 0.03"
        assert m["norm_rmse"] < 0.10,  f"norm_rmse={m['norm_rmse']:.4f} ≥ 0.10"
        assert m["cosine"]    > 0.999, f"cosine={m['cosine']:.6f} ≤ 0.999"

    # --- v2_evt should be at least as accurate as v2 ----------------------

    @pytest.mark.parametrize("m_splits, n, k", V2_CASES)
    @pytest.mark.parametrize("seed", [42])
    def test_v2_evt_no_worse_than_v2(self, m_splits, n, k, seed):
        """v2_evt does the per-token scale multiply in FP32 (vs BF16 in v2),
        so its norm_rmse / max_abs against the golden should be no worse than
        v2's — typically much better."""
        _require_sm100()
        device = "cuda"

        inputs, _, _ = _build_v2_inputs(m_splits, n, k, device, seed)

        out_v2  = grouped_cutlass_gemm_v2(inputs,     out_dtype=torch.bfloat16)
        out_evt = grouped_cutlass_gemm_v2_evt(inputs, out_dtype=torch.bfloat16)
        golden  = grouped_cutlass_gemm_v2_golden(inputs, out_dtype=torch.bfloat16)

        m_v2  = _error_metrics(out_v2,  golden)
        m_evt = _error_metrics(out_evt, golden)

        print(
            f"\n  m_splits={m_splits} n={n} k={k}\n"
            f"  v2:     norm_rmse={m_v2['norm_rmse']:.5f}  cos={m_v2['cosine']:.6f}\n"
            f"  v2_evt: norm_rmse={m_evt['norm_rmse']:.5f}  cos={m_evt['cosine']:.6f}\n"
            f"  delta:  norm_rmse={m_evt['norm_rmse'] - m_v2['norm_rmse']:+.5f}  "
            f"cos={m_evt['cosine'] - m_v2['cosine']:+.6f}"
        )

        # Small slack (1%) for measurement noise; in practice EVT is ~36 dB tighter.
        assert m_evt["norm_rmse"] <= m_v2["norm_rmse"] * 1.01 + 1e-5, (
            f"v2_evt norm_rmse {m_evt['norm_rmse']:.5f} regressed vs v2 {m_v2['norm_rmse']:.5f}"
        )
        assert m_evt["cosine"]    >= m_v2["cosine"] - 1e-4, (
            f"v2_evt cosine {m_evt['cosine']:.6f} regressed vs v2 {m_v2['cosine']:.6f}"
        )

    # --- v2 vs v2_evt sanity ---------------------------------------------

    @pytest.mark.parametrize("m_splits, n, k", V2_CASES)
    def test_v2_evt_close_to_v2(self, m_splits, n, k):
        """v2_evt and v2 should differ only in the per-token-scale dtype path —
        the difference should be small (cosine > 0.999) but not zero."""
        _require_sm100()
        device = "cuda"
        seed = 0

        inputs, _, _ = _build_v2_inputs(m_splits, n, k, device, seed)

        out_v2  = grouped_cutlass_gemm_v2(inputs,     out_dtype=torch.bfloat16)
        out_evt = grouped_cutlass_gemm_v2_evt(inputs, out_dtype=torch.bfloat16)

        m = _error_metrics(out_evt, out_v2)
        print(
            f"\n  m_splits={m_splits} n={n} k={k}  v2 vs v2_evt:\n"
            f"  max_abs={m['max_abs']:.4f}  norm_rmse={m['norm_rmse']:.4f}  "
            f"cosine={m['cosine']:.6f}"
        )

        # Two paths should agree to within bf16 LSB-level noise
        assert m["cosine"]    > 0.999,  f"cosine={m['cosine']:.6f} too low"
        assert m["norm_rmse"] < 0.05,   f"norm_rmse={m['norm_rmse']:.5f} too large"

    # --- multi-seed robustness -------------------------------------------

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_v2_evt_seed_sweep(self, seed):
        _require_sm100()
        m_splits, n, k = [128] * 4, 4096, 2048
        device = "cuda"

        inputs, _, _ = _build_v2_inputs(m_splits, n, k, device, seed)
        kernel_out = grouped_cutlass_gemm_v2_evt(inputs, out_dtype=torch.bfloat16)
        golden_out = grouped_cutlass_gemm_v2_golden(inputs, out_dtype=torch.bfloat16)

        m = _error_metrics(kernel_out, golden_out)
        assert m["max_abs"]  < 0.15, f"seed={seed} max_abs={m['max_abs']:.4f}"
        assert m["mean_abs"] < 0.03, f"seed={seed} mean_abs={m['mean_abs']:.4f}"
        assert m["cosine"]   > 0.999, f"seed={seed} cosine={m['cosine']:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
