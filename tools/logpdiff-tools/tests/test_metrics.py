"""Tests for logpdiff.metrics: known-value checks and integrity rules."""

from __future__ import annotations

import math

import pytest

from logpdiff.metrics import (
    DEFAULT_THRESHOLDS,
    equal_summary,
    evaluate_stages,
    find_first_failure,
    fp4_underflow_ratio,
    logprob_metrics,
    routing_metrics,
    tensor_metrics,
    weighted_summary,
    worst_entries,
)
from logpdiff.schema import PIPELINE_STAGES

from conftest import require_torch


class TestTensorMetrics:
    def test_identical_tensors(self):
        torch = require_torch()
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        m = tensor_metrics(a, a.clone())
        assert m["exact_match"] is True
        assert m["shape_match"] is True
        assert m["dtype_match"] is True
        assert m["checksum_match"] is True
        assert m["max_abs_error"] == 0.0
        assert m["mean_abs_error"] == 0.0
        assert m["mse"] == 0.0
        assert m["sqnr_db"] == float("inf")
        assert m["cosine"] == 1.0
        assert m["symmetric_rel_l2"] == pytest.approx(0.0)
        assert "error" not in m

    def test_known_values(self):
        torch = require_torch()
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b = torch.tensor([1.0, 2.0, 3.0, 5.0])  # single unit error at the end
        m = tensor_metrics(a, b)
        assert m["exact_match"] is False
        assert m["mean_abs_error"] == pytest.approx(0.25)
        assert m["max_abs_error"] == pytest.approx(1.0)
        assert m["mse"] == pytest.approx(0.25)
        assert m["rmse"] == pytest.approx(0.5)
        # sorted abs errors are [0, 0, 0, 1]; linear-interpolation percentiles
        assert m["p50_abs_error"] == pytest.approx(0.0)
        assert m["p95_abs_error"] == pytest.approx(0.85)
        assert m["p99_abs_error"] == pytest.approx(0.97)
        # SQNR = 10*log10(||a||^2 / ||a-b||^2) = 10*log10(30/1)
        assert m["sqnr_db"] == pytest.approx(10.0 * math.log10(30.0))
        # cosine = 34 / (sqrt(30) * sqrt(39))
        assert m["cosine"] == pytest.approx(34.0 / math.sqrt(30.0 * 39.0))
        assert m["symmetric_rel_l2"] == pytest.approx(
            1.0 / (math.sqrt(30.0) + math.sqrt(39.0) + 1e-12)
        )
        assert m["zero_count_a"] == 0
        assert m["nan_count_a"] == 0

    def test_percentile_single_element(self):
        torch = require_torch()
        m = tensor_metrics(torch.tensor([2.0]), torch.tensor([5.0]))
        assert m["p50_abs_error"] == pytest.approx(3.0)
        assert m["p95_abs_error"] == pytest.approx(3.0)

    def test_shape_mismatch_never_truncated(self):
        torch = require_torch()
        a = torch.ones(4)
        b = torch.ones(3)
        m = tensor_metrics(a, b)
        assert m["shape_match"] is False
        assert m["error"]  # explains the mismatch instead of truncating
        assert "no silent truncation" in m["error"]
        for key in ("exact_match", "mean_abs_error", "max_abs_error",
                    "sqnr_db", "symmetric_rel_l2", "cosine"):
            assert m[key] is None
        # shape-independent metrics are still computed
        assert m["numel_a"] == 4 and m["numel_b"] == 3
        assert m["zero_count_a"] == 0

    def test_nan_and_inf_counts(self):
        torch = require_torch()
        a = torch.tensor([float("nan"), 1.0, float("inf")])
        b = torch.tensor([0.0, 1.0, 2.0])
        m = tensor_metrics(a, b)
        assert m["nan_count_a"] == 1
        assert m["inf_count_a"] == 1
        assert m["nan_count_b"] == 0

    def test_empty_tensors(self):
        torch = require_torch()
        m = tensor_metrics(torch.empty(0), torch.empty(0))
        assert m["exact_match"] is True
        assert m["cosine"] is None
        assert "empty tensors" in m["error"]


class TestFp4UnderflowRatio:
    def test_known_ratio(self):
        torch = require_torch()
        a = torch.tensor([1.0, 0.0, 2.0, 3.0])
        b = torch.tensor([0.0, 0.0, 2.0, 3.0])
        assert fp4_underflow_ratio(a, b) == pytest.approx(1.0 / 3.0)

    def test_all_zero_reference(self):
        torch = require_torch()
        assert fp4_underflow_ratio(torch.zeros(4), torch.zeros(4)) == 0.0

    def test_shape_mismatch_raises(self):
        torch = require_torch()
        with pytest.raises(ValueError, match="shape mismatch"):
            fp4_underflow_ratio(torch.ones(4), torch.ones(3))


class TestRoutingMetrics:
    def test_perfect_match(self):
        torch = require_torch()
        ids = torch.tensor([[0, 1], [1, 2], [0, 2]])
        weights = torch.tensor([[0.6, 0.4], [0.5, 0.5], [0.7, 0.3]])
        m = routing_metrics(ids, weights, ids.clone(), weights.clone())
        assert m["num_tokens"] == 3
        assert m["top_k"] == 2
        assert m["position_match_ratio"] == 1.0
        assert m["set_match_ratio"] == 1.0
        assert m["first_mismatch_position"] is None
        assert m["weight_error_mean"] == 0.0
        assert m["weight_error_max"] == 0.0

    def test_known_mismatches(self):
        torch = require_torch()
        ids_a = torch.tensor([[0, 1], [1, 2], [0, 2]])
        ids_b = torch.tensor([[0, 1], [2, 1], [0, 3]])
        w = torch.zeros(3, 2)
        m = routing_metrics(ids_a, w, ids_b, w)
        # slot matches: token0 both, token1 none (swap), token2 slot0 only
        assert m["position_match_ratio"] == pytest.approx(3.0 / 6.0)
        # token1 keeps the same SET (order-insensitive), token2 does not
        assert m["set_match_ratio"] == pytest.approx(2.0 / 3.0)
        assert m["first_mismatch_position"] == 1
        assert m["topk_overlap_min"] == 1  # token2 shares only expert 0

    def test_token_count_mismatch_raises(self):
        torch = require_torch()
        ids_a = torch.zeros(3, 2, dtype=torch.long)
        ids_b = torch.zeros(2, 2, dtype=torch.long)
        w = torch.zeros(3, 2)
        with pytest.raises(ValueError, match="token count mismatch"):
            routing_metrics(ids_a, w, ids_b, torch.zeros(2, 2))

    def test_zero_tokens_raise(self):
        torch = require_torch()
        with pytest.raises(ValueError, match="no tokens"):
            routing_metrics(
                torch.empty(0, 2, dtype=torch.long), torch.empty(0, 2),
                torch.empty(0, 2, dtype=torch.long), torch.empty(0, 2),
            )

    def test_full_scores_give_boundary_margins(self):
        torch = require_torch()
        ids = torch.tensor([[0], [1]])
        # full router scores: clear margins for both tokens
        scores_a = torch.tensor([[3.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
        scores_b = torch.tensor([[2.5, 1.0, 0.0], [0.5, 2.0, 1.5]])
        m = routing_metrics(ids, scores_a, ids.clone(), scores_b)
        assert m["boundary_margin_mean_a"] == pytest.approx(2.0)
        assert m["boundary_margin_mean_b"] == pytest.approx(1.0)
        assert m["num_experts"] == 3
        assert m["expert_load_a"] == [1, 1, 0]


class TestLogprobMetrics:
    def test_over_threshold_count(self):
        torch = require_torch()
        lp_a = torch.tensor([0.0, -1.0, -2.0])
        lp_b = torch.tensor([0.05, -1.5, -2.0])
        m = logprob_metrics(lp_a, lp_b, abs_threshold=0.1)
        assert m["num_tokens"] == 3
        assert m["tokens_over_threshold"] == 1  # only the 0.5 diff
        assert m["mean_abs_error"] == pytest.approx(0.55 / 3.0)
        assert m["max_abs_error"] == pytest.approx(0.5)
        assert m["p95_abs_error"] == pytest.approx(0.455)
        # logit-derived metrics are None without logits
        assert m["greedy_mismatch_count"] is None
        assert m["kl_divergence_mean"] is None

    def test_shape_mismatch_raises(self):
        torch = require_torch()
        with pytest.raises(ValueError, match="logprob shape mismatch"):
            logprob_metrics(torch.zeros(4), torch.zeros(3))

    def test_logits_must_come_in_pairs(self):
        torch = require_torch()
        lp = torch.zeros(3)
        with pytest.raises(ValueError, match="together"):
            logprob_metrics(lp, lp, logits_a=torch.zeros(3, 5))

    def test_logit_derived_metrics(self):
        torch = require_torch()
        lp_a = torch.tensor([0.0, -1.0, -2.0])
        lp_b = torch.tensor([0.05, -1.5, -2.0])
        logits_a = torch.tensor(
            [[5.0, 1, 0, 0, 0], [0, 5, 1, 0, 0], [0, 1, 5, 0, 0]]
        )
        logits_b = torch.tensor(
            [[5.0, 1, 0, 0, 0], [5, 0, 1, 0, 0], [0, 1, 5, 0, 0]]
        )
        m = logprob_metrics(lp_a, lp_b, logits_a=logits_a, logits_b=logits_b,
                            top_k=2)
        assert m["greedy_mismatch_count"] == 1  # token 1 flips argmax
        assert m["top_k"] == 2
        assert m["topk_mismatch_count"] == 1
        assert m["kl_divergence_mean"] >= 0.0
        assert 0.0 <= m["js_divergence_mean"] <= math.log(2.0)


class TestThresholdsAndFirstFailure:
    def test_evaluate_stages_levels(self):
        thresholds = DEFAULT_THRESHOLDS
        stage_metrics = {
            "layer_input": {"symmetric_rel_l2": 1e-9},   # PASS
            "dispatch": {"symmetric_rel_l2": 1e-5},      # WARN
            "fc1_gemm": {"cosine": 0.5},                 # FAIL
        }
        status = evaluate_stages(stage_metrics, thresholds)
        assert status == {
            "layer_input": "PASS",
            "dispatch": "WARN",
            "fc1_gemm": "FAIL",
        }

    def test_nan_metric_bumps_to_warn(self):
        status = evaluate_stages(
            {"layer_input": {"symmetric_rel_l2": float("nan")}}
        )
        assert status["layer_input"] == "WARN"

    def test_unknown_stage_is_pass(self):
        assert evaluate_stages({"not_a_stage": {"x": 1}}) == {
            "not_a_stage": "PASS"
        }

    def test_first_failure_follows_pipeline_order(self):
        # FAIL beats WARN regardless of pipeline position
        assert find_first_failure(
            {"layer_input": "WARN", "logprob": "FAIL"}
        ) == ("logprob", "FAIL")
        # earliest FAIL in PIPELINE_STAGES order
        assert find_first_failure(
            {"logprob": "FAIL", "routing": "FAIL"}
        ) == ("routing", "FAIL")
        # earliest WARN when nothing fails
        assert find_first_failure(
            {"fc2_gemm": "WARN", "dispatch": "WARN"}
        ) == ("dispatch", "WARN")
        assert find_first_failure({s: "PASS" for s in PIPELINE_STAGES}) == (
            None,
            "PASS",
        )
        # unknown stage names are ignored
        assert find_first_failure({"bogus_stage": "FAIL"}) == (None, "PASS")


class TestSummaries:
    def test_weighted_summary(self):
        per_expert = {0: {"err": 1.0, "note": "x"}, 1: {"err": 3.0}}
        summary = weighted_summary(per_expert, {0: 1, 1: 3})
        assert summary["err"] == pytest.approx((1.0 * 1 + 3.0 * 3) / 4.0)
        assert summary["total_tokens"] == 4
        assert summary["note"] is None  # non-numeric values are skipped

    def test_weighted_summary_missing_weight_raises(self):
        with pytest.raises(KeyError):
            weighted_summary({0: {"err": 1.0}}, {})

    def test_equal_summary(self):
        summary = equal_summary({0: {"err": 1.0}, 1: {"err": 3.0}})
        assert summary["err"] == pytest.approx(2.0)

    def test_worst_entries_sorted_descending(self):
        entries = [
            {"expert_id": 0, "max_abs_err": 0.1},
            {"expert_id": 1, "max_abs_err": 0.5},
            {"expert_id": 2, "max_abs_err": None},
        ]
        worst = worst_entries(entries, "max_abs_err", topk=5)
        assert [w["value"] for w in worst] == [0.5, 0.1]
        assert worst[0]["entry"]["expert_id"] == 1
