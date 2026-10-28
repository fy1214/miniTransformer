"""Tests for logpdiff.reporter: HTML structure, escaping, plots toggle."""

from __future__ import annotations

import re

import pytest

from logpdiff.reporter import render_report


def _rich_report_data():
    return {
        "run": {
            "run_id": "run-1",
            "git_commit": "abc123",
            "model": "/models/x",
            "versions": {"torch": "2.7.1", "python": "3.13"},
            "env": {"QAT_PARAMS": "8"},
            "seed": 1234,
            "parallel": {"tp": 1, "ep": 1, "pp": 1},
            "force_routing": False,
        },
        "first_failure": {
            "stage": "fc1_gemm",
            "status": "FAIL",
            "description": "cosine below fail threshold",
        },
        "stage_table": [
            {"stage": "layer_input", "status": "PASS",
             "metrics": {"symmetric_rel_l2": 1e-9}},
            {"stage": "fc1_gemm", "status": "FAIL",
             "metrics": {"cosine": 0.5}, "reason": "cosine < 0.9999"},
        ],
        "expert_table": [
            {"layer_id": 0, "expert_id": 0, "max_abs_err": 0.5,
             "mean_abs_err": 0.2, "cosine": 0.5},
            {"layer_id": 0, "expert_id": 1, "max_abs_err": 0.1,
             "mean_abs_err": 0.05, "cosine": 0.99},
        ],
        "worst_tokens": [
            {"sample_id": "sample-0", "token_position": 3, "token_id": 42,
             "ref_logprob": -1.0, "target_logprob": -1.5, "abs_diff": 0.5},
        ],
        "provenance_legend": {
            "actual_kernel": ["layer_input"],
            "replayed_reference": ["fc1_output"],
        },
        "alignment": {"method": "dispatcher_index", "matched": 8,
                      "unmatched": 0, "duplicate": 0, "conflict": 0},
        "samples": [{"sample_id": "sample-0", "text": "hello"}],
    }


class TestSections:
    SECTION_IDS = [
        "run-header",
        "first-failure",
        "stage-table",
        "expert-table",
        "worst-tokens",
        "provenance-legend",
        "alignment",
        "samples",
    ]

    def test_key_sections_present(self, tmp_path):
        out = tmp_path / "report.html"
        render_report(_rich_report_data(), out)
        text = out.read_text(encoding="utf-8")
        for section_id in self.SECTION_IDS:
            assert f'id="{section_id}"' in text
        # first failure content is rendered
        assert "fc1_gemm" in text
        assert "st-fail" in text

    def test_empty_report_data_renders_placeholders(self, tmp_path):
        out = tmp_path / "report.html"
        render_report({}, out)  # must not raise on missing keys
        text = out.read_text(encoding="utf-8")
        for section_id in self.SECTION_IDS:
            assert f'id="{section_id}"' in text
        assert "no stage data." in text

    def test_invalid_alignment_rendering(self, tmp_path):
        data = _rich_report_data()
        data["stage_table"][1]["status"] = "INVALID_ALIGNMENT"
        data["alignment"] = {
            "method": "hash_exact",
            "conflict": 1,
            "invalid_experts": [{"layer_id": 0, "expert_id": 3,
                                 "reason": "shape conflict"}],
        }
        out = tmp_path / "report.html"
        render_report(data, out)
        text = out.read_text(encoding="utf-8")
        assert "INVALID_ALIGNMENT" in text
        assert 'id="invalid-alignment-banner"' in text
        assert "shape conflict" in text

    def test_force_routing_banner(self, tmp_path):
        data = _rich_report_data()
        data["run"]["force_routing"] = True
        out = tmp_path / "report.html"
        render_report(data, out)
        text = out.read_text(encoding="utf-8")
        assert 'id="force-routing-banner"' in text


class TestEscaping:
    EVIL = "<script>alert('xss')</script>"

    def _render(self, tmp_path, data):
        out = tmp_path / "report.html"
        render_report(data, out)
        return out.read_text(encoding="utf-8")

    def test_no_raw_script_tags_anywhere(self, tmp_path):
        data = _rich_report_data()
        data["run"]["run_id"] = self.EVIL
        data["run"]["model"] = self.EVIL
        data["first_failure"]["description"] = self.EVIL
        data["stage_table"][0]["reason"] = self.EVIL
        data["samples"] = [{"sample_id": self.EVIL, "note": self.EVIL}]
        data["worst_tokens"] = [{"sample_id": self.EVIL, "abs_diff": 0.5}]
        text = self._render(tmp_path, data)
        assert "<script>" not in text
        assert "&lt;script&gt;" in text  # escaped form is present instead

    def test_attribute_injection_escaped(self, tmp_path):
        evil = '"><img src=x onerror=alert(1)>'
        data = {"samples": [{"sample_id": evil}]}
        text = self._render(tmp_path, data)
        assert "<img" not in text
        assert "&lt;img" in text

    def test_out_path_parents_created(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "report.html"
        render_report({}, out)
        assert out.is_file()


class TestPlots:
    def test_no_svg_without_plots(self, tmp_path):
        out = tmp_path / "report.html"
        render_report(_rich_report_data(), out, with_plots=False)
        text = out.read_text(encoding="utf-8")
        assert "<svg" not in text
        assert 'id="plots"' not in text

    def test_svg_with_plots(self, tmp_path):
        out = tmp_path / "report.html"
        render_report(_rich_report_data(), out, with_plots=True)
        text = out.read_text(encoding="utf-8")
        assert 'id="plots"' in text
        assert "<svg" in text
        # expert bar chart uses the expert_table error column
        assert "expert error" in text

    def test_plots_section_survives_missing_data(self, tmp_path):
        out = tmp_path / "report.html"
        render_report({}, out, with_plots=True)  # no data: placeholders, no crash
        text = out.read_text(encoding="utf-8")
        assert 'id="plots"' in text
        assert "no expert error data." in text

    def test_svg_labels_escaped(self, tmp_path):
        data = _rich_report_data()
        data["scatter"] = {
            "points": [(1.0, 1.0), (2.0, 2.5)],
            "xlabel": "<script>bad()</script>",
        }
        out = tmp_path / "report.html"
        render_report(data, out, with_plots=True)
        text = out.read_text(encoding="utf-8")
        assert "<script>" not in text


class TestProvenanceLegend:
    def test_actual_and_replay_boxes_separate(self, tmp_path):
        legend = {
            "actual_kernel": ["layer_input", "expert_combined_output"],
            "replayed_reference": ["fc1_output", "fc1_qdata"],
            "derived_metric": ["token_logprobs"],
        }
        out = tmp_path / "report.html"
        render_report({"provenance_legend": legend}, out)
        text = out.read_text(encoding="utf-8")

        def box(cls):
            m = re.search(
                rf'<div class="cov-box {cls}">(.*?)</div>', text, re.S
            )
            assert m, f"coverage box {cls} missing"
            return m.group(1)

        actual = box("cov-actual")
        replay = box("cov-replay")
        assert "layer_input" in actual
        assert "fc1_output" not in actual
        assert "fc1_output" in replay
        assert "layer_input" not in replay
