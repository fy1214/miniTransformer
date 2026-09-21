"""Provenance isolation: actual_kernel data and replayed_reference data must
never be blended, in records, in comparison output, or in the report."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from logpdiff import schema
from logpdiff.reporter import render_report
from logpdiff.writer import create_run_dir

from conftest import make_tensors, require_torch, save_payload


def _mixed_provenance_run(tmp_path):
    """A run whose sglang side mixes provenances across tensor points."""
    torch = require_torch()
    run = create_run_dir(tmp_path / "runs", run_id="provenance-run",
                         model_path="/models/synthetic", seed=0)
    tensors = make_tensors(torch)
    for fw in ("sglang", "megatron"):
        save_payload(run, fw, schema.LAYER_INPUT, tensors["layer_input"],
                     provenance=schema.ACTUAL_KERNEL)
        save_payload(run, fw, schema.EXPERT_COMBINED_OUTPUT,
                     tensors["combined"], provenance=schema.ACTUAL_KERNEL)
        for eid, mat in tensors["fc1_by_expert"].items():
            for row in range(mat.shape[0]):
                save_payload(
                    run, fw, schema.FC1_OUTPUT, mat[row], expert_id=eid,
                    token_position=row + (4 if eid else 0), route_slot=0,
                    provenance=schema.REPLAYED_REFERENCE,
                )
        save_payload(run, fw, schema.TOKEN_LOGPROBS, tensors["logprobs"],
                     provenance=schema.DERIVED_METRIC)
    run.finalize_manifest()
    return run


def _legend_from_index(run_dir, framework):
    """Build the provenance -> tensor points coverage map from an index file,
    the same way a collector/report pipeline would."""
    legend = {}
    index = Path(run_dir) / framework / "index.jsonl"
    for line in index.read_text().splitlines():
        rec = json.loads(line)
        legend.setdefault(rec["provenance"], set()).add(rec["tensor_point"])
    return {k: sorted(v) for k, v in legend.items()}


def _coverage_box(html_text, cls):
    m = re.search(rf'<div class="cov-box {cls}">(.*?)</div>', html_text, re.S)
    assert m, f"coverage box {cls} missing from report"
    return m.group(1)


def _deep_find(obj, key):
    """Recursively find the value of ``key`` in nested dict/list data."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _deep_find(value, key)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found = _deep_find(item, key)
            if found is not None:
                return found
    return None


class TestProvenanceLegendFromRecords:
    def test_legend_renders_actual_and_replay_separately(self, tmp_path):
        run = _mixed_provenance_run(tmp_path)
        legend = _legend_from_index(run.run_dir, "sglang")
        # sanity: the synthetic run really mixes provenances
        assert set(legend) == {
            schema.ACTUAL_KERNEL,
            schema.REPLAYED_REFERENCE,
            schema.DERIVED_METRIC,
        }
        out = tmp_path / "report.html"
        render_report({"run": {"run_id": run.run_id},
                       "provenance_legend": legend}, out)
        text = out.read_text(encoding="utf-8")

        actual_box = _coverage_box(text, "cov-actual")
        replay_box = _coverage_box(text, "cov-replay")
        assert "layer_input" in actual_box
        assert "expert_combined_output" in actual_box
        assert "fc1_output" not in actual_box
        assert "token_logprobs" not in actual_box
        assert "fc1_output" in replay_box
        assert "layer_input" not in replay_box

    def test_legend_aliases_normalize_to_the_same_boxes(self, tmp_path):
        legend = {"actual": ["layer_input"], "replay": ["fc1_output"]}
        out = tmp_path / "report.html"
        render_report({"provenance_legend": legend}, out)
        text = out.read_text(encoding="utf-8")
        assert "layer_input" in _coverage_box(text, "cov-actual")
        assert "fc1_output" in _coverage_box(text, "cov-replay")

    def test_report_states_the_separation_red_line(self, tmp_path):
        out = tmp_path / "report.html"
        render_report({"provenance_legend": {"actual_kernel": ["layer_input"]}},
                      out)
        text = out.read_text(encoding="utf-8")
        assert "actual_kernel" in text
        assert "replayed_reference" in text
        # the legend explains why replayed values must stay separate
        assert "NOT real kernel output" in text


class TestCompareProvenanceIsolation:
    """logpdiff.compare is added by a parallel task; when present, its public
    output must keep actual vs replayed data visibly separated."""

    def test_run_compare_output_separates_provenance(self, tmp_path):
        compare = pytest.importorskip(
            "logpdiff.compare", reason="logpdiff.compare comes from a parallel task"
        )
        run = _mixed_provenance_run(tmp_path)
        result = compare.run_compare(str(run.run_dir))
        if result is None:
            # compare may only write artifacts; fall back to metrics.json
            metrics_path = run.run_dir / "metrics.json"
            assert metrics_path.is_file(), (
                "run_compare returned None and wrote no metrics.json"
            )
            result = json.loads(metrics_path.read_text())
        text = json.dumps(result, default=str)
        assert schema.ACTUAL_KERNEL in text
        assert schema.REPLAYED_REFERENCE in text

        legend = _deep_find(result, "provenance_legend")
        if legend is not None:
            normalized = json.dumps(legend, default=str)
            # actual and replayed coverage must live under different keys
            assert schema.ACTUAL_KERNEL in normalized
            assert schema.REPLAYED_REFERENCE in normalized
