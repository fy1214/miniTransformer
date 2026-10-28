"""CPU-only tests for the lightweight monitoring extensions."""

from __future__ import annotations

import csv
import json
import math
from types import SimpleNamespace

import pytest
from logpdiff import monitoring
from logpdiff.compare import run_compare
from logpdiff.monitoring import (
    aggregate_sweeps,
    bootstrap_amplification,
    compute_rl_amplification,
    export_routing,
    parse_layers,
    run_layer_sweep,
    write_launch_meta,
)
from logpdiff.schema import DERIVED_METRIC, TOKEN_LOGPROBS, TensorRecord


def test_write_launch_meta_records_without_model_hashing(synth_run):
    meta = write_launch_meta(
        synth_run.run_dir,
        checkpoint_path="/checkpoints/step-10",
        checkpoint_step=10,
        recipe="nvfp4_te",
        image="slime:test",
    )
    on_disk = json.loads(
        (synth_run.run_dir / "launch_meta.json").read_text(encoding="utf-8")
    )
    assert on_disk == meta
    assert meta["checkpoint_step"] == 10
    assert meta["recipe"] == "nvfp4_te"
    assert meta["identity_policy"] == "record_only_no_recursive_hash"
    assert "model_hash" not in meta
    assert "tokenizer_hash" not in meta


def test_export_routing_uses_existing_forced_routing_format(tmp_path, synth_run):
    torch = pytest.importorskip("torch")
    output = tmp_path / "routing.pt"
    summary = export_routing(synth_run.run_dir, output)
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert summary["samples"] == 1
    assert set(payload["sample-0"]) == {"topk_ids", "topk_weights"}
    assert payload["sample-0"]["topk_ids"].dtype == torch.long
    assert payload["sample-0"]["topk_ids"].shape == (8, 1)


def test_parse_layers_and_completed_sweep_resume(tmp_path):
    assert parse_layers("0,4,8,14:19") == [0, 4, 8, 14, 15, 16, 17, 18]
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "runs" / "g2" / "layer_0000"
    run_dir.mkdir(parents=True)
    required = (
        "config.resolved.json",
        "launch_meta.json",
        "manifest.json",
        "metrics.json",
        "report.html",
    )
    for name in required:
        (run_dir / name).write_text("{}", encoding="utf-8")
    summary = run_layer_sweep(config, [0], tmp_path / "runs", "g2")
    assert summary["completed"] == []
    assert summary["skipped"] == [0]


def test_sweep_keeps_complete_strict_fail_runs(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")

    def fake_run(cmd, check):
        run_id = cmd[cmd.index("--run-id") + 1]
        output_dir = cmd[cmd.index("--output-dir") + 1]
        run_dir = tmp_path / "runs" / "g1" / run_id
        assert str(run_dir.parent) == output_dir
        run_dir.mkdir(parents=True)
        for name in monitoring._RUN_COMPLETE_FILES:
            content = (
                {"first_failure": {"status": "FAIL"}}
                if name == "metrics.json"
                else {}
            )
            (run_dir / name).write_text(json.dumps(content), encoding="utf-8")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(monitoring.subprocess, "run", fake_run)
    summary = run_layer_sweep(config, [0], tmp_path / "runs", "g1")
    assert summary["completed"] == [0]
    assert summary["comparison_failures"] == [0]


def test_aggregate_sweeps_concatenates_existing_metrics(tmp_path, synth_run):
    run_compare(synth_run.run_dir)
    write_launch_meta(synth_run.run_dir, recipe="nvfp4_te")
    output = tmp_path / "layerwise.csv"
    summary = aggregate_sweeps(synth_run.run_dir.parent, output)
    assert summary["run_count"] == 1
    assert summary["row_count"] > 0
    with open(output, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert rows[0]["condition"] == synth_run.run_dir.name
    assert rows[0]["launch_recipe"] == "nvfp4_te"


def test_rl_metrics_sign_ratio_and_clip(tmp_path, synth_run_fail):
    output = tmp_path / "g1.csv"
    summary = compute_rl_amplification(
        synth_run_fail.run_dir,
        output,
        clip_epsilon=0.2,
        condition="g1",
    )
    assert summary["token_count"] == 8
    assert summary["abs_delta_p99"] == pytest.approx(0.5)
    assert summary["delta_mean"] == pytest.approx(-0.5)
    assert summary["below_clip_fraction"] == 1.0
    assert summary["above_clip_fraction"] == 0.0
    assert summary["ratio_p50"] == pytest.approx(math.exp(-0.5))
    with open(output, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    assert float(rows[0]["delta_logp"]) == pytest.approx(-0.5)


def test_rl_metrics_reject_multiple_layers(tmp_path, synth_run):
    torch = pytest.importorskip("torch")
    for framework in ("sglang", "megatron"):
        record = TensorRecord(
            run_id=synth_run.run_id,
            framework=framework,
            sample_id="sample-0",
            layer_id=1,
            tensor_point=TOKEN_LOGPROBS,
            provenance=DERIVED_METRIC,
        )
        synth_run.save_tensor(record, torch.zeros(8))
    with pytest.raises(ValueError, match="span layers"):
        compute_rl_amplification(
            synth_run.run_dir,
            tmp_path / "rl.csv",
            clip_epsilon=0.2,
        )


def test_export_routing_rejects_corrupt_run_id(tmp_path, synth_run):
    index = synth_run.run_dir / "sglang" / "index.jsonl"
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
    rows[0]["run_id"] = "wrong-run"
    index.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="run_id"):
        export_routing(synth_run.run_dir, tmp_path / "routing.pt")


def test_prompt_cluster_bootstrap_and_paired_difference(
    tmp_path, synth_run, synth_run_fail
):
    g0 = tmp_path / "g0.csv"
    g1 = tmp_path / "g1.csv"
    compute_rl_amplification(
        synth_run.run_dir, g0, clip_epsilon=0.2, condition="g0"
    )
    compute_rl_amplification(
        synth_run_fail.run_dir, g1, clip_epsilon=0.2, condition="g1"
    )
    output = tmp_path / "bootstrap.json"
    result = bootstrap_amplification(
        {"g0": g0, "g1": g1},
        output,
        pairs=["g1:g0"],
        resamples=20,
        seed=7,
    )
    paired = result["paired_differences"]["g1:g0"]
    assert paired["point"]["p99_abs_delta"] == pytest.approx(0.5)
    assert paired["point"]["clip_fraction"] == 1.0
    assert output.is_file()
