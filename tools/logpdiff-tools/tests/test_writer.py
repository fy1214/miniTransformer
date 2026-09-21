"""Tests for logpdiff.writer: run directory layout, manifest, payload safety."""

from __future__ import annotations

import json

import pytest

from logpdiff import schema
from logpdiff.writer import (
    FRAMEWORKS,
    assert_internal_payload,
    create_run_dir,
)

from conftest import make_record, require_torch


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class TestRunDirLayout:
    def test_directory_structure(self, tmp_path):
        run = create_run_dir(tmp_path, run_id="r1", model_path="/m", seed=7)
        assert run.run_dir == tmp_path.resolve() / "r1"
        for fw in FRAMEWORKS:
            assert (run.run_dir / fw / "tensors").is_dir()

    def test_no_overwrite(self, tmp_path):
        create_run_dir(tmp_path, run_id="r1")
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            create_run_dir(tmp_path, run_id="r1")

    @pytest.mark.parametrize("run_id", ["", ".", "..", "a/b", "a\\b"])
    def test_invalid_run_id(self, tmp_path, run_id):
        with pytest.raises(ValueError, match="invalid run_id"):
            create_run_dir(tmp_path, run_id=run_id)

    def test_multi_parallel_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="TP=EP=PP=1"):
            create_run_dir(tmp_path, run_id="r1", tp=2)


class TestIndexAppend:
    def test_append_record_appends_jsonl(self, tmp_path):
        run = create_run_dir(tmp_path, run_id="r1")
        for tok in (0, 1):
            rec = make_record("r1", "sglang", schema.FC1_OUTPUT,
                              expert_id=0, token_position=tok)
            run.append_record("sglang", rec)
        rows = _read_jsonl(run.run_dir / "sglang" / "index.jsonl")
        assert len(rows) == 2
        assert [r["token_position"] for r in rows] == [0, 1]
        assert rows[0]["shape"] == [4]  # tuple serialized as list
        other = run.run_dir / "megatron" / "index.jsonl"
        assert not other.exists()  # untouched framework has no index yet

    def test_framework_mismatch_rejected(self, tmp_path):
        run = create_run_dir(tmp_path, run_id="r1")
        rec = make_record("r1", "sglang", schema.FC1_OUTPUT)
        with pytest.raises(ValueError, match="does not match"):
            run.append_record("megatron", rec)

    def test_unknown_framework_rejected(self, tmp_path):
        run = create_run_dir(tmp_path, run_id="r1")
        rec = make_record("r1", "tensorrt", schema.FC1_OUTPUT)
        with pytest.raises(ValueError, match="unknown framework"):
            run.append_record("tensorrt", rec)

    def test_invalid_record_rejected(self, tmp_path):
        run = create_run_dir(tmp_path, run_id="r1")
        rec = make_record("r1", "sglang", schema.FC1_OUTPUT, payload_path="")
        with pytest.raises(ValueError):
            run.append_record("sglang", rec)


class TestSaveTensor:
    def test_save_tensor_backfills_and_roundtrips(self, tmp_path):
        torch = require_torch()
        run = create_run_dir(tmp_path, run_id="r1")
        tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        rec = schema.TensorRecord(
            run_id="r1", framework="megatron", sample_id="s0", layer_id=0,
            tensor_point=schema.DISPATCH_INPUT,
            provenance=schema.HOOK_BOUNDARY,
        )
        payload_path = run.save_tensor(rec, tensor)
        assert payload_path.startswith("megatron/tensors/")
        loaded = torch.load(
            assert_internal_payload(payload_path, run.run_dir),
            map_location="cpu",
        )
        assert torch.equal(loaded, tensor)
        rows = _read_jsonl(run.run_dir / "megatron" / "index.jsonl")
        assert rows[0]["shape"] == [2, 3]
        assert rows[0]["dtype"] == "torch.float32"
        assert rows[0]["payload_path"] == payload_path

    def test_non_tensor_rejected(self, tmp_path):
        run = create_run_dir(tmp_path, run_id="r1")
        rec = make_record("r1", "sglang", schema.FC1_OUTPUT)
        with pytest.raises(TypeError):
            run.save_tensor(rec, [[1.0]])


class TestWeightChecksumDedup:
    def test_identical_weights_deduplicated(self, tmp_path):
        torch = require_torch()
        run = create_run_dir(tmp_path, run_id="r1")
        weight = torch.randn(4, 8)

        rec1 = make_record("r1", "megatron", schema.FC1_QDATA,
                           expert_id=0, shape=(), dtype="",
                           payload_path="x", provenance=schema.ACTUAL_KERNEL)
        path1 = run.save_weight_tensor(rec1, weight)
        rec2 = make_record("r1", "megatron", schema.FC1_QDATA,
                           expert_id=1, shape=(), dtype="",
                           payload_path="x", provenance=schema.ACTUAL_KERNEL)
        path2 = run.save_weight_tensor(rec2, weight)

        assert path1 == path2  # second save reuses the first payload
        assert len(list((run.run_dir / "megatron" / "tensors").iterdir())) == 1

        rows = _read_jsonl(run.run_dir / "megatron" / "index.jsonl")
        assert len(rows) == 2
        assert rows[0]["weight_checksum"] == rows[1]["weight_checksum"]
        assert "dedup_ref" not in rows[0]
        assert rows[1]["dedup_ref"] is True

    def test_different_weights_not_deduplicated(self, tmp_path):
        torch = require_torch()
        run = create_run_dir(tmp_path, run_id="r1")
        rec_a = make_record("r1", "megatron", schema.FC1_QDATA, expert_id=0)
        rec_b = make_record("r1", "megatron", schema.FC1_QDATA, expert_id=1)
        path_a = run.save_weight_tensor(rec_a, torch.randn(4, 8))
        path_b = run.save_weight_tensor(rec_b, torch.randn(4, 8))
        assert path_a != path_b
        assert len(list((run.run_dir / "megatron" / "tensors").iterdir())) == 2


class TestManifest:
    def test_manifest_required_fields(self, tmp_path):
        run = create_run_dir(
            tmp_path, run_id="r1", model_path="/models/x",
            tokenizer_path="/models/x", seed=1234, backend="nvfp4",
            force_routing=True,
        )
        manifest = run.finalize_manifest()
        on_disk = json.loads((run.run_dir / "manifest.json").read_text())
        assert on_disk == manifest

        assert manifest["run_id"] == "r1"
        assert manifest["model_path"] == "/models/x"
        assert manifest["seed"] == 1234
        assert manifest["backend"] == "nvfp4"
        assert manifest["parallel"] == {"tp": 1, "ep": 1, "pp": 1}
        assert manifest["force_routing"] is True
        # git commit: a hex string inside the repo (never silently missing here)
        assert isinstance(manifest["git_commit"], str)
        assert len(manifest["git_commit"]) == 40

        versions = manifest["versions"]
        for key in ("python", "torch", "cuda", "transformer_engine",
                    "sglang", "megatron", "minite"):
            assert key in versions
        assert versions["torch"] is not None  # tests run with torch installed

    def test_manifest_env_allowlist(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QAT_PARAMS", "8")
        monkeypatch.setenv("MINITE_SYNTHETIC_FLAG", "1")
        monkeypatch.setenv("MY_LOGPDIFF_FLAG", "yes")
        run = create_run_dir(tmp_path, run_id="r1")
        manifest = run.finalize_manifest(extra_env_allowlist=["MY_LOGPDIFF_FLAG"])
        env = manifest["env"]
        assert env["QAT_PARAMS"] == "8"
        assert env["MINITE_SYNTHETIC_FLAG"] == "1"  # prefix match
        assert env["MY_LOGPDIFF_FLAG"] == "yes"  # explicit allowlist
        assert "HOME" not in env  # unrelated env vars are not recorded


class TestPayloadSafety:
    def test_internal_relative_path_ok(self, tmp_path):
        run = create_run_dir(tmp_path, run_id="r1")
        p = assert_internal_payload("sglang/tensors/x.pt", run.run_dir)
        assert p == run.run_dir / "sglang" / "tensors" / "x.pt"

    def test_internal_absolute_path_ok(self, tmp_path):
        run = create_run_dir(tmp_path, run_id="r1")
        p = assert_internal_payload(
            str(run.run_dir / "megatron" / "tensors" / "x.pt"), run.run_dir
        )
        assert p.is_relative_to(run.run_dir)

    @pytest.mark.parametrize(
        "path", ["../escape.pt", "../../etc/passwd", "/etc/passwd"]
    )
    def test_escape_rejected(self, tmp_path, path):
        run = create_run_dir(tmp_path, run_id="r1")
        with pytest.raises(ValueError, match="escapes run directory"):
            assert_internal_payload(path, run.run_dir)

    def test_sibling_prefix_is_not_inside(self, tmp_path):
        # "runs-evil/x.pt" must not pass for run dir "runs/r1"-style prefixes
        run = create_run_dir(tmp_path / "runs", run_id="r1")
        with pytest.raises(ValueError):
            assert_internal_payload(tmp_path / "runs-evil" / "x.pt", run.run_dir)
