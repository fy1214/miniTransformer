"""Small monitoring helpers built on top of logpdiff run directories.

The module keeps experiment orchestration and offline analysis outside slime.
It imports only the standard library; torch is loaded lazily when tensor
payloads are read or a forced-routing file is written.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .schema import (
    ROUTER_TOPK_IDS,
    ROUTER_TOPK_WEIGHTS,
    TOKEN_LOGPROBS,
    TensorRecord,
)
from .writer import assert_internal_payload

_ENV_EXACT = ("QAT_PARAMS", "SHOULD_REPLACE_TE_GROUPLINEAR")
_ENV_PREFIXES = ("MINITE_", "NVTE_NVFP4_", "SGLANG_NVFP4_")
_RUN_COMPLETE_FILES = (
    "config.resolved.json",
    "launch_meta.json",
    "manifest.json",
    "metrics.json",
    "report.html",
)
_IDENTITY_KEYS = ("model_path", "tokenizer_path", "checkpoint_path", "recipe")
_RL_FIELDS = (
    "condition",
    "run_id",
    "sample_id",
    "layer_id",
    "token_position",
    "token_id",
    "logp_rollout",
    "logp_learner",
    "delta_logp",
    "abs_delta_logp",
    "log_ratio_num",
    "ratio_num",
    "below_clip",
    "above_clip",
)


def _read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _config_value(config, key):
    if not isinstance(config, dict):
        return None
    if config.get(key) is not None:
        return config[key]
    for side in ("sglang", "megatron"):
        section = config.get(side)
        if isinstance(section, dict) and section.get(key) is not None:
            return section[key]
    return None


def _coalesce(*values):
    return next((value for value in values if value is not None), None)


def _git_commit(path) -> str | None:
    if path is None:
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = proc.stdout.strip()
    return value if proc.returncode == 0 and value else None


def _recorded_env() -> dict:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key in _ENV_EXACT or key.startswith(_ENV_PREFIXES)
    }


def parse_name_path_specs(items, *, label="value") -> dict[str, str]:
    """Parse repeated ``NAME=PATH`` CLI values."""
    result = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"{label} must be NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"{label} must be NAME=PATH, got {item!r}")
        if name in result:
            raise ValueError(f"duplicate {label} name {name!r}")
        result[name] = path
    return result


def write_launch_meta(
    run_dir,
    *,
    model_path=None,
    tokenizer_path=None,
    checkpoint_path=None,
    checkpoint_step=None,
    input_file=None,
    seed=None,
    recipe=None,
    image=None,
    gpu=None,
    source_repos=None,
    overwrite=False,
) -> dict:
    """Write human-auditable launch metadata without hashing model files."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory {run_dir} does not exist")
    output = run_dir / "launch_meta.json"
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; use overwrite=True")

    config_path = run_dir / "config.resolved.json"
    manifest_path = run_dir / "manifest.json"
    config = _read_json(config_path) if config_path.is_file() else {}
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    manifest_env = manifest.get("env") or {}

    sources = {}
    for name, path in sorted((source_repos or {}).items()):
        sources[name] = {
            "path": str(Path(path).resolve()),
            "commit": _git_commit(path),
        }

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "model_path": _coalesce(
            model_path, manifest.get("model_path"), _config_value(config, "model_path")
        ),
        "tokenizer_path": _coalesce(
            tokenizer_path,
            manifest.get("tokenizer_path"),
            _config_value(config, "tokenizer_path"),
        ),
        "checkpoint_path": _coalesce(
            checkpoint_path, _config_value(config, "ref_load")
        ),
        "checkpoint_step": _coalesce(
            checkpoint_step, _config_value(config, "checkpoint_step")
        ),
        "input_file": _coalesce(input_file, _config_value(config, "input_file")),
        "seed": _coalesce(seed, manifest.get("seed"), _config_value(config, "seed")),
        "recipe": _coalesce(
            recipe,
            os.environ.get("MINITE_RECIPE"),
            os.environ.get("QAT_PARAMS"),
            manifest_env.get("MINITE_RECIPE"),
            manifest_env.get("QAT_PARAMS"),
            _config_value(config, "recipe"),
        ),
        "image": _coalesce(image, _config_value(config, "image")),
        "gpu": _coalesce(gpu, manifest.get("gpu_name"), _config_value(config, "gpu")),
        "source_repos": sources,
        "env": _recorded_env(),
        "identity_policy": "record_only_no_recursive_hash",
    }
    _write_json(output, meta)
    return meta


def _read_index(run_dir: Path, framework: str) -> list[TensorRecord]:
    path = run_dir / framework / "index.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                record = TensorRecord.from_dict(raw)
                record.validate()
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{lineno}: invalid TensorRecord: {exc}"
                ) from exc
            if record.framework != framework:
                raise ValueError(
                    f"{path}:{lineno}: framework is {record.framework!r}, "
                    f"expected {framework!r}"
                )
            if record.run_id != run_dir.name:
                raise ValueError(
                    f"{path}:{lineno}: run_id is {record.run_id!r}, "
                    f"expected {run_dir.name!r}"
                )
            records.append(record)
    return records


def _import_torch():
    import torch

    return torch


def _load_payload(run_dir: Path, record: TensorRecord):
    torch = _import_torch()
    path = assert_internal_payload(record.payload_path, run_dir)
    if not path.is_file():
        raise FileNotFoundError(f"payload {path} does not exist")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def export_routing(run_dir, output, *, layer_idx=None, overwrite=False) -> dict:
    """Export SGLang top-k tensors to the existing forced-routing ``.pt`` format."""
    run_dir = Path(run_dir).resolve()
    output = Path(output).resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; use overwrite=True")

    records = _read_index(run_dir, "sglang")
    selected = [
        r
        for r in records
        if r.tensor_point in (ROUTER_TOPK_IDS, ROUTER_TOPK_WEIGHTS)
        and (layer_idx is None or r.layer_id == layer_idx)
    ]
    layers = sorted({r.layer_id for r in selected})
    if not selected:
        raise ValueError("no SGLang routing records found")
    if layer_idx is None and len(layers) != 1:
        raise ValueError(f"routing records span layers {layers}; pass layer_idx")

    grouped = {}
    for record in selected:
        key = str(record.sample_id)
        entry = grouped.setdefault(key, {})
        field = (
            "topk_ids" if record.tensor_point == ROUTER_TOPK_IDS else "topk_weights"
        )
        if field in entry:
            raise ValueError(
                f"duplicate {field} for sample {key!r}, layer {record.layer_id}"
            )
        entry[field] = _load_payload(run_dir, record)

    torch = _import_torch()
    for sample_id, entry in grouped.items():
        missing = {"topk_ids", "topk_weights"} - set(entry)
        if missing:
            raise ValueError(f"sample {sample_id!r} lacks {sorted(missing)}")
        ids, weights = entry["topk_ids"], entry["topk_weights"]
        if not isinstance(ids, torch.Tensor) or not isinstance(weights, torch.Tensor):
            raise TypeError(f"sample {sample_id!r}: routing payloads must be tensors")
        if ids.shape != weights.shape or ids.ndim != 2:
            raise ValueError(
                f"sample {sample_id!r}: routing shapes must match [tokens, top_k], "
                f"got {tuple(ids.shape)} and {tuple(weights.shape)}"
            )
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if ids.dtype not in integer_dtypes or ids.numel() == 0 or bool((ids < 0).any()):
            raise ValueError(
                f"sample {sample_id!r}: topk_ids must be non-negative integers"
            )
        if not weights.is_floating_point() or not bool(torch.isfinite(weights).all()):
            raise ValueError(
                f"sample {sample_id!r}: topk_weights must be finite floating point"
            )
        if bool((weights < 0).any()):
            raise ValueError(f"sample {sample_id!r}: topk_weights must be non-negative")
        row_sums = weights.float().sum(dim=1)
        if not torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            rtol=1e-3,
            atol=1e-3,
        ):
            raise ValueError(
                f"sample {sample_id!r}: topk_weights must sum to one per token"
            )
        entry["topk_ids"] = ids.to(dtype=torch.long)
        entry["topk_weights"] = weights

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(grouped, output)
    return {
        "output": str(output),
        "layer_idx": layer_idx if layer_idx is not None else layers[0],
        "samples": len(grouped),
    }


def parse_layers(spec: str) -> list[int]:
    """Parse ``0,4,8,14:19``; ranges use Python's stop-exclusive convention."""
    values = []
    for token in (part.strip() for part in spec.split(",")):
        if not token:
            continue
        if ":" not in token:
            values.append(int(token))
            continue
        parts = token.split(":")
        if len(parts) not in (2, 3) or not all(parts):
            raise ValueError(f"invalid layer range {token!r}")
        start, stop = int(parts[0]), int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
        if step <= 0:
            raise ValueError(f"layer range step must be positive: {token!r}")
        values.extend(range(start, stop, step))
    if not values or any(value < 0 for value in values):
        raise ValueError("layers must contain non-negative integers")
    return list(dict.fromkeys(values))


def _run_complete(run_dir: Path) -> bool:
    return all((run_dir / name).is_file() for name in _RUN_COMPLETE_FILES)


def _comparison_failed(run_dir: Path) -> bool:
    metrics = _read_json(run_dir / "metrics.json")
    first_failure = metrics.get("first_failure") or {}
    return first_failure.get("status") == "FAIL"


def run_layer_sweep(config, layers, runs_root, condition) -> dict:
    """Run existing ``logpdiff run`` once per requested layer."""
    if condition in ("", ".", "..") or Path(condition).name != condition:
        raise ValueError(f"invalid condition name {condition!r}")
    layers = list(layers)
    if not layers or any(not isinstance(layer, int) or layer < 0 for layer in layers):
        raise ValueError("layers must contain non-negative integers")
    config = Path(config).resolve()
    if not config.is_file():
        raise FileNotFoundError(f"config {config} does not exist")
    config_data = _read_json(config)
    if not isinstance(config_data, dict):
        raise TypeError("sweep config must contain a JSON object")
    if config_data.get("run_dir"):
        raise ValueError("sweep config must not set run_dir")

    condition_dir = Path(runs_root).resolve() / condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "condition": condition,
        "config": str(config),
        "layers": list(layers),
        "completed": [],
        "comparison_failures": [],
        "skipped": [],
    }
    for layer in layers:
        run_id = f"layer_{layer:04d}"
        run_dir = condition_dir / run_id
        if run_dir.exists():
            if _run_complete(run_dir):
                summary["skipped"].append(layer)
                if _comparison_failed(run_dir):
                    summary["comparison_failures"].append(layer)
                continue
            raise FileExistsError(
                f"partial run {run_dir} exists; inspect it before resuming"
            )
        cmd = [
            sys.executable,
            "-m",
            "logpdiff",
            "run",
            "--config",
            str(config),
            "--output-dir",
            str(condition_dir),
            "--run-id",
            run_id,
            "--layer-idx",
            str(layer),
        ]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            if _run_complete(run_dir):
                summary["completed"].append(layer)
                if _comparison_failed(run_dir):
                    summary["comparison_failures"].append(layer)
                _write_json(condition_dir / "sweep_manifest.json", summary)
                continue
            summary["failed_layer"] = layer
            summary["returncode"] = proc.returncode
            _write_json(condition_dir / "sweep_manifest.json", summary)
            raise RuntimeError(
                f"logpdiff run failed at layer {layer} with code {proc.returncode}"
            )
        summary["completed"].append(layer)
        if _comparison_failed(run_dir):
            summary["comparison_failures"].append(layer)
        _write_json(condition_dir / "sweep_manifest.json", summary)
    _write_json(condition_dir / "sweep_manifest.json", summary)
    return summary


def aggregate_sweeps(runs_root, output) -> dict:
    """Concatenate per-run ``metrics.csv`` files and report coverage drift."""
    runs_root = Path(runs_root).resolve()
    output = Path(output).resolve()
    metric_files = [
        path
        for path in sorted(runs_root.rglob("metrics.csv"))
        if path.resolve() != output
    ]
    if not metric_files:
        raise FileNotFoundError(f"no metrics.csv files found under {runs_root}")

    fieldnames = ["condition", "run_dir", "launch_recipe", "checkpoint_step"]
    runs = []
    identity_values = {}
    for metrics_path in metric_files:
        run_dir = metrics_path.parent
        relative = run_dir.relative_to(runs_root)
        condition = relative.parts[0] if len(relative.parts) > 1 else run_dir.name
        launch_path = run_dir / "launch_meta.json"
        launch = _read_json(launch_path) if launch_path.is_file() else {}
        with open(metrics_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"{metrics_path} has no CSV header")
            for name in reader.fieldnames:
                if name not in fieldnames:
                    fieldnames.append(name)
        runs.append((metrics_path, run_dir, condition, launch))
        for key in _IDENTITY_KEYS:
            value = launch.get(key)
            if value is not None:
                identity_values.setdefault((condition, key), set()).add(str(value))

    coverage = {}
    row_count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        for metrics_path, run_dir, condition, launch in runs:
            with open(metrics_path, newline="", encoding="utf-8") as in_file:
                reader = csv.DictReader(in_file)
                for row in reader:
                    merged = {
                        "condition": condition,
                        "run_dir": str(run_dir),
                        "launch_recipe": launch.get("recipe"),
                        "checkpoint_step": launch.get("checkpoint_step"),
                    }
                    merged.update(row)
                    writer.writerow(merged)
                    row_count += 1
                    coverage.setdefault(condition, {}).setdefault(
                        str(row.get("layer_id")), set()
                    ).add(str(row.get("sample_id")))

    warnings = []
    coverage_json = {}
    for condition, by_layer in sorted(coverage.items()):
        normalized = {
            layer: sorted(samples) for layer, samples in sorted(by_layer.items())
        }
        coverage_json[condition] = normalized
        if len({tuple(samples) for samples in normalized.values()}) > 1:
            warnings.append(f"{condition}: prompt coverage differs across layers")
    for (condition, key), values in sorted(identity_values.items()):
        if len(values) > 1:
            warnings.append(
                f"{condition}: launch metadata differs for {key}: {sorted(values)}"
            )

    summary = {
        "runs_root": str(runs_root),
        "output": str(output),
        "run_count": len(metric_files),
        "row_count": row_count,
        "coverage": coverage_json,
        "warnings": warnings,
    }
    summary_path = output.with_name(output.stem + ".summary.json")
    summary["summary_path"] = str(summary_path)
    _write_json(summary_path, summary)
    return summary


def _percentile_sorted(ordered, q: float) -> float:
    if not ordered:
        raise ValueError("cannot compute a percentile of no values")
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _percentiles(values, quantiles) -> dict[float, float]:
    ordered = sorted(float(value) for value in values)
    return {q: _percentile_sorted(ordered, q) for q in quantiles}


def _ci95(values) -> dict:
    q = _percentiles(values, (2.5, 50.0, 97.5))
    return {"lower": q[2.5], "median": q[50.0], "upper": q[97.5]}


def _safe_exp(value: float) -> float:
    if value > 709.0:
        return float("inf")
    if value < -745.0:
        return 0.0
    return math.exp(value)


def _input_ids_by_sample(run_dir: Path) -> dict[str, list]:
    path = run_dir / "config.resolved.json"
    if not path.is_file():
        return {}
    config = _read_json(path)
    for side in ("sglang", "megatron"):
        section = config.get(side)
        if not isinstance(section, dict):
            continue
        result = {}
        for sample in section.get("samples") or []:
            sample_id = sample.get("sample_id", sample.get("name"))
            input_ids = sample.get("input_ids")
            if sample_id is not None and isinstance(input_ids, list):
                result[str(sample_id)] = input_ids
        if result:
            return result
    return {}


def compute_rl_amplification(
    run_dir,
    output,
    *,
    clip_epsilon: float,
    condition=None,
    layer_idx=None,
) -> dict:
    """Compute learner-minus-rollout logp deltas from one paired run."""
    if not 0.0 < clip_epsilon < 1.0:
        raise ValueError("clip_epsilon must be between 0 and 1")
    run_dir = Path(run_dir).resolve()
    output = Path(output).resolve()
    condition = condition or run_dir.parent.name

    by_framework = {}
    for framework in ("sglang", "megatron"):
        records = [
            record
            for record in _read_index(run_dir, framework)
            if record.tensor_point == TOKEN_LOGPROBS
            and (layer_idx is None or record.layer_id == layer_idx)
        ]
        mapping = {}
        for record in records:
            key = (str(record.sample_id), record.layer_id)
            if key in mapping:
                raise ValueError(
                    f"duplicate token_logprobs record for {framework} {key}"
                )
            mapping[key] = record
        by_framework[framework] = mapping

    keys_a = set(by_framework["sglang"])
    keys_b = set(by_framework["megatron"])
    if keys_a != keys_b:
        raise ValueError(
            "token_logprobs sample/layer coverage differs: "
            f"sglang_only={sorted(keys_a - keys_b)}, "
            f"megatron_only={sorted(keys_b - keys_a)}"
        )
    if not keys_a:
        raise ValueError("no paired token_logprobs records found")
    layers = {layer for _, layer in keys_a}
    if layer_idx is None and len(layers) != 1:
        raise ValueError(
            f"token_logprobs records span layers {sorted(layers)}; pass layer_idx"
        )

    input_ids = _input_ids_by_sample(run_dir)
    below_log = math.log1p(-clip_epsilon)
    above_log = math.log1p(clip_epsilon)
    torch = _import_torch()
    paired_values = []
    for sample_id, layer in sorted(keys_a):
        rollout = _load_payload(run_dir, by_framework["sglang"][(sample_id, layer)])
        learner = _load_payload(run_dir, by_framework["megatron"][(sample_id, layer)])
        if not isinstance(rollout, torch.Tensor) or not isinstance(
            learner, torch.Tensor
        ):
            raise TypeError(
                f"sample {sample_id!r}, layer {layer}: logp payloads must be tensors"
            )
        if tuple(rollout.shape) != tuple(learner.shape):
            raise ValueError(
                f"sample {sample_id!r}, layer {layer}: logp shape mismatch "
                f"{tuple(rollout.shape)} vs {tuple(learner.shape)}"
            )
        rollout_flat = rollout.detach().float().reshape(-1)
        learner_flat = learner.detach().float().reshape(-1)
        if rollout_flat.numel() == 0:
            raise ValueError(f"sample {sample_id!r}, layer {layer}: empty logp tensor")
        if not bool(torch.isfinite(rollout_flat).all()) or not bool(
            torch.isfinite(learner_flat).all()
        ):
            raise ValueError(
                f"sample {sample_id!r}, layer {layer}: non-finite logp detected"
            )
        ids = input_ids.get(sample_id) or []
        if ids and len(ids) != rollout_flat.numel() + 1:
            raise ValueError(
                f"sample {sample_id!r}: {len(ids)} input ids but "
                f"{rollout_flat.numel()} chosen-token logps"
            )
        paired_values.append(
            (
                sample_id,
                layer,
                rollout_flat.tolist(),
                learner_flat.tolist(),
                ids,
            )
        )

    deltas = []
    abs_deltas = []
    below_count = above_count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_RL_FIELDS)
        writer.writeheader()
        for sample_id, layer, rollout_values, learner_values, ids in paired_values:
            for position, (rollout_logp, learner_logp) in enumerate(
                zip(rollout_values, learner_values)
            ):
                delta = float(learner_logp - rollout_logp)
                abs_delta = abs(delta)
                below = delta < below_log
                above = delta > above_log
                below_count += int(below)
                above_count += int(above)
                deltas.append(delta)
                abs_deltas.append(abs_delta)
                writer.writerow({
                    "condition": condition,
                    "run_id": run_dir.name,
                    "sample_id": sample_id,
                    "layer_id": layer,
                    "token_position": position,
                    "token_id": ids[position + 1] if position + 1 < len(ids) else "",
                    "logp_rollout": rollout_logp,
                    "logp_learner": learner_logp,
                    "delta_logp": delta,
                    "abs_delta_logp": abs_delta,
                    "log_ratio_num": delta,
                    "ratio_num": _safe_exp(delta),
                    "below_clip": below,
                    "above_clip": above,
                })

    token_count = len(deltas)
    delta_q = _percentiles(deltas, (1.0, 50.0, 99.0))
    abs_q = _percentiles(abs_deltas, (50.0, 95.0, 99.0))
    summary = {
        "run_dir": str(run_dir),
        "condition": condition,
        "sign_convention": "delta_logp = megatron_learner - sglang_rollout",
        "clip_epsilon": clip_epsilon,
        "prompt_count": len(paired_values),
        "token_count": token_count,
        "delta_mean": sum(deltas) / token_count,
        "delta_median": delta_q[50.0],
        "abs_delta_p50": abs_q[50.0],
        "abs_delta_p95": abs_q[95.0],
        "abs_delta_p99": abs_q[99.0],
        "abs_delta_max": max(abs_deltas),
        "ratio_p01": _safe_exp(delta_q[1.0]),
        "ratio_p50": _safe_exp(delta_q[50.0]),
        "ratio_p99": _safe_exp(delta_q[99.0]),
        "below_clip_fraction": below_count / token_count,
        "above_clip_fraction": above_count / token_count,
        "clip_fraction": (below_count + above_count) / token_count,
        "output": str(output),
    }
    summary_path = output.with_name(output.stem + ".summary.json")
    summary["summary_path"] = str(summary_path)
    _write_json(summary_path, summary)
    return summary


def _read_amplification_csv(path: Path) -> dict[str, list[dict]]:
    grouped = {}
    positions = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "sample_id",
            "layer_id",
            "token_position",
            "token_id",
            "abs_delta_logp",
            "below_clip",
            "above_clip",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} lacks columns {sorted(missing)}")
        for row in reader:
            sample_id = row["sample_id"]
            try:
                layer_id = int(row["layer_id"])
                position = int(row["token_position"])
                abs_delta = float(row["abs_delta_logp"])
            except ValueError as exc:
                raise ValueError(f"{path}: invalid numeric field: {exc}") from exc
            if (
                layer_id < 0
                or position < 0
                or not math.isfinite(abs_delta)
                or abs_delta < 0
            ):
                raise ValueError(
                    f"{path}: invalid row for sample {sample_id!r}, "
                    f"layer_id={layer_id}, token_position={position}, "
                    f"abs_delta_logp={abs_delta}"
                )
            seen = positions.setdefault(sample_id, set())
            if position in seen:
                raise ValueError(
                    f"{path}: duplicate token_position {position} "
                    f"for sample {sample_id!r}"
                )
            seen.add(position)
            row["layer_id"] = layer_id
            row["token_position"] = position
            row["abs_delta_logp"] = abs_delta
            for field in ("below_clip", "above_clip"):
                value = row[field].strip().lower()
                if value not in ("0", "1", "false", "true", "no", "yes"):
                    raise ValueError(f"{path}: invalid boolean {field}={row[field]!r}")
                row[field] = value in ("1", "true", "yes")
            grouped.setdefault(sample_id, []).append(row)
    if not grouped:
        raise ValueError(f"{path} contains no data rows")
    layers = {row["layer_id"] for rows in grouped.values() for row in rows}
    if len(layers) != 1:
        raise ValueError(f"{path}: rl metrics span layers {sorted(layers)}")
    return grouped


def _cluster_stats(rows: list[dict]) -> dict:
    values = [row["abs_delta_logp"] for row in rows]
    q = _percentiles(values, (95.0, 99.0))
    clipped = sum(row["below_clip"] or row["above_clip"] for row in rows)
    return {
        "mean_abs_delta": sum(values) / len(values),
        "p95_abs_delta": q[95.0],
        "p99_abs_delta": q[99.0],
        "clip_fraction": clipped / len(rows),
    }


def _selected_rows(clusters, prompt_ids):
    return [row for prompt in prompt_ids for row in clusters[prompt]]


def _bootstrap_condition(clusters, resamples, rng):
    prompt_ids = sorted(clusters)
    point = _cluster_stats(_selected_rows(clusters, prompt_ids))
    draws = {key: [] for key in point}
    for _ in range(resamples):
        selected = [rng.choice(prompt_ids) for _ in prompt_ids]
        stats = _cluster_stats(_selected_rows(clusters, selected))
        for key, value in stats.items():
            draws[key].append(value)
    return point, {key: _ci95(values) for key, values in draws.items()}


def bootstrap_amplification(
    inputs, output, *, pairs=None, resamples=2000, seed=0
) -> dict:
    """Prompt-cluster bootstrap, including paired condition differences."""
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    clusters = {
        name: _read_amplification_csv(Path(path).resolve())
        for name, path in sorted(inputs.items())
    }
    result = {
        "seed": seed,
        "resamples": resamples,
        "conditions": {},
        "paired_differences": {},
    }
    for offset, (name, grouped) in enumerate(sorted(clusters.items())):
        point, ci = _bootstrap_condition(
            grouped, resamples, random.Random(seed + offset)
        )
        result["conditions"][name] = {
            "prompt_count": len(grouped),
            "point": point,
            "ci95": ci,
        }

    for offset, pair in enumerate(pairs or []):
        if ":" not in pair:
            raise ValueError(f"pair must be TARGET:REFERENCE, got {pair!r}")
        target, reference = pair.split(":", 1)
        if target not in clusters or reference not in clusters:
            raise ValueError(f"unknown condition in pair {pair!r}")
        target_prompts = set(clusters[target])
        reference_prompts = set(clusters[reference])
        if target_prompts != reference_prompts:
            raise ValueError(f"pair {pair!r} has different prompt coverage")
        for prompt in target_prompts:
            target_coverage = {
                (row["token_position"], row["token_id"])
                for row in clusters[target][prompt]
            }
            reference_coverage = {
                (row["token_position"], row["token_id"])
                for row in clusters[reference][prompt]
            }
            if target_coverage != reference_coverage:
                raise ValueError(
                    f"pair {pair!r}, prompt {prompt!r} has different token coverage"
                )

        prompt_ids = sorted(target_prompts)
        point_target = _cluster_stats(_selected_rows(clusters[target], prompt_ids))
        point_reference = _cluster_stats(
            _selected_rows(clusters[reference], prompt_ids)
        )
        point = {
            key: point_target[key] - point_reference[key] for key in point_target
        }
        draws = {key: [] for key in point}
        rng = random.Random(seed + 1000 + offset)
        for _ in range(resamples):
            selected = [rng.choice(prompt_ids) for _ in prompt_ids]
            target_stats = _cluster_stats(_selected_rows(clusters[target], selected))
            reference_stats = _cluster_stats(
                _selected_rows(clusters[reference], selected)
            )
            for key, values in draws.items():
                values.append(target_stats[key] - reference_stats[key])
        result["paired_differences"][pair] = {
            "definition": f"{target} - {reference}",
            "point": point,
            "ci95": {key: _ci95(values) for key, values in draws.items()},
        }

    output = Path(output).resolve()
    result["output"] = str(output)
    _write_json(output, result)
    return result
