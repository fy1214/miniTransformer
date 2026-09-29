"""Cross-framework comparison driver for logpdiff run directories.

``run_compare`` is the P0 compare stage: it consumes one run directory
produced by the two collectors (see :mod:`logpdiff.writer` for the layout),
aligns the two frameworks' tensor records, computes metrics for the aligned
pairs, localizes the first failing pipeline stage, and writes ``metrics.json``
(full fidelity) plus ``metrics.csv`` (one row per aligned pair) next to the
inputs. The returned dict follows the :func:`logpdiff.reporter.render_report`
``report_data`` contract, so it can be rendered directly (and ``metrics.json``
round-trips back into the same contract for the ``report`` CLI subcommand).

Side convention: sglang is side A (the reference), megatron is side B (the
target). Pair metrics are always computed as "A vs B".

Provenance separation (hard rule): actual_kernel payloads and
replayed_reference payloads are never silently blended. Every aligned pair is
tagged with a ``comparison_class``:

- ``actual_kernel`` / ``replayed_reference`` / ``hook_boundary`` /
  ``derived_metric`` -- both sides share that provenance;
- ``actual_vs_replay`` -- one side is ``actual_kernel`` and the other is
  ``replayed_reference`` (the explicitly-labeled actual-vs-replay
  cross-check, e.g. megatron's real kernel tensors vs sglang's Python
  replay);
- ``"<prov_a>_vs_<prov_b>"`` -- any other mixed provenance pair (e.g.
  sglang's actual layer_input vs megatron's hook_boundary capture); these
  are cross-provenance cross-checks and stay explicitly labeled as well.

Stage rows in the report are emitted per (pipeline stage, comparison_class),
so different classes are never aggregated into one number.

Data integrity:

- ``config.resolved.json`` consistency between the two frameworks is hard
  --validated where both sides provide the value: sample set, per-sample
  input_ids SHA-256, per-sample token counts, total token count, expert
  count and router top-k. Any disagreement raises ``ValueError``; nothing is
  ever truncated to ``min(length)``. Values missing on one side (or a
  missing/partial config, e.g. a hand-built run) downgrade the
  corresponding check to an entry in ``warnings`` -- a check that cannot
  run is reported, not silently skipped.
- Records are parsed with :meth:`TensorRecord.from_dict` and
  :meth:`TensorRecord.validate`; payloads are loaded only via
  :func:`logpdiff.writer.assert_internal_payload`.
- Experts in ``AlignmentResult.invalid_experts`` get their value metrics
  marked ``ALIGNMENT_INVALID`` (no cosine is computed for them).

Importing this module does NOT require torch (lazy import, same convention
as :mod:`logpdiff.metrics`).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from . import schema
from .aligner import ALIGNMENT_INVALID, align_records
from .metrics import (
    DEFAULT_THRESHOLDS,
    equal_summary,
    evaluate_stages,
    find_first_failure,
    logprob_metrics,
    routing_metrics,
    tensor_metrics,
    weighted_summary,
    worst_entries,
)
from .schema import (
    ACTUAL_KERNEL,
    PIPELINE_STAGES,
    REPLAYED_REFERENCE,
    ROUTER_TOPK_IDS,
    ROUTER_TOPK_WEIGHTS,
    TOKEN_LOGPROBS,
    TensorRecord,
)
from .writer import assert_internal_payload

__all__ = [
    "run_compare",
    "TENSOR_POINT_TO_STAGE",
    "COMPARISON_ACTUAL_VS_REPLAY",
    "REFERENCE_FRAMEWORK",
    "TARGET_FRAMEWORK",
]

REFERENCE_FRAMEWORK = "sglang"
"""Side A of the comparison (the reference)."""

TARGET_FRAMEWORK = "megatron"
"""Side B of the comparison (the target under test)."""

COMPARISON_ACTUAL_VS_REPLAY = "actual_vs_replay"
"""Class label for the explicitly-labeled actual-kernel vs replayed-reference
cross-check."""

TENSOR_POINT_TO_STAGE = {
    schema.LAYER_INPUT: "layer_input",
    schema.ROUTER_LOGITS: "routing",
    schema.ROUTER_TOPK_IDS: "routing",
    schema.ROUTER_TOPK_WEIGHTS: "routing",
    schema.DISPATCH_INPUT: "dispatch",
    schema.FC1_RAW_INPUT: "fc1_quant",
    schema.FC1_QDATA: "fc1_quant",
    schema.FC1_SCALES: "fc1_quant",
    schema.FC1_OUTPUT: "fc1_gemm",
    schema.SWIGLU_OUTPUT: "swiglu",
    schema.FC2_RAW_INPUT: "fc2_quant",
    schema.FC2_QDATA: "fc2_quant",
    schema.FC2_SCALES: "fc2_quant",
    schema.FC2_OUTPUT: "fc2_gemm",
    schema.EXPERT_COMBINED_OUTPUT: "combined_output",
    schema.TOKEN_LOGPROBS: "logprob",
}

# value-level metric keys replaced by ALIGNMENT_INVALID for pairs whose
# expert has invalid alignment (same list as metrics._VALUE_METRIC_KEYS,
# duplicated to stay on public surface)
_TENSOR_VALUE_KEYS = (
    "exact_match",
    "mean_abs_error",
    "max_abs_error",
    "p50_abs_error",
    "p95_abs_error",
    "p99_abs_error",
    "mse",
    "rmse",
    "sqnr_db",
    "symmetric_rel_l2",
    "cosine",
)

_CSV_COLUMNS = (
    "run_id",
    "sample_id",
    "layer_id",
    "tensor_point",
    "expert_id",
    "token_position",
    "route_slot",
    "stage",
    "comparison_class",
    "provenance_a",
    "provenance_b",
    "align_method",
    "alignment_status",
    "shape_a",
    "shape_b",
    "dtype_a",
    "dtype_b",
    "numel_a",
    "numel_b",
    "shape_match",
    "dtype_match",
    "checksum_match",
    "exact_match",
    "mean_abs_error",
    "max_abs_error",
    "p50_abs_error",
    "p95_abs_error",
    "p99_abs_error",
    "mse",
    "rmse",
    "sqnr_db",
    "symmetric_rel_l2",
    "cosine",
    "position_match_ratio",
    "set_match_ratio",
    "topk_overlap_mean",
    "weight_error_mean",
    "weight_error_max",
    "first_mismatch_position",
    "num_tokens",
    "tokens_over_threshold",
    "greedy_mismatch_count",
    "topk_mismatch_count",
    "kl_divergence_mean",
    "kl_divergence_max",
    "js_divergence_mean",
    "js_divergence_max",
    "note",
)

_STATUS_RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _import_torch():
    import torch  # noqa: F401 -- lazy: compare must import without torch

    return torch


def _is_number(value) -> bool:
    """Finite-or-infinite numeric (not bool, not NaN), JSON-safe."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not (isinstance(value, float) and math.isnan(value))


def _comparison_class(prov_a: str, prov_b: str) -> str:
    """Label one aligned pair's provenance combination.

    Same-provenance pairs carry the provenance itself; an actual/replayed
    mix is the explicitly-labeled ``actual_vs_replay`` cross-check; every
    other mix is labeled with both provenances (side A first).
    """
    if prov_a == prov_b:
        return prov_a
    if {prov_a, prov_b} == {ACTUAL_KERNEL, REPLAYED_REFERENCE}:
        return COMPARISON_ACTUAL_VS_REPLAY
    return f"{prov_a}_vs_{prov_b}"


def _worse(status_a: str, status_b: str) -> str:
    """Worst of two statuses; INVALID_ALIGNMENT / NODATA count as WARN."""
    rank_a = _STATUS_RANK.get(status_a, 1)
    rank_b = _STATUS_RANK.get(status_b, 1)
    return status_a if rank_a >= rank_b else status_b


# ---------------------------------------------------------------------------
# config.resolved.json: two-sided consistency (hard validation)
# ---------------------------------------------------------------------------


def _framework_section(config: dict, framework: str):
    """Return framework's section of config.resolved.json, or None.

    Supports the sectioned layout (``config["sglang"]`` / ``config["megatron"]``
    written by the collectors) and the legacy flat layout where one
    framework's keys sit at the top level next to ``"framework": <name>``.
    """
    if not isinstance(config, dict):
        return None
    section = config.get(framework)
    if isinstance(section, dict):
        return section
    if config.get("framework") == framework:
        return config
    return None


def _sample_map(section: dict, framework: str) -> dict:
    """Normalize a config section's sample list to
    ``{sample_id: {"num_tokens": int|None, "input_ids_sha256": str|None}}``.

    Accepts both the sglang collector's ``sample_id`` key and the megatron
    collector's ``name`` key. The SHA-256 is taken from
    ``input_ids_sha256`` when recorded, else recomputed from ``input_ids``
    exactly the way the sglang collector computes it.
    """
    out = {}
    for entry in section.get("samples") or []:
        if not isinstance(entry, dict):
            continue
        sid = entry.get("sample_id", entry.get("name"))
        if sid is None:
            continue
        sid = str(sid)
        if sid in out:
            raise ValueError(
                f"duplicate sample_id {sid!r} in the {framework} section of "
                "config.resolved.json"
            )
        ids = entry.get("input_ids")
        sha = entry.get("input_ids_sha256")
        if sha is None and isinstance(ids, list):
            sha = hashlib.sha256(json.dumps(ids).encode("utf-8")).hexdigest()
        num_tokens = entry.get("num_tokens")
        if num_tokens is None and isinstance(ids, list):
            num_tokens = len(ids)
        out[sid] = {"num_tokens": num_tokens, "input_ids_sha256": sha}
    return out


def _validate_consistency(
    run_id: str, config, sec_a, sec_b, warnings: list
) -> dict:
    """Hard-validate the two frameworks' config sections against each other.

    Raises ValueError on the first real disagreement (sample set, per-sample
    input_ids SHA / token count, total tokens, expert count, top-k). Checks
    whose inputs are missing (partial config, older/hand-built runs) are
    downgraded to ``warnings`` entries. Returns a config_check summary dict.
    """
    check = {
        "config_present": isinstance(config, dict),
        "run_id": run_id,
        "sections": {
            REFERENCE_FRAMEWORK: sec_a is not None,
            TARGET_FRAMEWORK: sec_b is not None,
        },
        "samples_checked": 0,
        "num_experts": None,
        "top_k": None,
    }
    if not isinstance(config, dict):
        warnings.append(
            "config.resolved.json missing or unreadable; two-sided config "
            "consistency checks were skipped"
        )
        return check
    config_run_id = config.get("run_id")
    if config_run_id is not None and config_run_id != run_id:
        raise ValueError(
            f"config.resolved.json run_id {config_run_id!r} does not match the "
            f"run directory name {run_id!r}"
        )
    if sec_a is None or sec_b is None:
        missing = REFERENCE_FRAMEWORK if sec_a is None else TARGET_FRAMEWORK
        warnings.append(
            f"config.resolved.json has no {missing!r} section; two-sided "
            "consistency checks downgraded to record-level validation"
        )
        return check

    # -- samples: set equality, per-sample sha / token counts, totals -------
    samples_a = _sample_map(sec_a, REFERENCE_FRAMEWORK)
    samples_b = _sample_map(sec_b, TARGET_FRAMEWORK)
    if samples_a and samples_b:
        only_a = sorted(set(samples_a) - set(samples_b))
        only_b = sorted(set(samples_b) - set(samples_a))
        if only_a or only_b:
            raise ValueError(
                "sample set mismatch between the two frameworks: "
                f"only in {REFERENCE_FRAMEWORK}: {only_a}; "
                f"only in {TARGET_FRAMEWORK}: {only_b}"
            )
        total_a = total_b = 0
        have_all_tokens = True
        for sid in samples_a:
            sa, sb = samples_a[sid], samples_b[sid]
            sha_a, sha_b = sa["input_ids_sha256"], sb["input_ids_sha256"]
            if sha_a is not None and sha_b is not None:
                if sha_a != sha_b:
                    raise ValueError(
                        f"input_ids sha256 mismatch for sample {sid!r}: "
                        f"{REFERENCE_FRAMEWORK}={sha_a} vs "
                        f"{TARGET_FRAMEWORK}={sha_b} (the two sides captured "
                        "different inputs; refusing to compare)"
                    )
            else:
                warnings.append(
                    f"sample {sid!r}: input_ids sha256 missing on one side; "
                    "input identity not verifiable for this sample"
                )
            n_a, n_b = sa["num_tokens"], sb["num_tokens"]
            if n_a is not None and n_b is not None:
                if n_a != n_b:
                    raise ValueError(
                        f"token count mismatch for sample {sid!r}: "
                        f"{REFERENCE_FRAMEWORK}={n_a} vs {TARGET_FRAMEWORK}={n_b}"
                    )
                total_a += n_a
                total_b += n_b
            else:
                have_all_tokens = False
                warnings.append(
                    f"sample {sid!r}: token count missing on one side; "
                    "token-count check skipped for this sample"
                )
        if have_all_tokens and total_a != total_b:
            # unreachable given the per-sample check, kept as a belt-and-braces
            raise ValueError(
                f"total token count mismatch: {total_a} vs {total_b}"
            )
        check["samples_checked"] = len(samples_a)
        check["total_tokens"] = total_a if have_all_tokens else None
    else:
        warnings.append(
            "config.resolved.json lacks a sample list on at least one side; "
            "input consistency not verifiable"
        )

    # -- expert count / top-k -------------------------------------------------
    experts_a = sec_a.get("num_experts")
    experts_b = sec_b.get("num_experts")
    if experts_a is not None and experts_b is not None:
        if experts_a != experts_b:
            raise ValueError(
                f"num_experts mismatch: {REFERENCE_FRAMEWORK}={experts_a} vs "
                f"{TARGET_FRAMEWORK}={experts_b}"
            )
        check["num_experts"] = experts_a
    elif experts_a is not None or experts_b is not None:
        warnings.append("num_experts recorded on one side only; cannot cross-check")

    topk_a = sec_a.get("top_k", sec_a.get("router_topk"))
    topk_b = sec_b.get("top_k", sec_b.get("router_topk"))
    if topk_a is not None and topk_b is not None:
        if topk_a != topk_b:
            raise ValueError(
                f"router top-k mismatch: {REFERENCE_FRAMEWORK}={topk_a} vs "
                f"{TARGET_FRAMEWORK}={topk_b}"
            )
        check["top_k"] = topk_a
    elif topk_a is not None or topk_b is not None:
        warnings.append("router top-k recorded on one side only; cannot cross-check")

    layer_a = sec_a.get("layer_idx")
    layer_b = sec_b.get("layer_idx")
    if layer_a is not None and layer_b is not None and layer_a != layer_b:
        warnings.append(
            f"layer_idx differs between sides ({layer_a} vs {layer_b}); "
            "records of different layers will simply not align"
        )
    return check


# ---------------------------------------------------------------------------
# index reading / payload loading
# ---------------------------------------------------------------------------


def _read_index(run_dir: Path, framework: str, run_id: str) -> list:
    """Read and validate ``<run_dir>/<framework>/index.jsonl``."""
    path = run_dir / framework / "index.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run the {framework} collector first"
        )
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
            try:
                record = TensorRecord.from_dict(raw)
                record.validate()
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: invalid TensorRecord: {e}") from e
            if record.framework != framework:
                raise ValueError(
                    f"{path}:{lineno}: record.framework {record.framework!r} "
                    f"does not match its index file {framework!r}"
                )
            if record.run_id != run_id:
                raise ValueError(
                    f"{path}:{lineno}: record.run_id {record.run_id!r} does not "
                    f"match run {run_id!r}"
                )
            records.append(record)
    return records


class _PayloadCache:
    """torch.load payloads once, always via assert_internal_payload."""

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        self._torch = None
        self._cache = {}

    def load(self, record: TensorRecord):
        if record.payload_path not in self._cache:
            path = assert_internal_payload(record.payload_path, self._run_dir)
            if not path.is_file():
                raise FileNotFoundError(
                    f"payload {path} referenced by a "
                    f"{record.tensor_point!r} record (sample {record.sample_id!r}) "
                    "does not exist"
                )
            if self._torch is None:
                self._torch = _import_torch()
            self._cache[record.payload_path] = self._torch.load(
                path, map_location="cpu"
            )
        return self._cache[record.payload_path]


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _aggregate_metrics(stage: str, metrics_list: list, thresholds: dict) -> dict:
    """Aggregate per-pair metric dicts of one (stage, class) cell.

    Aggregation is worst-case in the threshold direction: "above" metrics
    take the max across pairs, "below" metrics the min; metrics without a
    threshold rule take the mean. ``exact_match`` booleans become
    ``exact_match_ratio``. Non-numeric values (None, NaN, strings such as
    ALIGNMENT_INVALID) never enter the aggregate.
    """
    rules = thresholds.get(stage) or {}
    bools = []
    values = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            if key == "exact_match" and isinstance(value, bool):
                bools.append(1.0 if value else 0.0)
                continue
            if _is_number(value):
                values.setdefault(key, []).append(float(value))
    out = {}
    if bools:
        out["exact_match_ratio"] = sum(bools) / len(bools)
    for key, vals in values.items():
        when = (rules.get(key) or {}).get("when")
        if when == "above":
            out[key] = max(vals)
        elif when == "below":
            out[key] = min(vals)
        else:
            out[key] = sum(vals) / len(vals)
    return out


def _invalid_metrics() -> dict:
    """Metric placeholder for pairs whose expert has invalid alignment."""
    out = {key: ALIGNMENT_INVALID for key in _TENSOR_VALUE_KEYS}
    out["note"] = (
        "expert alignment is invalid (unexplained conflict); value metrics "
        "suppressed instead of reporting a misleading cosine"
    )
    return out


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def run_compare(
    run_dir,
    *,
    allow_value_match_fallback: bool = False,
    thresholds=None,
) -> dict:
    """Compare the two frameworks captured in ``run_dir``.

    Reads ``config.resolved.json`` (hard two-sided consistency validation),
    both ``index.jsonl`` files, aligns records with
    :func:`logpdiff.aligner.align_records`, computes tensor/routing/logprob
    metrics for every aligned pair, aggregates them per pipeline stage and
    comparison class, and writes ``metrics.json`` + ``metrics.csv`` into
    ``run_dir``.

    ``thresholds`` overrides :data:`logpdiff.metrics.DEFAULT_THRESHOLDS`
    (same shape: ``{stage: {metric: {"when": "above"|"below", "warn": x,
    "fail": y}}}``).

    Returns the full result dict, which doubles as the ``report_data``
    contract of :func:`logpdiff.reporter.render_report`.

    Raises ValueError on any two-sided consistency violation, FileNotFoundError
    on missing run files or payloads. Never truncates to a common length.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    if not isinstance(thresholds, dict):
        raise TypeError(
            f"thresholds must be a dict like DEFAULT_THRESHOLDS, got "
            f"{type(thresholds).__name__}"
        )
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory {run_dir} does not exist")
    run_id = run_dir.name
    warnings: list = []

    # -- config + two-sided consistency --------------------------------------
    config_path = run_dir / "config.resolved.json"
    config = None
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    else:
        warnings.append(f"{config_path} not found")
    sec_a = _framework_section(config, REFERENCE_FRAMEWORK)
    sec_b = _framework_section(config, TARGET_FRAMEWORK)
    config_check = _validate_consistency(run_id, config, sec_a, sec_b, warnings)

    # -- records ---------------------------------------------------------------
    records_a = _read_index(run_dir, REFERENCE_FRAMEWORK, run_id)
    records_b = _read_index(run_dir, TARGET_FRAMEWORK, run_id)
    if not records_a or not records_b:
        warnings.append(
            "one side's index.jsonl is empty "
            f"({REFERENCE_FRAMEWORK}: {len(records_a)}, "
            f"{TARGET_FRAMEWORK}: {len(records_b)} records); "
            "little or nothing can be compared"
        )

    # -- alignment ---------------------------------------------------------------
    alignment = align_records(
        records_a,
        records_b,
        run_dir,
        run_dir,
        allow_value_match_fallback=allow_value_match_fallback,
    )

    # -- per-pair metrics ---------------------------------------------------------
    loader = _PayloadCache(run_dir)
    weights_pairs = {}  # (sample_id, layer_id) -> aligned pair of router_topk_weights
    for pair in alignment.matched:
        if pair.record_a.tensor_point == ROUTER_TOPK_WEIGHTS:
            weights_pairs.setdefault((pair.sample_id, pair.layer_id), pair)

    pair_entries = []
    token_entries = []  # per-token logprob diffs (worst tokens / plots)
    for pair in alignment.matched:
        rec_a, rec_b = pair.record_a, pair.record_b
        tensor_point = rec_a.tensor_point
        stage = TENSOR_POINT_TO_STAGE[tensor_point]
        entry = {
            "sample_id": pair.sample_id,
            "layer_id": pair.layer_id,
            "tensor_point": tensor_point,
            "expert_id": pair.expert_id,
            "token_position": pair.token_position,
            "route_slot": pair.route_slot,
            "stage": stage,
            "comparison_class": _comparison_class(
                rec_a.provenance, rec_b.provenance
            ),
            "provenance_a": rec_a.provenance,
            "provenance_b": rec_b.provenance,
            "align_method": pair.method,
            "alignment_status": "OK",
            "payload_a": rec_a.payload_path,
            "payload_b": rec_b.payload_path,
            "shape_a": list(rec_a.shape),
            "shape_b": list(rec_b.shape),
            "dtype_a": rec_a.dtype,
            "dtype_b": rec_b.dtype,
        }
        if pair.expert_id is not None and pair.expert_id in alignment.invalid_experts:
            entry["alignment_status"] = ALIGNMENT_INVALID
            entry["metrics"] = _invalid_metrics()
            pair_entries.append(entry)
            continue

        t_a = loader.load(rec_a)
        t_b = loader.load(rec_b)
        if tensor_point == ROUTER_TOPK_IDS:
            partner = weights_pairs.get((pair.sample_id, pair.layer_id))
            if partner is None:
                entry["metrics"] = {
                    "error": "no aligned router_topk_weights pair for this "
                             "sample/layer; routing metrics not computed"
                }
            else:
                w_a = loader.load(partner.record_a)
                w_b = loader.load(partner.record_b)
                try:
                    entry["metrics"] = routing_metrics(t_a, w_a, t_b, w_b)
                except ValueError as e:
                    entry["metrics"] = {"error": str(e)}
        elif tensor_point == TOKEN_LOGPROBS:
            try:
                entry["metrics"] = logprob_metrics(t_a, t_b)
            except ValueError as e:
                entry["metrics"] = {"error": str(e)}
            else:
                diff = (t_a.detach().float().reshape(-1)
                        - t_b.detach().float().reshape(-1)).abs()
                la = t_a.detach().float().reshape(-1)
                lb = t_b.detach().float().reshape(-1)
                for i in range(diff.numel()):
                    token_entries.append(
                        {
                            "sample_id": pair.sample_id,
                            "layer_id": pair.layer_id,
                            # index into the per-token logprob vector (both
                            # collectors store logprob of target token i+1 at i)
                            "token_position": i,
                            "ref_logprob": float(la[i]),
                            "target_logprob": float(lb[i]),
                            "abs_diff": float(diff[i]),
                        }
                    )
        elif tensor_point == ROUTER_TOPK_WEIGHTS:
            # per-pair float compare; the ids partner row carries routing_metrics
            entry["metrics"] = tensor_metrics(t_a, t_b)
        else:
            entry["metrics"] = tensor_metrics(t_a, t_b)
        pair_entries.append(entry)

    # -- stage x comparison-class cells --------------------------------------------
    cells = {}  # (stage, comparison_class) -> list of metric dicts
    cell_invalid = {}  # (stage, comparison_class) -> invalid pair count
    for entry in pair_entries:
        key = (entry["stage"], entry["comparison_class"])
        if entry["alignment_status"] == ALIGNMENT_INVALID:
            cell_invalid[key] = cell_invalid.get(key, 0) + 1
        cells.setdefault(key, []).append(entry["metrics"])

    stage_rows = []
    stage_status = {}
    stage_metrics_full = {}
    for stage in PIPELINE_STAGES:
        stage_cells = [(k[1], cells[k]) for k in sorted(cells) if k[0] == stage]
        if not stage_cells:
            stage_rows.append(
                {
                    "stage": stage,
                    "status": "NODATA",
                    "metrics": {},
                    "reason": "no aligned pairs for this stage",
                }
            )
            continue
        combined = "PASS"
        per_class = {}
        for cls, metrics_list in stage_cells:
            aggregated = _aggregate_metrics(stage, metrics_list, thresholds)
            invalid_count = cell_invalid.get((stage, cls), 0)
            error_count = sum(1 for m in metrics_list if "error" in m)
            row_metrics = {
                "comparison_class": cls,
                "pair_count": len(metrics_list),
            }
            if invalid_count:
                row_metrics["invalid_pair_count"] = invalid_count
            if error_count:
                row_metrics["error_pair_count"] = error_count
            row_metrics.update(aggregated)
            per_class[cls] = row_metrics
            if aggregated:
                status = evaluate_stages({stage: row_metrics}, thresholds)[stage]
            elif invalid_count:
                status = "INVALID_ALIGNMENT"
            else:
                status = "NODATA"
            row = {"stage": stage, "status": status, "metrics": row_metrics}
            if status == "INVALID_ALIGNMENT":
                row["reason"] = (
                    f"{invalid_count} pair(s) with invalid expert alignment; "
                    "their value metrics are suppressed"
                )
            elif status == "NODATA":
                first_error = next(
                    (m["error"] for m in metrics_list if "error" in m), None
                )
                row["reason"] = first_error or "no computable metrics"
            stage_rows.append(row)
            combined = _worse(combined, status)
        stage_metrics_full[stage] = per_class
        stage_status[stage] = combined

    first_stage, first_status = find_first_failure(stage_status)
    if first_stage is None:
        first_failure = None
    else:
        first_failure = {
            "stage": first_stage,
            "status": first_status,
            "description": (
                f"earliest {first_status} pipeline stage in "
                f"PIPELINE_STAGES order: {first_stage!r}"
            ),
        }

    # -- expert table ----------------------------------------------------------------
    expert_groups = {}
    for entry in pair_entries:
        if entry["expert_id"] is None:
            continue
        key = (entry["layer_id"], entry["expert_id"])
        expert_groups.setdefault(key, []).append(entry)
    expert_rows = []
    for (layer_id, expert_id), entries in sorted(expert_groups.items()):
        invalid = expert_id in alignment.invalid_experts
        valid_metrics = [
            e["metrics"] for e in entries if e["alignment_status"] != ALIGNMENT_INVALID
        ]
        row = {
            "layer_id": layer_id,
            "expert_id": expert_id,
            "pair_count": len(entries),
            "status": ALIGNMENT_INVALID if invalid else "OK",
            "tensor_points": sorted({e["tensor_point"] for e in entries}),
            "comparison_classes": sorted({e["comparison_class"] for e in entries}),
            "max_abs_err": None,
            "mean_abs_err": None,
            "cosine": None,
        }
        max_errs = [
            m["max_abs_error"] for m in valid_metrics if _is_number(m.get("max_abs_error"))
        ]
        mean_errs = [
            m["mean_abs_error"]
            for m in valid_metrics
            if _is_number(m.get("mean_abs_error"))
        ]
        cosines = [m["cosine"] for m in valid_metrics if _is_number(m.get("cosine"))]
        if max_errs:
            row["max_abs_err"] = max(max_errs)
        if mean_errs:
            row["mean_abs_err"] = sum(mean_errs) / len(mean_errs)
        if cosines:
            row["cosine"] = min(cosines)
        expert_rows.append(row)
    # untrusted (invalid-alignment) experts first, then worst numeric error
    expert_rows.sort(
        key=lambda r: (
            0 if r["status"] == ALIGNMENT_INVALID else 1,
            -(r["max_abs_err"] if r["max_abs_err"] is not None else -1.0),
        )
    )

    # -- per-expert summaries ---------------------------------------------------------
    per_expert = {}
    token_counts = {}
    for entry in pair_entries:
        eid = entry["expert_id"]
        if eid is None or entry["alignment_status"] == ALIGNMENT_INVALID:
            continue
        metrics = entry["metrics"]
        agg = per_expert.setdefault(eid, {})
        for key in ("mean_abs_error", "symmetric_rel_l2"):
            if _is_number(metrics.get(key)):
                agg.setdefault("_sum_" + key, [0.0, 0])
                agg["_sum_" + key][0] += float(metrics[key])
                agg["_sum_" + key][1] += 1
        for key in ("max_abs_error",):
            if _is_number(metrics.get(key)):
                agg[key] = max(float(metrics[key]), agg.get(key, float("-inf")))
        if _is_number(metrics.get("cosine")):
            agg["cosine"] = min(float(metrics["cosine"]), agg.get("cosine", float("inf")))
        shape = entry["shape_a"]
        token_counts[eid] = token_counts.get(eid, 0) + (shape[0] if shape else 1)
    for eid, agg in per_expert.items():
        for key in ("mean_abs_error", "symmetric_rel_l2"):
            acc = agg.pop("_sum_" + key, None)
            if acc and acc[1]:
                agg[key] = acc[0] / acc[1]
        if agg.get("cosine") == float("inf"):
            agg["cosine"] = None
        if agg.get("max_abs_error") == float("-inf"):
            agg["max_abs_error"] = None
    summaries = {}
    if per_expert:
        summaries["per_expert_weighted"] = weighted_summary(per_expert, token_counts)
        summaries["per_expert_equal"] = equal_summary(per_expert)

    # -- worst tokens (per-token logprob diffs) -----------------------------------------
    worst_token_rows = [
        item["entry"] for item in worst_entries(token_entries, "abs_diff", topk=10)
    ]

    # -- report sections ------------------------------------------------------------------
    legend = {}
    for record in list(records_a) + list(records_b):
        legend.setdefault(record.provenance, set()).add(record.tensor_point)
    provenance_legend = {k: sorted(v) for k, v in sorted(legend.items())}

    expert_layer = {}
    for record in list(records_a) + list(records_b):
        if record.expert_id is not None and record.expert_id not in expert_layer:
            expert_layer[record.expert_id] = record.layer_id
    config_layer = None
    for section in (sec_a, sec_b):
        if isinstance(section, dict) and section.get("layer_idx") is not None:
            config_layer = section["layer_idx"]
            break
    alignment_info = {
        "method": alignment.method,
        "matched": len(alignment.matched),
        "unmatched": len(alignment.unmatched_a) + len(alignment.unmatched_b),
        f"unmatched_{REFERENCE_FRAMEWORK}": len(alignment.unmatched_a),
        f"unmatched_{TARGET_FRAMEWORK}": len(alignment.unmatched_b),
        "duplicate": alignment.duplicate_count,
        "conflict": alignment.conflict_count,
        "min_fallback_cosine": alignment.min_fallback_cosine,
        "one_to_one_per_expert": {
            str(k): v for k, v in sorted(alignment.one_to_one_per_expert.items())
        },
        "invalid_experts": [
            {
                "layer_id": expert_layer.get(eid, config_layer),
                "expert_id": eid,
                "reason": "unexplained alignment conflict (dispatcher-keyed "
                          "pair with contradictory shapes)",
            }
            for eid in sorted(alignment.invalid_experts)
        ],
    }

    sample_rows = _build_sample_rows(sec_a, sec_b, pair_entries, alignment)
    run_info = _build_run_info(run_dir, run_id, config, sec_a, sec_b)

    # -- plot inputs (consumed by render_report only when with_plots=True) --------------
    heat_tps = sorted({e["tensor_point"] for e in pair_entries})
    heat_layers = sorted({e["layer_id"] for e in pair_entries})
    heat_rows = []
    for layer_id in heat_layers:
        values = []
        for tensor_point in heat_tps:
            errs = [
                e["metrics"]["max_abs_error"]
                for e in pair_entries
                if e["layer_id"] == layer_id
                and e["tensor_point"] == tensor_point
                and _is_number(e["metrics"].get("max_abs_error"))
            ]
            values.append(max(errs) if errs else None)
        heat_rows.append({"layer_id": layer_id, "values": values})
    heatmap = {"tensor_points": heat_tps, "rows": heat_rows} if heat_tps else None

    token_error_series = [
        {"token_position": i, "abs_diff": entry["abs_diff"]}
        for i, entry in enumerate(token_entries)
    ]
    error_values = [
        e["metrics"]["max_abs_error"]
        for e in pair_entries
        if _is_number(e["metrics"].get("max_abs_error"))
    ]
    scatter_points = [
        (entry["ref_logprob"], entry["target_logprob"]) for entry in token_entries
    ]
    if len(scatter_points) > 2000:
        stride = math.ceil(len(scatter_points) / 2000)
        scatter_points = scatter_points[::stride]
    scatter = {
        "points": scatter_points,
        "xlabel": f"{REFERENCE_FRAMEWORK} logprob (reference)",
        "ylabel": f"{TARGET_FRAMEWORK} logprob (target)",
    }

    result = {
        # -- reporter contract --------------------------------------------------
        "run": run_info,
        "first_failure": first_failure,
        "stage_table": stage_rows,
        "expert_table": expert_rows,
        "worst_tokens": worst_token_rows,
        "provenance_legend": provenance_legend,
        "alignment": alignment_info,
        "samples": sample_rows,
        # -- plot inputs --------------------------------------------------------
        "heatmap": heatmap,
        "token_error_series": token_error_series,
        "error_values": error_values,
        "scatter": scatter,
        # -- full-fidelity extras -------------------------------------------------
        "run_id": run_id,
        "run_dir": str(run_dir),
        "generated_by": "logpdiff.compare.run_compare",
        "reference_framework": REFERENCE_FRAMEWORK,
        "target_framework": TARGET_FRAMEWORK,
        "allow_value_match_fallback": bool(allow_value_match_fallback),
        "thresholds": thresholds,
        "config_check": config_check,
        "warnings": warnings,
        "stage_status": stage_status,
        "stage_metrics": stage_metrics_full,
        "summaries": summaries,
        "comparison_classes": _class_counts(pair_entries),
        "pairs": pair_entries,
        "unmatched": {
            REFERENCE_FRAMEWORK: [r.to_dict() for r in alignment.unmatched_a],
            TARGET_FRAMEWORK: [r.to_dict() for r in alignment.unmatched_b],
        },
    }

    _write_metrics_json(run_dir / "metrics.json", result)
    _write_metrics_csv(run_dir / "metrics.csv", run_id, pair_entries)
    return result


# ---------------------------------------------------------------------------
# report section builders
# ---------------------------------------------------------------------------


def _class_counts(pair_entries) -> dict:
    counts = {}
    for entry in pair_entries:
        cls = entry["comparison_class"]
        counts[cls] = counts.get(cls, 0) + 1
    return dict(sorted(counts.items()))


def _build_sample_rows(sec_a, sec_b, pair_entries, alignment) -> list:
    """One row per sample: config facts plus per-sample alignment counts."""
    samples = {}
    for framework, section in (
        (REFERENCE_FRAMEWORK, sec_a),
        (TARGET_FRAMEWORK, sec_b),
    ):
        if not isinstance(section, dict):
            continue
        for sid, info in _sample_map(section, framework).items():
            samples.setdefault(sid, info)
    if not samples:
        # configs without sample lists: derive the sample set from the records
        for entry in pair_entries:
            samples.setdefault(
                str(entry["sample_id"]), {"num_tokens": None, "input_ids_sha256": None}
            )
        for record in list(alignment.unmatched_a) + list(alignment.unmatched_b):
            samples.setdefault(
                str(record.sample_id), {"num_tokens": None, "input_ids_sha256": None}
            )
    rows = []
    for sid in sorted(samples):
        matched = sum(1 for e in pair_entries if str(e["sample_id"]) == sid)
        unmatched_a = sum(
            1 for r in alignment.unmatched_a if str(r.sample_id) == sid
        )
        unmatched_b = sum(
            1 for r in alignment.unmatched_b if str(r.sample_id) == sid
        )
        rows.append(
            {
                "sample_id": sid,
                "num_tokens": samples[sid]["num_tokens"],
                "input_ids_sha256": samples[sid]["input_ids_sha256"],
                "matched_pairs": matched,
                f"unmatched_{REFERENCE_FRAMEWORK}": unmatched_a,
                f"unmatched_{TARGET_FRAMEWORK}": unmatched_b,
            }
        )
    return rows


def _build_run_info(run_dir, run_id, config, sec_a, sec_b) -> dict:
    """The ``run`` section of the report: manifest.json first, config as fallback."""
    info = {"run_id": run_id}
    manifest_path = Path(run_dir) / "manifest.json"
    if manifest_path.is_file():
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            manifest = None
        if isinstance(manifest, dict):
            info.update(
                {
                    "git_commit": manifest.get("git_commit"),
                    "model": manifest.get("model_path"),
                    "versions": manifest.get("versions"),
                    "env": manifest.get("env"),
                    "seed": manifest.get("seed"),
                    "parallel": manifest.get("parallel"),
                    "force_routing": manifest.get("force_routing"),
                }
            )
    config = config if isinstance(config, dict) else {}
    if info.get("model") is None:
        info["model"] = (
            (sec_a or {}).get("model_path")
            or (sec_b or {}).get("model_path")
            or config.get("model_path")
        )
    if info.get("parallel") is None:
        parallel = (sec_b or {}).get("parallel") or (sec_a or {}).get("parallel")
        if parallel is None and all(k in config for k in ("tp", "ep", "pp")):
            parallel = {"tp": config["tp"], "ep": config["ep"], "pp": config["pp"]}
        info["parallel"] = parallel if parallel is not None else {"tp": 1, "ep": 1, "pp": 1}
    if info.get("seed") is None:
        info["seed"] = config.get("seed")
    if info.get("force_routing") is None:
        info["force_routing"] = bool(
            config.get("force_routing")
            or (sec_a or {}).get("force_routing")
            or (sec_b or {}).get("force_routing")
        )
    return {k: v for k, v in info.items() if v is not None}


# ---------------------------------------------------------------------------
# output files
# ---------------------------------------------------------------------------


def _write_metrics_json(path: Path, result: dict) -> None:
    """Full-fidelity metrics dump (also re-loadable as report_data)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_metrics_csv(path: Path, run_id: str, pair_entries) -> None:
    """One row per aligned pair: primary keys + tensor_point + provenance +
    the key metric columns (empty when a metric does not apply to the row's
    tensor point, ALIGNMENT_INVALID where suppressed)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(_CSV_COLUMNS))
        writer.writeheader()
        for entry in pair_entries:
            metrics = entry["metrics"]
            row = {
                "run_id": run_id,
                "sample_id": entry["sample_id"],
                "layer_id": entry["layer_id"],
                "tensor_point": entry["tensor_point"],
                "expert_id": entry["expert_id"],
                "token_position": entry["token_position"],
                "route_slot": entry["route_slot"],
                "stage": entry["stage"],
                "comparison_class": entry["comparison_class"],
                "provenance_a": entry["provenance_a"],
                "provenance_b": entry["provenance_b"],
                "align_method": entry["align_method"],
                "alignment_status": entry["alignment_status"],
                "shape_a": entry["shape_a"],
                "shape_b": entry["shape_b"],
                "dtype_a": entry["dtype_a"],
                "dtype_b": entry["dtype_b"],
                "note": metrics.get("error") or metrics.get("note"),
            }
            for key in _CSV_COLUMNS:
                if key not in row and key in metrics:
                    row[key] = metrics[key]
            writer.writerow(
                {k: ("" if row.get(k) is None else row.get(k)) for k in _CSV_COLUMNS}
            )
