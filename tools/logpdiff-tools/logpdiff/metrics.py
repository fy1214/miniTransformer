"""Metric computation for logpdiff train/inference consistency diagnostics.

All metrics are implemented with torch. Inputs may live on any device (CPU
or CUDA) and may have any real-valued dtype: values are cast to float32 for
computation (float64 accumulation for sums/norms), so callers do not need to
normalize device or dtype first.

Data-integrity rules enforced by this module:

- No silent truncation. Shape or token-count mismatches either raise
  ``ValueError`` (:func:`routing_metrics`, :func:`logprob_metrics`,
  :func:`fp4_underflow_ratio`) or are reported explicitly with
  ``shape_match=False`` plus an ``error`` note (:func:`tensor_metrics`).
- Metrics that cannot be computed are reported as ``None`` (with an
  ``error`` note where applicable), never as made-up numbers.

Importing this module does NOT require torch (lazy import, same convention
as :mod:`logpdiff.writer`); calling any metric function does.

All returned dicts contain only JSON-serializable Python values
(str/int/float/bool/None/list/dict). One caveat: an infinite SQNR (exact
signals) is ``float("inf")``, which Python's default ``json.dumps`` writes
as ``Infinity`` (valid for Python's decoder, non-strict JSON).
"""

from __future__ import annotations

import hashlib
import math

from .schema import PIPELINE_STAGES

__all__ = [
    "tensor_metrics",
    "fp4_underflow_ratio",
    "routing_metrics",
    "logprob_metrics",
    "weighted_summary",
    "equal_summary",
    "worst_entries",
    "DEFAULT_THRESHOLDS",
    "evaluate_stages",
    "find_first_failure",
]

# Epsilon for the symmetric relative-L2 denominator. Small enough to leave
# non-degenerate cases untouched; large enough that two exactly-zero tensors
# yield 0 / eps = 0 ("identical") instead of 0/0 = nan.
_EPS = 1e-12

# Value-level metric keys nulled out when a comparison is not computable.
_VALUE_METRIC_KEYS = (
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


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------


def _import_torch():
    import torch  # noqa: F401 -- lazy: metrics must import without torch

    return torch


def _require_tensor(t, name: str, torch) -> None:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}")


def _tensor_bytes_sha256(tensor, torch) -> str:
    """SHA-256 over the raw payload bytes of a contiguous CPU copy.

    Byte-level only: dtype and shape are NOT mixed into the hash, so
    ``checksum_match`` answers "same bytes?" while ``dtype_match`` /
    ``shape_match`` answer the remaining metadata questions.
    """
    t = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    if t.numel() > 0:
        flat = t.reshape(-1)
        # reinterpret as uint8 so the hash covers non-f32 dtypes (bf16/fp16/...)
        try:
            raw = flat.view(torch.uint8).numpy().tobytes()
        except (TypeError, RuntimeError):
            import io

            buf = io.BytesIO()
            torch.save(t, buf)
            raw = buf.getvalue()
        h.update(raw)
    return h.hexdigest()


def _percentile(sorted_vals, q: float) -> float:
    """Linear-interpolation percentile (numpy "linear" method).

    ``sorted_vals`` must be a sorted 1-D tensor with at least one element.
    Implemented manually (not ``torch.quantile``) because ``torch.quantile``
    rejects tensors with more than 2**24 elements.
    """
    n = sorted_vals.numel()
    if n == 1:
        return float(sorted_vals[0])
    rank = (q / 100.0) * (n - 1)
    lo = int(math.floor(rank))
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return float(sorted_vals[lo]) * (1.0 - frac) + float(sorted_vals[hi]) * frac


# ---------------------------------------------------------------------------
# tensor-level metrics
# ---------------------------------------------------------------------------


def tensor_metrics(a, b) -> dict:
    """Comparison metrics between tensors ``a`` (reference) and ``b``.

    Both tensors are cast to float32 for computation; nothing about device
    or storage dtype is assumed. Returned keys:

    - ``shape_a``/``shape_b``, ``dtype_a``/``dtype_b``, ``numel_a``/``numel_b``
    - ``shape_match`` / ``dtype_match``: metadata consistency flags
    - ``checksum_a`` / ``checksum_b`` / ``checksum_match``: SHA-256 of the raw
      payload bytes (byte-level; dtype/shape not mixed into the hash)
    - ``nan_count_*`` / ``inf_count_*`` / ``zero_count_*``: per-tensor element
      counts, always computed (even on shape mismatch)
    - ``exact_match``: ``torch.equal`` (value-level; promotes dtype, treats
      NaN as unequal; consult ``dtype_match`` for storage dtype)
    - ``mean/max/p50/p95/p99_abs_error``, ``mse``, ``rmse``
    - ``sqnr_db``: 10*log10(||a||^2 / ||a-b||^2); ``+inf`` when the error is
      exactly zero, ``-inf`` when the signal is exactly zero but the error
      is not
    - ``symmetric_rel_l2``: ||a-b|| / (||a|| + ||b|| + eps)
    - ``cosine``: flattened cosine similarity; defined as 1.0 when both
      tensors are exactly zero and 0.0 when exactly one is
    - ``error``: present only when some metrics could not be computed

    Shape mismatch is NEVER silently truncated: ``shape_match`` is False,
    only the shape-independent metrics (checksums, NaN/Inf/zero counts) are
    computed, every value-level metric is None, and ``error`` explains why.
    """
    torch = _import_torch()
    _require_tensor(a, "a", torch)
    _require_tensor(b, "b", torch)

    af = a.detach().float()
    bf = b.detach().float()

    result = {
        "shape_a": [int(d) for d in a.shape],
        "shape_b": [int(d) for d in b.shape],
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "numel_a": int(a.numel()),
        "numel_b": int(b.numel()),
        "shape_match": tuple(a.shape) == tuple(b.shape),
        "dtype_match": a.dtype == b.dtype,
        "checksum_a": _tensor_bytes_sha256(a, torch),
        "checksum_b": _tensor_bytes_sha256(b, torch),
        # per-tensor element counts: safe for any shape, including mismatched
        "nan_count_a": int(torch.isnan(af).sum()),
        "nan_count_b": int(torch.isnan(bf).sum()),
        "inf_count_a": int(torch.isinf(af).sum()),
        "inf_count_b": int(torch.isinf(bf).sum()),
        "zero_count_a": int((af == 0).sum()),
        "zero_count_b": int((bf == 0).sum()),
    }
    result["checksum_match"] = result["checksum_a"] == result["checksum_b"]

    if not result["shape_match"]:
        for key in _VALUE_METRIC_KEYS:
            result[key] = None
        result["error"] = (
            f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}: "
            "value-level metrics not computed (no silent truncation)"
        )
        return result

    result["exact_match"] = bool(torch.equal(a, b))

    if af.numel() == 0:
        for key in _VALUE_METRIC_KEYS:
            if key != "exact_match":
                result[key] = None
        result["error"] = "empty tensors: value-level metrics not computed"
        return result

    xf = af.reshape(-1)
    yf = bf.reshape(-1)
    # float64 accumulation for sums/norms: data integrity over CPU cost
    x = xf.double()
    y = yf.double()
    err = x - y
    abs_err_f32 = (xf - yf).abs()
    sorted_abs = torch.sort(abs_err_f32).values

    n = float(x.numel())
    err_sq = float((err * err).sum())
    sig_power = float((x * x).sum())
    mse = err_sq / n

    if err_sq == 0.0:
        sqnr_db = float("inf")
    elif sig_power == 0.0:
        sqnr_db = float("-inf")
    else:
        sqnr_db = 10.0 * math.log10(sig_power / err_sq)

    norm_a = math.sqrt(sig_power)
    norm_b = math.sqrt(float((y * y).sum()))
    norm_err = math.sqrt(err_sq)

    if norm_a == 0.0 and norm_b == 0.0:
        cosine = 1.0
    elif norm_a == 0.0 or norm_b == 0.0:
        cosine = 0.0
    else:
        dot = float((x * y).sum())
        # clamp against float overshoot just outside [-1, 1]
        cosine = max(-1.0, min(1.0, dot / (norm_a * norm_b)))

    result.update(
        {
            "mean_abs_error": float(err.abs().sum()) / n,
            "max_abs_error": float(abs_err_f32.max()),
            "p50_abs_error": _percentile(sorted_abs, 50.0),
            "p95_abs_error": _percentile(sorted_abs, 95.0),
            "p99_abs_error": _percentile(sorted_abs, 99.0),
            "mse": mse,
            "rmse": math.sqrt(mse),
            "sqnr_db": sqnr_db,
            "symmetric_rel_l2": norm_err / (norm_a + norm_b + _EPS),
            "cosine": cosine,
        }
    )
    return result


def fp4_underflow_ratio(a, b) -> float:
    """Fraction of positions where ``a`` is nonzero but ``b`` is zero.

    Intended for NVFP4 dequant views: ``a`` is the original (e.g. bf16)
    activation, ``b`` the same positions after quantize->dequant, so this
    measures how many nonzero activations collapsed to zero on the coarse
    e2m1 grid (min nonzero magnitude 0.5 x block scale).

    Returns 0.0 when ``a`` has no nonzero elements (nothing to underflow).
    Raises ValueError on shape mismatch -- never a truncated comparison.
    """
    torch = _import_torch()
    _require_tensor(a, "a", torch)
    _require_tensor(b, "b", torch)
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(
            f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}: "
            "refusing to compare a truncated prefix"
        )
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    nonzero = af != 0
    n = int(nonzero.sum())
    if n == 0:
        return 0.0
    underflowed = int(((bf == 0) & nonzero).sum())
    return underflowed / n


# ---------------------------------------------------------------------------
# routing metrics
# ---------------------------------------------------------------------------


def _as_2d_ids(ids, name: str, torch):
    t = ids.detach()
    if t.dim() == 1:
        t = t.unsqueeze(1)
    if t.dim() != 2:
        raise ValueError(f"{name} must be [num_tokens] or [num_tokens, k], got {tuple(t.shape)}")
    return t.long()


def routing_metrics(ids_a, weights_a, ids_b, weights_b) -> dict:
    """Comparison metrics for router top-k selections.

    ``ids_*`` are top-k expert ids, shape ``[num_tokens, k]`` (a 1-D
    ``[num_tokens]`` input is treated as k=1). ``weights_*`` are either

    - aligned top-k weights, shape ``[num_tokens, k]`` (e.g. the
      ``router_topk_weights`` tensor point), or
    - full router scores, shape ``[num_tokens, num_experts]`` (e.g.
      ``router_logits`` or full probabilities), which additionally enable
      the top-k boundary margin.

    Token-count (or any other shape) mismatch raises ValueError -- comparing
    a truncated prefix is never acceptable. Zero tokens also raises.

    Returned keys:

    - ``num_tokens``, ``top_k``
    - ``position_match_ratio``: fraction of (token, slot) positions with
      identical expert id
    - ``set_match_ratio``: fraction of tokens whose selected expert SETS
      are identical (order-insensitive)
    - ``topk_overlap_mean`` / ``topk_overlap_min``: per-token intersection
      size statistics
    - ``matched_position_count``, ``weight_error_mean``, ``weight_error_max``:
      |w_a - w_b| over slots where both sides selected the same expert id
      (None when no slot matches); for full scores the weights are gathered
      at the selected ids
    - ``boundary_margin_mean_*`` / ``boundary_margin_min_*``: per-token
      margin between the smallest selected score (k-th) and the largest
      non-selected score ((k+1)-th), for both sides; None unless full
      scores are given and k < num_experts. A margin near zero means the
      top-k selection can flip under tiny numeric noise.
    - ``num_experts``, ``expert_load_a`` / ``expert_load_b``: per-expert
      token counts as lists indexed by expert id
    - ``first_mismatch_position``: first token index where any slot's id
      differs, else None
    """
    torch = _import_torch()
    ia = _as_2d_ids(ids_a, "ids_a", torch)
    ib = _as_2d_ids(ids_b, "ids_b", torch)
    if ia.shape[0] != ib.shape[0]:
        raise ValueError(
            f"token count mismatch: {ia.shape[0]} vs {ib.shape[0]}: "
            "refusing to compare a truncated prefix"
        )
    if ia.shape != ib.shape:
        raise ValueError(f"top-k id shape mismatch: {tuple(ia.shape)} vs {tuple(ib.shape)}")
    num_tokens, k = ia.shape
    if num_tokens == 0:
        raise ValueError("no tokens to compare")

    wa = weights_a.detach().float()
    wb = weights_b.detach().float()
    if wa.shape != wb.shape:
        raise ValueError(f"weight shape mismatch: {tuple(wa.shape)} vs {tuple(wb.shape)}")
    if wa.dim() != 2 or wa.shape[0] != num_tokens:
        raise ValueError(
            f"weights must be [num_tokens, k] or [num_tokens, num_experts], got {tuple(wa.shape)}"
        )
    # [T, k] -> aligned top-k weights; anything wider -> full router scores.
    # (If num_experts == k a full-score tensor is indistinguishable from
    # aligned weights; the margin is vacuous then anyway.)
    full_scores = wa.shape[1] != k

    slot_match = ia == ib  # [T, k]
    position_match_ratio = float(slot_match.float().mean())
    set_match = (torch.sort(ia, dim=1).values == torch.sort(ib, dim=1).values).all(dim=1)
    overlap = (ia.unsqueeze(2) == ib.unsqueeze(1)).any(dim=2).sum(dim=1)  # [T]

    if full_scores:
        wa_sel = wa.gather(1, ia)
        wb_sel = wb.gather(1, ia)
    else:
        wa_sel, wb_sel = wa, wb
    matched_count = int(slot_match.sum())
    if matched_count > 0:
        wdiff = (wa_sel - wb_sel).abs()[slot_match]
        weight_error_mean = float(wdiff.mean())
        weight_error_max = float(wdiff.max())
    else:
        weight_error_mean = None
        weight_error_max = None

    margin_mean_a = margin_min_a = margin_mean_b = margin_min_b = None
    if full_scores:
        num_experts = wa.shape[1]
        if k < num_experts:
            sel_a = torch.zeros_like(wa, dtype=torch.bool).scatter_(1, ia, True)
            sel_b = torch.zeros_like(wb, dtype=torch.bool).scatter_(1, ib, True)
            sel_min_a = wa.masked_fill(~sel_a, float("inf")).min(dim=1).values
            unsel_max_a = wa.masked_fill(sel_a, float("-inf")).max(dim=1).values
            sel_min_b = wb.masked_fill(~sel_b, float("inf")).min(dim=1).values
            unsel_max_b = wb.masked_fill(sel_b, float("-inf")).max(dim=1).values
            margin_a = sel_min_a - unsel_max_a
            margin_b = sel_min_b - unsel_max_b
            margin_mean_a = float(margin_a.mean())
            margin_min_a = float(margin_a.min())
            margin_mean_b = float(margin_b.mean())
            margin_min_b = float(margin_b.min())
    else:
        num_experts = int(max(int(ia.max()), int(ib.max()))) + 1

    load_a = torch.bincount(ia.reshape(-1), minlength=num_experts)
    load_b = torch.bincount(ib.reshape(-1), minlength=num_experts)

    position_match = slot_match.all(dim=1)
    if bool(position_match.all()):
        first_mismatch = None
    else:
        first_mismatch = int((~position_match).nonzero(as_tuple=False)[0].item())

    return {
        "num_tokens": int(num_tokens),
        "top_k": int(k),
        "position_match_ratio": position_match_ratio,
        "set_match_ratio": float(set_match.float().mean()),
        "topk_overlap_mean": float(overlap.float().mean()),
        "topk_overlap_min": int(overlap.min()),
        "matched_position_count": matched_count,
        "weight_error_mean": weight_error_mean,
        "weight_error_max": weight_error_max,
        "boundary_margin_mean_a": margin_mean_a,
        "boundary_margin_min_a": margin_min_a,
        "boundary_margin_mean_b": margin_mean_b,
        "boundary_margin_min_b": margin_min_b,
        "num_experts": int(num_experts),
        "expert_load_a": [int(c) for c in load_a],
        "expert_load_b": [int(c) for c in load_b],
        "first_mismatch_position": first_mismatch,
    }


# ---------------------------------------------------------------------------
# logprob metrics
# ---------------------------------------------------------------------------


def logprob_metrics(lp_a, lp_b, *, logits_a=None, logits_b=None, abs_threshold=0.1, top_k=5) -> dict:
    """Comparison metrics for per-token logprobs (and optionally full logits).

    ``lp_a`` / ``lp_b`` are per-token logprobs; both must have the same
    shape (any shape, compared flattened) -- a token-count mismatch raises
    ValueError, never a truncated comparison.

    ``logits_a`` / ``logits_b`` are optional full logits of shape
    ``[num_tokens, vocab]`` and must be provided together. When given, they
    enable the greedy/top-k mismatch counts and the KL/JS divergences; those
    keys are None otherwise. ``top_k`` (default 5) is the k for the top-k
    set mismatch count.

    Returned keys:

    - ``num_tokens``, ``abs_threshold``
    - ``mean_abs_error`` / ``max_abs_error`` / ``p95_abs_error`` of |lp_a - lp_b|
    - ``tokens_over_threshold``: count of tokens with |diff| > abs_threshold
    - ``greedy_mismatch_count``: positions where argmax(logits) differs
    - ``topk_mismatch_count``, ``top_k``: positions where the top-k token
      SETS differ
    - ``kl_divergence_mean`` / ``kl_divergence_max``: KL(p_a || p_b) per
      position (a = reference side), from log-softmax in float32
    - ``js_divergence_mean`` / ``js_divergence_max``: Jensen-Shannon
      divergence via logaddexp (numerically stable, bounded by ln 2)
    """
    torch = _import_torch()
    _require_tensor(lp_a, "lp_a", torch)
    _require_tensor(lp_b, "lp_b", torch)
    if tuple(lp_a.shape) != tuple(lp_b.shape):
        raise ValueError(
            f"logprob shape mismatch {tuple(lp_a.shape)} vs {tuple(lp_b.shape)}: "
            "refusing to compare a truncated prefix"
        )
    la = lp_a.detach().float().reshape(-1)
    lb = lp_b.detach().float().reshape(-1)
    n = la.numel()
    if n == 0:
        raise ValueError("no tokens to compare")

    diff = (la - lb).abs()
    result = {
        "num_tokens": int(n),
        "abs_threshold": float(abs_threshold),
        "mean_abs_error": float(diff.double().mean()),
        "max_abs_error": float(diff.max()),
        "p95_abs_error": _percentile(torch.sort(diff).values, 95.0),
        "tokens_over_threshold": int((diff > abs_threshold).sum()),
    }

    logit_keys = (
        "greedy_mismatch_count",
        "topk_mismatch_count",
        "top_k",
        "kl_divergence_mean",
        "kl_divergence_max",
        "js_divergence_mean",
        "js_divergence_max",
    )
    if (logits_a is None) != (logits_b is None):
        raise ValueError("logits_a and logits_b must be provided together")
    if logits_a is None:
        result.update({key: None for key in logit_keys})
        return result

    ga = logits_a.detach().float()
    gb = logits_b.detach().float()
    if ga.shape != gb.shape:
        raise ValueError(f"logits shape mismatch: {tuple(ga.shape)} vs {tuple(gb.shape)}")
    if ga.dim() != 2:
        raise ValueError(f"logits must be [num_tokens, vocab], got {tuple(ga.shape)}")
    if ga.shape[0] != n:
        raise ValueError(f"logits have {ga.shape[0]} positions but logprobs have {n} tokens")

    greedy_mismatch = int((ga.argmax(dim=1) != gb.argmax(dim=1)).sum())
    kk = min(int(top_k), ga.shape[1])
    top_a = torch.sort(ga.topk(kk, dim=1).indices, dim=1).values
    top_b = torch.sort(gb.topk(kk, dim=1).indices, dim=1).values
    topk_mismatch = int((top_a != top_b).any(dim=1).sum())

    logp_a = torch.log_softmax(ga, dim=-1)
    logp_b = torch.log_softmax(gb, dim=-1)
    p_a = logp_a.exp()
    p_b = logp_b.exp()
    # KL(p_a || p_b) = sum p_a * (log p_a - log p_b); clamp float noise at 0
    kl = ((p_a * (logp_a - logp_b)).sum(dim=-1)).clamp_min(0.0)
    # JS: m = (p_a + p_b) / 2 computed in log space via logaddexp
    log_m = torch.logaddexp(logp_a, logp_b) - math.log(2.0)
    js = (
        0.5 * (p_a * (logp_a - log_m)).sum(dim=-1)
        + 0.5 * (p_b * (logp_b - log_m)).sum(dim=-1)
    ).clamp_min(0.0)

    result.update(
        {
            "greedy_mismatch_count": greedy_mismatch,
            "topk_mismatch_count": topk_mismatch,
            "top_k": kk,
            "kl_divergence_mean": float(kl.double().mean()),
            "kl_divergence_max": float(kl.max()),
            "js_divergence_mean": float(js.double().mean()),
            "js_divergence_max": float(js.max()),
        }
    )
    return result


# ---------------------------------------------------------------------------
# per-expert summaries
# ---------------------------------------------------------------------------


def _metric_keys(per_expert: dict) -> list:
    keys = []
    for metrics in per_expert.values():
        for key in metrics:
            if key not in keys:
                keys.append(key)
    return keys


def _numeric_value(metrics: dict, key: str):
    """Return the metric as float, or None if missing/non-numeric/NaN."""
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def weighted_summary(per_expert: dict, token_counts: dict) -> dict:
    """Token-count-weighted mean of each scalar metric across experts.

    ``per_expert`` maps expert_id -> metrics dict; ``token_counts`` maps
    expert_id -> number of tokens routed to that expert. Only scalar
    numeric (non-bool, non-NaN) values are aggregated; None entries and
    non-scalar values (lists, dicts, strings) are skipped. An expert with a
    numeric metric but no entry in ``token_counts`` raises KeyError rather
    than being silently dropped. Keys whose total weight is zero map to
    None. The result also carries ``total_tokens``.
    """
    result = {}
    for key in _metric_keys(per_expert):
        weighted_sum = 0.0
        weight_sum = 0.0
        for expert_id, metrics in per_expert.items():
            value = _numeric_value(metrics, key)
            if value is None:
                continue
            weight = token_counts[expert_id]  # KeyError on purpose: see docstring
            if weight < 0:
                raise ValueError(f"negative token count for expert {expert_id}: {weight}")
            weighted_sum += value * weight
            weight_sum += weight
        result[key] = (weighted_sum / weight_sum) if weight_sum > 0 else None
    result["total_tokens"] = int(sum(token_counts.get(e, 0) for e in per_expert))
    return result


def equal_summary(per_expert: dict) -> dict:
    """Unweighted (per-expert equal) mean of each scalar metric.

    Same aggregation rules as :func:`weighted_summary` but every expert
    counts once. Keys with no numeric values map to None.
    """
    result = {}
    for key in _metric_keys(per_expert):
        values = [
            v
            for v in (_numeric_value(m, key) for m in per_expert.values())
            if v is not None
        ]
        result[key] = (sum(values) / len(values)) if values else None
    return result


def worst_entries(records_with_metrics, key: str, topk: int = 10) -> list:
    """Return the ``topk`` entries with the largest value of metric ``key``.

    ``records_with_metrics`` is an iterable of dicts; the metric is read
    from ``entry[key]`` or, failing that, ``entry["metrics"][key]``, so both
    flat records and ``{"record": ..., "metrics": {...}}`` wrappers work.
    Entries whose value is missing, None, NaN, or non-numeric are skipped.
    "Worst" means largest value -- pass an error/divergence metric (for
    match ratios or cosine, pass a derived error or pre-sort ascending).

    Each output item is ``{"value": float, "entry": <original dict>}``,
    sorted by descending value.
    """
    scored = []
    for entry in records_with_metrics:
        value = None
        if isinstance(entry, dict):
            if key in entry:
                value = entry[key]
            elif isinstance(entry.get("metrics"), dict):
                value = entry["metrics"].get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        scored.append({"value": float(value), "entry": entry})
    scored.sort(key=lambda item: item["value"], reverse=True)
    return scored[: int(topk)]


# ---------------------------------------------------------------------------
# stage thresholds, evaluation, first-failure localization
# ---------------------------------------------------------------------------

# Default PASS/WARN/FAIL thresholds per pipeline stage (schema.PIPELINE_STAGES).
#
# Rule format per metric: {"when": "above"|"below", "warn": x, "fail": y}.
# "above" metrics (errors, ratios-of-bad-things) breach when value > limit;
# "below" metrics (match ratios, cosine, SQNR) breach when value < limit.
# FAIL is checked before WARN. All values are plain JSON numbers.
#
# The numbers are deliberately conservative starting points for NVFP4
# train/inference consistency on single-device runs; tune per model once
# real baseline noise is measured.
DEFAULT_THRESHOLDS = {
    # layer_input: both frameworks consume the same captured input; bf16
    # storage explains at most ~1 ulp. Any larger divergence here
    # invalidates every downstream stage comparison, hence very tight.
    "layer_input": {
        "symmetric_rel_l2": {"when": "above", "warn": 1e-7, "fail": 1e-5},
    },
    # routing: a flipped top-k selection changes which experts run, so match
    # ratios are strict. Top-k weights are softmax outputs in [0, 1]; 1e-3
    # absolute drift is bf16-logit noise, 1e-2 starts moving real probability.
    "routing": {
        "position_match_ratio": {"when": "below", "warn": 0.999, "fail": 0.99},
        "set_match_ratio": {"when": "below", "warn": 0.999, "fail": 0.99},
        "weight_error_max": {"when": "above", "warn": 1e-3, "fail": 1e-2},
    },
    # dispatch: permutation/copy only, no arithmetic; expect near-bit-exact.
    "dispatch": {
        "symmetric_rel_l2": {"when": "above", "warn": 1e-6, "fail": 1e-4},
    },
    # fc1_quant / fc2_quant: NVFP4 (e2m1) quantization views. A small
    # underflow fraction (nonzero activations rounding to zero on the coarse
    # e2m1 grid, min nonzero magnitude 0.5 x block scale) is expected when
    # scale computation differs slightly; >1% signals a real scale/rounding
    # divergence. SQNR is computed between the two frameworks' dequantized
    # views of the same input, so it should sit far above the ~25 dB fp4
    # noise floor: <50 dB warns, <35 dB means the quantized payloads
    # materially diverge.
    "fc1_quant": {
        "fp4_underflow_ratio": {"when": "above", "warn": 1e-3, "fail": 1e-2},
        "sqnr_db": {"when": "below", "warn": 50.0, "fail": 35.0},
    },
    "fc2_quant": {
        "fp4_underflow_ratio": {"when": "above", "warn": 1e-3, "fail": 1e-2},
        "sqnr_db": {"when": "below", "warn": 50.0, "fail": 35.0},
    },
    # gemm / swiglu / combined_output: bf16 tensor-core outputs. Different
    # kernels (split-k, accumulation order) produce cosine ~1 and rel-L2 well
    # under 1e-3; rel-L2 above 1e-2 is a real numerical divergence, not
    # kernel noise.
    "fc1_gemm": {
        "cosine": {"when": "below", "warn": 0.99999, "fail": 0.9999},
        "symmetric_rel_l2": {"when": "above", "warn": 1e-3, "fail": 1e-2},
    },
    "swiglu": {
        "cosine": {"when": "below", "warn": 0.99999, "fail": 0.9999},
        "symmetric_rel_l2": {"when": "above", "warn": 1e-3, "fail": 1e-2},
    },
    "fc2_gemm": {
        "cosine": {"when": "below", "warn": 0.99999, "fail": 0.9999},
        "symmetric_rel_l2": {"when": "above", "warn": 1e-3, "fail": 1e-2},
    },
    "combined_output": {
        "cosine": {"when": "below", "warn": 0.99999, "fail": 0.9999},
        "symmetric_rel_l2": {"when": "above", "warn": 1e-3, "fail": 1e-2},
    },
    # logprob: per-token logprobs in nats. 0.01 mean abs error is sampler
    # noise; 0.1 visibly shifts sampling. Any greedy-token flip is a hard
    # failure for train/inference consistency (warn == fail == 0). JS
    # divergence is bounded by ln 2 ~ 0.693; 1e-4/1e-3 are tight
    # full-distribution gates.
    "logprob": {
        "mean_abs_error": {"when": "above", "warn": 1e-2, "fail": 1e-1},
        "greedy_mismatch_count": {"when": "above", "warn": 0, "fail": 0},
        "js_divergence_mean": {"when": "above", "warn": 1e-4, "fail": 1e-3},
    },
}

_STATUS_RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}


def _worse(status_a: str, status_b: str) -> str:
    return status_a if _STATUS_RANK[status_a] >= _STATUS_RANK[status_b] else status_b


def evaluate_stages(stage_metrics: dict, thresholds: dict | None = None) -> dict:
    """Evaluate per-stage metric dicts against thresholds.

    ``stage_metrics`` maps pipeline stage name -> metrics dict (as produced
    by :func:`tensor_metrics`, :func:`routing_metrics`,
    :func:`logprob_metrics`, or an aggregation of them). ``thresholds``
    defaults to :data:`DEFAULT_THRESHOLDS`.

    Returns ``{stage: "PASS"|"WARN"|"FAIL"}`` for every stage present in
    ``stage_metrics``. A stage without threshold rules is PASS. Rules whose
    metric is missing, None, or non-numeric are skipped; a NaN metric value
    is anomalous data and bumps the stage to at least WARN.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    result = {}
    for stage, metrics in stage_metrics.items():
        status = "PASS"
        for metric_key, rule in (thresholds.get(stage) or {}).items():
            value = metrics.get(metric_key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if isinstance(value, float) and math.isnan(value):
                status = _worse(status, "WARN")
                continue
            when = rule.get("when", "above")
            warn = rule.get("warn")
            fail = rule.get("fail")
            if when == "above":
                if fail is not None and value > fail:
                    status = _worse(status, "FAIL")
                elif warn is not None and value > warn:
                    status = _worse(status, "WARN")
            elif when == "below":
                if fail is not None and value < fail:
                    status = _worse(status, "FAIL")
                elif warn is not None and value < warn:
                    status = _worse(status, "WARN")
            else:
                raise ValueError(
                    f"unknown threshold direction {when!r} for {stage}.{metric_key}"
                )
        result[stage] = status
    return result


def find_first_failure(stage_status: dict) -> tuple:
    """Locate the first anomalous pipeline stage.

    ``stage_status`` maps stage name -> "PASS"/"WARN"/"FAIL" (see
    :func:`evaluate_stages`). Stages are scanned in schema.PIPELINE_STAGES
    order. Returns ``(stage, "FAIL")`` for the earliest FAIL; if no stage
    FAILs, ``(stage, "WARN")`` for the earliest WARN; if everything is
    PASS, ``(None, "PASS")``. Stages absent from ``stage_status`` and names
    outside PIPELINE_STAGES are ignored.
    """
    first_warn = None
    for stage in PIPELINE_STAGES:
        status = stage_status.get(stage)
        if status == "FAIL":
            return stage, "FAIL"
        if status == "WARN" and first_warn is None:
            first_warn = stage
    if first_warn is not None:
        return first_warn, "WARN"
    return None, "PASS"
