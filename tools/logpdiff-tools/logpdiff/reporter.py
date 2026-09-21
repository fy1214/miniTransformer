"""HTML report renderer for logpdiff runs.

Pure standard library (``html.escape`` + hand-written markup and inline
SVG); no templating or plotting dependencies, consistent with the rest of
the ``logpdiff`` package. Importing this module does not require torch.

Public API
----------

``render_report(report_data, out_path, *, with_plots=False) -> None``

``report_data`` contract (produced by the compare stage; every key is
optional -- a missing key renders an explicit "no data" placeholder
instead of failing, because a partial report is more useful than none):

- ``run``: dict with ``run_id``, ``git_commit``, ``model``, ``versions``
  (dict), ``env`` (dict), ``seed``, ``parallel`` (dict with tp/ep/pp),
  ``force_routing`` (bool).
- ``first_failure``: ``None`` or dict ``{"stage", "status",
  "description"}`` (``reason``/``message`` also accepted) describing the
  earliest failing pipeline stage.
- ``stage_table``: list of dicts ``{"stage", "status", "metrics": {...},
  "reason": str (optional)}``; ``status`` is one of ``PASS`` / ``WARN`` /
  ``FAIL`` / ``INVALID_ALIGNMENT``. Metric keys may also be placed
  directly on the row instead of nested under ``"metrics"``.
- ``expert_table``: list of dicts, pre-sorted by error (worst first);
  typical keys: ``layer_id``, ``expert_id``, ``max_abs_err``,
  ``mean_abs_err``, ``cosine``.
- ``worst_tokens``: list of dicts; typical keys: ``sample_id``,
  ``token_position``, ``token_id``, ``ref_logprob``, ``target_logprob``,
  ``abs_diff``.
- ``provenance_legend``: dict mapping provenance to the tensor points it
  covers in this run, e.g. ``{"actual_kernel": [...],
  "replayed_reference": [...]}`` (aliases ``actual`` / ``replayed`` /
  ``replay`` / ``derived`` accepted), optionally nested under a
  ``"coverage"`` key. The four provenance meanings are always rendered
  from :mod:`logpdiff.schema` constants.
- ``alignment``: dict with ``method``, ``matched``, ``unmatched``,
  ``duplicate``, ``conflict``, ``min_fallback_cosine`` and
  ``invalid_experts`` (list of dicts ``{"layer_id", "expert_id",
  "reason"}``, plain expert ids, or a ``{expert: reason}`` dict).
- ``samples``: list of dicts rendered as collapsible ``<details>``
  blocks (``sample_id`` / ``title`` used as the summary).

Plot inputs (used only when ``with_plots=True``; all optional):

- ``heatmap``: layer x tensor-point error matrix, either
  ``{"tensor_points": [...], "rows": [{"layer_id": int, "values":
  [...]}]}``, ``{"layers": [...], "tensor_points": [...], "values":
  [[...]]}``, or a flat list of ``{"layer_id", "tensor_point",
  "value"}`` cells. Rendered as a pure-HTML table with background
  colors (not SVG).
- ``token_error_series``: ordered per-token errors: list of numbers,
  ``(position, value)`` pairs, or dicts with ``token_position`` /
  ``position`` / ``x`` and a value key (``abs_diff`` / ``value`` /
  ``y`` / ``error`` / ``abs_err``). Rendered as an SVG line chart.
- ``error_values``: flat list of error magnitudes for the SVG error
  histogram; falls back to the expert-table error column.
- ``scatter``: reference-vs-target point cloud: list of ``(ref,
  target)`` pairs or dicts (``ref``/``reference``/``x`` and
  ``target``/``y``), or ``{"points": [...], "xlabel": str, "ylabel":
  str}``. Rendered as an SVG scatter plot with a y=x guide.
- The expert-error bar chart is built from ``expert_table``.

Security: every data string is HTML-escaped before embedding. The report
never loads ``.pt`` payloads; they are internal logpdiff artifacts (see
:mod:`logpdiff.writer`) and the header says so explicitly.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

from .schema import (
    ACTUAL_KERNEL,
    DERIVED_METRIC,
    HOOK_BOUNDARY,
    REPLAYED_REFERENCE,
)

STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"
STATUS_INVALID_ALIGNMENT = "INVALID_ALIGNMENT"

_STATUS_CLASSES = {
    STATUS_PASS: "st-pass",
    STATUS_WARN: "st-warn",
    STATUS_FAIL: "st-fail",
    STATUS_INVALID_ALIGNMENT: "st-invalid-alignment",
}

_PROVENANCE_ORDER = (ACTUAL_KERNEL, REPLAYED_REFERENCE, HOOK_BOUNDARY, DERIVED_METRIC)

_PROVENANCE_MEANINGS = {
    ACTUAL_KERNEL: "Captured from the real forward pass of the framework under test.",
    REPLAYED_REFERENCE: (
        "Re-computed by replaying a reference implementation on captured "
        "inputs; NOT real kernel output -- always kept separate from "
        "actual_kernel data."
    ),
    HOOK_BOUNDARY: (
        "Captured at a module hook boundary; may include hook-side transforms."
    ),
    DERIVED_METRIC: (
        "Computed from other payloads (e.g. per-token logprob diffs); "
        "not raw captured data."
    ),
}

_PROVENANCE_ALIASES = {
    "actual": ACTUAL_KERNEL,
    ACTUAL_KERNEL: ACTUAL_KERNEL,
    "replayed": REPLAYED_REFERENCE,
    "replay": REPLAYED_REFERENCE,
    REPLAYED_REFERENCE: REPLAYED_REFERENCE,
    HOOK_BOUNDARY: HOOK_BOUNDARY,
    "hook": HOOK_BOUNDARY,
    "derived": DERIVED_METRIC,
    DERIVED_METRIC: DERIVED_METRIC,
}

# keys probed (in order) when an "error value" is needed from a row
_ERROR_KEYS = ("max_abs_err", "mean_abs_err", "abs_err", "max_err", "error", "abs_diff")

_CSS = """
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
       margin: 24px; color: #222; }
h1 { font-size: 22px; margin-bottom: 8px; }
h2 { font-size: 17px; border-bottom: 1px solid #ccc; padding-bottom: 4px;
     margin-top: 28px; }
h3 { font-size: 14px; margin-bottom: 6px; }
table { border-collapse: collapse; margin: 8px 0; font-size: 13px; }
th, td { border: 1px solid #bbb; padding: 4px 8px; text-align: left;
         vertical-align: top; }
th { background: #f0f0f0; }
code { background: #f6f6f6; padding: 0 3px; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 3px;
         font-weight: 600; font-size: 12px; }
.st-pass { background: #d9f2d9; color: #1c6b1c; }
.st-warn { background: #fff3c4; color: #8a6d00; }
.st-fail { background: #ffd9d9; color: #a01818; }
.st-invalid-alignment { background: #e6d9f7; color: #5b2d90;
                        border: 1px dashed #5b2d90; }
.st-unknown { background: #e8e8e8; color: #555; }
tr.row-fail td { background: #fff0f0; }
tr.row-warn td { background: #fffbe8; }
tr.row-invalid td { background: #f6effc; }
.banner { padding: 10px 14px; border-radius: 4px; margin: 10px 0;
          font-size: 14px; }
.banner-internal { background: #e8eef7; border: 1px solid #9ab; }
.banner-force-routing { background: #fff0a8; border: 2px solid #d4a900;
                        font-weight: 600; }
.banner-invalid { background: #efe2fb; border: 2px dashed #5b2d90; }
.meta dt { font-weight: 600; float: left; width: 170px; clear: left; }
.meta dd { margin-left: 180px; margin-bottom: 4px; }
details { margin: 6px 0; }
summary { cursor: pointer; font-weight: 600; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 8px;
      overflow-x: auto; }
.section-note { color: #555; font-size: 12px; }
.heat-cell { text-align: right; font-variant-numeric: tabular-nums; }
.cov-box { display: inline-block; vertical-align: top; margin: 0 24px 8px 0;
           padding: 8px 12px; border-radius: 4px; min-width: 220px; }
.cov-actual { background: #e3f4e3; border: 1px solid #4a904a; }
.cov-replay { background: #fdeacc; border: 1px solid #c98a1b; }
.cov-other { background: #ececec; border: 1px solid #999; }
svg text { font-family: inherit; }
"""

# SVG chart geometry
_CHART_W = 720
_CHART_H = 340
_ML, _MR, _MT, _MB = 64, 20, 40, 64  # margins: left/right/top/bottom


# ---------------------------------------------------------------------------
# formatting / escaping helpers
# ---------------------------------------------------------------------------


def _esc(value) -> str:
    """HTML-escape any value (tags, attributes and quotes included)."""
    return html.escape(str(value), quote=True)


def _fmt(value) -> str:
    """Format a scalar for display; the result must still be escaped."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return format(value, ".6g")
    return str(value)


def _as_float(value):
    """Return ``float(value)`` for finite non-bool numbers, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _cell(value) -> str:
    """Render one table cell (already escaped / safe markup)."""
    if isinstance(value, (dict, list, tuple)):
        return f"<code>{_esc(json.dumps(value, ensure_ascii=False, default=str))}</code>"
    return _esc(_fmt(value))


def _status_badge(status) -> str:
    text = "UNKNOWN" if status is None else str(status).upper()
    cls = _STATUS_CLASSES.get(text, "st-unknown")
    return f'<span class="badge {cls}">{_esc(text)}</span>'


def _row_class(status) -> str:
    s = "" if status is None else str(status).upper()
    if s == STATUS_FAIL:
        return ' class="row-fail"'
    if s == STATUS_WARN:
        return ' class="row-warn"'
    if s == STATUS_INVALID_ALIGNMENT:
        return ' class="row-invalid"'
    return ""


# ---------------------------------------------------------------------------
# generic dict-row table
# ---------------------------------------------------------------------------


def _render_dict_table(rows, *, preferred=()) -> str:
    """Render a list of dicts as a table.

    Columns are ``preferred`` keys that occur at least once, followed by
    every other key in first-seen order. A ``status`` column is rendered
    as a colored badge.
    """
    cols = [k for k in preferred if any(k in r for r in rows)]
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    out = ["<table><thead><tr>"]
    for c in cols:
        out.append(f"<th>{_esc(c)}</th>")
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append(f"<tr{_row_class(r.get('status'))}>")
        for c in cols:
            v = r.get(c)
            out.append(f"<td>{_status_badge(v) if c == 'status' else _cell(v)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


# ---------------------------------------------------------------------------
# report sections
# ---------------------------------------------------------------------------


def _render_run_header(run, alignment) -> str:
    run = run if isinstance(run, dict) else {}
    parts = ['<header id="run-header">', "<h1>logpdiff run report</h1>"]
    parts.append(
        '<div class="banner banner-internal" id="internal-data-banner">'
        "All <code>.pt</code> payloads behind this report are internal "
        "logpdiff artifacts: produced and consumed only by this tooling "
        "(see <code>writer.assert_internal_payload</code>). Never point "
        "this toolchain at external or untrusted dump files."
        "</div>"
    )
    if bool(run.get("force_routing")):
        parts.append(
            '<div class="banner banner-force-routing" id="force-routing-banner">'
            "FORCE-ROUTING was ACTIVE for this run: expert routing was "
            "overridden by the harness. Router-related metrics "
            "(router_logits, router_topk_ids/weights, dispatch) describe "
            "the forced routing, not natural model behavior."
            "</div>"
        )
    else:
        parts.append(
            '<div class="section-note" id="force-routing-note">'
            "force_routing: not active for this run (natural routing).</div>"
        )
    alignment_method = None
    if isinstance(alignment, dict):
        alignment_method = alignment.get("method")
    parts.append(
        '<div class="banner banner-internal" id="alignment-method-banner">'
        "Expert alignment method: "
        f"<code>{_esc(_fmt(alignment_method))}</code></div>"
    )
    parallel = run.get("parallel")
    if isinstance(parallel, dict) and parallel:
        parallel_str = ", ".join(f"{k}={_fmt(parallel[k])}" for k in sorted(parallel))
    else:
        parallel_str = _fmt(parallel)
    meta = [
        ("run_id", run.get("run_id")),
        ("git_commit", run.get("git_commit")),
        ("model", run.get("model", run.get("model_path"))),
        ("seed", run.get("seed")),
        ("parallel", parallel_str),
        ("alignment_method", alignment_method),
    ]
    parts.append('<dl class="meta">')
    for key, value in meta:
        parts.append(f"<dt>{_esc(key)}</dt><dd>{_esc(_fmt(value))}</dd>")
    parts.append("</dl>")
    parts.append(
        '<p class="section-note">P0 scope: single-device capture '
        "(TP=EP=PP=1); multi-parallel runs are rejected by the writer.</p>"
    )
    versions = run.get("versions")
    if isinstance(versions, dict) and versions:
        parts.append("<h3>versions</h3><table><tbody>")
        for k in sorted(versions):
            parts.append(
                f"<tr><th>{_esc(k)}</th><td>{_esc(_fmt(versions[k]))}</td></tr>"
            )
        parts.append("</tbody></table>")
    env = run.get("env")
    if isinstance(env, dict) and env:
        rows = "".join(
            f"<tr><th>{_esc(k)}</th><td><code>{_esc(_fmt(env[k]))}</code></td></tr>"
            for k in sorted(env)
        )
        parts.append(
            f'<details id="env-details"><summary>environment '
            f"({len(env)} vars)</summary><table><tbody>{rows}</tbody></table>"
            "</details>"
        )
    parts.append("</header>")
    return "\n".join(parts)


def _render_first_failure(first_failure) -> str:
    parts = ['<section id="first-failure">', "<h2>First failure</h2>"]
    if not first_failure:
        parts.append(
            '<p><span class="badge st-pass">NONE</span> '
            "no failing stage recorded.</p>"
        )
    else:
        if isinstance(first_failure, dict):
            stage = first_failure.get("stage")
            status = first_failure.get("status", STATUS_FAIL)
            desc = (
                first_failure.get("description")
                or first_failure.get("reason")
                or first_failure.get("message")
                or ""
            )
        else:
            stage, status, desc = first_failure, STATUS_FAIL, ""
        parts.append(
            f"<p>stage <code>{_esc(_fmt(stage))}</code> "
            f"&mdash; {_status_badge(status)}</p>"
        )
        if desc:
            parts.append(f"<p>{_esc(desc)}</p>")
    parts.append("</section>")
    return "\n".join(parts)


def _render_stage_table(stage_table) -> str:
    parts = ['<section id="stage-table">', "<h2>Pipeline stages</h2>"]
    rows = [r for r in (stage_table or []) if isinstance(r, dict)]
    if not rows:
        parts.append('<p class="section-note">no stage data.</p></section>')
        return "\n".join(parts)
    metric_keys: list = []
    for r in rows:
        metrics = r.get("metrics")
        keys = (
            list(metrics)
            if isinstance(metrics, dict)
            else [k for k in r if k not in ("stage", "status", "reason")]
        )
        for k in keys:
            if k not in metric_keys:
                metric_keys.append(k)
    want_reason = any(r.get("reason") for r in rows)
    header = ["stage", "status"] + metric_keys + (["reason"] if want_reason else [])
    parts.append("<table><thead><tr>")
    for h in header:
        parts.append(f"<th>{_esc(h)}</th>")
    parts.append("</tr></thead><tbody>")
    has_invalid = False
    for r in rows:
        status = r.get("status")
        if status is not None and str(status).upper() == STATUS_INVALID_ALIGNMENT:
            has_invalid = True
        metrics = (
            r.get("metrics")
            if isinstance(r.get("metrics"), dict)
            else {
                k: v for k, v in r.items() if k not in ("stage", "status", "reason")
            }
        )
        parts.append(f"<tr{_row_class(status)}>")
        parts.append(f"<td><code>{_esc(_fmt(r.get('stage')))}</code></td>")
        parts.append(f"<td>{_status_badge(status)}</td>")
        for k in metric_keys:
            parts.append(f"<td>{_cell(metrics.get(k))}</td>")
        if want_reason:
            parts.append(f"<td>{_cell(r.get('reason'))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    if has_invalid:
        parts.append(
            '<p class="section-note"><span class="badge '
            'st-invalid-alignment">INVALID_ALIGNMENT</span> expert-pair '
            "alignment was unreliable for the marked stage(s) (see the "
            "alignment section for per-expert reasons); their metrics "
            "must not be trusted as framework-vs-framework diffs.</p>"
        )
    parts.append("</section>")
    return "\n".join(parts)


def _render_expert_table(expert_table) -> str:
    parts = [
        '<section id="expert-table">',
        "<h2>Experts (sorted by error, worst first)</h2>",
    ]
    rows = [r for r in (expert_table or []) if isinstance(r, dict)]
    if not rows:
        parts.append('<p class="section-note">no expert data.</p></section>')
        return "\n".join(parts)
    parts.append(_render_dict_table(rows, preferred=("layer_id", "expert_id")))
    parts.append("</section>")
    return "\n".join(parts)


def _render_worst_tokens(worst_tokens) -> str:
    parts = ['<section id="worst-tokens">', "<h2>Worst tokens</h2>"]
    rows = [r for r in (worst_tokens or []) if isinstance(r, dict)]
    if not rows:
        parts.append('<p class="section-note">no token data.</p></section>')
        return "\n".join(parts)
    parts.append(
        _render_dict_table(
            rows, preferred=("sample_id", "token_position", "token_id")
        )
    )
    parts.append("</section>")
    return "\n".join(parts)


def _normalize_coverage(legend) -> dict:
    cov = {prov: [] for prov in _PROVENANCE_ORDER}
    if not isinstance(legend, dict):
        return cov
    src = legend.get("coverage")
    if not isinstance(src, dict):
        src = legend
    for key, value in src.items():
        prov = _PROVENANCE_ALIASES.get(str(key))
        if prov is not None and isinstance(value, (list, tuple)):
            cov[prov] = [str(v) for v in value]
    return cov


def _render_provenance_legend(legend) -> str:
    parts = ['<section id="provenance-legend">', "<h2>Provenance legend</h2>"]
    parts.append(
        "<table><thead><tr><th>provenance</th><th>meaning</th></tr></thead><tbody>"
    )
    for prov in _PROVENANCE_ORDER:
        parts.append(
            f"<tr><td><code>{_esc(prov)}</code></td>"
            f"<td>{_esc(_PROVENANCE_MEANINGS[prov])}</td></tr>"
        )
    parts.append("</tbody></table>")
    cov = _normalize_coverage(legend)
    parts.append("<h3>Coverage in this run</h3>")
    parts.append(
        '<p class="section-note">actual_kernel and replayed_reference '
        "coverage are shown separately on purpose: replayed values are "
        "reference recomputations, not real kernel outputs, and must "
        "never be mixed into actual captures.</p>"
    )
    boxes = [
        (ACTUAL_KERNEL, "cov-actual", "actual_kernel coverage"),
        (REPLAYED_REFERENCE, "cov-replay", "replayed_reference coverage"),
        (HOOK_BOUNDARY, "cov-other", "hook_boundary coverage"),
        (DERIVED_METRIC, "cov-other", "derived_metric coverage"),
    ]
    parts.append("<div>")
    for prov, cls, title in boxes:
        points = cov.get(prov) or []
        body = (
            ", ".join(f"<code>{_esc(p)}</code>" for p in points)
            if points
            else '<span class="section-note">none</span>'
        )
        parts.append(
            f'<div class="cov-box {cls}"><strong>{_esc(title)}</strong>'
            f"<br>{body}</div>"
        )
    parts.append("</div></section>")
    return "\n".join(parts)


def _render_alignment(alignment) -> str:
    parts = ['<section id="alignment">', "<h2>Expert alignment</h2>"]
    if not isinstance(alignment, dict) or not alignment:
        parts.append('<p class="section-note">no alignment data.</p></section>')
        return "\n".join(parts)
    fields = (
        "method",
        "matched",
        "unmatched",
        "duplicate",
        "conflict",
        "min_fallback_cosine",
    )
    parts.append("<table><thead><tr><th>field</th><th>value</th></tr></thead><tbody>")
    for f in fields:
        if f in alignment:
            parts.append(
                f"<tr><td>{_esc(f)}</td><td>{_cell(alignment.get(f))}</td></tr>"
            )
    parts.append("</tbody></table>")
    invalid = alignment.get("invalid_experts") or []
    if isinstance(invalid, dict):
        invalid = [{"expert": k, "reason": v} for k, v in invalid.items()]
    if not isinstance(invalid, list):
        invalid = [invalid]
    conflict = alignment.get("conflict")
    n_conflict = conflict if isinstance(conflict, int) else 0
    if invalid or n_conflict > 0:
        parts.append(
            '<div class="banner banner-invalid" id="invalid-alignment-banner">'
            '<span class="badge st-invalid-alignment">INVALID_ALIGNMENT</span> '
            f"{_esc(_fmt(conflict))} conflict(s), {len(invalid)} invalid "
            "expert(s). Expert-pair matching is unreliable for these "
            "entries; their comparisons must be treated as untrusted. "
            "Per-expert reasons are listed below.</div>"
        )
    if invalid:
        rows = [
            item if isinstance(item, dict) else {"expert": item, "reason": "no reason recorded"}
            for item in invalid
        ]
        parts.append(
            _render_dict_table(
                rows, preferred=("layer_id", "expert_id", "expert", "reason")
            )
        )
    parts.append("</section>")
    return "\n".join(parts)


def _render_samples(samples) -> str:
    parts = ['<section id="samples">', "<h2>Samples</h2>"]
    items = list(samples or [])
    if not items:
        parts.append('<p class="section-note">no samples recorded.</p></section>')
        return "\n".join(parts)
    for i, item in enumerate(items):
        if isinstance(item, dict):
            title = item.get("sample_id", item.get("title", f"sample {i}"))
            body = ["<table><tbody>"]
            for k in item:
                if k in ("sample_id", "title"):
                    continue
                v = item[k]
                if isinstance(v, (dict, list, tuple)):
                    rendered = (
                        f"<pre>{_esc(json.dumps(v, ensure_ascii=False, indent=2, default=str))}</pre>"
                    )
                else:
                    rendered = _esc(_fmt(v))
                body.append(f"<tr><th>{_esc(k)}</th><td>{rendered}</td></tr>")
            body.append("</tbody></table>")
            body_html = "".join(body)
        else:
            title = f"sample {i}"
            body_html = f"<pre>{_esc(_fmt(item))}</pre>"
        parts.append(
            f'<details class="sample" id="sample-{i}">'
            f"<summary>{_esc(_fmt(title))}</summary>{body_html}</details>"
        )
    parts.append("</section>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# SVG chart helpers (standard library only)
# ---------------------------------------------------------------------------


def _svg_root(width, height, inner) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img">'
        f"{inner}</svg>"
    )


def _svg_text(x, y, text, *, size=11, anchor="middle", rotate=None) -> str:
    transform = (
        f' transform="rotate({rotate} {float(x):.1f} {float(y):.1f})"'
        if rotate is not None
        else ""
    )
    return (
        f'<text x="{float(x):.1f}" y="{float(y):.1f}" font-size="{size}" '
        f'text-anchor="{anchor}"{transform}>{_esc(text)}</text>'
    )


def _ticks(vmin, vmax, count=5):
    """Linear tick positions from vmin to vmax (inclusive)."""
    if not (math.isfinite(vmin) and math.isfinite(vmax)) or vmax <= vmin:
        return [vmin if math.isfinite(vmin) else 0.0]
    return [vmin + (vmax - vmin) * i / (count - 1) for i in range(count)]


def _y_axis_parts(y_of, ymin, ymax, plot_w, plot_h):
    """Y axis line, baseline, gridlines, tick marks and tick labels."""
    parts = []
    y_base = y_of(0.0) if ymin < 0.0 < ymax else y_of(ymin)
    parts.append(
        f'<line x1="{_ML}" y1="{_MT}" x2="{_ML}" y2="{_MT + plot_h}" stroke="#333"/>'
    )
    parts.append(
        f'<line x1="{_ML}" y1="{y_base:.1f}" x2="{_ML + plot_w}" '
        f'y2="{y_base:.1f}" stroke="#333"/>'
    )
    for t in _ticks(ymin, ymax, 5):
        y = y_of(t)
        parts.append(
            f'<line x1="{_ML}" y1="{y:.1f}" x2="{_ML + plot_w}" y2="{y:.1f}" '
            'stroke="#e5e5e5"/>'
        )
        parts.append(
            f'<line x1="{_ML - 4}" y1="{y:.1f}" x2="{_ML}" y2="{y:.1f}" stroke="#333"/>'
        )
        parts.append(_svg_text(_ML - 8, y + 4, _fmt(t), anchor="end", size=10))
    return parts


def _x_axis_num_parts(x_of, xmin, xmax, plot_h):
    """X axis tick marks and labels for a numeric axis."""
    parts = []
    for t in _ticks(xmin, xmax, 6):
        x = x_of(t)
        parts.append(
            f'<line x1="{x:.1f}" y1="{_MT + plot_h}" x2="{x:.1f}" '
            f'y2="{_MT + plot_h + 4}" stroke="#333"/>'
        )
        parts.append(_svg_text(x, _MT + plot_h + 16, _fmt(t), size=10))
    return parts


def _axis_titles(width, plot_w, plot_h, xlabel, ylabel):
    return [
        _svg_text(_ML + plot_w / 2, _MT + plot_h + _MB - 10, xlabel, size=12),
        _svg_text(16, _MT + plot_h / 2, ylabel, size=12, rotate=-90),
    ]


def _svg_bar_chart(items, *, title, xlabel, ylabel, width=_CHART_W, height=_CHART_H):
    """Vertical bar chart; ``items`` is a list of (label, value) pairs."""
    plot_w = width - _ML - _MR
    plot_h = height - _MT - _MB
    vmax = max(v for _, v in items)
    vmin = min(0.0, min(v for _, v in items))
    if vmax <= vmin:
        vmax = vmin + 1.0

    def y_of(v):
        return _MT + plot_h - (v - vmin) / (vmax - vmin) * plot_h

    y_base = y_of(0.0)
    n = len(items)
    slot = plot_w / n
    bar_w = max(1.0, slot * 0.7)
    rotate = -35 if n > 6 else None
    parts = [_svg_text(width / 2, 20, title, size=14)]
    parts.extend(_y_axis_parts(y_of, vmin, vmax, plot_w, plot_h))
    for i, (label, value) in enumerate(items):
        x = _ML + slot * i + (slot - bar_w) / 2
        y_top = y_of(max(value, 0.0))
        h = max(abs(y_of(value) - y_base), 0.5)
        parts.append(
            f'<rect x="{x:.1f}" y="{y_top:.1f}" width="{bar_w:.1f}" '
            f'height="{h:.1f}" fill="#4a7ebb"/>'
        )
        lx = _ML + slot * (i + 0.5)
        if rotate is not None:
            parts.append(
                _svg_text(lx, _MT + plot_h + 8, label, size=10, anchor="end", rotate=rotate)
            )
        else:
            parts.append(_svg_text(lx, _MT + plot_h + 16, label, size=10))
    parts.extend(_axis_titles(width, plot_w, plot_h, xlabel, ylabel))
    return _svg_root(width, height, "".join(parts))


def _svg_line_chart(points, *, title, xlabel, ylabel, width=_CHART_W, height=_CHART_H):
    """Line chart; ``points`` is a list of (x, y) numeric pairs."""
    plot_w = width - _ML - _MR
    plot_h = height - _MT - _MB
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    if xmax <= xmin:
        xmax = xmin + 1.0
    ymin = min(0.0, min(ys))
    ymax = max(ys)
    if ymax <= ymin:
        ymax = ymin + 1.0

    def x_of(x):
        return _ML + (x - xmin) / (xmax - xmin) * plot_w

    def y_of(y):
        return _MT + plot_h - (y - ymin) / (ymax - ymin) * plot_h

    parts = [_svg_text(width / 2, 20, title, size=14)]
    parts.extend(_y_axis_parts(y_of, ymin, ymax, plot_w, plot_h))
    parts.extend(_x_axis_num_parts(x_of, xmin, xmax, plot_h))
    pts = " ".join(f"{x_of(x):.1f},{y_of(y):.1f}" for x, y in points)
    parts.append(
        f'<polyline points="{pts}" fill="none" stroke="#4a7ebb" stroke-width="1.5"/>'
    )
    for x, y in points:
        parts.append(
            f'<circle cx="{x_of(x):.1f}" cy="{y_of(y):.1f}" r="2.5" fill="#4a7ebb"/>'
        )
    parts.extend(_axis_titles(width, plot_w, plot_h, xlabel, ylabel))
    return _svg_root(width, height, "".join(parts))


def _histogram_items(values, bins=None):
    """Bin ``values`` into (range-label, count) pairs for a bar chart."""
    vmin, vmax = min(values), max(values)
    if vmax <= vmin:
        vmax = vmin + 1.0
    if bins is None:
        bins = max(5, min(20, int(math.sqrt(len(values))) or 1))
    step = (vmax - vmin) / bins
    counts = [0] * bins
    for v in values:
        idx = int((v - vmin) / step)
        idx = min(max(idx, 0), bins - 1)
        counts[idx] += 1
    labels = [
        f"{_fmt(vmin + i * step)}..{_fmt(vmin + (i + 1) * step)}" for i in range(bins)
    ]
    return list(zip(labels, counts))


def _svg_scatter(points, *, title, xlabel, ylabel, width=_CHART_W, height=_CHART_H):
    """Reference-vs-target scatter with a dashed y=x guide.

    Both axes share one scale so the guide line is meaningful.
    """
    plot_w = width - _ML - _MR
    plot_h = height - _MT - _MB
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    lo = min(min(xs), min(ys))
    hi = max(max(xs), max(ys))
    if hi <= lo:
        hi = lo + 1.0

    def x_of(x):
        return _ML + (x - lo) / (hi - lo) * plot_w

    def y_of(y):
        return _MT + plot_h - (y - lo) / (hi - lo) * plot_h

    parts = [_svg_text(width / 2, 20, title, size=14)]
    parts.extend(_y_axis_parts(y_of, lo, hi, plot_w, plot_h))
    parts.extend(_x_axis_num_parts(x_of, lo, hi, plot_h))
    parts.append(
        f'<line x1="{x_of(lo):.1f}" y1="{y_of(lo):.1f}" x2="{x_of(hi):.1f}" '
        f'y2="{y_of(hi):.1f}" stroke="#999" stroke-dasharray="5,4"/>'
    )
    for x, y in points:
        parts.append(
            f'<circle cx="{x_of(x):.1f}" cy="{y_of(y):.1f}" r="3" '
            'fill="#c0504d" fill-opacity="0.65"/>'
        )
    parts.extend(_axis_titles(width, plot_w, plot_h, xlabel, ylabel))
    return _svg_root(width, height, "".join(parts))


# ---------------------------------------------------------------------------
# plot input normalization
# ---------------------------------------------------------------------------


def _expert_errors(expert_table):
    """(label, error) pairs from expert_table rows, in the given order."""
    items = []
    for r in expert_table or []:
        if not isinstance(r, dict):
            continue
        value = None
        for k in _ERROR_KEYS:
            value = _as_float(r.get(k))
            if value is not None:
                break
        if value is None:
            continue
        label = ""
        if "layer_id" in r:
            label += f"L{_fmt(r['layer_id'])}"
        label += f"e{_fmt(r.get('expert_id', '?'))}"
        items.append((label, value))
    return items


def _xy_points(raw):
    """Normalize a token error series to a list of (x, y) float pairs."""
    out = []
    for i, item in enumerate(raw or []):
        if isinstance(item, dict):
            x = item.get("token_position", item.get("position", item.get("x", i)))
            y = None
            for k in ("abs_diff", "value", "y", "error", "abs_err"):
                if k in item:
                    y = item[k]
                    break
            if y is None:
                continue
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            x, y = item
        else:
            x, y = i, item
        xf, yf = _as_float(x), _as_float(y)
        if xf is None:
            xf = float(i)
        if yf is not None:
            out.append((xf, yf))
    return out


def _scatter_points(raw):
    """Normalize scatter input to (points, xlabel, ylabel)."""
    xlabel, ylabel = "reference", "target"
    if isinstance(raw, dict):
        points = raw.get("points") or []
        xlabel = str(raw.get("xlabel", xlabel))
        ylabel = str(raw.get("ylabel", ylabel))
    else:
        points = raw or []
    out = []
    for item in points:
        if isinstance(item, dict):
            pair = (
                item.get("ref", item.get("reference", item.get("x"))),
                item.get("target", item.get("y")),
            )
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            pair = item
        else:
            continue
        xf, yf = _as_float(pair[0]), _as_float(pair[1])
        if xf is not None and yf is not None:
            out.append((xf, yf))
    return out, xlabel, ylabel


def _normalize_heatmap(hm):
    """Normalize heatmap input to (tensor_points, [(layer_label, [values])])."""
    if isinstance(hm, dict) and "tensor_points" in hm:
        tps = [str(t) for t in hm.get("tensor_points") or []]
        rows = []

        def _row_from_values(label, values):
            if isinstance(values, dict):
                rows.append((label, [_as_float(values.get(tp)) for tp in tps]))
            elif isinstance(values, (list, tuple)):
                vals = [_as_float(v) for v in values][: len(tps)]
                vals += [None] * (len(tps) - len(vals))
                rows.append((label, vals))

        if isinstance(hm.get("rows"), list):
            for row in hm["rows"]:
                if isinstance(row, dict):
                    label = _fmt(row.get("layer_id", row.get("layer", len(rows))))
                    _row_from_values(label, row.get("values"))
        elif isinstance(hm.get("values"), list) and isinstance(hm.get("layers"), list):
            matrix = hm["values"]
            for li, layer in enumerate(hm["layers"]):
                values = matrix[li] if li < len(matrix) else None
                _row_from_values(_fmt(layer), values)
        return (tps, rows) if tps and rows else None
    if isinstance(hm, list):
        cells = [c for c in hm if isinstance(c, dict)]
        tps: list = []
        layers: list = []
        for c in cells:
            tp = str(c.get("tensor_point"))
            layer = c.get("layer_id", c.get("layer"))
            if tp not in tps:
                tps.append(tp)
            if layer not in layers:
                layers.append(layer)
        if not tps or not layers:
            return None
        grid = {layer: {tp: None for tp in tps} for layer in layers}
        for c in cells:
            layer = c.get("layer_id", c.get("layer"))
            grid[layer][str(c.get("tensor_point"))] = _as_float(c.get("value"))
        rows = [(_fmt(layer), [grid[layer][tp] for tp in tps]) for layer in layers]
        return tps, rows
    return None


def _render_heatmap(hm) -> str:
    norm = _normalize_heatmap(hm)
    if norm is None:
        return '<p class="section-note">no heatmap data.</p>'
    tps, rows = norm
    flat = [v for _, vals in rows for v in vals if v is not None]
    vmax = max(flat) if flat else 0.0
    if vmax <= 0:
        vmax = 1.0
    parts = ['<table class="heatmap" id="heatmap-table"><thead><tr><th>layer / tensor_point</th>']
    for tp in tps:
        parts.append(f"<th>{_esc(tp)}</th>")
    parts.append("</tr></thead><tbody>")
    for label, vals in rows:
        parts.append(f"<tr><th>{_esc(label)}</th>")
        for v in vals:
            if v is None:
                parts.append(
                    '<td class="heat-cell" style="background:#eee">n/a</td>'
                )
            else:
                t = min(1.0, max(0.0, v / vmax))
                g = int(round(255 - 200 * t))
                parts.append(
                    f'<td class="heat-cell" style="background:rgb(255,{g},{g})">'
                    f"{_esc(_fmt(v))}</td>"
                )
        parts.append("</tr>")
    parts.append("</tbody></table>")
    parts.append(
        f'<p class="section-note">color scale: 0 (white) .. {_esc(_fmt(vmax))} '
        "(red); n/a = not captured.</p>"
    )
    return "".join(parts)


def _render_plots(report_data) -> str:
    parts = ['<section id="plots">', "<h2>Plots</h2>"]

    parts.append("<h3>layer x tensor-point error heatmap</h3>")
    parts.append(_render_heatmap(report_data.get("heatmap")))

    expert_items = _expert_errors(report_data.get("expert_table"))
    parts.append("<h3>expert error</h3>")
    if expert_items:
        parts.append(
            _svg_bar_chart(
                expert_items,
                title="expert error (worst first)",
                xlabel="expert",
                ylabel="error",
            )
        )
    else:
        parts.append('<p class="section-note">no expert error data.</p>')

    token_points = _xy_points(report_data.get("token_error_series"))
    parts.append("<h3>token error trend</h3>")
    if token_points:
        parts.append(
            _svg_line_chart(
                token_points,
                title="per-token error",
                xlabel="token position",
                ylabel="error",
            )
        )
    else:
        parts.append('<p class="section-note">no token error series.</p>')

    values = report_data.get("error_values")
    if not isinstance(values, list) or not values:
        values = [v for _, v in expert_items]
    values = [v for v in (_as_float(x) for x in values) if v is not None]
    parts.append("<h3>error histogram</h3>")
    if values:
        parts.append(
            _svg_bar_chart(
                _histogram_items(values),
                title="error histogram",
                xlabel="error bin",
                ylabel="count",
            )
        )
    else:
        parts.append('<p class="section-note">no error values.</p>')

    scatter_points, sx, sy = _scatter_points(report_data.get("scatter"))
    parts.append("<h3>reference vs target</h3>")
    if scatter_points:
        parts.append(
            _svg_scatter(
                scatter_points,
                title="reference vs target",
                xlabel=sx,
                ylabel=sy,
            )
        )
    else:
        parts.append('<p class="section-note">no scatter data.</p>')

    parts.append("</section>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def render_report(report_data: dict, out_path, *, with_plots: bool = False) -> None:
    """Render ``report_data`` as a self-contained HTML report at ``out_path``.

    With ``with_plots=False`` (default) only tables and text are emitted --
    the output contains no ``<svg>`` at all. With ``with_plots=True`` an
    additional plots section is rendered: a layer x tensor-point CSS
    heatmap (pure HTML table) plus inline-SVG expert-error bars, token
    error trend, error histogram and reference-target scatter, all with
    labeled axes and ticks. ``out_path`` may be a str or Path; missing
    parent directories are created.
    """
    if not isinstance(report_data, dict):
        raise TypeError(
            f"report_data must be a dict, got {type(report_data).__name__}"
        )
    run = report_data.get("run")
    run = run if isinstance(run, dict) else {}
    alignment = report_data.get("alignment")
    run_id = run.get("run_id", "")
    body = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>logpdiff report {_esc(_fmt(run_id))}</title>",
        f"<style>{_CSS}</style>",
        "</head><body>",
        _render_run_header(run, alignment),
        _render_first_failure(report_data.get("first_failure")),
        _render_stage_table(report_data.get("stage_table")),
        _render_expert_table(report_data.get("expert_table")),
        _render_worst_tokens(report_data.get("worst_tokens")),
        _render_provenance_legend(report_data.get("provenance_legend")),
        _render_alignment(alignment),
        _render_samples(report_data.get("samples")),
    ]
    if with_plots:
        body.append(_render_plots(report_data))
    body.append("</body></html>")
    out = Path(out_path)
    if str(out.parent) not in ("", ".") and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(body) + "\n", encoding="utf-8")
