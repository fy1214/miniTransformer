"""Unified command line interface for logpdiff.

Subcommands:

- ``run`` -- drive the full P0 flow in one go: collect-sglang ->
  collect-megatron -> compare -> report, all into one shared run directory.
- ``collect-sglang`` / ``collect-megatron`` -- forward every argument
  verbatim to ``dump_expert_fc1_and_fc2_sglang.py`` /
  ``dump_expert_fc1_and_fc2_megatron.py`` next to this package (the megatron
  collector is usually launched under ``torchrun --nproc_per_node=1``; the
  passthrough works either way).
- ``compare --run-dir <dir>`` -- run :func:`logpdiff.compare.run_compare`,
  writing ``metrics.json`` and ``metrics.csv`` into the run directory.
- ``report --run-dir <dir>`` -- re-render ``report.html`` from the existing
  ``metrics.json`` (running compare first if it is missing).
- ``write-launch-meta`` / ``export-routing`` -- write launch metadata and
  export captured SGLang routing for a forced-routing run.
- ``sweep`` / ``aggregate`` -- run selected layers and concatenate their
  existing ``metrics.csv`` artifacts with coverage warnings.
- ``rl-metrics`` / ``bootstrap`` -- compute learner-minus-rollout numerical
  ratio metrics and prompt-cluster confidence intervals offline.

Config merge order for ``run`` / ``compare`` / ``report``:
built-in defaults < ``--config`` JSON file < CLI flags. Unknown config keys
are rejected. ``collect-*`` subcommands have their own ``--config`` handling
inside the collector scripts (arguments are forwarded, not merged here).

Exit codes: comparison verdicts never affect the exit code by default (a
FAIL comparison still exits 0); with ``--strict`` a FAIL first-failure exits
1. Usage errors and data-integrity violations (config inconsistency, missing
files) exit 2 with the error message on stderr.

Importing this module requires only the standard library; torch is needed
only when compare actually runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent.parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

_SGLANG_COLLECTOR = "dump_expert_fc1_and_fc2_sglang.py"
_MEGATRON_COLLECTOR = "dump_expert_fc1_and_fc2_megatron.py"

_COMPARE_DEFAULTS = {
    "run_dir": None,
    "allow_value_match_fallback": False,
    "thresholds": None,
    "strict": False,
}

_REPORT_DEFAULTS = {
    "run_dir": None,
    "plots": False,
}

_RUN_DEFAULTS = {
    "run_dir": None,
    "output_dir": "runs",
    "run_id": None,
    "model_path": None,
    "tokenizer_path": None,
    "ref_load": None,
    "input_file": None,
    "layer_idx": 0,
    "ep_size": 1,
    "moe_backend": None,
    "mem_fraction": None,
    "hf_weights": None,
    "force_routing": None,
    "seed": None,
    "allow_value_match_fallback": False,
    "thresholds": None,
    "strict": False,
    "plots": False,
    "checkpoint_step": None,
    "image": None,
    "gpu": None,
    "source_repos": None,
}


# ---------------------------------------------------------------------------
# config merge helpers
# ---------------------------------------------------------------------------


def _load_config_file(path, parser, valid_keys) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        parser.error(f"cannot read --config {path}: {e}")
    except json.JSONDecodeError as e:
        parser.error(f"--config {path} is not valid JSON: {e}")
    if not isinstance(data, dict):
        parser.error(f"--config {path} must contain a JSON object")
    unknown = set(data) - set(valid_keys)
    if unknown:
        parser.error(
            f"unknown keys in --config {path}: {sorted(unknown)}; "
            f"valid keys: {sorted(valid_keys)}"
        )
    return data


def _merge_config(args, parser, defaults: dict) -> dict:
    """defaults < --config file < CLI flags (only explicitly-given flags)."""
    cfg = dict(defaults)
    if getattr(args, "config", None) is not None:
        cfg.update(_load_config_file(args.config, parser, set(defaults)))
    for key in cfg:
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    for key in ("allow_value_match_fallback", "strict", "plots"):
        if key in cfg:
            cfg[key] = bool(cfg[key])
    return cfg


def _load_thresholds(value, parser):
    """``--thresholds`` accepts an inline JSON object or a path to one."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    text = None
    if os.path.isfile(value):
        try:
            with open(value, encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            parser.error(f"cannot read --thresholds {value}: {e}")
    else:
        text = value
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        parser.error(
            f"--thresholds is neither a readable file nor valid JSON: {e}"
        )
    if not isinstance(data, dict):
        parser.error("--thresholds must be a JSON object "
                     "({stage: {metric: {when, warn, fail}}})")
    return data


# ---------------------------------------------------------------------------
# collector forwarding
# ---------------------------------------------------------------------------


def _forward_sglang(collector_args) -> int:
    import dump_expert_fc1_and_fc2_sglang as collector

    try:
        collector.main(collector_args)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0
    return 0


def _forward_megatron(collector_args) -> int:
    import dump_expert_fc1_and_fc2_megatron as collector

    # the megatron collector parses sys.argv itself (torchrun-style)
    saved_argv = sys.argv
    sys.argv = [_MEGATRON_COLLECTOR] + list(collector_args)
    try:
        collector.main()
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0
    finally:
        sys.argv = saved_argv
    return 0


# ---------------------------------------------------------------------------
# subcommand handlers
# ---------------------------------------------------------------------------


def _print_compare_summary(result: dict) -> None:
    print(f"run_id: {result.get('run_id')}")
    for row in result.get("stage_table") or []:
        metrics = row.get("metrics") or {}
        cls = metrics.get("comparison_class")
        label = f"{row.get('stage')} [{cls}]" if cls else str(row.get("stage"))
        print(f"  {label:<58} {row.get('status')}")
    first_failure = result.get("first_failure")
    if first_failure:
        print(
            f"first_failure: {first_failure.get('stage')} "
            f"({first_failure.get('status')})"
        )
    else:
        print("first_failure: none (all captured stages PASS)")
    for warning in result.get("warnings") or []:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"metrics: {result.get('run_dir')}/metrics.json")
    print(f"csv:     {result.get('run_dir')}/metrics.csv")


def _run_compare(cfg, parser):
    """Run compare for the merged config; returns (result, exit_code)."""
    from .compare import run_compare

    thresholds = _load_thresholds(cfg.get("thresholds"), parser)
    try:
        result = run_compare(
            cfg["run_dir"],
            allow_value_match_fallback=bool(
                cfg.get("allow_value_match_fallback", False)
            ),
            thresholds=thresholds,
        )
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR: compare failed: {e}", file=sys.stderr)
        return None, 2
    return result, 0


def _strict_exit_code(result, strict: bool) -> int:
    first_failure = (result or {}).get("first_failure")
    if strict and isinstance(first_failure, dict):
        if first_failure.get("status") == "FAIL":
            print(
                f"STRICT: first failure is FAIL at stage "
                f"{first_failure.get('stage')!r}; exiting 1",
                file=sys.stderr,
            )
            return 1
    return 0


def _cmd_compare(args, parser) -> int:
    cfg = _merge_config(args, parser, _COMPARE_DEFAULTS)
    if not cfg["run_dir"]:
        parser.error('--run-dir is required (or set "run_dir" in --config)')
    result, rc = _run_compare(cfg, parser)
    if result is None:
        return rc
    _print_compare_summary(result)
    return _strict_exit_code(result, cfg["strict"])


def _cmd_report(args, parser) -> int:
    from .reporter import render_report

    cfg = _merge_config(args, parser, _REPORT_DEFAULTS)
    if not cfg["run_dir"]:
        parser.error('--run-dir is required (or set "run_dir" in --config)')
    run_dir = Path(cfg["run_dir"]).resolve()
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            with open(metrics_path, encoding="utf-8") as f:
                report_data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"ERROR: cannot load {metrics_path}: {e}", file=sys.stderr)
            return 2
        if not isinstance(report_data, dict):
            print(f"ERROR: {metrics_path} is not a JSON object", file=sys.stderr)
            return 2
    else:
        print(
            f"{metrics_path} not found; running compare first...",
            file=sys.stderr,
        )
        report_data, rc = _run_compare(cfg, parser)
        if report_data is None:
            return rc
    out_path = run_dir / "report.html"
    render_report(report_data, out_path, with_plots=cfg["plots"])
    print(f"report: {out_path}")
    return 0


def _cmd_run(args, parser) -> int:
    from .monitoring import parse_name_path_specs, write_launch_meta
    from .reporter import render_report
    from .writer import create_run_dir

    cfg = _merge_config(args, parser, _RUN_DEFAULTS)
    if cfg["run_dir"] and cfg["run_id"]:
        parser.error("--run-id only applies when creating a run (with "
                     "--output-dir), not when joining one via --run-dir")
    for required in ("model_path", "ref_load", "input_file"):
        if not cfg[required]:
            parser.error(
                f"--{required.replace('_', '-')} is required for `run` "
                f'(or set "{required}" in --config)'
            )

    if cfg["run_dir"]:
        run_dir = Path(cfg["run_dir"]).resolve()
        if not run_dir.is_dir():
            parser.error(f"--run-dir {run_dir} does not exist")
    else:
        writer = create_run_dir(
            cfg["output_dir"],
            run_id=cfg["run_id"],
            model_path=cfg["model_path"],
            tokenizer_path=cfg["tokenizer_path"],
            seed=cfg["seed"],
            backend=cfg["moe_backend"],
            ep=cfg["ep_size"],
            force_routing=bool(cfg["force_routing"]),
        )
        run_dir = writer.run_dir
    print(f"run directory: {run_dir}")

    launch_meta_path = run_dir / "launch_meta.json"
    if not launch_meta_path.exists():
        source_repos = cfg.get("source_repos") or []
        if isinstance(source_repos, dict):
            sources = {str(k): str(v) for k, v in source_repos.items()}
        else:
            try:
                sources = parse_name_path_specs(
                    source_repos, label="source repository"
                )
            except ValueError as e:
                parser.error(str(e))
        sources.setdefault("minite", str(_TOOLS_DIR.parents[1]))
        write_launch_meta(
            run_dir,
            model_path=cfg["model_path"],
            tokenizer_path=cfg["tokenizer_path"] or cfg["model_path"],
            checkpoint_path=cfg["ref_load"],
            checkpoint_step=cfg.get("checkpoint_step"),
            input_file=cfg["input_file"],
            seed=cfg["seed"],
            image=cfg.get("image"),
            gpu=cfg.get("gpu"),
            source_repos=sources,
        )
        print(f"launch metadata: {launch_meta_path}")

    sglang_argv = [
        "--run-dir", str(run_dir),
        "--model-path", cfg["model_path"],
        "--input-file", cfg["input_file"],
        "--layer-idx", str(cfg["layer_idx"]),
    ]
    if cfg["tokenizer_path"]:
        sglang_argv += ["--tokenizer-path", cfg["tokenizer_path"]]
    if cfg["moe_backend"]:
        sglang_argv += ["--moe-backend", cfg["moe_backend"]]
    if cfg["mem_fraction"] is not None:
        sglang_argv += ["--mem-fraction", str(cfg["mem_fraction"])]
    if cfg["hf_weights"]:
        sglang_argv += ["--hf-weights", cfg["hf_weights"]]
    if cfg["force_routing"]:
        sglang_argv += ["--force-routing", cfg["force_routing"]]
    if cfg["seed"] is not None:
        sglang_argv += ["--seed", str(cfg["seed"])]

    megatron_argv = [
        "--run-dir", str(run_dir),
        "--model-path", cfg["model_path"],
        "--ref-load", cfg["ref_load"],
        "--input-file", cfg["input_file"],
        "--layer-idx", str(cfg["layer_idx"]),
        "--ep-size", str(cfg["ep_size"]),
    ]
    if cfg["force_routing"]:
        megatron_argv += ["--force-routing", cfg["force_routing"]]

    print("\n=== [1/4] collect-sglang ===")
    rc = _forward_sglang(sglang_argv)
    if rc != 0:
        print(f"ERROR: sglang collector exited {rc}", file=sys.stderr)
        return rc
    print("\n=== [2/4] collect-megatron ===")
    rc = _forward_megatron(megatron_argv)
    if rc != 0:
        print(f"ERROR: megatron collector exited {rc}", file=sys.stderr)
        return rc
    print("\n=== [3/4] compare ===")
    result, rc = _run_compare(cfg, parser)
    if result is None:
        return rc
    _print_compare_summary(result)
    print("\n=== [4/4] report ===")
    out_path = run_dir / "report.html"
    render_report(result, out_path, with_plots=cfg["plots"])
    print(f"report: {out_path}")
    return _strict_exit_code(result, cfg["strict"])


def _cmd_collect_sglang(args, parser) -> int:
    return _forward_sglang(_collector_args(args))


def _cmd_collect_megatron(args, parser) -> int:
    return _forward_megatron(_collector_args(args))


def _collector_args(args) -> list:
    forwarded = list(args.collector_args or [])
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    return forwarded


# ---------------------------------------------------------------------------
# offline monitoring helpers
# ---------------------------------------------------------------------------


def _print_json(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))


def _monitoring_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs), 0
    except (
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return None, 2


def _cmd_write_launch_meta(args, parser) -> int:
    from .monitoring import parse_name_path_specs, write_launch_meta

    try:
        sources = parse_name_path_specs(
            args.source_repos, label="source repository"
        )
    except ValueError as e:
        parser.error(str(e))
    result, rc = _monitoring_call(
        write_launch_meta,
        args.run_dir,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        checkpoint_path=args.checkpoint_path,
        checkpoint_step=args.checkpoint_step,
        input_file=args.input_file,
        seed=args.seed,
        recipe=args.recipe,
        image=args.image,
        gpu=args.gpu,
        source_repos=sources,
        overwrite=args.overwrite,
    )
    if result is not None:
        _print_json(result)
    return rc


def _cmd_export_routing(args, parser) -> int:
    from .monitoring import export_routing

    result, rc = _monitoring_call(
        export_routing,
        args.run_dir,
        args.output,
        layer_idx=args.layer_idx,
        overwrite=args.overwrite,
    )
    if result is not None:
        _print_json(result)
    return rc


def _cmd_sweep(args, parser) -> int:
    from .monitoring import parse_layers, run_layer_sweep

    try:
        layers = parse_layers(args.layers)
    except ValueError as e:
        parser.error(str(e))
    result, rc = _monitoring_call(
        run_layer_sweep,
        args.config,
        layers,
        args.runs_root,
        args.condition,
    )
    if result is not None:
        _print_json(result)
    return rc


def _cmd_aggregate(args, parser) -> int:
    from .monitoring import aggregate_sweeps

    result, rc = _monitoring_call(
        aggregate_sweeps, args.runs_root, args.output
    )
    if result is not None:
        for warning in result.get("warnings") or []:
            print(f"WARNING: {warning}", file=sys.stderr)
        _print_json(result)
    return rc


def _cmd_rl_metrics(args, parser) -> int:
    from .monitoring import compute_rl_amplification

    result, rc = _monitoring_call(
        compute_rl_amplification,
        args.run_dir,
        args.output,
        clip_epsilon=args.clip_epsilon,
        condition=args.condition,
        layer_idx=args.layer_idx,
    )
    if result is not None:
        _print_json(result)
    return rc


def _cmd_bootstrap(args, parser) -> int:
    from .monitoring import bootstrap_amplification, parse_name_path_specs

    try:
        inputs = parse_name_path_specs(args.inputs, label="bootstrap input")
    except ValueError as e:
        parser.error(str(e))
    result, rc = _monitoring_call(
        bootstrap_amplification,
        inputs,
        args.output,
        pairs=args.pairs,
        resamples=args.resamples,
        seed=args.seed,
    )
    if result is not None:
        _print_json(result)
    return rc


# ---------------------------------------------------------------------------
# parser / entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="logpdiff",
        description="logpdiff: train/inference consistency diagnostics "
                    "(sglang vs megatron, P0 single-device runs).",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p_run = sub.add_parser(
        "run",
        help="collect both frameworks into one run dir, then compare + report",
        description="Drive collect-sglang -> collect-megatron -> compare -> "
                    "report. The run directory is created under --output-dir "
                    "(or reused via --run-dir) and shared by both collectors.",
    )
    p_run.add_argument("--config", default=None,
                       help="JSON config; defaults < config < CLI flags")
    p_run.add_argument("--run-dir", default=None,
                       help="reuse an existing run directory (mutually "
                            "exclusive with --run-id)")
    p_run.add_argument("--output-dir", default=None,
                       help="run ROOT; the run is created at <output-dir>/<run-id>/ "
                            "(default: ./runs)")
    p_run.add_argument("--run-id", default=None, help="explicit run id")
    p_run.add_argument("--model-path", default=None, help="HF model path (required)")
    p_run.add_argument("--tokenizer-path", default=None)
    p_run.add_argument("--ref-load", default=None,
                       help="megatron torch_dist checkpoint dir (required)")
    p_run.add_argument(
        "--input-file",
        default=None,
        help="input JSONL with {\"name\", \"input_ids\"} records (required)",
    )
    p_run.add_argument("--layer-idx", type=int, default=None)
    p_run.add_argument("--ep-size", type=int, default=None,
                       help="expert parallel size; P0 supports only 1")
    p_run.add_argument("--moe-backend", default=None,
                       help="sglang moe_runner_backend (default: triton)")
    p_run.add_argument("--mem-fraction", type=float, default=None)
    p_run.add_argument("--hf-weights", default=None,
                       help="HF raw w13 weights .pt for the sglang fc1/fc2 replay")
    p_run.add_argument("--force-routing", default=None, metavar="PT",
                       help="legacy dump .pt with forced topk_ids/topk_weights")
    p_run.add_argument("--seed", type=int, default=None)
    p_run.add_argument("--allow-value-match-fallback", action="store_true",
                       default=None, help="allow cosine-fallback pairing in alignment")
    p_run.add_argument("--thresholds", default=None,
                       help="stage thresholds as inline JSON or a JSON file path")
    p_run.add_argument("--strict", action="store_true", default=None,
                       help="exit 1 when the first failure is FAIL "
                            "(default: comparison verdicts never change the exit code)")
    p_run.add_argument("--plots", action="store_true", default=None,
                       help="render report.html with plots")
    p_run.add_argument("--checkpoint-step", type=int, default=None,
                       help="recorded in launch_meta.json; no checkpoint hashing")
    p_run.add_argument("--image", default=None,
                       help="container/image label recorded in launch_meta.json")
    p_run.add_argument("--gpu", default=None,
                       help="GPU label recorded in launch_meta.json")
    p_run.add_argument("--source-repo", dest="source_repos", action="append",
                       default=None, metavar="NAME=PATH",
                       help="source checkout to record (repeatable; commit only)")
    p_run.set_defaults(func=_cmd_run)

    p_cs = sub.add_parser(
        "collect-sglang",
        help=f"forward verbatim to {_SGLANG_COLLECTOR}",
        description=f"Every argument after 'collect-sglang' is forwarded "
                    f"verbatim to {_SGLANG_COLLECTOR} (its own --config applies).",
    )
    p_cs.add_argument("collector_args", nargs=argparse.REMAINDER)
    p_cs.set_defaults(func=_cmd_collect_sglang)

    p_cm = sub.add_parser(
        "collect-megatron",
        help=f"forward verbatim to {_MEGATRON_COLLECTOR}",
        description=f"Every argument after 'collect-megatron' is forwarded "
                    f"verbatim to {_MEGATRON_COLLECTOR} (usually launched via "
                    "torchrun --nproc_per_node=1; its own --config applies).",
    )
    p_cm.add_argument("collector_args", nargs=argparse.REMAINDER)
    p_cm.set_defaults(func=_cmd_collect_megatron)

    p_cmp = sub.add_parser(
        "compare",
        help="compare one run directory (writes metrics.json / metrics.csv)",
        description="Run logpdiff.compare.run_compare on a run directory. "
                    "Exits 0 even when the comparison FAILs, unless --strict.",
    )
    p_cmp.add_argument("--config", default=None,
                       help="JSON config; defaults < config < CLI flags")
    p_cmp.add_argument("--run-dir", default=None,
                       help="run directory with sglang/ and megatron/ captures")
    p_cmp.add_argument("--allow-value-match-fallback", action="store_true",
                       default=None, help="allow cosine-fallback pairing in alignment")
    p_cmp.add_argument("--thresholds", default=None,
                       help="stage thresholds as inline JSON or a JSON file path")
    p_cmp.add_argument("--strict", action="store_true", default=None,
                       help="exit 1 when the first failure is FAIL "
                            "(default: exit code is always 0)")
    p_cmp.set_defaults(func=_cmd_compare)

    p_rep = sub.add_parser(
        "report",
        help="re-render report.html from an existing metrics.json",
        description="Render report.html from the run's metrics.json; runs "
                    "compare first when metrics.json is missing.",
    )
    p_rep.add_argument("--config", default=None,
                       help="JSON config; defaults < config < CLI flags")
    p_rep.add_argument("--run-dir", default=None, help="run directory")
    p_rep.add_argument("--plots", action="store_true", default=None,
                       help="include SVG plots")
    p_rep.set_defaults(func=_cmd_report)

    p_meta = sub.add_parser(
        "write-launch-meta",
        help="write record-only launch_meta.json (no recursive file hashes)",
    )
    p_meta.add_argument("--run-dir", required=True)
    p_meta.add_argument("--model-path", default=None)
    p_meta.add_argument("--tokenizer-path", default=None)
    p_meta.add_argument("--checkpoint-path", default=None)
    p_meta.add_argument("--checkpoint-step", type=int, default=None)
    p_meta.add_argument("--input-file", default=None)
    p_meta.add_argument("--seed", type=int, default=None)
    p_meta.add_argument("--recipe", default=None)
    p_meta.add_argument("--image", default=None)
    p_meta.add_argument("--gpu", default=None)
    p_meta.add_argument("--source-repo", dest="source_repos", action="append",
                        default=[], metavar="NAME=PATH")
    p_meta.add_argument("--overwrite", action="store_true")
    p_meta.set_defaults(func=_cmd_write_launch_meta)

    p_route = sub.add_parser(
        "export-routing",
        help="export SGLang top-k capture to a forced-routing .pt file",
    )
    p_route.add_argument("--run-dir", required=True)
    p_route.add_argument("--output", required=True)
    p_route.add_argument("--layer-idx", type=int, default=None)
    p_route.add_argument("--overwrite", action="store_true")
    p_route.set_defaults(func=_cmd_export_routing)

    p_sweep = sub.add_parser(
        "sweep",
        help="run the existing paired collector for selected layers",
    )
    p_sweep.add_argument("--config", required=True,
                         help="logpdiff run config (must not set run_dir)")
    p_sweep.add_argument("--layers", required=True,
                         help="comma/range spec, e.g. 0,4,8,14:19")
    p_sweep.add_argument("--runs-root", required=True)
    p_sweep.add_argument("--condition", required=True)
    p_sweep.set_defaults(func=_cmd_sweep)

    p_agg = sub.add_parser(
        "aggregate",
        help="concatenate sweep metrics.csv files with coverage warnings",
    )
    p_agg.add_argument("--runs-root", required=True)
    p_agg.add_argument("--output", required=True)
    p_agg.set_defaults(func=_cmd_aggregate)

    p_rl = sub.add_parser(
        "rl-metrics",
        help="compute signed delta-logp, numerical ratio and clip potential",
    )
    p_rl.add_argument("--run-dir", required=True)
    p_rl.add_argument("--output", required=True, help="per-token CSV output")
    p_rl.add_argument("--clip-epsilon", required=True, type=float)
    p_rl.add_argument("--condition", default=None)
    p_rl.add_argument("--layer-idx", type=int, default=None)
    p_rl.set_defaults(func=_cmd_rl_metrics)

    p_boot = sub.add_parser(
        "bootstrap",
        help="prompt-cluster bootstrap for one or more rl-metrics CSV files",
    )
    p_boot.add_argument("--input", dest="inputs", action="append", required=True,
                        metavar="CONDITION=CSV")
    p_boot.add_argument("--pair", dest="pairs", action="append", default=[],
                        metavar="TARGET:REFERENCE")
    p_boot.add_argument("--resamples", type=int, default=2000)
    p_boot.add_argument("--seed", type=int, default=0)
    p_boot.add_argument("--output", required=True)
    p_boot.set_defaults(func=_cmd_bootstrap)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args, parser)


if __name__ == "__main__":
    sys.exit(main())
