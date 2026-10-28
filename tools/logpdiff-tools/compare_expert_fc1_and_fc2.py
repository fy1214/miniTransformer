"""Backwards-compatible entry point for the expert fc1/fc2 comparison.

This is now a thin wrapper around the logpdiff run-directory compare
(:func:`logpdiff.compare.run_compare`). The legacy single-file .pt dumps
(``fc1_gemm_sglang_v2.pt`` / ``megatron_expert_fc1.pt``) are gone: the two
collectors write schema-compatible run directories (see
``logpdiff/writer.py`` for the layout), and comparison operates on a whole
run directory.

New usage:

    # compare an existing run directory (writes metrics.json + metrics.csv):
    python compare_expert_fc1_and_fc2.py --run-dir runs/<run-id>

    # same via the unified CLI (which can also render report.html):
    PYTHONPATH=<this dir> python -m logpdiff compare --run-dir runs/<run-id>
    PYTHONPATH=<this dir> python -m logpdiff report  --run-dir runs/<run-id>

    # end-to-end (create run dir, collect both sides, compare, report):
    ./run_expert_fc1_compare.sh [input_file] [dump_root]

``--help`` works without torch/sglang/megatron installed.
"""

from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

_LEGACY_ERROR = """\
ERROR: the legacy single-file comparison is no longer supported.
The collectors do not produce fc1_gemm_sglang_v2.pt / megatron_expert_fc1.pt
anymore; they write schema-compatible logpdiff run directories instead.
The old weight modes (dump-hf / dump-mg / weights) are gone as well.

New workflow:
  1. collect sglang:   python dump_expert_fc1_and_fc2_sglang.py \
                           --model-path ... --input-file ... --output-dir runs/
                       (creates runs/<run-id>/ with the sglang/ capture)
  2. collect megatron: torchrun --nproc_per_node=1 \
                           dump_expert_fc1_and_fc2_megatron.py \
                           --run-dir runs/<run-id> --model-path ... \
                           --ref-load ... --input-file ...
  3. compare:          python compare_expert_fc1_and_fc2.py --run-dir runs/<run-id>
  4. report:           python -m logpdiff report --run-dir runs/<run-id>
                       (with PYTHONPATH=%(script_dir)s)
or run everything end-to-end via run_expert_fc1_compare.sh.
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="compare_expert_fc1_and_fc2.py",
        description="Compare one logpdiff run directory (sglang vs megatron). "
                    "Thin wrapper around `python -m logpdiff compare`.",
        epilog="The legacy --sglang-dump/--megatron-dump single-file .pt "
               "comparison is no longer supported; see the error text when "
               "invoked without --run-dir.",
    )
    parser.add_argument("--run-dir", default=None,
                        help="logpdiff run directory to compare (new interface)")
    parser.add_argument("--allow-value-match-fallback", action="store_true",
                        help="forwarded to logpdiff compare (cosine-fallback pairing)")
    parser.add_argument("--thresholds", default=None,
                        help="stage thresholds as inline JSON or a JSON file path, "
                             "forwarded to logpdiff compare")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 when the first failing stage is FAIL "
                             "(default: exit code is always 0)")
    parser.add_argument("--plots", action="store_true",
                        help="also render report.html (with plots) after comparing")
    # legacy flags: still parsed, but any use without --run-dir gets a clear
    # migration error instead of silently doing nothing
    parser.add_argument("--mode", default=None,
                        help="LEGACY (unsupported): activations/dump-hf/dump-mg/weights")
    parser.add_argument("--sglang-dump", default=None,
                        help="LEGACY (unsupported): sglang activation dump .pt")
    parser.add_argument("--megatron-dump", default=None,
                        help="LEGACY (unsupported): Megatron activation dump .pt")
    parser.add_argument("--layer-idx", type=int, default=None,
                        help="LEGACY (unsupported); the layer is taken from the run")
    parser.add_argument("--output-dir", default=None, help="LEGACY (unsupported)")
    parser.add_argument("--ep-size", type=int, default=None, help="LEGACY (unsupported)")
    parser.add_argument("--model-path", default=None, help="LEGACY (unsupported)")
    parser.add_argument("--ref-load-path", default=None, help="LEGACY (unsupported)")
    parser.add_argument("--hf-dump", default=None, help="LEGACY (unsupported)")
    parser.add_argument("--mg-dump", default=None, help="LEGACY (unsupported)")
    parser.add_argument("--hf-weights", default=None, help="LEGACY (unsupported)")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"WARNING: ignoring unrecognized args: {unknown}", file=sys.stderr)

    if args.run_dir:
        from logpdiff.cli import main as cli_main

        cli_args = ["compare", "--run-dir", args.run_dir]
        if args.allow_value_match_fallback:
            cli_args.append("--allow-value-match-fallback")
        if args.thresholds:
            cli_args += ["--thresholds", args.thresholds]
        if args.strict:
            cli_args.append("--strict")
        rc = cli_main(cli_args)
        if rc == 0 and args.plots:
            rc = cli_main(["report", "--run-dir", args.run_dir, "--plots"])
        elif rc == 0:
            print(
                "hint: render the HTML report with: python -m logpdiff report "
                f"--run-dir {args.run_dir} (PYTHONPATH={_SCRIPT_DIR})"
            )
        return rc

    print(_LEGACY_ERROR % {"script_dir": _SCRIPT_DIR}, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
