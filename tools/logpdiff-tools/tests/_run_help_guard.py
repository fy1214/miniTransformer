"""Subprocess helper: run a legacy script with --help while guarding that no
heavy third-party dependency gets imported.

Usage: python _run_help_guard.py <script-path>

Exit codes:
  0  -- --help ran cleanly (SystemExit(0) from argparse)
  3  -- the script raised SystemExit with a non-zero code
  42 -- a blocked (heavy) module import was attempted
"""

from __future__ import annotations

import builtins
import runpy
import sys

# Heavy dependencies that must not be needed just to print --help.
# (torch is an allowed dependency of the logpdiff toolchain, so it is
# deliberately not in this list.)
BLOCKED = frozenset(
    {
        "sglang",
        "sgl_kernel",
        "megatron",
        "megatron_core",
        "slime",
        "minite",
        "safetensors",
        "transformer_engine",
        "flashinfer",
        "flash_attn",
        "vllm",
    }
)


def main() -> int:
    script = sys.argv[1]
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in BLOCKED:
            print(f"heavy dependency import during --help: {name}",
                  file=sys.stderr)
            raise SystemExit(42)
        return real_import(name, *args, **kwargs)

    builtins.__import__ = guarded_import
    sys.argv = [script, "--help"]
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as exc:
        if exc.code in (None, 0):
            return 0
        if isinstance(exc.code, int):
            return exc.code
        print(exc.code, file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
