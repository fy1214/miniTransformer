#!/usr/bin/env python3
"""Force-disable FP8/FP4 in TE Linear / LayerNormLinear forward.

Why: when training with --fp4, TE quantizers will run on every Linear layer.
For layers that are NOT replaced by miniTransformer (those outside MoE experts),
we want them to stay bf16. The simplest patch is to assign `fp8 = False` at the
top of `_Linear.forward` / `_LayerNormLinear.forward`, after the non_tensor_args
tuple is unpacked.

This script is idempotent — re-running it is a no-op.

Target paths (resolved automatically from python -c "import transformer_engine"):
    transformer_engine/pytorch/module/linear.py            -> class _Linear
    transformer_engine/pytorch/module/layernorm_linear.py  -> class _LayerNormLinear
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

SENTINEL = "# miniTE-b200: disable FP8 fallback for unreplaced Linear layers"
MARKER_RE = re.compile(r"^(?P<indent>[ \t]+)\) = non_tensor_args\s*$", re.MULTILINE)


def locate_te_dir() -> Path:
    try:
        import transformer_engine  # type: ignore
    except ImportError as e:
        sys.exit(f"transformer_engine not importable: {e}")
    return Path(transformer_engine.__file__).resolve().parent


def patch_file(path: Path) -> bool:
    src = path.read_text()
    if SENTINEL in src:
        print(f"  [skip] {path.name}: already patched")
        return False

    m = MARKER_RE.search(src)
    if not m:
        print(f"  [WARN] {path.name}: no ') = non_tensor_args' marker found — "
              f"is this TE version expected to unpack non_tensor_args?",
              file=sys.stderr)
        return False

    indent = m.group("indent")
    insertion = f"\n{indent}{SENTINEL}\n{indent}fp8 = False\n"
    end = m.end()
    new_src = src[:end] + insertion + src[end:]
    backup = path.with_suffix(path.suffix + ".miniTE.bak")
    if not backup.exists():
        backup.write_text(src)
    path.write_text(new_src)
    print(f"  [done] {path.name}: inserted `fp8 = False` after non_tensor_args unpack "
          f"(backup -> {backup.name})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--te-dir", type=Path, default=None,
                    help="Override TE install dir (default: auto-detect)")
    args = ap.parse_args()
    te_dir = args.te_dir or locate_te_dir()
    print(f"transformer_engine dir: {te_dir}")
    targets = [
        te_dir / "pytorch" / "module" / "linear.py",
        te_dir / "pytorch" / "module" / "layernorm_linear.py",
    ]
    missing = [p for p in targets if not p.exists()]
    if missing:
        sys.exit(f"missing target(s): {missing}")
    any_changed = False
    for p in targets:
        any_changed |= patch_file(p)
    return 0 if any_changed or all((p.read_text().count(SENTINEL) == 1) for p in targets) else 1


if __name__ == "__main__":
    sys.exit(main())
