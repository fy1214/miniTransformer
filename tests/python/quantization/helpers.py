"""Shared runners for quantization precision eval scripts."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from miniTransformer.ops.exp.quant_utils import print_metrics, print_summary_table


def print_input_stats(x: torch.Tensor) -> None:
    print(f"Input: shape={x.shape}, dtype={x.dtype}")
    print(
        f"Range: [{x.float().min():.4f}, {x.float().max():.4f}],"
        f" std={x.float().std():.4f}"
    )


def run_named_configs(
    x: torch.Tensor,
    configs: Sequence[tuple[str, Callable[[], torch.Tensor]]],
) -> dict:
    """Run (name, fn) configs, print per-scheme metrics, then a summary table."""
    results = {}
    for name, fn in configs:
        try:
            results[name] = print_metrics(name, x, fn())
        except Exception as e:
            print(f"\n  [{name}] skipped: {e}")
    print_summary_table(results)
    return results
