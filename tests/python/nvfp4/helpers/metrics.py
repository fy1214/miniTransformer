"""Real-input quant eval metrics and reporting (shared by eval/*.py scripts)."""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch

__all__ = [
    "SEP",
    "load_real_input",
    "compute_metrics",
    "run_quant_eval",
    "print_metrics_table",
    "print_row_comparison",
]

SEP = "-" * 72


def load_real_input(
    path: str,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Load a dump tensor, flatten to 2D [tokens, hidden], move to device."""
    x = torch.load(path, map_location="cpu")
    x = x.reshape(-1, x.shape[-1]).to(dtype=dtype, device=device)
    xf = x.to(torch.float32)
    print(f"Loaded : {path}")
    print(f"  shape={list(x.shape)}, dtype={dtype}")
    print(
        f"  min={xf.min():.4f}  max={xf.max():.4f}  "
        f"abs_mean={xf.abs().mean():.4f}  abs_std={xf.abs().std():.4f}"
    )
    return x


def compute_metrics(
    x_orig: torch.Tensor,
    x_dq: torch.Tensor,
) -> Dict:
    """Per-row and aggregate precision metrics (eff-bits, MSE, cosine, rel-L2)."""
    xf = x_orig.to(torch.float32)
    dqf = x_dq.to(torch.float32)

    err = xf - dqf
    mse_rows = err.pow(2).mean(dim=1)
    rel_sq = err.pow(2).sum(dim=1) / xf.pow(2).sum(dim=1).clamp(min=1e-9)
    rel_l2 = rel_sq.sqrt()
    num = (xf * dqf).sum(dim=1)
    denom = xf.norm(dim=1) * dqf.norm(dim=1)
    cos_rows = num / denom.clamp(min=1e-9)
    eff_bits = (-torch.log2(rel_sq.mean()) / 2).item()

    return dict(
        eff_bits=eff_bits,
        mse_rows=mse_rows,
        cos_rows=cos_rows,
        rel_l2_rows=rel_l2,
        mse_mean=mse_rows.mean().item(),
        cos_mean=cos_rows.mean().item(),
        cos_min=cos_rows.min().item(),
    )


def run_quant_eval(
    x: torch.Tensor,
    methods: Dict[str, Callable[[torch.Tensor], torch.Tensor]],
    *,
    title: str = "",
) -> Dict[str, Dict]:
    """Run quant→dequant wrappers on ``x`` and print a metrics table."""
    results: Dict[str, Dict] = {}
    for label, fn in methods.items():
        x_dq = fn(x)
        results[label] = compute_metrics(x, x_dq)

    print(f"\n{'=' * 72}")
    if title:
        print(f"  {title}")
        print(SEP)
    print_metrics_table(results)

    if len(results) == 2:
        labels = list(results.keys())
        print()
        print_row_comparison(
            results[labels[0]],
            results[labels[1]],
            ref_label=labels[0],
            cmp_label=labels[1],
        )

    return results


def print_metrics_table(
    results: Dict[str, Dict],
    *,
    cos_thresholds: Tuple[float, ...] = (0.99, 0.95, 0.90),
    rel_thresholds: Tuple[float, ...] = (0.05, 0.10),
) -> None:
    """Side-by-side metrics table for multiple methods."""
    labels = list(results.keys())
    metrics = list(results.values())
    col_w = 16

    def row(name, vals):
        return f"  {name:<33}" + "".join(f"{v:>{col_w}}" for v in vals)

    print(f"{'Metric':<35}" + "".join(f"{l:>{col_w}}" for l in labels))
    print(SEP)
    print(row("Eff-bits", [f"{m['eff_bits']:.4f}" for m in metrics]))
    print(row("Mean MSE", [f"{m['mse_mean']:.2e}" for m in metrics]))
    print(row("Mean cosine", [f"{m['cos_mean']:.6f}" for m in metrics]))
    print(row("Min  cosine", [f"{m['cos_min']:.6f}" for m in metrics]))
    for thr in cos_thresholds:
        print(row(
            f"Rows cos < {thr}",
            [str((m["cos_rows"] < thr).sum().item()) for m in metrics],
        ))
    for thr in rel_thresholds:
        print(row(
            f"Rows rel-L2 > {int(thr * 100)}%",
            [str((m["rel_l2_rows"] > thr).sum().item()) for m in metrics],
        ))


def print_row_comparison(
    ref_metrics: Dict,
    cmp_metrics: Dict,
    ref_label: str = "baseline",
    cmp_label: str = "candidate",
) -> None:
    """Per-row win/lose/tie breakdown (rel-L2 as criterion)."""
    rel_ref = ref_metrics["rel_l2_rows"]
    rel_cmp = cmp_metrics["rel_l2_rows"]
    m = rel_ref.shape[0]

    better = rel_cmp < rel_ref
    worse = rel_cmp > rel_ref
    same = ~better & ~worse
    bn, wn, sn = better.sum().item(), worse.sum().item(), same.sum().item()

    mse_d = cmp_metrics["mse_rows"] - ref_metrics["mse_rows"]
    cos_d = cmp_metrics["cos_rows"] - ref_metrics["cos_rows"]

    print(f"  Per-row: {cmp_label} vs {ref_label}")
    print(SEP)
    print(
        f"  Better : {bn:>8} rows ({100.0 * bn / m:.1f}%)  "
        f"MSE mean={mse_d[better].mean():.2e}  Cos mean={cos_d[better].mean():+.2e}"
    )
    print(
        f"  Worse  : {wn:>8} rows ({100.0 * wn / m:.1f}%)  "
        f"MSE mean={mse_d[worse].mean():.2e}  Cos mean={cos_d[worse].mean():+.2e}"
    )
    print(f"  Same   : {sn:>8} rows ({100.0 * sn / m:.1f}%)")
    print(
        f"  Net    : MSE delta={mse_d.mean():+.2e}  "
        f"Cos delta={cos_d.mean():+.2e}  "
        f"Eff-bits delta={cmp_metrics['eff_bits'] - ref_metrics['eff_bits']:+.4f}"
    )
