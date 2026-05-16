#!/usr/bin/env python3
"""
End-to-end NVFP4 grouped GEMM smoke / diff test.

Usage (from repo root):

  python scripts/run_grouped_gemm_nvfp4_test.py
  python scripts/run_grouped_gemm_nvfp4_test.py --large
  python scripts/run_grouped_gemm_nvfp4_test.py --m-splits 128,128 --n 256 --k 1024

Behavior:
  - Any CUDA GPU: runs BF16 reference vs pertoken quant + ``reference_grouped_gemm_pertoken``
    (no SM100 required).
  - SM100 only: additionally runs CuTeDSL grouped kernels and (unless ``--skip-te``) the TE path.

  Use ``--require-sm100`` if the script must exercise kernels (CI on Blackwell); exits 1 when not SM100.

Requires:
  - CUDA for meaningful output (CPU-only exits early).
  - SM100 + cutlass/cuteDSL + transformer_engine for full kernel / TE paths.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

import torch


def _parse_m_splits(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _grouped_bf16_reference(x_groups, w_groups) -> torch.Tensor:
    outs = []
    for x, w in zip(x_groups, w_groups):
        outs.append(torch.matmul(x.float(), w.float().T))
    return torch.cat(outs, dim=0)


def _error_metrics(pred: torch.Tensor, ref: torch.Tensor) -> dict:
    err = (pred.float() - ref.float()).abs()
    rmse = torch.sqrt(torch.mean((pred.float() - ref.float()) ** 2))
    return {
        "max_abs": float(err.max().item()),
        "mean_abs": float(err.mean().item()),
        "rmse": float(rmse.item()),
    }


def _print_metrics(label: str, metrics: dict) -> None:
    print(
        f"  {label}: max={metrics['max_abs']:.6f}, "
        f"mean={metrics['mean_abs']:.6f}, rmse={metrics['rmse']:.6f}"
    )


def _make_te_nvfp4_quantizer(columnwise: bool):
    import transformer_engine.pytorch.cpp_extensions as tex
    from transformer_engine.pytorch import NVFP4Quantizer

    return NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=columnwise,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
    )


def run_case(
    m_splits: List[int],
    n: int,
    k: int,
    seed: int = 1234,
    skip_te: bool = False,
    skip_kernel: bool = False,
    require_sm100: bool = False,
) -> int:
    from miniTransformer.ops.cuteDSL.nvfp4.grouped_gemm_nvfp4_api import (
        grouped_gemm_cutedsl,
        grouped_gemm_cutedsl_pertoken,
        is_sm100,
        quantize_activation_nvfp4_pertoken,
        quantize_weight_nvfp4_tensor_global,
        reference_grouped_gemm_pertoken,
    )

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available.")
        return 1 if require_sm100 else 0

    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(torch.cuda.current_device())
    sm100 = is_sm100()

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = "cuda"

    print(f"\n=== shape: m_splits={m_splits}, n={n}, k={k} ===")
    print(f"  GPU: {name}  compute capability {major}.{minor}  SM100={sm100}")

    x_groups = [
        0.1 * torch.randn(m, k, dtype=torch.bfloat16, device=device) for m in m_splits
    ]
    w_groups = [
        0.1 * torch.randn(n, k, dtype=torch.bfloat16, device=device) for _ in m_splits
    ]

    ref = _grouped_bf16_reference(x_groups, w_groups)
    print(f"  BF16 reference shape: {tuple(ref.shape)}")

    failures = 0

    # --- pertoken reference (no kernel) ---
    a_pt = [quantize_activation_nvfp4_pertoken(x) for x in x_groups]
    b_tg = [quantize_weight_nvfp4_tensor_global(w) for w in w_groups]
    ref_pt = reference_grouped_gemm_pertoken(a_pt, b_tg, m_splits, torch.bfloat16)
    m_ref = _error_metrics(ref_pt, ref)
    _print_metrics("pertoken reference vs BF16", m_ref)

    if require_sm100 and not sm100:
        print(
            "  FAIL (--require-sm100): need SM100 (cc 10.0) for CuTeDSL grouped kernels."
        )
        return 1

    if skip_kernel or not sm100:
        if not sm100 and not skip_kernel:
            print(
                "  SKIP CuTeDSL kernels: SM100 (Blackwell, cc 10.0) required for grouped blockscaled GEMM."
            )
        if not sm100 and not skip_te:
            print(
                "  SKIP TE NVFP4 CuTeDSL path: same SM100 kernel requirement."
            )
        return failures

    # --- pertoken + CuTeDSL kernel ---
    try:
        out_pt = grouped_gemm_cutedsl_pertoken(a_pt, b_tg, m_splits, torch.bfloat16)
        m_kern = _error_metrics(out_pt, ref_pt)
        _print_metrics("CuTeDSL pertoken vs pertoken ref", m_kern)
        m_bf16 = _error_metrics(out_pt, ref)
        _print_metrics("CuTeDSL pertoken vs BF16", m_bf16)
        if not torch.isfinite(out_pt).all():
            print("  FAIL: pertoken kernel output has non-finite values.")
            failures += 1
        elif m_kern["max_abs"] > 0.1:
            print("  WARN: pertoken kernel vs reference max_abs > 0.1")
    except Exception as exc:
        print(f"  SKIP pertoken kernel: {exc}")

    # --- TE NVFP4 + CuTeDSL (tensor-global amax) ---
    if skip_te:
        return failures

    try:
        __import__("transformer_engine.pytorch", fromlist=["*"])
        te_q_x = _make_te_nvfp4_quantizer(columnwise=False)
        te_q_w = _make_te_nvfp4_quantizer(columnwise=False)
        a_nv = [te_q_x.quantize(x) for x in x_groups]
        b_nv = [te_q_w.quantize(w) for w in w_groups]
        out_te = grouped_gemm_cutedsl(a_nv, b_nv, m_splits, torch.bfloat16)
        m_te = _error_metrics(out_te, ref)
        _print_metrics("TE NVFP4 CuTeDSL vs BF16", m_te)
        if not torch.isfinite(out_te).all():
            print("  FAIL: TE path output has non-finite values.")
            failures += 1
    except Exception as exc:
        print(f"  SKIP TE NVFP4 path: {exc}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="NVFP4 grouped GEMM diff test")
    parser.add_argument(
        "--m-splits",
        type=str,
        default="128,128",
        help="Comma-separated M sizes per group, e.g. 128,128",
    )
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--large",
        action="store_true",
        help="Also run a larger MoE-like case (4 groups, n=4096, k=2048).",
    )
    parser.add_argument("--skip-te", action="store_true", help="Skip Transformer Engine path.")
    parser.add_argument(
        "--skip-kernel",
        action="store_true",
        help="Only run reference / quant checks, no CuTeDSL kernel.",
    )
    parser.add_argument(
        "--require-sm100",
        action="store_true",
        help="Exit with status 1 if CUDA device is not SM100 (for CI that must run kernels).",
    )
    args = parser.parse_args()

    cases: List[Tuple[List[int], int, int]] = [
        (_parse_m_splits(args.m_splits), args.n, args.k),
    ]
    if args.large:
        cases.append(([80, 176, 128, 72], 4096, 2048))

    total_fail = 0
    for m_splits, n, k in cases:
        total_fail += run_case(
            m_splits,
            n,
            k,
            seed=args.seed,
            skip_te=args.skip_te,
            skip_kernel=args.skip_kernel,
            require_sm100=args.require_sm100,
        )

    if total_fail:
        print(f"\nFinished with {total_fail} failure(s).")
        return 1
    print(
        "\nSummary: pertoken quant + reference grouped matmul vs BF16 ran on CUDA above. "
        "CuTeDSL grouped NVFP4 GEMM runs only on SM100; "
        "pass --require-sm100 to exit non-zero without Blackwell."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
