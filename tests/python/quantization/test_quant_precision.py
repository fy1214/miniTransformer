"""NVFP4 vs MXFP8 quantization precision comparison using TransformerEngine.

Supports multiple input distributions to simulate real LLM activation patterns.

Usage:
    python test_quant_precision.py
    python test_quant_precision.py --shape 1024 4096 --dtype bfloat16
    python test_quant_precision.py --dist outlier
    python test_quant_precision.py --all-dist
"""
import argparse

import torch
import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer


# ── Quantize + dequantize ─────────────────────────────────────────────────────

def quant_dequant_nvfp4(x: torch.Tensor, rowwise: bool = True) -> torch.Tensor:
    quantizer = NVFP4Quantizer(rowwise=rowwise, columnwise=False)
    qt = quantizer.make_empty(x.shape, dtype=x.dtype, device=x.device)
    qt = quantizer.update_quantized(x, qt)
    return qt.dequantize(dtype=x.dtype)


def quant_dequant_mxfp8(x: torch.Tensor, rowwise: bool = True) -> torch.Tensor:
    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3, rowwise=rowwise, columnwise=False,
    )
    qt = quantizer.make_empty(x.shape, dtype=x.dtype, device=x.device)
    qt = quantizer.update_quantized(x, qt)
    return qt.dequantize(dtype=x.dtype)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(x_orig: torch.Tensor, x_dq: torch.Tensor, name: str) -> dict:
    x_f, d_f = x_orig.float(), x_dq.float()
    err = (x_f - d_f).abs()
    snr_db = (10 * torch.log10(x_f.pow(2).mean() / (err.pow(2).mean() + 1e-12))).item()
    cos_sim = torch.nn.functional.cosine_similarity(
        x_f.flatten().unsqueeze(0), d_f.flatten().unsqueeze(0)
    ).item()
    m = dict(
        max_abs=err.max().item(),
        mean_abs=err.mean().item(),
        rel_err=(err / (x_f.abs() + 1e-6)).mean().item(),
        snr_db=snr_db,
        cos_sim=cos_sim,
    )
    print(f"  [{name}]")
    print(f"    max abs err  : {m['max_abs']:.6f}")
    print(f"    mean abs err : {m['mean_abs']:.6f}")
    print(f"    mean rel err : {m['rel_err']:.6f}")
    print(f"    SNR (dB)     : {m['snr_db']:.2f}")
    print(f"    cosine sim   : {m['cos_sim']:.8f}")
    return m


# ── Input distributions ───────────────────────────────────────────────────────

def make_activation_tensor(shape, dtype, dist: str, device: str = "cuda") -> torch.Tensor:
    torch.manual_seed(42)
    if dist == "randn":
        x = torch.randn(shape, device=device, dtype=torch.float32)
    elif dist == "uniform":
        x = torch.rand(shape, device=device, dtype=torch.float32) * 2 - 1
    elif dist == "outlier":
        x = torch.randn(shape, device=device, dtype=torch.float32)
        n = max(1, int(x.numel() * 0.01))
        x.flatten()[torch.randperm(x.numel(), device=device)[:n]] = \
            torch.randn(n, device=device) * 50
    elif dist == "small_outlier":
        x = torch.randn(shape, device=device, dtype=torch.float32)
        n = max(1, int(x.numel() * 0.001))
        x.flatten()[torch.randperm(x.numel(), device=device)[:n]] = \
            torch.randn(n, device=device) * 200
    else:
        raise ValueError(f"unknown dist: {dist!r}  (randn/uniform/outlier/small_outlier)")
    return x.to(dtype)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_nvfp4_vs_mxfp8(shape, dtype, dist: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = make_activation_tensor(shape, dtype, dist, device)
    x_f = x.float()

    print(f"\n{'=' * 60}")
    print(f"Input tensor")
    print(f"{'=' * 60}")
    print(f"  shape       : {list(x.shape)}")
    print(f"  dtype       : {x.dtype}")
    print(f"  dist        : {dist}")
    print(f"  abs max     : {x_f.abs().max():.4f}")
    print(f"  abs mean    : {x_f.abs().mean():.4f}")
    print(f"  std         : {x_f.std():.4f}")
    print(f"  outlier>3σ  : {(x_f.abs() > 3 * x_f.std()).float().mean():.4%}")

    print(f"\n{'=' * 60}")
    print(f"Quantization precision comparison")
    print(f"{'=' * 60}")

    r_nvfp4, r_mxfp8 = None, None
    try:
        r_nvfp4 = compute_metrics(x, quant_dequant_nvfp4(x), "NVFP4")
    except Exception as e:
        print(f"  [NVFP4] skipped: {e}")
    try:
        r_mxfp8 = compute_metrics(x, quant_dequant_mxfp8(x), "MXFP8")
    except Exception as e:
        print(f"  [MXFP8] skipped: {e}")

    if r_nvfp4 and r_mxfp8:
        winner = "NVFP4" if r_nvfp4["snr_db"] > r_mxfp8["snr_db"] else "MXFP8"
        diff = abs(r_nvfp4["snr_db"] - r_mxfp8["snr_db"])
        print(f"\n{'=' * 60}")
        print(f"  NVFP4  SNR : {r_nvfp4['snr_db']:6.2f} dB")
        print(f"  MXFP8  SNR : {r_mxfp8['snr_db']:6.2f} dB")
        print(f"  Winner     : {winner}  (gap {diff:.2f} dB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--dist", choices=["randn", "uniform", "outlier", "small_outlier"],
                        default="randn")
    parser.add_argument("--all-dist", action="store_true")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    shape = tuple(args.shape)
    dists = ["randn", "uniform", "outlier", "small_outlier"] if args.all_dist else [args.dist]
    for dist in dists:
        run_nvfp4_vs_mxfp8(shape, dtype, dist)
