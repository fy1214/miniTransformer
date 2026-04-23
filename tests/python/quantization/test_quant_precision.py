"""
NVFP4 vs MXFP8 量化精度对比脚本
===================================

对同一个 tensor x，分别用 NVFP4Quantizer 和 MXFP8Quantizer 量化后反量化，
与原始 x 对比精度（abs error、rel error、SNR）。

运行：
    python3 test_quant_precision.py
    python3 test_quant_precision.py --shape 1024 4096 --dtype bfloat16
    python3 test_quant_precision.py --dist randn      # 正态分布
    python3 test_quant_precision.py --dist outlier    # 含 outlier 的分布
"""

import argparse
import torch

import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer


# ==================== 精度指标 ====================

def precision_metrics(x_orig: torch.Tensor, x_dq: torch.Tensor, name: str):
    """计算量化误差的各项指标并打印。"""
    x_f = x_orig.float()
    d_f = x_dq.float()
    err = (x_f - d_f).abs()

    max_abs   = err.max().item()
    mean_abs  = err.mean().item()
    # 相对误差：|err| / (|x| + eps)
    rel_err   = (err / (x_f.abs() + 1e-6)).mean().item()
    # 信噪比 SNR（dB）：10 * log10(signal_power / noise_power)
    signal_power = (x_f ** 2).mean()
    noise_power  = (err ** 2).mean()
    snr_db = (10 * torch.log10(signal_power / (noise_power + 1e-12))).item()
    # 余弦相似度
    cos_sim = torch.nn.functional.cosine_similarity(
        x_f.flatten().unsqueeze(0),
        d_f.flatten().unsqueeze(0)
    ).item()

    print(f"  [{name}]")
    print(f"    max abs err  : {max_abs:.6f}")
    print(f"    mean abs err : {mean_abs:.6f}")
    print(f"    mean rel err : {rel_err:.6f}")
    print(f"    SNR (dB)     : {snr_db:.2f}")
    print(f"    cosine sim   : {cos_sim:.8f}")
    return dict(max_abs=max_abs, mean_abs=mean_abs, rel_err=rel_err,
                snr_db=snr_db, cos_sim=cos_sim)


# ==================== 量化 + 反量化 ====================

def quant_dequant_nvfp4(x: torch.Tensor, rowwise: bool = True) -> torch.Tensor:
    """NVFP4 量化 → 反量化。"""
    quantizer = NVFP4Quantizer(
        rowwise=rowwise,
        columnwise=False,
    )
    qt = quantizer.make_empty(x.shape, dtype=x.dtype, device=x.device)
    qt = quantizer.update_quantized(x, qt)
    return qt.dequantize(dtype=x.dtype)


def quant_dequant_mxfp8(x: torch.Tensor, rowwise: bool = True) -> torch.Tensor:
    """MXFP8 量化 → 反量化。"""
    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=rowwise,
        columnwise=False,
    )
    qt = quantizer.make_empty(x.shape, dtype=x.dtype, device=x.device)
    qt = quantizer.update_quantized(x, qt)
    return qt.dequantize(dtype=x.dtype)


# ==================== 测试分布 ====================

def make_tensor(shape, dtype, dist: str, device: str = "cuda") -> torch.Tensor:
    """生成不同分布的测试 tensor。"""
    torch.manual_seed(42)
    if dist == "randn":
        x = torch.randn(shape, device=device, dtype=torch.float32)
    elif dist == "uniform":
        x = torch.rand(shape, device=device, dtype=torch.float32) * 2 - 1
    elif dist == "outlier":
        # 正态分布 + 1% 的极端 outlier（模拟 LLM 激活值）
        x = torch.randn(shape, device=device, dtype=torch.float32)
        n_outlier = max(1, int(x.numel() * 0.01))
        idx = torch.randperm(x.numel(), device=device)[:n_outlier]
        x.flatten()[idx] = torch.randn(n_outlier, device=device) * 50
    elif dist == "small_outlier":
        # 更极端：0.1% 的 outlier，幅度更大
        x = torch.randn(shape, device=device, dtype=torch.float32)
        n_outlier = max(1, int(x.numel() * 0.001))
        idx = torch.randperm(x.numel(), device=device)[:n_outlier]
        x.flatten()[idx] = torch.randn(n_outlier, device=device) * 200
    else:
        raise ValueError(f"未知分布: {dist}，支持 randn / uniform / outlier / small_outlier")
    return x.to(dtype)


# ==================== 主函数 ====================

def run(shape, dtype, dist: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = make_tensor(shape, dtype, dist, device)

    print(f"\n{'='*60}")
    print(f"输入 tensor 统计")
    print(f"{'='*60}")
    x_f = x.float()
    print(f"  shape     : {list(x.shape)}")
    print(f"  dtype     : {x.dtype}")
    print(f"  dist      : {dist}")
    print(f"  abs max   : {x_f.abs().max():.4f}")
    print(f"  abs mean  : {x_f.abs().mean():.4f}")
    print(f"  std       : {x_f.std():.4f}")
    outlier_ratio = (x_f.abs() > 3 * x_f.std()).float().mean().item()
    print(f"  outlier>3σ: {outlier_ratio:.4%}")

    print(f"\n{'='*60}")
    print(f"量化精度对比")
    print(f"{'='*60}")

    # NVFP4
    try:
        x_nvfp4 = quant_dequant_nvfp4(x)
        r_nvfp4 = precision_metrics(x, x_nvfp4, "NVFP4")
    except Exception as e:
        print(f"  [NVFP4] 失败: {e}")
        r_nvfp4 = None

    # MXFP8
    try:
        x_mxfp8 = quant_dequant_mxfp8(x)
        r_mxfp8 = precision_metrics(x, x_mxfp8, "MXFP8")
    except Exception as e:
        print(f"  [MXFP8] 失败: {e}")
        r_mxfp8 = None

    # 总结
    if r_nvfp4 and r_mxfp8:
        print(f"\n{'='*60}")
        print(f"SNR 对比总结  (高 = 好)")
        print(f"{'='*60}")
        print(f"  NVFP4  SNR: {r_nvfp4['snr_db']:6.2f} dB")
        print(f"  MXFP8  SNR: {r_mxfp8['snr_db']:6.2f} dB")
        winner = "NVFP4" if r_nvfp4['snr_db'] > r_mxfp8['snr_db'] else "MXFP8"
        diff = abs(r_nvfp4['snr_db'] - r_mxfp8['snr_db'])
        print(f"  Winner: {winner}  (差距 {diff:.2f} dB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument(
        "--dist",
        choices=["randn", "uniform", "outlier", "small_outlier"],
        default="randn",
        help="输入分布（默认 randn）"
    )
    parser.add_argument(
        "--all-dist", action="store_true",
        help="依次测试所有分布"
    )
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    shape = tuple(args.shape)

    if args.all_dist:
        for dist in ["randn", "uniform", "outlier", "small_outlier"]:
            run(shape, dtype, dist)
    else:
        run(shape, dtype, args.dist)
