"""Fixed E=2, varying M=1/2/3/4: quantization precision vs mantissa bits.

Schemes (8 total): E2M4 / E2M3 / E2M2 / E2M1  ×  E4M3 scale / E8M0 scale

Key finding: each additional mantissa bit improves SNR by ~6 dB (information-theoretic).
Metrics: MAE / Mean Rel Err / SNR / Rel_L2
"""
import torch

from miniTransformer.ops.exp.quant_utils import (
    fp_format_max, quant_dequant_block,
    _quantize_e2m4, _quantize_e2m3, _quantize_e2m2, _quantize_e2m1,
    print_metrics, print_summary_table, make_test_tensor,
)


def run_mantissa_bits_experiment() -> None:
    x = make_test_tensor(32, 32)
    print(f"Input: shape={x.shape}, dtype={x.dtype}")
    print(f"Range: [{x.float().min():.4f}, {x.float().max():.4f}],"
          f" std={x.float().std():.4f}")

    configs = [
        ("E2M4 | E4M3", lambda: quant_dequant_block(x, 16, fp_format_max(2, 4), _quantize_e2m4, "e4m3")),
        ("E2M3 | E4M3", lambda: quant_dequant_block(x, 16, fp_format_max(2, 3), _quantize_e2m3, "e4m3")),
        ("E2M2 | E4M3", lambda: quant_dequant_block(x, 16, fp_format_max(2, 2), _quantize_e2m2, "e4m3")),
        ("E2M1 | E4M3", lambda: quant_dequant_block(x, 16, fp_format_max(2, 1), _quantize_e2m1, "e4m3")),
        ("E2M4 | E8M0", lambda: quant_dequant_block(x, 16, fp_format_max(2, 4), _quantize_e2m4, "e8m0")),
        ("E2M3 | E8M0", lambda: quant_dequant_block(x, 16, fp_format_max(2, 3), _quantize_e2m3, "e8m0")),
        ("E2M2 | E8M0", lambda: quant_dequant_block(x, 16, fp_format_max(2, 2), _quantize_e2m2, "e8m0")),
        ("E2M1 | E8M0", lambda: quant_dequant_block(x, 16, fp_format_max(2, 1), _quantize_e2m1, "e8m0")),
    ]

    results = {}
    for name, fn in configs:
        results[name] = print_metrics(name, x, fn())
    print_summary_table(results)


if __name__ == "__main__":
    run_mantissa_bits_experiment()
