"""Fixed M=1, varying E=2/3/4/5: quantization precision vs exponent bits.

Schemes (8 total): E5M1 / E4M1 / E3M1 / E2M1  ×  E4M3 scale / E8M0 scale

Key finding: wider exponent increases dynamic range but not mantissa resolution.
Metrics: MAE / Mean Rel Err / SNR / Rel_L2
"""
import torch

from miniTransformer.ops.exp.quant_utils import (
    fp_format_max, quant_dequant_block,
    _quantize_e5m1, _quantize_e4m1, _quantize_e3m1, _quantize_e2m1,
    print_metrics, print_summary_table, make_test_tensor,
)


def run_exponent_bits_experiment() -> None:
    x = make_test_tensor(32, 32)
    print(f"Input: shape={x.shape}, dtype={x.dtype}")
    print(f"Range: [{x.float().min():.4f}, {x.float().max():.4f}],"
          f" std={x.float().std():.4f}")

    configs = [
        ("E5M1 | E4M3", lambda: quant_dequant_block(x, 16, fp_format_max(5, 1), _quantize_e5m1, "e4m3")),
        ("E4M1 | E4M3", lambda: quant_dequant_block(x, 16, fp_format_max(4, 1), _quantize_e4m1, "e4m3")),
        ("E3M1 | E4M3", lambda: quant_dequant_block(x, 16, fp_format_max(3, 1), _quantize_e3m1, "e4m3")),
        ("E2M1 | E4M3", lambda: quant_dequant_block(x, 16, fp_format_max(2, 1), _quantize_e2m1, "e4m3")),
        ("E5M1 | E8M0", lambda: quant_dequant_block(x, 16, fp_format_max(5, 1), _quantize_e5m1, "e8m0")),
        ("E4M1 | E8M0", lambda: quant_dequant_block(x, 16, fp_format_max(4, 1), _quantize_e4m1, "e8m0")),
        ("E3M1 | E8M0", lambda: quant_dequant_block(x, 16, fp_format_max(3, 1), _quantize_e3m1, "e8m0")),
        ("E2M1 | E8M0", lambda: quant_dequant_block(x, 16, fp_format_max(2, 1), _quantize_e2m1, "e8m0")),
    ]

    results = {}
    for name, fn in configs:
        results[name] = print_metrics(name, x, fn())
    print_summary_table(results)


if __name__ == "__main__":
    run_exponent_bits_experiment()
