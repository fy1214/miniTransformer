"""Fixed E=2, varying M=1/2/3/4 with per-token global amax two-level quantization.

Extends test_quantize_M_effect.py by adding a per-token global scale before
per-block quantization, reducing the impact of inter-row dynamic range differences.

Level-1  global_scale[i] = amax(|x[i,:]|) / (_NORM_MAX * fp_max)   — fp32, per row
Level-2  decode_scale    = amax(|x_norm_block|) / fp_max             — e4m3 or e8m0

_NORM_MAX = 240.0 (instead of 448.0) intentionally leaves headroom for block scale
precision by keeping the normalized range within the lower FP8 representable region.

Metrics: MAE / Mean Rel Err / SNR / Rel_L2
"""
import torch

from miniTransformer.ops.exp.quant_utils import (
    fp_format_max, quant_dequant_pertoken,
    _quantize_e2m4, _quantize_e2m3, _quantize_e2m2, _quantize_e2m1,
    print_metrics, print_summary_table, make_test_tensor,
)

_NORM_MAX = 240.0


def run_pertoken_amax_experiment() -> None:
    x = make_test_tensor(32, 128)
    print(f"Input: shape={x.shape}, dtype={x.dtype}")
    print(f"Range: [{x.float().min():.4f}, {x.float().max():.4f}],"
          f" std={x.float().std():.4f}")

    configs = [
        ("E2M4+global | E4M3", lambda: quant_dequant_pertoken(x, 16, fp_format_max(2, 4), _quantize_e2m4, "e4m3", _NORM_MAX)),
        ("E2M3+global | E4M3", lambda: quant_dequant_pertoken(x, 16, fp_format_max(2, 3), _quantize_e2m3, "e4m3", _NORM_MAX)),
        ("E2M2+global | E4M3", lambda: quant_dequant_pertoken(x, 16, fp_format_max(2, 2), _quantize_e2m2, "e4m3", _NORM_MAX)),
        ("E2M1+global | E4M3", lambda: quant_dequant_pertoken(x, 16, fp_format_max(2, 1), _quantize_e2m1, "e4m3", _NORM_MAX)),
        ("E2M4+global | E8M0", lambda: quant_dequant_pertoken(x, 16, fp_format_max(2, 4), _quantize_e2m4, "e8m0", _NORM_MAX)),
        ("E2M3+global | E8M0", lambda: quant_dequant_pertoken(x, 16, fp_format_max(2, 3), _quantize_e2m3, "e8m0", _NORM_MAX)),
        ("E2M2+global | E8M0", lambda: quant_dequant_pertoken(x, 16, fp_format_max(2, 2), _quantize_e2m2, "e8m0", _NORM_MAX)),
        ("E2M1+global | E8M0", lambda: quant_dequant_pertoken(x, 16, fp_format_max(2, 1), _quantize_e2m1, "e8m0", _NORM_MAX)),
    ]

    results = {}
    for name, fn in configs:
        results[name] = print_metrics(name, x, fn())
    print_summary_table(results)


if __name__ == "__main__":
    run_pertoken_amax_experiment()
