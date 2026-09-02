"""Fixed E=2, vary M=1/2/3/4 with per-token global amax (two-level quant).

Level-1: per-row global_scale from row amax / (_NORM_MAX * fp_max).
Level-2: per-block scale in e4m3 or e8m0 on the normalized values.

_NORM_MAX=240 (vs FP8 max 448) leaves headroom so block scales stay in a
better-represented FP8 region.

Metrics: MAE / mean rel err / SNR / Rel_L2.
"""

from miniTransformer.ops.exp.quant_utils import (
    fp_format_max,
    quant_dequant_pertoken,
    _quantize_e2m4,
    _quantize_e2m3,
    _quantize_e2m2,
    _quantize_e2m1,
    make_test_tensor,
)

from helpers import print_input_stats, run_named_configs

_NORM_MAX = 240.0


def run_pertoken_amax_experiment() -> None:
    x = make_test_tensor(32, 128)
    print_input_stats(x)
    run_named_configs(
        x,
        [
            ("E2M4+global | E4M3", lambda: quant_dequant_pertoken(x, 16, fp_format_max(2, 4), _quantize_e2m4, "e4m3", _NORM_MAX)),
            ("E2M3+global | E4M3", lambda: quant_dequant_pertoken(x, 16, fp_format_max(2, 3), _quantize_e2m3, "e4m3", _NORM_MAX)),
            ("E2M2+global | E4M3", lambda: quant_dequant_pertoken(x, 16, fp_format_max(2, 2), _quantize_e2m2, "e4m3", _NORM_MAX)),
            ("E2M1+global | E4M3", lambda: quant_dequant_pertoken(x, 16, fp_format_max(2, 1), _quantize_e2m1, "e4m3", _NORM_MAX)),
            ("E2M4+global | E8M0", lambda: quant_dequant_pertoken(x, 16, fp_format_max(2, 4), _quantize_e2m4, "e8m0", _NORM_MAX)),
            ("E2M3+global | E8M0", lambda: quant_dequant_pertoken(x, 16, fp_format_max(2, 3), _quantize_e2m3, "e8m0", _NORM_MAX)),
            ("E2M2+global | E8M0", lambda: quant_dequant_pertoken(x, 16, fp_format_max(2, 2), _quantize_e2m2, "e8m0", _NORM_MAX)),
            ("E2M1+global | E8M0", lambda: quant_dequant_pertoken(x, 16, fp_format_max(2, 1), _quantize_e2m1, "e8m0", _NORM_MAX)),
        ],
    )


if __name__ == "__main__":
    run_pertoken_amax_experiment()
