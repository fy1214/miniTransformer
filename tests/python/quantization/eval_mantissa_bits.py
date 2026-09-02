"""Fixed E=2, vary M=1/2/3/4: precision vs mantissa bits.

Eight schemes: E2M4 / E2M3 / E2M2 / E2M1  x  E4M3 / E8M0 block scale.
Each extra mantissa bit typically adds ~6 dB SNR.

Metrics: MAE / mean rel err / SNR / Rel_L2.
"""

from miniTransformer.ops.exp.quant_utils import (
    fp_format_max,
    quant_dequant_block,
    _quantize_e2m4,
    _quantize_e2m3,
    _quantize_e2m2,
    _quantize_e2m1,
    make_test_tensor,
)

from helpers import print_input_stats, run_named_configs


def run_mantissa_bits_experiment() -> None:
    x = make_test_tensor(32, 32)
    print_input_stats(x)
    run_named_configs(
        x,
        [
            ("E2M4 | E4M3", lambda: quant_dequant_block(x, 16, fp_format_max(2, 4), _quantize_e2m4, "e4m3")),
            ("E2M3 | E4M3", lambda: quant_dequant_block(x, 16, fp_format_max(2, 3), _quantize_e2m3, "e4m3")),
            ("E2M2 | E4M3", lambda: quant_dequant_block(x, 16, fp_format_max(2, 2), _quantize_e2m2, "e4m3")),
            ("E2M1 | E4M3", lambda: quant_dequant_block(x, 16, fp_format_max(2, 1), _quantize_e2m1, "e4m3")),
            ("E2M4 | E8M0", lambda: quant_dequant_block(x, 16, fp_format_max(2, 4), _quantize_e2m4, "e8m0")),
            ("E2M3 | E8M0", lambda: quant_dequant_block(x, 16, fp_format_max(2, 3), _quantize_e2m3, "e8m0")),
            ("E2M2 | E8M0", lambda: quant_dequant_block(x, 16, fp_format_max(2, 2), _quantize_e2m2, "e8m0")),
            ("E2M1 | E8M0", lambda: quant_dequant_block(x, 16, fp_format_max(2, 1), _quantize_e2m1, "e8m0")),
        ],
    )


if __name__ == "__main__":
    run_mantissa_bits_experiment()
