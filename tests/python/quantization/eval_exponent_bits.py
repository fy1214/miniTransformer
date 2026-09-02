"""Fixed M=1, vary E=2/3/4/5: precision vs exponent bits.

Eight schemes: E5M1 / E4M1 / E3M1 / E2M1  x  E4M3 / E8M0 block scale.
Wider E grows dynamic range; it does not add mantissa resolution.

Metrics: MAE / mean rel err / SNR / Rel_L2.
"""

from miniTransformer.ops.exp.quant_utils import (
    fp_format_max,
    quant_dequant_block,
    _quantize_e5m1,
    _quantize_e4m1,
    _quantize_e3m1,
    _quantize_e2m1,
    make_test_tensor,
)

from helpers import print_input_stats, run_named_configs


def run_exponent_bits_experiment() -> None:
    x = make_test_tensor(32, 32)
    print_input_stats(x)
    run_named_configs(
        x,
        [
            ("E5M1 | E4M3", lambda: quant_dequant_block(x, 16, fp_format_max(5, 1), _quantize_e5m1, "e4m3")),
            ("E4M1 | E4M3", lambda: quant_dequant_block(x, 16, fp_format_max(4, 1), _quantize_e4m1, "e4m3")),
            ("E3M1 | E4M3", lambda: quant_dequant_block(x, 16, fp_format_max(3, 1), _quantize_e3m1, "e4m3")),
            ("E2M1 | E4M3", lambda: quant_dequant_block(x, 16, fp_format_max(2, 1), _quantize_e2m1, "e4m3")),
            ("E5M1 | E8M0", lambda: quant_dequant_block(x, 16, fp_format_max(5, 1), _quantize_e5m1, "e8m0")),
            ("E4M1 | E8M0", lambda: quant_dequant_block(x, 16, fp_format_max(4, 1), _quantize_e4m1, "e8m0")),
            ("E3M1 | E8M0", lambda: quant_dequant_block(x, 16, fp_format_max(3, 1), _quantize_e3m1, "e8m0")),
            ("E2M1 | E8M0", lambda: quant_dequant_block(x, 16, fp_format_max(2, 1), _quantize_e2m1, "e8m0")),
        ],
    )


if __name__ == "__main__":
    run_exponent_bits_experiment()
