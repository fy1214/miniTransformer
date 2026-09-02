"""Multi-format quantization comparison (MXFP8 / custom E*M* / NVFP4 paths).

Schemes:
  MXFP8  E4M3  | E8M0          — TE MXFP8
  E4M1         | E4M3          — fp_max=192 (NaN-aware FP8-style max)
  E2M3         | E4M3          — fp_max=3.75
  MXFP4  E2M1  | E4M3          — block E2M1, no global scale
  NVFP4  E2M1  | global+E4M3   — one global amax + per-block E4M3
  NVFP4  E2M1  | pertoken+E4M3 — per-row amax + per-block E4M3
  NVFP4  E2M1  | TE            — TE NVFP4Quantizer
  NVFP4  E2M1  | REF           — miniTransformer Nvfp4TiledQuantizeRefOp

Block scale:
  decode_scale = round(block_amax / fp_max)
  encode_scale = 1 / decode_scale
  x_q = quantize(clamp(x * encode_scale, +/-fp_max)); x_dq = x_q * decode_scale
"""

import torch
import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch import NVFP4Quantizer, MXFP8Quantizer

from miniTransformer.ops.exp.quant_utils import (
    quant_dequant_block,
    quant_dequant_pertoken,
    _quantize_e4m1,
    _quantize_e2m3,
    _quantize_e2m1,
    make_test_tensor,
)
from helpers import print_input_stats, run_named_configs

_FP4_E2M1_MAX = 6.0
_E4M1_MAX = 192.0
_E2M3_MAX = 3.75


def quant_dequant_mxfp8(x: torch.Tensor) -> torch.Tensor:
    quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    return quantizer(x).dequantize()


def quant_dequant_nvfp4_te(x: torch.Tensor) -> torch.Tensor:
    quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
    )
    return quantizer(x).dequantize()


def quant_dequant_nvfp4_ref(x: torch.Tensor) -> torch.Tensor:
    from miniTransformer.quantiztion.nvfp4_quantize_op_ref import Nvfp4TiledQuantizeRefOp
    from miniTransformer.quantiztion import quantization

    op = Nvfp4TiledQuantizeRefOp()
    params = quantization.QParams(
        scaling_dtype=torch.float8_e4m3fn,
        output_dtype=torch.uint8,
        eps=0.0,
        pow_2_scales=False,
        quant_tile_shape=(1, 16),
        with_2d_quantization=False,
        with_rht=False,
    )
    qx, sx, _, _, global_amax_row, _ = op._quantize(x, params, False, False)
    return op.dequantize(qx, sx, global_amax_row)


# Alias kept for docs that historically used this spelling.
quant_dequant_nvfp4_REF = quant_dequant_nvfp4_ref


def quant_dequant_nvfp4_global(x: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Single global amax (via reshape to one row) + per-block E4M3."""
    return quant_dequant_pertoken(
        x.reshape(1, -1), block_size, _FP4_E2M1_MAX, _quantize_e2m1, "e4m3",
    ).reshape(x.shape)


def quant_dequant_nvfp4_pertoken(x: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Per-row global scale + per-block E4M3."""
    return quant_dequant_pertoken(x, block_size, _FP4_E2M1_MAX, _quantize_e2m1, "e4m3")


def run_format_comparison() -> None:
    x = make_test_tensor(32, 32, device="cuda")
    print_input_stats(x)
    run_named_configs(
        x,
        [
            ("MXFP8 E4M3 | E8M0 (baseline)", lambda: quant_dequant_mxfp8(x)),
            ("E4M1       | E4M3", lambda: quant_dequant_block(x, 16, _E4M1_MAX, _quantize_e4m1, "e4m3")),
            ("E2M3       | E4M3", lambda: quant_dequant_block(x, 16, _E2M3_MAX, _quantize_e2m3, "e4m3")),
            ("MXFP4 E2M1 | E4M3", lambda: quant_dequant_block(x, 16, _FP4_E2M1_MAX, _quantize_e2m1, "e4m3")),
            ("NVFP4 E2M1 | global + E4M3", lambda: quant_dequant_nvfp4_global(x)),
            ("NVFP4 E2M1 | pertoken + E4M3", lambda: quant_dequant_nvfp4_pertoken(x)),
            ("NVFP4 E2M1 | TE", lambda: quant_dequant_nvfp4_te(x)),
            ("NVFP4 E2M1 | REF", lambda: quant_dequant_nvfp4_ref(x)),
        ],
    )


if __name__ == "__main__":
    run_format_comparison()
