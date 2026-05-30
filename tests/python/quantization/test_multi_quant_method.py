"""Multi-format quantization comparison: MXFP8, NVFP4, and custom low-precision formats.

Schemes:
  MXFP8  E4M3  | E8M0 scale        — TransformerEngine baseline
  E4M1         | E4M3 scale        — fp_max=192.0 (FP8 NaN-aware convention)
  E2M3         | E4M3 scale        — fp_max=3.75
  MXFP4  E2M1  | E4M3 scale        — MX spec, no global scale
  NVFP4  E2M1  | global + E4M3     — global amax + per-block E4M3 scale
  NVFP4  E2M1  | pertoken + E4M3   — per-token amax + per-block E4M3 scale
  NVFP4  E2M1  | TE                — TransformerEngine official implementation
  NVFP4  E2M1  | REF               — miniTransformer reference implementation

Block scale design (mirrors nvfp4_quantize_op_ref._quantize_blockwise_reference):
  decode_scale = round-to-nearest(block_amax / fp_max)   — no ceil
  encode_scale = 1 / decode_scale                         — derived from quantized value
  x_scaled     = x * encode_scale + clamp guard
"""
import torch
import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch import NVFP4Quantizer, MXFP8Quantizer

from miniTransformer.ops.exp.quant_utils import (
    cast_to_e4m3, quant_dequant_block, quant_dequant_pertoken,
    _quantize_e4m1, _quantize_e2m3, _quantize_e2m1,
    print_metrics, print_summary_table, make_test_tensor,
)

_FP4_E2M1_MAX = 6.0
_FP8_E4M3_MAX = 448.0

# fp_max values for FP8-convention formats (max_exp - 1 reserved for NaN/Inf):
#   E4M1: 2^(14-7) * 1.5 = 192.0
#   E2M3: 2^(2-1) * 1.875 = 3.75
_E4M1_MAX = 192.0
_E2M3_MAX = 3.75


# ── TransformerEngine-backed implementations ──────────────────────────────────

def quant_dequant_mxfp8(x: torch.Tensor) -> torch.Tensor:
    quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    return quantizer(x).dequantize()


def quant_dequant_nvfp4_te(x: torch.Tensor) -> torch.Tensor:
    quantizer = NVFP4Quantizer(
        rowwise=True, columnwise=False,
        with_amax_reduction=False, amax_reduction_group=None,
        with_rht=False, with_post_rht_amax=False,
        with_2d_quantization=False, stochastic_rounding=False,
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


# ── Custom NVFP4 implementations ──────────────────────────────────────────────

def quant_dequant_nvfp4_global(x: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """NVIDIA NVFP4: single global amax scale + per-block E4M3 scale."""
    return quant_dequant_pertoken(
        x.reshape(1, -1), block_size, _FP4_E2M1_MAX, _quantize_e2m1, "e4m3",
    ).reshape(x.shape)


def quant_dequant_nvfp4_pertoken(x: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """NVFP4 per-token: per-row global scale + per-block E4M3 scale."""
    return quant_dequant_pertoken(x, block_size, _FP4_E2M1_MAX, _quantize_e2m1, "e4m3")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_format_comparison() -> None:
    x = make_test_tensor(32, 32, device="cuda")
    print(f"Input: shape={x.shape}, dtype={x.dtype}")
    print(f"Range: [{x.float().min():.4f}, {x.float().max():.4f}],"
          f" std={x.float().std():.4f}")

    configs = [
        ("MXFP8 E4M3 | E8M0 (baseline)",    lambda: quant_dequant_mxfp8(x)),
        ("E4M1       | E4M3",                lambda: quant_dequant_block(x, 16, _E4M1_MAX, _quantize_e4m1, "e4m3")),
        ("E2M3       | E4M3",                lambda: quant_dequant_block(x, 16, _E2M3_MAX, _quantize_e2m3, "e4m3")),
        ("MXFP4 E2M1 | E4M3",                lambda: quant_dequant_block(x, 16, _FP4_E2M1_MAX, _quantize_e2m1, "e4m3")),
        ("NVFP4 E2M1 | global + E4M3",       lambda: quant_dequant_nvfp4_global(x)),
        ("NVFP4 E2M1 | pertoken + E4M3",     lambda: quant_dequant_nvfp4_pertoken(x)),
        ("NVFP4 E2M1 | TE",                  lambda: quant_dequant_nvfp4_te(x)),
        ("NVFP4 E2M1 | REF",                 lambda: quant_dequant_nvfp4_ref(x)),
    ]

    results = {}
    for name, fn in configs:
        try:
            results[name] = print_metrics(name, x, fn())
        except Exception as e:
            print(f"\n  [{name}] skipped: {e}")
    print_summary_table(results)


if __name__ == "__main__":
    run_format_comparison()
