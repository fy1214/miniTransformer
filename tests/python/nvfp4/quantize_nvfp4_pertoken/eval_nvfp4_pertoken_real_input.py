#!/usr/bin/env python3
"""
Real-input precision evaluation: quantize_nvfp4_pertoken family
+ quantize_weight_nvfp4_swizzled (tensor-global amax).

Demonstrates the eval framework pattern — to test a different operator,
copy this file and replace the wrapper functions in Section A.

Run (inside container, from this directory):
  cd tests/python/nvfp4/quantize_nvfp4_pertoken
  python3 eval_nvfp4_pertoken_real_input.py

Requires miniTransformer installed: pip install -e .
"""
import torch

from miniTransformer.ops.nvfp4_quantize import (
    quantize_nvfp4_pertoken,
    quantize_nvfp4_pertoken_swizzled,
    dequantize_nvfp4_pertoken,
    dequantize_nvfp4_pertoken_swizzled,
    quantize_weight_nvfp4_swizzled,
    dequantize_weight_nvfp4_swizzled,
    _unswizzle_scale,
)
from miniTransformer.ops.nvfp4_ref_quantize import (
    quant_dequant_nvfp4_ref_standalone,
    quantize_nvfp4_ref_pertoken_amax,
    dequantize_nvfp4_ref_pertoken_amax,
    quantize_nvfp4_ref_global_amax,
    dequantize_nvfp4_ref_global_amax,
)
from real_input_eval_utils import load_real_input, run_quant_eval, SEP

DUMP_PATH = '/tmp/dump/fc1_step0000_inp.pt'

# ===========================================================================
# Section A: operator wrappers  ← only this section changes per operator
#
# Contract: fn(x: Tensor[m,n] bfloat16 CUDA) -> Tensor[m,n] bfloat16 CUDA
# ===========================================================================

def pertoken_rtn(x: torch.Tensor) -> torch.Tensor:
    return dequantize_nvfp4_pertoken(quantize_nvfp4_pertoken(x, four_six=False))

def pertoken_46(x: torch.Tensor) -> torch.Tensor:
    return dequantize_nvfp4_pertoken(quantize_nvfp4_pertoken(x, four_six=True))

def pertoken_swizzled_rtn(x: torch.Tensor) -> torch.Tensor:
    return dequantize_nvfp4_pertoken_swizzled(quantize_nvfp4_pertoken_swizzled(x, four_six=False))

def ref_global_amax(x: torch.Tensor) -> torch.Tensor:
    return quant_dequant_nvfp4_ref_standalone(x, per_token_amax=False, pow_2_scales=False)

def ref_pertoken_amax(x: torch.Tensor) -> torch.Tensor:
    qx, sx, am = quantize_nvfp4_ref_pertoken_amax(x)
    return dequantize_nvfp4_ref_pertoken_amax(qx, sx, am)

def weight_rtn(x: torch.Tensor) -> torch.Tensor:
    return dequantize_weight_nvfp4_swizzled(quantize_weight_nvfp4_swizzled(x, four_six=False))

def weight_46(x: torch.Tensor) -> torch.Tensor:
    return dequantize_weight_nvfp4_swizzled(quantize_weight_nvfp4_swizzled(x, four_six=True))

# ===========================================================================
# Section B: operator-specific checks (bit-exact against golden)
# ===========================================================================

def check_bit_exact(x: torch.Tensor) -> None:
    cuda_res               = quantize_nvfp4_pertoken(x, four_six=False)
    ref_qx, ref_sx, ref_am = quantize_nvfp4_ref_pertoken_amax(x)

    data_match  = torch.equal(cuda_res.data, ref_qx)
    scale_match = torch.equal(cuda_res.scale.view(torch.uint8), ref_sx.view(torch.uint8))

    print(f"\n{'=' * 72}")
    print("  Bit-exact: CUDA kernel vs Python golden (per-token amax, RTN)")
    print(SEP)
    print(f"  FP4 data  bit-exact : {data_match}")
    print(f"  FP8 scale bit-exact : {scale_match}")
    if not data_match:
        nd = (cuda_res.data != ref_qx).sum().item()
        print(f"  -> data mismatches : {nd} / {cuda_res.data.numel()} ({100.*nd/cuda_res.data.numel():.4f}%)")
    if not scale_match:
        cu8, ru8 = cuda_res.scale.view(torch.uint8), ref_sx.view(torch.uint8)
        nd = (cu8 != ru8).sum().item()
        print(f"  -> scale mismatches: {nd} / {cu8.numel()} ({100.*nd/cu8.numel():.4f}%)")

    # tensor_scale cross-check (non-zero rows only)
    ts    = cuda_res.tensor_scale
    ref_ts = ref_am.squeeze(-1).to(torch.float32) / (448.0 * 6.0)
    valid  = ref_ts > 0
    ratio  = ts[valid] / ref_ts.to(ts.device)[valid]
    print(f"  tensor_scale ratio  : min={ratio.min():.6f}  max={ratio.max():.6f}"
          f"  (non-zero rows: {valid.sum().item()})")


def check_bit_exact_weight(x: torch.Tensor) -> None:
    res                     = quantize_weight_nvfp4_swizzled(x, four_six=False)
    ref_qx, ref_sx, ref_am  = quantize_nvfp4_ref_global_amax(x)

    data_match = torch.equal(res.data_uint8, ref_qx)

    n, k = res.data_uint8.shape[0], res.data_uint8.shape[1] * 2
    n_padded = ((n + 128 - 1) // 128) * 128
    scale_rm_f32 = _unswizzle_scale(res.block_scale_fp8.to(torch.float32), n_padded, k)[:n]
    scale_rm     = scale_rm_f32.to(torch.float8_e4m3fn)
    scale_match  = torch.equal(scale_rm.view(torch.uint8), ref_sx.view(torch.uint8))

    ref_gds = ref_am.to(torch.float32) / (448.0 * 6.0)   # global decode scale
    impl_gds = res.amax.to(torch.float32) / (448.0 * 6.0)
    gs_ratio = impl_gds / ref_gds

    print(f"\n{'=' * 72}")
    print("  Bit-exact: weight_nvfp4_swizzled vs Python golden (global amax, RTN)")
    print(SEP)
    print(f"  FP4 data  bit-exact : {data_match}")
    print(f"  FP8 scale bit-exact : {scale_match}")
    if not data_match:
        nd = (res.data_uint8 != ref_qx).sum().item()
        print(f"  -> data mismatches : {nd} / {res.data_uint8.numel()} ({100.*nd/res.data_uint8.numel():.4f}%)")
    if not scale_match:
        cu8, ru8 = scale_rm.view(torch.uint8), ref_sx.view(torch.uint8)
        nd = (cu8 != ru8).sum().item()
        print(f"  -> scale mismatches: {nd} / {cu8.numel()} ({100.*nd/cu8.numel():.4f}%)")
    print(f"  global_scale ratio  : {gs_ratio.item():.6f}")

# ===========================================================================

def main() -> None:
    x = load_real_input(DUMP_PATH)

    # 1. Bit-exact check: quantize_nvfp4_pertoken vs Python golden
    check_bit_exact(x)

    # 2. Bit-exact check: quantize_weight_nvfp4_swizzled vs Python golden
    check_bit_exact_weight(x)

    # 3. CUDA RTN vs 4/6
    run_quant_eval(x, {
        "CUDA RTN":         pertoken_rtn,
        "CUDA 4/6":         pertoken_46,
    }, title="Precision: CUDA RTN vs Round-4-or-6")

    # 4. rowmajor vs swizzled (should be identical in precision)
    run_quant_eval(x, {
        "rowmajor RTN":     pertoken_rtn,
        "swizzled RTN":     pertoken_swizzled_rtn,
    }, title="Precision: row-major vs swizzled layout")

    # 5. global amax vs per-token amax (Python ref)
    run_quant_eval(x, {
        "global amax (ref)":    ref_global_amax,
        "per-token amax (ref)": ref_pertoken_amax,
    }, title="Strategy: global amax vs per-token amax (Python ref)")

    # 6. weight (global amax) vs pertoken (per-token amax)
    run_quant_eval(x, {
        "weight RTN (global)":  weight_rtn,
        "pertoken RTN":         pertoken_rtn,
    }, title="Strategy: weight global amax vs per-token amax (CUDA)")

    # 7. weight RTN vs 4/6
    # NOTE: weight 4/6 (four_six_fp4 kernel with tensor-global amax) shows
    # severely degraded precision (eff-bits ~1.2) on this input — likely a
    # kernel limitation for the global-amax path. Kept here to document the behavior.
    run_quant_eval(x, {
        "weight RTN": weight_rtn,
        "weight 4/6": weight_46,
    }, title="Precision: weight RTN vs Round-4-or-6")

    print()


if __name__ == "__main__":
    main()
