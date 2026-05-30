#!/usr/bin/env python3
"""
Real-input precision evaluation: group_quantize_nvfp4_pertoken_swizzled.

Run (inside container, from this directory):
  cd tests/python/nvfp4/group_quantize_nvfp4_pertoken_swizzled
  python3 eval_group_quantize_pertoken_swizzled_real_input.py

Requires miniTransformer installed: pip install -e .
"""
import torch

from miniTransformer.ops.nvfp4_quantize import (
    group_quantize_nvfp4_pertoken_swizzled,
    dequantize_group_nvfp4_pertoken_swizzled,
    quantize_nvfp4_pertoken,
    _unswizzle_scale,
)
from miniTransformer.ops.nvfp4_ref_quantize import (
    quantize_nvfp4_ref_pertoken_amax,
    dequantize_nvfp4_ref_pertoken_amax,
)
from real_input_eval_utils import load_real_input, run_quant_eval, SEP

DUMP_PATH = '/tmp/dump/fc1_step0000_inp.pt'

# Number of equal groups used for the multi-group consistency check.
# 148480 / 8 = 18560, 18560 % 128 == 0 — evenly divisible.
N_GROUPS = 8

# ===========================================================================
# Section A: operator wrappers  ← only this section changes per operator
#
# Contract: fn(x: Tensor[m,n] bfloat16 CUDA) -> Tensor[m,n] bfloat16 CUDA
# ===========================================================================

def group_rtn(x: torch.Tensor) -> torch.Tensor:
    res = group_quantize_nvfp4_pertoken_swizzled(x, [x.shape[0]], four_six=False)
    return dequantize_group_nvfp4_pertoken_swizzled(res)

def group_46(x: torch.Tensor) -> torch.Tensor:
    res = group_quantize_nvfp4_pertoken_swizzled(x, [x.shape[0]], four_six=True)
    return dequantize_group_nvfp4_pertoken_swizzled(res)

def ref_pertoken_amax(x: torch.Tensor) -> torch.Tensor:
    qx, sx, am = quantize_nvfp4_ref_pertoken_amax(x)
    return dequantize_nvfp4_ref_pertoken_amax(qx, sx, am)

# ===========================================================================
# Section B: bit-exact checks vs golden and multi-group consistency
# ===========================================================================

def _make_equal_splits(m: int, n_groups: int) -> list:
    """Split m rows into n_groups equal parts (each multiple of 128)."""
    m_per = m // n_groups
    assert m_per % 128 == 0, (
        f"m_per={m_per} is not a multiple of 128 — adjust N_GROUPS"
    )
    return [m_per] * (n_groups - 1) + [m - m_per * (n_groups - 1)]


def check_bit_exact(x: torch.Tensor) -> None:
    m, k = x.shape

    # Single-group result
    res = group_quantize_nvfp4_pertoken_swizzled(x, [m], four_six=False)

    # Python golden
    ref_qx, _ref_sx, ref_am = quantize_nvfp4_ref_pertoken_amax(x)

    data_match = torch.equal(res.data, ref_qx)

    # Unswizzle scale_flat back to row-major [m, k//16] for comparison.
    # _unswizzle_scale operates on float32; cast back to float8 before byte-compare.
    m_padded = ((m + 127) // 128) * 128
    scale_rm_f32 = _unswizzle_scale(res.scale_flat.to(torch.float32), m_padded, k)[:m]
    scale_rm = scale_rm_f32.to(torch.float8_e4m3fn)
    scale_match = torch.equal(scale_rm.view(torch.uint8), _ref_sx.view(torch.uint8))

    ref_ts = ref_am.squeeze(-1).to(torch.float32) / (448.0 * 6.0)
    valid  = ref_ts > 0
    ratio  = res.global_scale[valid] / ref_ts.to(res.global_scale.device)[valid]

    print(f"\n{'=' * 72}")
    print("  Bit-exact: single group vs Python golden (per-token amax, RTN)")
    print(SEP)
    print(f"  FP4 data  bit-exact : {data_match}")
    if not data_match:
        nd = (res.data != ref_qx).sum().item()
        print(f"  -> data mismatches : {nd} / {res.data.numel()} "
              f"({100.*nd/res.data.numel():.4f}%)")
    print(f"  FP8 scale bit-exact : {scale_match}")
    if not scale_match:
        cu8, ru8 = scale_rm.view(torch.uint8), _ref_sx.view(torch.uint8)
        nd = (cu8 != ru8).sum().item()
        print(f"  -> scale mismatches: {nd} / {cu8.numel()} "
              f"({100.*nd/cu8.numel():.4f}%)")
    print(f"  global_scale ratio  : min={ratio.min():.6f}  max={ratio.max():.6f}"
          f"  (non-zero rows: {valid.sum().item()})")

    # Multi-group consistency check
    splits    = _make_equal_splits(m, N_GROUPS)
    res_multi = group_quantize_nvfp4_pertoken_swizzled(x, splits, four_six=False)

    data_multi_match = torch.equal(res.data, res_multi.data)
    gs_multi_match   = torch.equal(res.global_scale, res_multi.global_scale)

    print(f"\n{'=' * 72}")
    print(f"  Multi-group consistency ({N_GROUPS} equal groups vs single group)")
    print(SEP)
    print(f"  FP4 data   bit-exact   : {data_multi_match}")
    print(f"  global_scale bit-exact : {gs_multi_match}")
    if not data_multi_match:
        nd = (res.data != res_multi.data).sum().item()
        print(f"  -> data mismatches : {nd} / {res.data.numel()} "
              f"({100.*nd/res.data.numel():.4f}%)")


def check_bit_exact_vs_pertoken(x: torch.Tensor) -> None:
    m, k = x.shape

    res    = group_quantize_nvfp4_pertoken_swizzled(x, [m], four_six=False)
    res_pt = quantize_nvfp4_pertoken(x, four_six=False)

    data_pt_match = torch.equal(res.data, res_pt.data)
    gs_pt_match   = torch.equal(res.global_scale, res_pt.tensor_scale)

    m_padded     = ((m + 127) // 128) * 128
    scale_rm_f32 = _unswizzle_scale(res.scale_flat.to(torch.float32), m_padded, k)[:m]
    scale_rm     = scale_rm_f32.to(torch.float8_e4m3fn)
    scale_pt_match = torch.equal(scale_rm.view(torch.uint8), res_pt.scale.view(torch.uint8))

    print(f"\n{'=' * 72}")
    print("  Bit-exact: group (single) vs quantize_nvfp4_pertoken (row-major)")
    print(SEP)
    print(f"  FP4 data   bit-exact   : {data_pt_match}")
    print(f"  FP8 scale  bit-exact   : {scale_pt_match}")
    print(f"  global_scale bit-exact : {gs_pt_match}")
    if not data_pt_match:
        nd = (res.data != res_pt.data).sum().item()
        print(f"  -> data mismatches : {nd} / {res.data.numel()} "
              f"({100.*nd/res.data.numel():.4f}%)")

# ===========================================================================

def main() -> None:
    x = load_real_input(DUMP_PATH)

    # 1. Bit-exact vs golden + multi-group consistency
    check_bit_exact(x)

    # 2. Bit-exact vs quantize_nvfp4_pertoken (row-major sibling)
    check_bit_exact_vs_pertoken(x)

    # 3. RTN vs 4/6
    run_quant_eval(x, {
        "group RTN": group_rtn,
        "group 4/6": group_46,
    }, title="Precision: group RTN vs Round-4-or-6")

    # 3. group vs Python golden (per-token amax ref)
    run_quant_eval(x, {
        "group RTN":    group_rtn,
        "ref pertoken": ref_pertoken_amax,
    }, title="group_quantize vs Python golden (per-token amax)")

    print()


if __name__ == "__main__":
    main()
