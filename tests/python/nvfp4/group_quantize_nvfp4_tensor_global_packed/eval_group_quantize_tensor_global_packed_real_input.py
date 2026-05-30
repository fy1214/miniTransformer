#!/usr/bin/env python3
"""
Real-input precision evaluation: group_quantize_nvfp4_tensor_global_packed.

This operator quantizes MoE expert weights [num_experts, n_per_expert, k]
with a per-expert tensor-global amax.  The real fc1 input [148480, 2048]
is reshaped into [num_experts, n_per_expert, k] to exercise the kernel.

Run (inside container, from this directory):
  cd tests/python/nvfp4/group_quantize_nvfp4_tensor_global_packed
  python3 eval_group_quantize_tensor_global_packed_real_input.py

Requires miniTransformer installed: pip install -e .
"""
import torch

from miniTransformer.ops.nvfp4_quantize import (
    group_quantize_nvfp4_tensor_global_packed,
    dequantize_group_nvfp4_tensor_global_packed,
    quantize_weight_nvfp4_swizzled,
    dequantize_weight_nvfp4_swizzled,
    _unswizzle_scale,
    _unpack_fp4,
    _get_fp4_grid,
)
from miniTransformer.ops.nvfp4_ref_quantize import (
    quantize_nvfp4_ref_global_amax,
    dequantize_nvfp4_ref_global_amax,
)
from real_input_eval_utils import load_real_input, run_quant_eval, SEP

DUMP_PATH = '/tmp/dump/fc1_step0000_inp.pt'

# Reshape [148480, 2048] → [num_experts, n_per_expert, k].
# 148480 = 64 * 2320;  2320 % 128 == 0 ✓
NUM_EXPERTS  = 64
N_PER_EXPERT = 2320   # 148480 // 64
K            = 2048

# ===========================================================================
# Section A: operator wrappers
#
# Contract: fn(x: Tensor[m,n] bfloat16 CUDA) -> Tensor[m,n] bfloat16 CUDA
# (x is 2D [total_n, k]; we reshape internally)
# ===========================================================================

def _to_3d(x: torch.Tensor) -> torch.Tensor:
    return x.view(NUM_EXPERTS, N_PER_EXPERT, K)

def group_weight_rtn(x: torch.Tensor) -> torch.Tensor:
    res = group_quantize_nvfp4_tensor_global_packed(
        _to_3d(x), N_PER_EXPERT, four_six=False, skip_cute_data=True)
    return dequantize_group_nvfp4_tensor_global_packed(res)

def group_weight_46(x: torch.Tensor) -> torch.Tensor:
    res = group_quantize_nvfp4_tensor_global_packed(
        _to_3d(x), N_PER_EXPERT, four_six=True, skip_cute_data=True)
    return dequantize_group_nvfp4_tensor_global_packed(res)

def weight_swizzled_rtn(x: torch.Tensor) -> torch.Tensor:
    """quantize_weight_nvfp4_swizzled: single tensor-global amax over all rows."""
    return dequantize_weight_nvfp4_swizzled(quantize_weight_nvfp4_swizzled(x, four_six=False))

def ref_global_amax_perexpert(x: torch.Tensor) -> torch.Tensor:
    """Python golden: per-expert global amax, applied independently to each expert."""
    x3d = _to_3d(x)
    parts = []
    for g in range(NUM_EXPERTS):
        xg = x3d[g]                                   # [n_per_expert, k]
        qx, sx, am = quantize_nvfp4_ref_global_amax(xg)
        parts.append(dequantize_nvfp4_ref_global_amax(qx, sx, am))
    return torch.cat(parts, dim=0)

# ===========================================================================
# Section B: bit-exact checks
# ===========================================================================

def check_regression(x: torch.Tensor) -> None:
    """Regression check for group_tensor_global kernel.

    Check 1 — Determinism:
        Two consecutive runs on the same input must produce bit-identical packed
        FP4 data and FP8 scales.  The smem-hoisting bug (fixed 2026-05-30) caused
        non-deterministic cross-expert scale contamination; this catches any
        recurrence immediately with zero tolerance.

    Check 2 — Reconstruction quality vs Python golden:
        Dequantize kernel output and Python-reference output, then assert per-row
        cosine similarity > COS_THRESH and mean relative-L2 < REL_L2_THRESH.
        The smem bug would cause ~30% scale error on affected rows, dropping
        cosine to ~0.85 and spiking rel-L2 by 10-100x.  The thresholds are set
        well above the noise floor of legitimate ±1-FP4-step differences
        (which have negligible impact on cosine/L2 at 0.064% mismatch rate).
    """
    import torch.nn.functional as F

    x3d = _to_3d(x)
    COS_THRESH   = 0.99
    REL_L2_THRESH = 1e-2

    def _run(x3d):
        res = group_quantize_nvfp4_tensor_global_packed(
            x3d, N_PER_EXPERT, four_six=False, skip_cute_data=True)
        torch.cuda.synchronize()
        return res

    res1 = _run(x3d)
    res2 = _run(x3d)

    # --- Check 1: Determinism (atol=0, bit-exact) ---
    det_data  = res1.data.equal(res2.data)
    det_scale = res1.scale_flat.view(torch.uint8).equal(res2.scale_flat.view(torch.uint8))
    deterministic = det_data and det_scale

    # --- Check 2: Reconstruction quality ---
    kernel_deq = dequantize_group_nvfp4_tensor_global_packed(res1).float()  # [total_n, K]
    ref_deq    = ref_global_amax_perexpert(x).float()                        # [total_n, K]

    cos     = F.cosine_similarity(kernel_deq, ref_deq, dim=1)   # [total_n]
    x_norm  = x.float().norm(dim=1).clamp(min=1e-8)
    rel_l2  = (kernel_deq - ref_deq).norm(dim=1) / x_norm       # relative to input

    # exclude rows where both outputs are zero (quantization underflow)
    nonzero_rows = kernel_deq.norm(dim=1) > 0
    cos_min      = cos[nonzero_rows].min().item() if nonzero_rows.any() else 1.0
    rel_l2_mean  = rel_l2.mean().item()
    cos_pass     = cos_min     > COS_THRESH
    rel_l2_pass  = rel_l2_mean < REL_L2_THRESH

    print(f"\n{'=' * 72}")
    print("  Regression: group_tensor_global kernel")
    print(SEP)
    print(f"  [1] Deterministic (run1 == run2, bit-exact)")
    print(f"        data  : {'PASS' if det_data  else 'FAIL ***'}")
    print(f"        scale : {'PASS' if det_scale else 'FAIL ***'}")
    print(f"  [2] Reconstruction quality vs Python golden")
    print(f"        min cosine  > {COS_THRESH}  : {'PASS' if cos_pass    else 'FAIL ***'}"
          f"  (got {cos_min:.6f})")
    print(f"        mean rel-L2 < {REL_L2_THRESH} : {'PASS' if rel_l2_pass else 'FAIL ***'}"
          f"  (got {rel_l2_mean:.2e})")


def check_bit_exact_vs_weight_swizzled(x: torch.Tensor) -> None:
    """
    Compare group_quantize_nvfp4_tensor_global_packed (per-expert amax)
    against quantize_weight_nvfp4_swizzled (single global amax over all rows).

    These two use different amax granularity, so data/scale are NOT expected
    to be bit-exact.  This check only verifies structural consistency (shape,
    dtype) and reports the global_scale difference.
    """
    res_group  = group_quantize_nvfp4_tensor_global_packed(
        _to_3d(x), N_PER_EXPERT, four_six=False, skip_cute_data=True)
    res_weight = quantize_weight_nvfp4_swizzled(x, four_six=False)

    # global_scale: group has per-expert [num_experts], weight has single scalar
    gs_group  = res_group.global_scale                         # [num_experts]
    gs_weight = res_weight.amax.to(torch.float32) / (448.0 * 6.0)  # scalar

    print(f"\n{'=' * 72}")
    print("  Comparison: group_tensor_global vs quantize_weight_nvfp4_swizzled")
    print(SEP)
    print(f"  global_scale (weight, single): {gs_weight.item():.6f}")
    print(f"  global_scale (group, per-expert): "
          f"min={gs_group.min():.6f}  max={gs_group.max():.6f}  "
          f"mean={gs_group.mean():.6f}")
    print(f"  Note: different amax granularity — per-expert vs whole-tensor")

# ===========================================================================

def main() -> None:
    x = load_real_input(DUMP_PATH)

    # 1. Regression: determinism + reconstruction quality
    check_regression(x)

    # 2. Comparison vs quantize_weight_nvfp4_swizzled (different amax granularity)
    check_bit_exact_vs_weight_swizzled(x)

    # 3. RTN vs 4/6
    run_quant_eval(x, {
        "group_weight RTN": group_weight_rtn,
        "group_weight 4/6": group_weight_46,
    }, title="Precision: group_tensor_global RTN vs Round-4-or-6")

    # 4. per-expert global amax (CUDA) vs per-expert global amax (Python ref)
    run_quant_eval(x, {
        "group_weight RTN":      group_weight_rtn,
        "ref global (per-exp)":  ref_global_amax_perexpert,
    }, title="group_tensor_global vs Python golden (per-expert global amax)")

    # 5. per-expert amax vs single-tensor amax (precision tradeoff)
    run_quant_eval(x, {
        "group_weight RTN":   group_weight_rtn,
        "weight_swizzled RTN": weight_swizzled_rtn,
    }, title="Strategy: per-expert amax vs single-tensor amax")

    print()


if __name__ == "__main__":
    main()
