#!/usr/bin/env python3
"""
Real-input precision evaluation for the **fused-pad** NVFP4 grouped pertoken
quantize kernel (``fp4_gemm.rtn_fp4_group_pertoken_pad_fuse``).

Uses the same dump as the per-row sibling eval but exercises the real Megatron
flow: the dump contains the *unpadded* permuted activation
(``148480 × 2048``) plus the per-expert token counts
(``fc1_step0000_m_splits.json`` — 64 groups, 39 non-128-aligned, 4 empty).
The pad-fuse kernel takes these directly; the baseline emulates Megatron's
``Fp8Padding`` (zero buffer + ``index_copy_``) followed by per-row quantize.

Run inside the build container:
    cd /mnt/scfd/mingfa/python/miniTransformer
    PYTHONPATH=. python tests/python/nvfp4/group_quantize_nvfp4_pertoken_swizzled/eval_pad_fuse_real_input.py
"""
from __future__ import annotations

import json
from typing import List, Tuple

import torch

import fp4_gemm
from real_input_eval_utils import (
    SEP,
    compute_metrics,
    load_real_input,
    print_metrics_table,
    print_row_comparison,
)

from miniTransformer.ops.nvfp4_ref_quantize import (
    dequantize_nvfp4_ref_pertoken_amax,
    quantize_nvfp4_ref_pertoken_amax,
)

DUMP_PATH    = "/tmp/dump/fc1_step0000_inp.pt"
SPLITS_PATH  = "/tmp/dump/fc1_step0000_m_splits.json"

NVFP4_DATA   = torch.uint8
NVFP4_SCALE  = torch.float8_e4m3fn
PAD          = 128
_MMA_TILE_ROWS = PAD

_FP4_GRID = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


# ---------------------------------------------------------------------------
# Dequant helpers (mirror what the unit test uses)
# ---------------------------------------------------------------------------

def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    p = packed.to(torch.int32)
    return torch.stack([p & 0x0F, (p >> 4) & 0x0F], dim=-1).flatten(start_dim=-2)


def _unswizzle_scale(flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    nMT = rows // _MMA_TILE_ROWS
    nKT = cols // 64
    s = flat.reshape(nMT * nKT, 32, 4, 4)
    s = s.permute(0, 2, 1, 3).reshape(nMT, nKT, _MMA_TILE_ROWS, 4)
    return s.permute(0, 2, 1, 3).reshape(rows, nKT * 4)


def _dequant_real_rows(data, scale_flat, global_scale, m_unp, m_pad, K, grid):
    scale_rm = _unswizzle_scale(scale_flat.to(torch.float32), m_pad, K)[:m_unp]
    fp4_vals = grid[_unpack_fp4(data[:m_unp])].to(torch.float32)
    scale_e  = scale_rm.repeat_interleave(16, dim=1)
    gs       = global_scale[:m_unp].to(torch.float32).unsqueeze(1)
    return (fp4_vals * scale_e * gs).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Offset & layout helpers
# ---------------------------------------------------------------------------

def _build_offsets(
    m_unp: List[int], m_pad: List[int], K: int, device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unpad = [0]; pad = [0]; sf = [0]
    for mu, mp in zip(m_unp, m_pad):
        unpad.append(unpad[-1] + mu)
        pad  .append(pad[-1]   + mp)
        sf   .append(sf[-1]    + mp * (K // 16))
    return (
        torch.tensor(unpad, dtype=torch.int32, device=device),
        torch.tensor(pad,   dtype=torch.int32, device=device),
        torch.tensor(sf,    dtype=torch.int64, device=device),
    )


def _build_scatter_indices(m_unp: List[int], m_pad: List[int], device="cuda"):
    dst = []
    p_off = 0
    for mu, mp in zip(m_unp, m_pad):
        for i in range(mu):
            dst.append(p_off + i)
        p_off += mp
    return torch.tensor(dst, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Quant paths
# ---------------------------------------------------------------------------

def _baseline_pad_then_quant(x_unp, m_unp, m_pad, K, four_six=False, scale=1.0):
    device = x_unp.device
    _, pad_off, sf_off = _build_offsets(m_unp, m_pad, K, str(device))
    total_pad = sum(m_pad)
    dst       = _build_scatter_indices(m_unp, m_pad, str(device))
    x_pad     = torch.zeros(total_pad, K, dtype=torch.bfloat16, device=device)
    x_pad.index_copy_(0, dst, x_unp)
    sf_bytes  = total_pad * (K // 16)
    data = torch.empty((total_pad, K // 2), dtype=NVFP4_DATA, device=device)
    sf   = torch.zeros(sf_bytes,             dtype=NVFP4_SCALE, device=device)
    gs   = torch.empty(total_pad,            dtype=torch.float32, device=device)
    fn   = (fp4_gemm.four_six_fp4_group_pertoken if four_six
            else fp4_gemm.rtn_fp4_group_pertoken)
    fn(data, sf, gs, x_pad, pad_off, sf_off, scale)
    return data, sf, gs


def _run_pad_fuse(x_unp, m_unp, m_pad, K, four_six=False, scale=1.0):
    device = x_unp.device
    unpad_off, pad_off, sf_off = _build_offsets(m_unp, m_pad, K, str(device))
    total_pad = sum(m_pad)
    sf_bytes  = total_pad * (K // 16)
    data = torch.empty((total_pad, K // 2), dtype=NVFP4_DATA, device=device)
    sf   = torch.zeros(sf_bytes,             dtype=NVFP4_SCALE, device=device)
    gs   = torch.empty(total_pad,            dtype=torch.float32, device=device)
    fn   = (fp4_gemm.four_six_fp4_group_pertoken_pad_fuse if four_six
            else fp4_gemm.rtn_fp4_group_pertoken_pad_fuse)
    fn(data, sf, gs, x_unp, pad_off, unpad_off, sf_off, scale)
    return data, sf, gs


# ---------------------------------------------------------------------------
# Eval routines
# ---------------------------------------------------------------------------

def check_bit_exact_vs_baseline(x_unp, m_unp, m_pad, K):
    print(f"\n{'=' * 72}")
    print("  Bit-exact: pad-fuse vs (Fp8Padding mock + per-row kernel)")
    print(SEP)
    for tag, four_six in [("RTN", False), ("4/6", True)]:
        data_a, sf_a, gs_a = _baseline_pad_then_quant(x_unp, m_unp, m_pad, K, four_six=four_six)
        data_b, sf_b, gs_b = _run_pad_fuse        (x_unp, m_unp, m_pad, K, four_six=four_six)
        d_ok  = torch.equal(data_a, data_b)
        s_ok  = torch.equal(sf_a.view(torch.uint8), sf_b.view(torch.uint8))
        g_ok  = torch.equal(gs_a, gs_b)
        print(f"  [{tag}]  FP4 data ={'✓' if d_ok else '✗'}   "
              f"SF ={'✓' if s_ok else '✗'}   global_scale ={'✓' if g_ok else '✗'}")
        if not (d_ok and s_ok and g_ok):
            nd = (data_a != data_b).sum().item()
            ns = (sf_a.view(torch.uint8) != sf_b.view(torch.uint8)).sum().item()
            ng = (gs_a != gs_b).sum().item()
            print(f"        diff: data={nd}/{data_a.numel()}  "
                  f"sf={ns}/{sf_a.numel()}  gs={ng}/{gs_a.numel()}")


def check_pad_rows_zero(x_unp, m_unp, m_pad, K):
    """Pre-fill output buffers with 0xAB/0xCD junk; verify pad rows get overwritten with 0."""
    device     = x_unp.device
    unpad_off, pad_off_t, sf_off_t = _build_offsets(m_unp, m_pad, K, str(device))
    total_pad  = sum(m_pad)
    sf_bytes   = total_pad * (K // 16)
    data = torch.full((total_pad, K // 2), 0xAB, dtype=NVFP4_DATA, device=device)
    sf   = torch.full((sf_bytes,),         0xCD, dtype=torch.uint8, device=device).view(NVFP4_SCALE)
    gs   = torch.empty(total_pad, dtype=torch.float32, device=device)
    fp4_gemm.rtn_fp4_group_pertoken_pad_fuse(
        data, sf, gs, x_unp, pad_off_t, unpad_off, sf_off_t, 1.0)

    bad_data = bad_sf = total_pad_rows = 0
    pad_acc = sf_acc = 0
    for mu, mp in zip(m_unp, m_pad):
        if mu < mp:
            total_pad_rows += (mp - mu)
            pad_rows = data[pad_acc + mu:pad_acc + mp]
            bad_data += (pad_rows != 0).sum().item()
            sf_g  = sf[sf_acc:sf_acc + mp * (K // 16)]
            sf_rm = _unswizzle_scale(sf_g.view(torch.uint8).to(torch.int32), mp, K)
            bad_sf += (sf_rm[mu:mp] != 0).sum().item()
        pad_acc += mp
        sf_acc  += mp * (K // 16)
    print(f"\n{'=' * 72}")
    print(f"  Pad-row exact-zero check ({total_pad_rows} pad rows across all groups)")
    print(SEP)
    print(f"  Non-zero FP4 bytes in pad rows : {bad_data}  →  {'✓' if bad_data==0 else '✗'}")
    print(f"  Non-zero SF  bytes in pad rows : {bad_sf}   →  {'✓' if bad_sf==0   else '✗'}")


def precision_metrics_vs_ref(x_unp, m_unp, m_pad, K):
    """Dequant real rows; compare to (a) the same input via per-row reference path
    and (b) the Python golden per-token amax reference (full-precision FP4)."""
    device = x_unp.device
    grid   = _FP4_GRID.to(device)

    # Path B: fuse — dequant real rows
    data_b, sf_b, gs_b = _run_pad_fuse(x_unp, m_unp, m_pad, K)
    fuse_dq = torch.empty_like(x_unp)
    unp_off = pad_off = sf_off = 0
    for mu, mp in zip(m_unp, m_pad):
        if mu == 0:
            pad_off += mp; sf_off += mp * (K // 16); continue
        dq = _dequant_real_rows(
            data_b[pad_off:pad_off + mp],
            sf_b[sf_off:sf_off + mp * (K // 16)],
            gs_b[pad_off:pad_off + mp],
            mu, mp, K, grid,
        )
        fuse_dq[unp_off:unp_off + mu] = dq
        unp_off += mu; pad_off += mp; sf_off += mp * (K // 16)

    # Path R: Python golden (per-token amax on the unpadded tensor)
    ref_qx, ref_sx, ref_am = quantize_nvfp4_ref_pertoken_amax(x_unp)
    ref_dq = dequantize_nvfp4_ref_pertoken_amax(ref_qx, ref_sx, ref_am)

    print(f"\n{'=' * 72}")
    print("  Precision: pad-fuse (real rows) vs Python golden per-token amax")
    print(SEP)
    fuse_m = compute_metrics(x_unp, fuse_dq)
    ref_m  = compute_metrics(x_unp, ref_dq)
    print_metrics_table({"pad-fuse": fuse_m, "ref pertoken": ref_m})
    print()
    print_row_comparison(ref_m, fuse_m, ref_label="ref pertoken", cmp_label="pad-fuse")

    return fuse_m


def precision_per_group(x_unp, m_unp, m_pad, K):
    """Per-expert eff-bits + worst-case row to confirm no group degrades."""
    device  = x_unp.device
    grid    = _FP4_GRID.to(device)
    data_b, sf_b, gs_b = _run_pad_fuse(x_unp, m_unp, m_pad, K)
    print(f"\n{'=' * 72}")
    print("  Per-group eff-bits / min cosine (real rows only)")
    print(SEP)
    print(f"  {'g':>3}  {'m_unp':>6}  {'m_pad':>6}  {'eff-bits':>9}  "
          f"{'min-cos':>9}  {'mean-cos':>9}")
    unp_off = pad_off = sf_off = 0
    eff_list = []
    worst_min = 1.0
    worst_g   = -1
    for g, (mu, mp) in enumerate(zip(m_unp, m_pad)):
        if mu == 0:
            print(f"  {g:>3}  {mu:>6}  {mp:>6}      empty   --         --")
            pad_off += mp; sf_off += mp * (K // 16); continue
        x_g  = x_unp[unp_off:unp_off + mu]
        dq   = _dequant_real_rows(
            data_b[pad_off:pad_off + mp],
            sf_b[sf_off:sf_off + mp * (K // 16)],
            gs_b[pad_off:pad_off + mp],
            mu, mp, K, grid,
        )
        m  = compute_metrics(x_g, dq)
        eff_list.append(m["eff_bits"])
        if m["cos_min"] < worst_min:
            worst_min = m["cos_min"]; worst_g = g
        print(f"  {g:>3}  {mu:>6}  {mp:>6}  {m['eff_bits']:>9.4f}  "
              f"{m['cos_min']:>9.6f}  {m['cos_mean']:>9.6f}")
        unp_off += mu; pad_off += mp; sf_off += mp * (K // 16)
    print(SEP)
    print(f"  mean eff-bits across non-empty groups : {sum(eff_list)/len(eff_list):.4f}")
    print(f"  worst min-cos : {worst_min:.6f}  (group {worst_g})")


# ---------------------------------------------------------------------------

def main():
    x_unp = load_real_input(DUMP_PATH)
    with open(SPLITS_PATH) as f:
        m_unp = json.load(f)

    assert sum(m_unp) == x_unp.shape[0], (
        f"m_splits sum {sum(m_unp)} != x rows {x_unp.shape[0]}"
    )
    m_pad = [((m + PAD - 1) // PAD) * PAD for m in m_unp]
    K = x_unp.shape[1]

    print(f"\nGroups: {len(m_unp)}")
    print(f"Empty groups   : {sum(1 for m in m_unp if m == 0)}")
    print(f"Non-128 groups : {sum(1 for m in m_unp if m % 128 != 0 and m > 0)}")
    print(f"sum(m_unp)     : {sum(m_unp)}")
    print(f"sum(m_pad)     : {sum(m_pad)}  "
          f"(overhead +{sum(m_pad)-sum(m_unp)} rows, "
          f"{100.*(sum(m_pad)-sum(m_unp))/sum(m_unp):.2f}%)")

    # 1. Bit-exact vs baseline (RTN + 4/6)
    check_bit_exact_vs_baseline(x_unp, m_unp, m_pad, K)

    # 2. Pad-row exact-zero (kernel must overwrite 0xAB/0xCD junk with zeros)
    check_pad_rows_zero(x_unp, m_unp, m_pad, K)

    # 3. Aggregate precision vs Python golden
    precision_metrics_vs_ref(x_unp, m_unp, m_pad, K)

    # 4. Per-group eff-bits / worst-case cosine
    precision_per_group(x_unp, m_unp, m_pad, K)
    print()


if __name__ == "__main__":
    main()
