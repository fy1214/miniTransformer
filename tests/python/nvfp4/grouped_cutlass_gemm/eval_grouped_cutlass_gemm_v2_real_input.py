#!/usr/bin/env python3
"""
Real-input precision evaluation: grouped_cutlass_gemm_v2.

Data layout in /tmp/dump/:
  fc1_step0000_inp.pt          [148480, 2048]  bf16  activations
  fc1_step0000_weight_NNN.pt   [  1536, 2048]  bf16  per-expert weight (000-063)
  fc1_step0000_m_splits.json   list[64]              rows per expert
  fc1_step0000_output.pt       [148480, 1536]  bf16  reference output

Three comparisons are reported:
  1. kernel vs Python golden  — implementation correctness
  2. kernel vs reference out  — end-to-end FP4 approximation error
  3. golden vs reference out  — FP4 quantisation error vs reference

Run (inside container, from this directory or repo root):
  export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH
  python3 tests/python/nvfp4/grouped_cutlass_gemm/eval_grouped_cutlass_gemm_v2_real_input.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import List

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from miniTransformer.quantiztion import quantization
from miniTransformer.quantiztion.nvfp4_quantize_op_ref import Nvfp4TiledQuantizeRefOp
from miniTransformer.ops.nvfp4_grouped_cutlass_gemm import V2GemmInputs, grouped_cutlass_gemm_v2
from miniTransformer.ops.nvfp4_qgemm import swizzle_scale
from miniTransformer.ops.nvfp4_quantize import _get_fp4_grid, _unpack_fp4, _unswizzle_scale

DUMP_PATH   = "/tmp/dump"
_NVFP4_FACTOR = 448.0 * 6.0

_QPARAMS = quantization.QParams(
    scaling_dtype=torch.float8_e4m3fn,
    output_dtype=torch.uint8,
    eps=0.0,
    pow_2_scales=False,
    quant_tile_shape=(1, 16),
    with_2d_quantization=False,
    with_rht=False,
)

SEP = "-" * 72


# ---------------------------------------------------------------------------
# Python golden (same logic as test file)
# ---------------------------------------------------------------------------

def _deq_fp4_block_scaled(
    data:         torch.Tensor,
    sf_flat:      torch.Tensor,
    global_scale: torch.Tensor,
    m:            int,
    k:            int,
) -> torch.Tensor:
    m_pad    = ((m + 127) // 128) * 128
    grid     = _get_fp4_grid(data.device)
    fp4_vals = grid[_unpack_fp4(data)].float()
    sf       = _unswizzle_scale(sf_flat.float(), m_pad, k)[:m]
    if global_scale.dim() == 1:
        gs = global_scale.float().unsqueeze(1)
    else:
        gs = global_scale.float()
    return (fp4_vals.view(m, k // 16, 16) * (sf * gs).unsqueeze(-1)).view(m, k)


def grouped_cutlass_gemm_v2_golden(
    inputs:    V2GemmInputs,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    ng   = len(inputs.m_splits)
    prob = inputs.prob_host.view(ng, 3)
    parts = []
    for g in range(ng):
        m_g = int(prob[g, 0])
        n_g = int(prob[g, 1])
        k_g = int(prob[g, 2])
        ar_s = int(inputs.a_rows[g]);       ar_e = int(inputs.a_rows[g + 1])
        br_s = int(inputs.b_rows[g]);       br_e = int(inputs.b_rows[g + 1])
        as_s = int(inputs.a_sf_offsets[g]); as_e = int(inputs.a_sf_offsets[g + 1])
        bs_s = int(inputs.b_sf_offsets[g]); bs_e = int(inputs.b_sf_offsets[g + 1])
        a_f = _deq_fp4_block_scaled(
            inputs.act_data[ar_s:ar_e], inputs.act_sf[as_s:as_e],
            inputs.act_ts[ar_s:ar_e], m_g, k_g,
        )
        b_f = _deq_fp4_block_scaled(
            inputs.wgt_data[br_s:br_e], inputs.wgt_sf[bs_s:bs_e],
            inputs.wgt_ts[g], n_g, k_g,
        )
        parts.append(torch.mm(a_f, b_f.t()).to(out_dtype))
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def error_metrics(pred: torch.Tensor, ref: torch.Tensor) -> dict:
    p, r  = pred.float(), ref.float()
    diff  = (p - r).abs()
    rmse  = torch.sqrt(torch.mean((p - r) ** 2))
    denom = torch.sqrt(torch.mean(r ** 2)).clamp_min(1e-8)
    cos   = torch.nn.functional.cosine_similarity(
        p.flatten().unsqueeze(0), r.flatten().unsqueeze(0)
    ).item()
    return {
        "max_abs":   float(diff.max()),
        "mean_abs":  float(diff.mean()),
        "norm_rmse": float(rmse / denom),
        "cosine":    cos,
    }


def print_metrics(label: str, m: dict) -> None:
    print(f"  {label}")
    print(f"    max_abs   = {m['max_abs']:.4f}")
    print(f"    mean_abs  = {m['mean_abs']:.4f}")
    print(f"    norm_rmse = {m['norm_rmse']:.4f}")
    print(f"    cosine    = {m['cosine']:.6f}")


# ---------------------------------------------------------------------------
# Quantise one expert
# ---------------------------------------------------------------------------

def _quantise_expert(
    x: torch.Tensor,   # [m, k] fp32
    op: Nvfp4TiledQuantizeRefOp,
) -> tuple:
    """Return (packed_data, swizzled_sf_flat, ts_scalar)."""
    q   = op.quantize(x, _QPARAMS, return_transpose=False)
    ts  = q.tensor_scale if q.tensor_scale is not None else x.abs().max().view(1)
    if ts.numel() > 1:
        ts = ts.max().view(1)
    ts = ts.to(torch.float32).to(x.device)
    sf_flat = swizzle_scale(q.scale.float(), x.shape[0], x.shape[1]) \
                  .to(torch.float8_e4m3fn).reshape(-1)
    return q.data, sf_flat, ts


# ---------------------------------------------------------------------------
# Build V2GemmInputs from real data
# ---------------------------------------------------------------------------

def build_v2_inputs(
    inp:      torch.Tensor,        # [total_m, k] float32 CUDA
    weights:  List[torch.Tensor],  # ng × [n, k] float32 CUDA
    m_splits: List[int],
    device:   str,
) -> V2GemmInputs:
    ng = len(m_splits)
    n, k = weights[0].shape

    op = Nvfp4TiledQuantizeRefOp()

    act_data_list, wgt_data_list = [], []
    act_sf_list,   wgt_sf_list   = [], []
    act_ts_list,   wgt_ts_list   = [], []

    k = inp.shape[1]
    row = 0
    for g, m_g in enumerate(m_splits):
        w_g = weights[g].float()
        b_data, b_sf, b_ts = _quantise_expert(w_g, op)
        wgt_ts_g = (b_ts / _NVFP4_FACTOR).squeeze()
        wgt_data_list.append(b_data)
        wgt_sf_list.append(b_sf)
        wgt_ts_list.append(wgt_ts_g.unsqueeze(0))

        if m_g == 0:
            act_data_list.append(torch.empty((0, k // 2), dtype=torch.uint8,  device=device))
            act_sf_list.append(  torch.empty((0,),        dtype=torch.float8_e4m3fn, device=device))
            act_ts_list.append(  torch.empty((0,),        dtype=torch.float32, device=device))
        else:
            x_g = inp[row:row + m_g].float()
            a_data, a_sf, a_ts = _quantise_expert(x_g, op)
            act_ts_g = torch.full((m_g,), a_ts.item() / _NVFP4_FACTOR,
                                  dtype=torch.float32, device=device)
            act_data_list.append(a_data)
            act_sf_list.append(a_sf)
            act_ts_list.append(act_ts_g)
        row += m_g

    act_data = torch.cat(act_data_list).contiguous()
    wgt_data = torch.cat(wgt_data_list).contiguous()
    act_sf   = torch.cat(act_sf_list).contiguous()
    wgt_sf   = torch.cat(wgt_sf_list).contiguous()
    act_ts   = torch.cat(act_ts_list).contiguous()
    wgt_ts   = torch.cat(wgt_ts_list).contiguous()

    a_rows    = torch.zeros(ng + 1, dtype=torch.int32,  device=device)
    b_rows    = torch.zeros(ng + 1, dtype=torch.int32,  device=device)
    a_sf_offs = torch.zeros(ng + 1, dtype=torch.int64,  device=device)
    b_sf_offs = torch.zeros(ng + 1, dtype=torch.int64,  device=device)

    for g, (m_g, a_sf_g, b_sf_g) in enumerate(zip(m_splits, act_sf_list, wgt_sf_list)):
        a_rows[g + 1]    = a_rows[g]    + m_g
        b_rows[g + 1]    = b_rows[g]    + n
        a_sf_offs[g + 1] = a_sf_offs[g] + a_sf_g.numel()
        b_sf_offs[g + 1] = b_sf_offs[g] + b_sf_g.numel()

    prob_host = torch.zeros(ng * 3, dtype=torch.int32)
    for g, m_g in enumerate(m_splits):
        prob_host[g * 3 + 0] = m_g
        prob_host[g * 3 + 1] = n
        prob_host[g * 3 + 2] = k

    return V2GemmInputs(
        act_data=act_data, act_sf=act_sf, act_ts=act_ts,
        a_rows=a_rows, a_sf_offsets=a_sf_offs,
        wgt_data=wgt_data, wgt_sf=wgt_sf, wgt_ts=wgt_ts,
        b_rows=b_rows, b_sf_offsets=b_sf_offs,
        prob_host=prob_host, m_splits=m_splits,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = "cuda"

    print("Loading data ...", flush=True)
    inp     = torch.load(f"{DUMP_PATH}/fc1_step0000_inp.pt",    map_location=device)
    ref_out = torch.load(f"{DUMP_PATH}/fc1_step0000_output.pt", map_location=device)
    with open(f"{DUMP_PATH}/fc1_step0000_m_splits.json") as f:
        m_splits = json.load(f)

    ng = len(m_splits)   # 64
    n  = 1536
    print(f"  experts={ng}  n={n}  k={inp.shape[1]}  total_m={inp.shape[0]}")

    weights = [
        torch.load(f"{DUMP_PATH}/fc1_step0000_weight_{g:03d}.pt",
                   map_location=device)
        for g in range(ng)
    ]

    print("Quantising and building V2GemmInputs ...", flush=True)
    inputs = build_v2_inputs(inp, weights, m_splits, device)
    torch.cuda.synchronize()

    print("Running grouped_cutlass_gemm_v2 ...", flush=True)
    kernel_out = grouped_cutlass_gemm_v2(inputs, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()

    print("Running Python golden ...", flush=True)
    golden_out = grouped_cutlass_gemm_v2_golden(inputs, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()

    print(f"\n{'=' * 72}")
    print("  grouped_cutlass_gemm_v2  —  real fc1 input (step 0000)")
    print(f"  shape: [{inp.shape[0]}, {n}]  (64 experts)")
    print(SEP)
    print_metrics("[1] kernel  vs  Python golden  (implementation correctness)", error_metrics(kernel_out, golden_out))
    print(SEP)
    print_metrics("[2] kernel  vs  reference out  (end-to-end FP4 error)", error_metrics(kernel_out, ref_out.to(device)))
    print(SEP)
    print_metrics("[3] golden  vs  reference out  (FP4 quant error vs ref)", error_metrics(golden_out, ref_out.to(device)))
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
