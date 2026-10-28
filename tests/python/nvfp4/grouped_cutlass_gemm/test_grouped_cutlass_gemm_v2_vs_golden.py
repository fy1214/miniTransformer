"""CUTLASS grouped NVFP4 GEMM v2 vs Python golden (QAT 8 / 12 / 14).

What: ``grouped_cutlass_gemm_v2`` (contiguous packed A/B + swizzled SF).
Where: ``nvfp4_cursor2_op_cutlass`` / ``_cutlass_backward`` / shared by v2 EVT path.
Compare: dequant-then-float32 GEMM golden; also FP32 ground-truth bounds.

Markers: ``production``. Requires fp4_gemm, CUDA sm_100+.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import pytest
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

pytest.importorskip("fp4_gemm")

pytestmark = pytest.mark.production

from miniTransformer.quantiztion import quantization
from miniTransformer.quantiztion.nvfp4_quantize_op_ref import Nvfp4TiledQuantizeRefOp
from miniTransformer.ops.nvfp4_grouped_cutlass_gemm import (
    V2GemmInputs,
    grouped_cutlass_gemm_v2,
)
from miniTransformer.ops.nvfp4_qgemm import swizzle_scale
from miniTransformer.ops.nvfp4_quantize import _get_fp4_grid, _unpack_fp4, _unswizzle_scale

_QPARAMS = quantization.QParams(
    scaling_dtype=torch.float8_e4m3fn,
    output_dtype=torch.uint8,
    eps=0.0,
    pow_2_scales=False,
    quant_tile_shape=(1, 16),
    with_2d_quantization=False,
    with_rht=False,
)

_NVFP4_FACTOR = 448.0 * 6.0   # normalisation denominator for global scales


# ---------------------------------------------------------------------------
# Python golden reference
# ---------------------------------------------------------------------------

def _deq_fp4_block_scaled(
    data:         torch.Tensor,               # [m, k//2]  uint8, packed FP4
    sf_flat:      torch.Tensor,               # flat FP8 swizzled, len = m_pad*k//16
    global_scale: torch.Tensor,               # [m] float32  OR  0-d scalar
    m:            int,
    k:            int,
) -> torch.Tensor:                            # [m, k]  float32
    """Dequantize one contiguous block of packed FP4 rows.

    value[i, j] = fp4[i, j]  ×  sf[i, j//16]  ×  global_scale[i]

    sf is stored in tcgen05 swizzled layout; rows are padded to the next
    multiple of 128 for unswizzling, then sliced back to [m].
    """
    m_pad = ((m + 127) // 128) * 128
    grid  = _get_fp4_grid(data.device)

    fp4_vals = grid[_unpack_fp4(data)].float()                      # [m, k]
    sf       = _unswizzle_scale(sf_flat.float(), m_pad, k)[:m]      # [m, k//16]

    if global_scale.dim() == 1:
        gs = global_scale.float().unsqueeze(1)   # [m, 1]  — per-row broadcast
    else:
        gs = global_scale.float()                # 0-d     — scalar broadcast

    return (fp4_vals.view(m, k // 16, 16) * (sf * gs).unsqueeze(-1)).view(m, k)


def grouped_cutlass_gemm_v2_golden(
    inputs:    V2GemmInputs,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Pure-Python golden for grouped_cutlass_gemm_v2.

    For each group g with problem (m_g, n_g, k_g):
        A_f = deq(act_data[g], act_sf[g], act_ts[g])   # [m_g, k_g] float32
        B_f = deq(wgt_data[g], wgt_sf[g], wgt_ts[g])   # [n_g, k_g] float32
        out_g = A_f @ B_f.T                             # [m_g, n_g]

    Numerically equivalent to the kernel's fused computation because
    the per-token effective scale acts[ts] * wgt_ts[g] is a scalar per row and
    can be moved outside the inner product.
    """
    ng   = len(inputs.m_splits)
    prob = inputs.prob_host.view(ng, 3)    # [ng, (m_g, n_g, k_g)]
    parts: List[torch.Tensor] = []

    for g in range(ng):
        m_g = int(prob[g, 0])
        n_g = int(prob[g, 1])
        k_g = int(prob[g, 2])

        ar_s = int(inputs.a_rows[g]);       ar_e = int(inputs.a_rows[g + 1])
        br_s = int(inputs.b_rows[g]);       br_e = int(inputs.b_rows[g + 1])
        as_s = int(inputs.a_sf_offsets[g]); as_e = int(inputs.a_sf_offsets[g + 1])
        bs_s = int(inputs.b_sf_offsets[g]); bs_e = int(inputs.b_sf_offsets[g + 1])

        a_f = _deq_fp4_block_scaled(
            inputs.act_data[ar_s:ar_e],
            inputs.act_sf[as_s:as_e],
            inputs.act_ts[ar_s:ar_e],   # [m_g] per-row global scale
            m_g, k_g,
        )

        b_f = _deq_fp4_block_scaled(
            inputs.wgt_data[br_s:br_e],
            inputs.wgt_sf[bs_s:bs_e],
            inputs.wgt_ts[g],           # 0-d per-expert global scale
            n_g, k_g,
        )

        parts.append(torch.mm(a_f, b_f.t()).to(out_dtype))   # [m_g, n_g]

    return torch.cat(parts, dim=0)   # [total_m, n]


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    prop = torch.cuda.get_device_properties(0)
    if prop.major < 10:
        pytest.skip(f"Requires SM100 (Blackwell); got SM{prop.major}{prop.minor}")


def _error_metrics(pred: torch.Tensor, ref: torch.Tensor) -> dict:
    p, r  = pred.float(), ref.float()
    diff  = (p - r).abs()
    rmse  = torch.sqrt(torch.mean((p - r) ** 2))
    norm_rmse = rmse / torch.sqrt(torch.mean(r ** 2)).clamp_min(1e-8)
    cos   = torch.nn.functional.cosine_similarity(
        p.flatten().unsqueeze(0), r.flatten().unsqueeze(0)
    ).item()
    return {
        "max_abs":   float(diff.max()),
        "mean_abs":  float(diff.mean()),
        "rmse":      float(rmse),
        "norm_rmse": float(norm_rmse),
        "cosine":    cos,
    }


def _build_one_group(
    m: int, n: int, k: int, device: str, seed: int
) -> Tuple[
    torch.Tensor, torch.Tensor,   # x_fp32, w_fp32
    torch.Tensor, torch.Tensor,   # act_data [m,k//2], wgt_data [n,k//2]
    torch.Tensor, torch.Tensor,   # act_sf_flat, wgt_sf_flat  (swizzled FP8)
    torch.Tensor,                 # act_ts [m] float32  (per-row global scale)
    torch.Tensor,                 # wgt_ts scalar float32
]:
    """Quantize one (m×k)×(n×k) pair and return packed V2 tensors."""
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    x_fp32 = torch.randn(m, k, device=device, dtype=torch.float32)
    w_fp32 = torch.empty(n, k, device=device, dtype=torch.float32)
    torch.nn.init.normal_(w_fp32, std=0.023)

    op = Nvfp4TiledQuantizeRefOp()
    qx = op.quantize(x_fp32, _QPARAMS, return_transpose=False)
    qw = op.quantize(w_fp32, _QPARAMS, return_transpose=False)

    # Global amax tensors — normalise to scalar [1] on CUDA
    def _scalar_amax(res, fallback):
        ts = res.tensor_scale if res.tensor_scale is not None else fallback
        if ts.numel() > 1:
            ts = ts.max().view(1)
        return ts.to(torch.float32).to(device)

    amax_x = _scalar_amax(qx, x_fp32.abs().max().view(1))
    amax_w = _scalar_amax(qw, w_fp32.abs().max().view(1))

    # Swizzled 4D scale factors → flattened for V2GemmInputs
    sx_flat = swizzle_scale(qx.scale.float(), m, k).to(torch.float8_e4m3fn).reshape(-1)
    sw_flat = swizzle_scale(qw.scale.float(), n, k).to(torch.float8_e4m3fn).reshape(-1)

    # Per-row activation scale: each row gets the same global decode scale.
    # Multiply by a random per-token factor (in [0.5, 1.5]) to exercise non-trivial pts.
    torch.manual_seed(seed + 1000)
    pts = torch.rand(m, device=device, dtype=torch.float32) * 1.0 + 0.5
    act_ts = pts * amax_x.item() / _NVFP4_FACTOR     # [m] float32
    wgt_ts = amax_w / _NVFP4_FACTOR                   # [1] float32

    # pts-scaled x_fp32: the kernel output approximates (x_fp32 * pts) @ w_fp32.T,
    # so use this as the fp32 ground truth for absolute-precision comparison.
    x_fp32_pts = x_fp32 * pts.unsqueeze(1)

    return (
        x_fp32_pts, w_fp32,
        qx.data, qw.data,
        sx_flat, sw_flat,
        act_ts, wgt_ts.squeeze(),
    )


def _build_v2_inputs(m_splits: List[int], n: int, k: int, device: str, seed: int) -> Tuple[V2GemmInputs, torch.Tensor, torch.Tensor]:
    """Build V2GemmInputs for a grouped problem plus fp32 data for ground-truth."""
    ng = len(m_splits)

    act_data_list, wgt_data_list = [], []
    act_sf_list,   wgt_sf_list   = [], []
    act_ts_list,   wgt_ts_list   = [], []
    x_fp32_list,   w_fp32_list   = [], []

    for g, m in enumerate(m_splits):
        x_fp32, w_fp32, act_d, wgt_d, act_sf, wgt_sf, act_ts, wgt_ts = \
            _build_one_group(m, n, k, device, seed + g)

        act_data_list.append(act_d)
        wgt_data_list.append(wgt_d)
        act_sf_list.append(act_sf)
        wgt_sf_list.append(wgt_sf)
        act_ts_list.append(act_ts)
        wgt_ts_list.append(wgt_ts.unsqueeze(0))
        x_fp32_list.append(x_fp32)
        w_fp32_list.append(w_fp32)

    act_data = torch.cat(act_data_list, dim=0).contiguous()   # [total_m, k//2]
    wgt_data = torch.cat(wgt_data_list, dim=0).contiguous()   # [ng*n, k//2]
    act_sf   = torch.cat(act_sf_list).contiguous()
    wgt_sf   = torch.cat(wgt_sf_list).contiguous()
    act_ts   = torch.cat(act_ts_list).contiguous()            # [total_m]
    wgt_ts   = torch.cat(wgt_ts_list).contiguous()            # [ng]

    a_rows    = torch.zeros(ng + 1, dtype=torch.int32,  device=device)
    b_rows    = torch.zeros(ng + 1, dtype=torch.int32,  device=device)
    a_sf_offs = torch.zeros(ng + 1, dtype=torch.int64,  device=device)
    b_sf_offs = torch.zeros(ng + 1, dtype=torch.int64,  device=device)

    for g, (m, act_sf_g, wgt_sf_g) in enumerate(zip(m_splits, act_sf_list, wgt_sf_list)):
        a_rows[g + 1]    = a_rows[g]    + m
        b_rows[g + 1]    = b_rows[g]    + n
        a_sf_offs[g + 1] = a_sf_offs[g] + act_sf_g.numel()
        b_sf_offs[g + 1] = b_sf_offs[g] + wgt_sf_g.numel()

    prob_host = torch.zeros(ng * 3, dtype=torch.int32)
    for g, m in enumerate(m_splits):
        prob_host[g * 3 + 0] = m
        prob_host[g * 3 + 1] = n
        prob_host[g * 3 + 2] = k

    inputs = V2GemmInputs(
        act_data=act_data, act_sf=act_sf, act_ts=act_ts,
        a_rows=a_rows, a_sf_offsets=a_sf_offs,
        wgt_data=wgt_data, wgt_sf=wgt_sf, wgt_ts=wgt_ts,
        b_rows=b_rows, b_sf_offsets=b_sf_offs,
        prob_host=prob_host, m_splits=m_splits,
    )
    return inputs, x_fp32_list, w_fp32_list


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

V2_CASES = [
    pytest.param([128],            128,  128,  id="single_small"),
    pytest.param([128, 256],       256,  256,  id="two_groups"),
    pytest.param([128, 128, 128],  512,  256,  id="three_groups"),
    pytest.param([256, 512],       256,  512,  id="varied_M"),
    pytest.param([128],           7168, 4096,  id="deepgemm_shape"),
    pytest.param([128] * 8,       4096, 7168,  id="moe_8experts_fc1"),
]


class TestGroupedCutlassGemmV2VsGolden:
    """Validate grouped_cutlass_gemm_v2 output against Python golden reference."""

    @pytest.mark.parametrize("m_splits, n, k", V2_CASES)
    @pytest.mark.parametrize("seed", [42, 7])
    def test_v2_vs_golden(self, m_splits, n, k, seed):
        """Kernel vs dequant-then-GEMM golden: cosine > 0.999, bounded abs error."""
        _require_sm100()
        device = "cuda"

        inputs, x_fp32_list, w_fp32_list = _build_v2_inputs(m_splits, n, k, device, seed)

        kernel_out = grouped_cutlass_gemm_v2(inputs, out_dtype=torch.bfloat16)
        golden_out = grouped_cutlass_gemm_v2_golden(inputs, out_dtype=torch.bfloat16)

        assert kernel_out.shape == golden_out.shape, \
            f"shape mismatch: kernel={kernel_out.shape} golden={golden_out.shape}"
        assert not torch.isnan(kernel_out).any(), "NaN in kernel output"
        assert not torch.isinf(kernel_out).any(), "Inf in kernel output"

        m = _error_metrics(kernel_out, golden_out)
        print(
            f"\n  m_splits={m_splits} n={n} k={k} seed={seed}\n"
            f"  kernel vs golden:"
            f"  max_abs={m['max_abs']:.4f}"
            f"  mean_abs={m['mean_abs']:.4f}"
            f"  norm_rmse={m['norm_rmse']:.4f}"
            f"  cosine={m['cosine']:.6f}"
        )

        assert m["max_abs"]   < 0.15,  f"max_abs={m['max_abs']:.4f} ≥ 0.15"
        assert m["mean_abs"]  < 0.03,  f"mean_abs={m['mean_abs']:.4f} ≥ 0.03"
        assert m["norm_rmse"] < 0.10,  f"norm_rmse={m['norm_rmse']:.4f} ≥ 0.10"
        assert m["cosine"]    > 0.999, f"cosine={m['cosine']:.6f} ≤ 0.999"

    @pytest.mark.parametrize("m_splits, n, k", V2_CASES)
    @pytest.mark.parametrize("seed", [42])
    def test_v2_vs_fp32_ground_truth(self, m_splits, n, k, seed):
        """Verify FP4 quantization error vs FP32 ground truth is within expected bounds."""
        _require_sm100()
        device = "cuda"

        inputs, x_fp32_list, w_fp32_list = _build_v2_inputs(m_splits, n, k, device, seed)

        kernel_out = grouped_cutlass_gemm_v2(inputs, out_dtype=torch.bfloat16)

        # FP32 ground truth: apply same per-row act_ts and per-expert wgt_ts scaling
        fp32_parts = []
        row = 0
        for g, m_g in enumerate(m_splits):
            pts_g = inputs.act_ts[row:row + m_g]        # [m_g]  (= pts * amax / factor)
            wts_g = inputs.wgt_ts[g]                    # scalar
            x_g   = x_fp32_list[g]
            w_g   = w_fp32_list[g]
            # effective per-row scale: pts_g already encodes the joint act_ts
            # wgt_ts corrects for the weight normalisation
            y = torch.mm(x_g.float(), w_g.float().t())  # [m_g, n] fp32 GEMM
            # scale: act_ts = pts * amax_x / (448*6) and wgt_ts = amax_w / (448*6)
            # but the ground truth uses the original fp32 data, so we need to account
            # for quantization-introduced pts separately — here we just check absolute
            # precision of the quantized output vs unscaled fp32 GEMM (normalised RMSE)
            fp32_parts.append(y.to(torch.bfloat16))
            row += m_g

        fp32_out = torch.cat(fp32_parts, dim=0)
        m = _error_metrics(kernel_out, fp32_out)
        print(
            f"\n  m_splits={m_splits} n={n} k={k} seed={seed}\n"
            f"  kernel vs fp32 GT:"
            f"  norm_rmse={m['norm_rmse']:.4f}"
            f"  cosine={m['cosine']:.6f}"
        )
        # FP4 quantisation introduces ~0.1–0.2 normalised RMSE vs FP32
        assert m["norm_rmse"] < 0.30, f"norm_rmse={m['norm_rmse']:.4f} ≥ 0.30"

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_seed_sweep(self, seed):
        """Multi-seed robustness check on a standard MoE shape."""
        _require_sm100()
        m_splits, n, k = [128] * 4, 4096, 2048
        device = "cuda"

        inputs, _, _ = _build_v2_inputs(m_splits, n, k, device, seed)
        kernel_out = grouped_cutlass_gemm_v2(inputs, out_dtype=torch.bfloat16)
        golden_out = grouped_cutlass_gemm_v2_golden(inputs, out_dtype=torch.bfloat16)

        m = _error_metrics(kernel_out, golden_out)
        assert m["max_abs"]  < 0.15, f"seed={seed} max_abs={m['max_abs']:.4f}"
        assert m["mean_abs"] < 0.03, f"seed={seed} mean_abs={m['mean_abs']:.4f}"
        assert m["cosine"]   > 0.999, f"seed={seed} cosine={m['cosine']:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
