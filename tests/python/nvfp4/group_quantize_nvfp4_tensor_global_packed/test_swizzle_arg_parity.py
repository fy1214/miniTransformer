#!/usr/bin/env python3
"""Swizzle-arg parity for ``rtn_fp4_group_tensor_global`` (MoE weight quant).

What: ``swizzle=True`` then ``_unswizzle_scale`` == ``swizzle=False`` (bit-exact SF).
Where: underlying kernel for ``group_quantize_nvfp4_tensor_global_packed`` (QAT 6–14 W).
Compare: same kernel with opposite ``swizzle`` flag on identical input.

Markers: ``production``.
"""
import pytest
import torch

import fp4_gemm
from miniTransformer.ops.nvfp4_quantize import _unswizzle_scale

pytestmark = pytest.mark.production


def _alloc_buffers(rows, cols, num_groups, device):
    T = 128
    n_padded = ((rows + T - 1) // T) * T
    sf_stride = n_padded * (cols // 16)

    data = torch.empty((rows, cols // 2), dtype=torch.uint8, device=device)
    gs = torch.empty(num_groups, dtype=torch.float32, device=device)
    sf = torch.zeros(num_groups * sf_stride, dtype=torch.float8_e4m3fn, device=device)
    return data, gs, sf, n_padded, sf_stride


@pytest.mark.parametrize("rows,cols", [
    (128, 2048),     # Qwen3-30B-A3B hidden
    (256, 4096),     # Qwen3-235B-A22B hidden
    (1024, 7168),    # DeepSeek-V3 hidden (max we support)
    (768, 2048),     # DeepSeek-V3 moe_intermediate down_proj
    (96, 2048),      # non-128-aligned rows
])
def test_swizzle_arg_parity_single_group(rows, cols):
    """Single group: swizzled+unswizzle SF matches linear SF; FP4 data identical."""
    device = "cuda"
    torch.manual_seed(0)
    x = torch.randn(rows, cols, dtype=torch.bfloat16, device=device) * 4.0

    row_offsets = torch.tensor([0, rows], dtype=torch.int32, device=device)
    amax = x.abs().max().to(torch.float32).reshape(1).contiguous()

    # --- Run with swizzle=True ---
    data_t, gs_t, sf_t, n_padded, sf_stride = _alloc_buffers(rows, cols, 1, device)
    scale_offsets = torch.tensor([0, sf_stride], dtype=torch.int64, device=device)
    fp4_gemm.rtn_fp4_group_tensor_global(
        data_t, sf_t, gs_t, x, row_offsets, scale_offsets, amax, 1.0,
        swizzle=True,
    )
    sf_t_unswiz = _unswizzle_scale(sf_t.to(torch.float32), n_padded, cols)[:rows]

    # --- Run with swizzle=False ---
    data_f, gs_f, sf_f, _, _ = _alloc_buffers(rows, cols, 1, device)
    fp4_gemm.rtn_fp4_group_tensor_global(
        data_f, sf_f, gs_f, x, row_offsets, scale_offsets, amax, 1.0,
        swizzle=False,
    )
    sf_f_rowmajor = sf_f.view(n_padded, cols // 16)[:rows].to(torch.float32)

    # --- Compare ---
    assert torch.equal(data_t, data_f), "packed FP4 must be identical"
    assert torch.equal(gs_t, gs_f), "global_scale must be identical"
    assert torch.equal(sf_t_unswiz, sf_f_rowmajor), \
        f"SF (rowmajor) mismatch: swizzle=True→unswizzle vs swizzle=False"


@pytest.mark.parametrize("num_candidates_fn_name", [
    "rtn_fp4_group_tensor_global",
    "four_six_fp4_group_tensor_global",
])
def test_swizzle_arg_parity_multi_group(num_candidates_fn_name):
    """Multi-group RTN/4-6: swizzle on/off SF parity after unswizzle."""
    device = "cuda"
    torch.manual_seed(1)
    cols = 2048
    group_rows = [128, 256, 128]
    total = sum(group_rows)
    x = torch.randn(total, cols, dtype=torch.bfloat16, device=device) * 4.0

    row_offsets = torch.tensor([0] + list(torch.tensor(group_rows).cumsum(0).tolist()),
                               dtype=torch.int32, device=device)

    amax_list = []
    cur = 0
    for m_g in group_rows:
        amax_list.append(x[cur:cur + m_g].abs().max().to(torch.float32))
        cur += m_g
    amax = torch.stack(amax_list).contiguous()

    T = 128
    sf_strides = [((m_g + T - 1) // T) * T * (cols // 16) for m_g in group_rows]
    scale_offsets = torch.tensor([0] + list(torch.tensor(sf_strides).cumsum(0).tolist()),
                                 dtype=torch.int64, device=device)
    total_sf = scale_offsets[-1].item()

    fn = getattr(fp4_gemm, num_candidates_fn_name)

    data_t = torch.empty((total, cols // 2), dtype=torch.uint8, device=device)
    gs_t = torch.empty(len(group_rows), dtype=torch.float32, device=device)
    sf_t = torch.zeros(total_sf, dtype=torch.float8_e4m3fn, device=device)
    fn(data_t, sf_t, gs_t, x, row_offsets, scale_offsets, amax, 1.0, swizzle=True)

    data_f = torch.empty_like(data_t)
    gs_f = torch.empty_like(gs_t)
    sf_f = torch.zeros_like(sf_t)
    fn(data_f, sf_f, gs_f, x, row_offsets, scale_offsets, amax, 1.0, swizzle=False)

    assert torch.equal(data_t, data_f), f"{num_candidates_fn_name}: FP4 data mismatch"
    assert torch.equal(gs_t, gs_f), f"{num_candidates_fn_name}: global_scale mismatch"

    # Compare SF group-by-group.
    for g, m_g in enumerate(group_rows):
        s0, s1 = scale_offsets[g].item(), scale_offsets[g + 1].item()
        m_padded = ((m_g + T - 1) // T) * T
        sf_t_g = _unswizzle_scale(sf_t[s0:s1].to(torch.float32), m_padded, cols)[:m_g]
        sf_f_g = sf_f[s0:s1].view(m_padded, cols // 16)[:m_g].to(torch.float32)
        assert torch.equal(sf_t_g, sf_f_g), \
            f"{num_candidates_fn_name}: group {g} SF mismatch"


def test_swizzle_arg_cap_relaxation_cols_8192():
    """Smoke: K=8192 accepted (cap raised from 4096); must not raise."""
    device = "cuda"
    rows, cols = 128, 8192  # DeepSeek-V3 hidden + headroom
    x = torch.randn(rows, cols, dtype=torch.bfloat16, device=device)

    row_offsets = torch.tensor([0, rows], dtype=torch.int32, device=device)
    amax = x.abs().max().to(torch.float32).reshape(1).contiguous()

    T = 128
    n_padded = ((rows + T - 1) // T) * T
    sf_stride = n_padded * (cols // 16)
    scale_offsets = torch.tensor([0, sf_stride], dtype=torch.int64, device=device)

    data = torch.empty((rows, cols // 2), dtype=torch.uint8, device=device)
    gs = torch.empty(1, dtype=torch.float32, device=device)
    sf = torch.zeros(sf_stride, dtype=torch.float8_e4m3fn, device=device)

    # Should not raise.
    fp4_gemm.rtn_fp4_group_tensor_global(
        data, sf, gs, x, row_offsets, scale_offsets, amax, 1.0, swizzle=False,
    )
    torch.cuda.synchronize()
