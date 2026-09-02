"""MoE grouped activation + weight NVFP4 quant (QAT 6–14 production path).

What:
  - ``group_quantize_nvfp4_pertoken_swizzled`` (GroupedLinear X)
  - ``group_quantize_nvfp4_tensor_global_packed`` / ``quantize_weight_nvfp4_swizzled`` (W)
Where: ``nvfp4_cursor2_op*.group_quantize`` for QAT 6–8 / 12 / 14.
Compare: per-group ``rtn_fp4_pertoken`` reference; per-expert weight swizzled.

Markers: ``production``; weight tests also production; ``bench``+``slow`` for micro-benches.
Requires: fp4_gemm, CUDA sm_100+.
"""

from __future__ import annotations

import time
from typing import List

import pytest
import torch

try:
    import fp4_gemm
except ImportError:
    fp4_gemm = None

# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------

_SKIP_REASON: str | None = None
if fp4_gemm is None:
    _SKIP_REASON = "fp4_gemm extension not built"
elif not torch.cuda.is_available():
    _SKIP_REASON = "CUDA is not available"
else:
    _maj, _min = torch.cuda.get_device_capability()
    if _maj < 10:
        _SKIP_REASON = (
            f"group FP4 pertoken kernel requires sm_100+ (got {_maj}.{_min})"
        )

pytestmark = [
    pytest.mark.production,
    pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or ""),
]

# ---------------------------------------------------------------------------
# Import the new operator
# ---------------------------------------------------------------------------

try:
    from miniTransformer.ops.nvfp4_quantize import (
        GroupQuantizeSwizzledResult,
        GroupWeightQuantizeResult,
        group_quantize_nvfp4_pertoken_swizzled,
        group_quantize_nvfp4_tensor_global_packed,
        quantize_weight_nvfp4_swizzled,
    )
except ImportError as _e:
    pytestmark = pytest.mark.skip(reason=f"nvfp4_quantization not importable: {_e}")
    group_quantize_nvfp4_pertoken_swizzled = None  # type: ignore[assignment]
    group_quantize_nvfp4_tensor_global_packed = None  # type: ignore[assignment]
    quantize_weight_nvfp4_swizzled = None  # type: ignore[assignment]

try:
    from miniTransformer.ops.cuteDSL.nvfp4.grouped_gemm_nvfp4_impl import (
        PackedWeightNvfp4Tensor,
        _make_fp4_tensor_for_cutedsl,
    )
except ImportError:
    PackedWeightNvfp4Tensor = None  # type: ignore[assignment,misc]
    _make_fp4_tensor_for_cutedsl = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Dequantization helpers  (same as test_nvfp4_pertoken_quantize.py)
# ---------------------------------------------------------------------------

_FP4_GRID = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

_MMA_TILE_ROWS = 128


def _fp4_grid(device) -> torch.Tensor:
    return _FP4_GRID.to(device)


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    p = packed.to(torch.int32)
    return torch.stack([p & 0x0F, (p >> 4) & 0x0F], dim=-1).flatten(start_dim=-2)


def _unswizzle_scale(scale_flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Undo tcgen05 swizzle: flat [nMT*nKT*512] → row-major [rows, cols//16]."""
    nMT = rows // _MMA_TILE_ROWS
    nKT = cols // 64
    s = scale_flat.reshape(nMT * nKT, 32, 4, 4)
    s = s.permute(0, 2, 1, 3).reshape(nMT, nKT, _MMA_TILE_ROWS, 4)
    return s.permute(0, 2, 1, 3).reshape(rows, nKT * 4)


def _get_scale_views(gw) -> list:
    """Build per-expert scale_views from GroupWeightQuantizeResult fields.

    Returns a list of tensors with shape (32,4,nMT,4,nKT,1) — the CUTE swizzled
    layout matching quantize_weight_nvfp4_swizzled's _cute_sfb.
    """
    num_experts  = gw.global_scale.shape[0]
    n_per_expert = int(gw.row_offsets_gpu[1].item())
    k            = gw.data.shape[1] * 2
    sf_stride    = int(gw.scale_offsets_gpu[1].item())
    n_padded     = ((n_per_expert + _MMA_TILE_ROWS - 1) // _MMA_TILE_ROWS) * _MMA_TILE_ROWS
    nMT          = n_padded // _MMA_TILE_ROWS
    nKT          = k // 64
    views = []
    for g in range(num_experts):
        s = gw.scale_flat[g * sf_stride:(g + 1) * sf_stride]
        views.append(s.view(1, nMT, nKT, 32, 4, 4).permute(3, 4, 1, 5, 2, 0))
    return views


def _dequant_swizzled_group(
    data: torch.Tensor,
    scale_view: torch.Tensor,    # (32,4,nMT,4,nKT,1) — from scale_views[g]
    global_scale: torch.Tensor,  # [m_i]
    m_i: int,
    k: int,
) -> torch.Tensor:
    """Dequantize one group using the scale_view tensor from GroupQuantizeSwizzledResult."""
    nMT = (m_i + _MMA_TILE_ROWS - 1) // _MMA_TILE_ROWS
    nKT = k // 64
    # Inverse-permute (3,4,1,5,2,0)^-1 = (5,2,4,0,1,3) to get back (1,nMT,nKT,32,4,4)
    # Simpler: just flatten to the raw swizzled buffer and unswizzle.
    flat = scale_view.permute(5, 2, 4, 0, 1, 3).reshape(-1).to(torch.float32)
    # flat is now [nMT*nKT*32*4*4] = [nMT*128*nKT*4] in swizzled order
    scale_rm = _unswizzle_scale(flat, nMT * _MMA_TILE_ROWS, k)[:m_i]  # [m_i, k//16]
    grid = _fp4_grid(data.device)
    fp4_vals = grid[_unpack_fp4(data)].to(torch.float32)       # [m_i, k]
    scale_exp = scale_rm.repeat_interleave(16, dim=1)           # [m_i, k]
    gs = global_scale.to(torch.float32).unsqueeze(1)            # [m_i, 1]
    return (fp4_vals * scale_exp * gs).to(torch.bfloat16)


def _eff_bitwidth(x: torch.Tensor, x_dq: torch.Tensor) -> float:
    x_f = x.to(torch.float32)
    d_f = x_dq.to(torch.float32)
    row_err = (x_f - d_f).pow(2).sum(dim=-1) / x_f.pow(2).sum(dim=-1).clamp(min=1e-9)
    return (-torch.log2(row_err.mean()) / 2).item()


# ---------------------------------------------------------------------------
# Reference: call rtn_fp4_pertoken per group with a padded scale buffer
# ---------------------------------------------------------------------------

def _ref_per_group(x: torch.Tensor, m_splits: List[int]):
    """Reference: call fp4_gemm.rtn_fp4_group_pertoken once per group.

    Uses the same kernel as group_quantize_nvfp4_pertoken_swizzled so results
    are bit-exact (rtn_fp4_pertoken uses a different quantize path).
    Scale buffer is zero-initialized and padded to ceil(m_i/128)*128 * k//16.
    """
    total_m, k = x.shape
    device = x.device
    data_ref   = torch.empty((total_m, k // 2), dtype=torch.uint8, device=device)
    gs_ref     = torch.empty(total_m, dtype=torch.float32, device=device)
    scales_ref = []   # list of flat padded swizzled scale tensors per group

    offset = 0
    for m_i in m_splits:
        if m_i == 0:
            scales_ref.append(None)
            continue
        m_pad    = ((m_i + _MMA_TILE_ROWS - 1) // _MMA_TILE_ROWS) * _MMA_TILE_ROWS
        sf_size  = m_pad * k // 16
        scale_i  = torch.zeros(sf_size, dtype=torch.float8_e4m3fn, device=device)
        row_off_g   = torch.tensor([0, m_i], dtype=torch.int32, device=device)
        scale_off_g = torch.tensor([0, sf_size], dtype=torch.int64, device=device)
        fp4_gemm.rtn_fp4_group_pertoken(
            data_ref[offset:offset + m_i],
            scale_i,
            gs_ref[offset:offset + m_i],
            x[offset:offset + m_i].contiguous(),
            row_off_g,
            scale_off_g,
            1.0,
        )
        scales_ref.append(scale_i)
        offset += m_i
    return data_ref, gs_ref, scales_ref


def _get_swizzled_scale_views(res, m_splits: List[int], k: int) -> list:
    """Build per-group scale_views from GroupQuantizeSwizzledResult fields.

    Returns a list of tensors with shape (32,4,nMT,4,nKT,1), or None for empty groups.
    """
    nKT = k // 64
    views = []
    for g, m_i in enumerate(m_splits):
        if m_i == 0:
            views.append(None)
            continue
        s0  = int(res.scale_offsets_gpu[g].item())
        s1  = int(res.scale_offsets_gpu[g + 1].item())
        nMT = ((m_i + _MMA_TILE_ROWS - 1) // _MMA_TILE_ROWS)
        s   = res.scale_flat[s0:s1]
        views.append(s.view(1, nMT, nKT, 32, 4, 4).permute(3, 4, 1, 5, 2, 0))
    return views


# ---------------------------------------------------------------------------
# Typical MoE shapes used in production
# ---------------------------------------------------------------------------

# (num_groups, m_per_group, k)  — chunked kernel requires m_per_group % 128 == 0
TYPICAL_MOE_SHAPES = [
    (8,  128, 1024),    # small, clean multiple of 128
    (16, 128, 2048),    # baseline
    (64, 256, 4096),    # m > 128, clean multiple
    (62, 128, 768),     # realistic: 62 non-empty experts, m=128 per expert
]

# Irregular m_splits like RL training, all 128-aligned (upstream Fp8Padding
# rounds each per-expert count up to 128 before quantize).
_RL_M_SPLITS = [
    128, 0, 128, 256, 128, 128, 0, 128, 128, 256,
    128, 128, 0, 128, 128, 128, 128, 0, 128, 128,
    128, 128, 128, 0, 128, 128, 128, 128, 0, 128,
    128, 128, 128, 128, 128, 0, 128, 128, 128, 128,
    0, 128, 128, 128, 128, 0, 128, 128, 128, 128,
    128, 128, 128, 0, 128, 128, 128, 128, 128, 0,
    128, 128,
]  # 62 groups, mixed empty/non-empty (all non-empty are 128-aligned)


# ---------------------------------------------------------------------------
# Precision tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_groups,m_per_group,k", TYPICAL_MOE_SHAPES)
def test_fp4_data_matches_reference(num_groups: int, m_per_group: int, k: int):
    """Grouped kernel FP4 data must be bit-for-bit identical to per-group reference."""
    torch.manual_seed(42)
    m_splits = [m_per_group] * num_groups
    total_m  = sum(m_splits)
    x = torch.randn(total_m, k, dtype=torch.bfloat16, device="cuda")

    res = group_quantize_nvfp4_pertoken_swizzled(x, m_splits)
    data_ref, gs_ref, _ = _ref_per_group(x, m_splits)

    torch.testing.assert_close(
        res.data, data_ref, atol=0, rtol=0,
        msg="FP4 data mismatch vs per-group reference"
    )
    torch.testing.assert_close(
        res.global_scale, gs_ref, atol=0, rtol=0,
        msg="global_scale mismatch vs per-group reference"
    )


@pytest.mark.parametrize("num_groups,m_per_group,k", TYPICAL_MOE_SHAPES)
def test_scale_matches_reference(num_groups: int, m_per_group: int, k: int):
    """Grouped kernel swizzled scales must equal per-group reference scales."""
    torch.manual_seed(7)
    m_splits = [m_per_group] * num_groups
    total_m  = sum(m_splits)
    x = torch.randn(total_m, k, dtype=torch.bfloat16, device="cuda")

    res = group_quantize_nvfp4_pertoken_swizzled(x, m_splits)
    _, _, scales_ref = _ref_per_group(x, m_splits)
    scale_views = _get_swizzled_scale_views(res, m_splits, k)

    for g, (sv, sr) in enumerate(zip(scale_views, scales_ref)):
        if sv is None:
            continue
        m_i  = m_splits[g]
        m_pad = ((m_i + _MMA_TILE_ROWS - 1) // _MMA_TILE_ROWS) * _MMA_TILE_ROWS
        # Extract flat swizzled data from scale_view and compare to reference
        flat_ours = sv.permute(5, 2, 4, 0, 1, 3).reshape(-1)
        torch.testing.assert_close(
            flat_ours.view(torch.uint8),
            sr[:m_pad * k // 16].view(torch.uint8),
            atol=0, rtol=0,
            msg=f"scale mismatch for group {g} (m_i={m_i}, k={k})"
        )


@pytest.mark.parametrize("num_groups,m_per_group,k", TYPICAL_MOE_SHAPES)
def test_dequant_precision(num_groups: int, m_per_group: int, k: int):
    """Dequantized grouped output must meet effective bit-width > 3.3."""
    torch.manual_seed(0)
    m_splits = [m_per_group] * num_groups
    total_m  = sum(m_splits)
    x = torch.randn(total_m, k, dtype=torch.bfloat16, device="cuda")

    res  = group_quantize_nvfp4_pertoken_swizzled(x, m_splits)
    scale_views = _get_swizzled_scale_views(res, m_splits, k)

    offset = 0
    all_bits: List[float] = []
    for g, m_i in enumerate(m_splits):
        if m_i == 0 or scale_views[g] is None:
            offset += m_i
            continue
        xi     = x[offset:offset + m_i]
        di     = res.data[offset:offset + m_i]
        gsi    = res.global_scale[offset:offset + m_i]
        xi_dq  = _dequant_swizzled_group(di, scale_views[g], gsi, m_i, k)
        all_bits.append(_eff_bitwidth(xi, xi_dq))
        offset += m_i

    mean_bits = sum(all_bits) / len(all_bits)
    assert mean_bits > 3.3, (
        f"mean effective bit-width {mean_bits:.3f} < 3.3 "
        f"(shape: {num_groups}x{m_per_group}x{k})"
    )


def test_dequant_precision_rl_splits():
    """Realistic RL m_splits: mixed empty/non-empty experts, irregular m_i."""
    k = 2048
    torch.manual_seed(13)
    m_splits = _RL_M_SPLITS
    total_m  = sum(m_splits)
    x = torch.randn(total_m, k, dtype=torch.bfloat16, device="cuda")

    res = group_quantize_nvfp4_pertoken_swizzled(x, m_splits)
    scale_views = _get_swizzled_scale_views(res, m_splits, k)

    offset = 0
    all_bits: List[float] = []
    for g, m_i in enumerate(m_splits):
        if m_i == 0 or scale_views[g] is None:
            offset += m_i
            continue
        xi     = x[offset:offset + m_i]
        di     = res.data[offset:offset + m_i]
        gsi    = res.global_scale[offset:offset + m_i]
        xi_dq  = _dequant_swizzled_group(di, scale_views[g], gsi, m_i, k)
        all_bits.append(_eff_bitwidth(xi, xi_dq))
        offset += m_i

    mean_bits = sum(all_bits) / len(all_bits)
    assert mean_bits > 3.3, f"RL splits: mean eff bits {mean_bits:.3f} < 3.3"


def test_fp4_data_matches_reference_rl_splits():
    """RL m_splits: FP4 data and global_scale must match per-group reference."""
    k = 2048
    torch.manual_seed(99)
    m_splits = _RL_M_SPLITS
    total_m  = sum(m_splits)
    x = torch.randn(total_m, k, dtype=torch.bfloat16, device="cuda")

    res = group_quantize_nvfp4_pertoken_swizzled(x, m_splits)
    data_ref, gs_ref, _ = _ref_per_group(x, m_splits)

    torch.testing.assert_close(res.data, data_ref, atol=0, rtol=0,
                                msg="FP4 data mismatch (RL splits)")
    torch.testing.assert_close(res.global_scale, gs_ref, atol=0, rtol=0,
                                msg="global_scale mismatch (RL splits)")


def test_empty_groups_handled():
    """m_splits with some zeros: empty groups must produce None scale_views."""
    k = 512
    m_splits = [128, 0, 128, 0, 128]
    total_m  = sum(m_splits)
    x = torch.randn(total_m, k, dtype=torch.bfloat16, device="cuda")

    res = group_quantize_nvfp4_pertoken_swizzled(x, m_splits)
    scale_views = _get_swizzled_scale_views(res, m_splits, k)

    assert scale_views[1] is None, "group 1 (m=0) must have None scale_view"
    assert scale_views[3] is None, "group 3 (m=0) must have None scale_view"
    assert scale_views[0] is not None
    assert scale_views[2] is not None
    assert scale_views[4] is not None
    assert res.data.shape    == (total_m, k // 2)
    assert res.global_scale.shape == (total_m,)


def test_all_zeros_stable():
    """All-zero input must not produce NaN or Inf."""
    k = 256
    m_splits = [128] * 4
    total_m  = sum(m_splits)
    x = torch.zeros(total_m, k, dtype=torch.bfloat16, device="cuda")

    res = group_quantize_nvfp4_pertoken_swizzled(x, m_splits)
    scale_views = _get_swizzled_scale_views(res, m_splits, k)
    assert torch.isfinite(res.global_scale).all(), "NaN/Inf in global_scale"
    for sv in scale_views:
        if sv is not None:
            assert torch.isfinite(sv.to(torch.float32)).all(), "NaN/Inf in scale_view"


def test_scale_view_shape():
    """scale_views must have the shape (32,4,nMT,4,nKT,1) matching _make_cute_sf output."""
    k = 1024
    m_splits = [128, 256, 256, 128]
    total_m  = sum(m_splits)
    x = torch.randn(total_m, k, dtype=torch.bfloat16, device="cuda")

    res = group_quantize_nvfp4_pertoken_swizzled(x, m_splits)
    scale_views = _get_swizzled_scale_views(res, m_splits, k)

    nKT = k // 64
    for g, m_i in enumerate(m_splits):
        sv  = scale_views[g]
        nMT = ((m_i + _MMA_TILE_ROWS - 1) // _MMA_TILE_ROWS)
        expected = (32, 4, nMT, 4, nKT, 1)
        assert sv.shape == expected, (
            f"group {g}: scale_view shape {sv.shape} != expected {expected}"
        )


def test_scale_offsets_row_offsets_correct():
    """scale_offsets and row_offsets must reflect the actual padded allocation."""
    k = 512
    m_splits = [128, 256, 128, 128]
    total_m  = sum(m_splits)
    x = torch.randn(total_m, k, dtype=torch.bfloat16, device="cuda")

    res = group_quantize_nvfp4_pertoken_swizzled(x, m_splits)

    # row_offsets
    assert res.row_offsets_gpu[0].item() == 0
    expected_row = 0
    for i, m_i in enumerate(m_splits):
        expected_row += m_i
        assert res.row_offsets_gpu[i + 1].item() == expected_row

    # scale_offsets: each group occupies ceil(m_i/128)*128 * k//16
    assert res.scale_offsets_gpu[0].item() == 0
    expected_sf = 0
    for i, m_i in enumerate(m_splits):
        m_pad = ((m_i + _MMA_TILE_ROWS - 1) // _MMA_TILE_ROWS) * _MMA_TILE_ROWS
        expected_sf += m_pad * k // 16
        assert res.scale_offsets_gpu[i + 1].item() == expected_sf


# ---------------------------------------------------------------------------
# Speed benchmarks
# ---------------------------------------------------------------------------

WARMUP  = 10
REPEATS = 100

_MOE_BENCH_CONFIGS = [
    # (m_splits,                              k,    label)
    ([128] * 64,                              2048, "64x128 k=2048"),
    (_RL_M_SPLITS,                            2048, "RL-62groups k=2048"),
    ([128] * 64,                              4096, "64x128 k=4096"),
    ([256] * 32,                              4096, "32x256 k=4096"),
]


def _time_fn(fn, *args, warmup=WARMUP, repeats=REPEATS) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1e6   # µs


def _old_path(x: torch.Tensor, m_splits: List[int]) -> None:
    """Simulate the old D2H → CPU CuTeDSL → H2D path for comparison."""
    from miniTransformer.ops.cuteDSL.nvfp4.grouped_gemm_nvfp4_impl import (
        block_scale_to_sf_fp8,
        quantize_activation_nvfp4_pertoken,
        _make_cute_sf,
    )
    act = quantize_activation_nvfp4_pertoken(x)
    total_m, k = x.shape
    sf_k = k // 16
    offset = 0
    for m_i in m_splits:
        if m_i == 0:
            continue
        sf = block_scale_to_sf_fp8(act.block_scale[offset:offset + m_i], m_i, k)
        _make_cute_sf(sf, m_i, sf_k)
        offset += m_i


@pytest.mark.bench
@pytest.mark.slow
@pytest.mark.parametrize("m_splits,k,label", _MOE_BENCH_CONFIGS)
def test_bench_grouped_vs_old_path(m_splits, k, label):
    """Bench: grouped kernel vs per-group loop + CuTe scale path (prints; always passes)."""
    total_m = sum(m_splits)
    x = torch.randn(total_m, k, dtype=torch.bfloat16, device="cuda")

    # Warm up imports and JIT compilations before timing
    group_quantize_nvfp4_pertoken_swizzled(x, m_splits)
    try:
        _old_path(x, m_splits)
    except Exception:
        pass  # old path may not be available; skip its timing

    us_new = _time_fn(group_quantize_nvfp4_pertoken_swizzled, x, m_splits)

    try:
        us_old = _time_fn(_old_path, x, m_splits)
        speedup = us_old / us_new
        print(
            f"\n[{label}]  new={us_new:.1f} µs  old={us_old:.1f} µs  "
            f"speedup={speedup:.1f}x"
        )
    except Exception:
        print(f"\n[{label}]  new={us_new:.1f} µs  (old path unavailable)")


@pytest.mark.bench
@pytest.mark.slow
@pytest.mark.parametrize("m_splits,k,label", _MOE_BENCH_CONFIGS)
def test_bench_grouped_kernel_only(m_splits, k, label):
    """Bench: ``group_quantize_nvfp4_pertoken_swizzled`` alone (prints; always passes)."""
    total_m = sum(m_splits)
    x = torch.randn(total_m, k, dtype=torch.bfloat16, device="cuda")
    group_quantize_nvfp4_pertoken_swizzled(x, m_splits)  # warmup JIT

    us = _time_fn(group_quantize_nvfp4_pertoken_swizzled, x, m_splits)
    bw_GBs = total_m * k * 2 / (us * 1e-6) / 1e9  # BF16 read BW
    print(f"\n[{label}]  {us:.1f} µs  ~{bw_GBs:.0f} GB/s")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if _SKIP_REASON:
        print(f"Skipped: {_SKIP_REASON}")
        raise SystemExit(0)

    print("=" * 72)
    print("Precision")
    print("=" * 72)
    for num_groups, m_per_group, k in TYPICAL_MOE_SHAPES:
        torch.manual_seed(42)
        m_splits = [m_per_group] * num_groups
        total_m  = sum(m_splits)
        x = torch.randn(total_m, k, dtype=torch.bfloat16, device="cuda")

        res      = group_quantize_nvfp4_pertoken_swizzled(x, m_splits)
        scale_views = _get_swizzled_scale_views(res, m_splits, k)
        data_ref, gs_ref, _ = _ref_per_group(x, m_splits)

        data_match = torch.equal(res.data, data_ref)
        gs_match   = torch.equal(res.global_scale, gs_ref)

        offset = 0
        bits_list: List[float] = []
        for g, m_i in enumerate(m_splits):
            if m_i == 0 or scale_views[g] is None:
                offset += m_i
                continue
            xi    = x[offset:offset + m_i]
            dq    = _dequant_swizzled_group(
                res.data[offset:offset + m_i],
                scale_views[g],
                res.global_scale[offset:offset + m_i],
                m_i, k,
            )
            bits_list.append(_eff_bitwidth(xi, dq))
            offset += m_i

        mean_bits = sum(bits_list) / len(bits_list) if bits_list else float("nan")
        status    = "OK" if data_match and gs_match else "MISMATCH"
        print(
            f"  {num_groups}x{m_per_group} k={k}:  "
            f"data={'✓' if data_match else '✗'}  "
            f"gs={'✓' if gs_match else '✗'}  "
            f"eff-bits={mean_bits:.3f}  [{status}]"
        )

    print("\n" + "=" * 72)
    print("Speed  (µs per call, averaged over 100 iterations)")
    print("=" * 72)
    print(f"{'Config':<28} {'new (µs)':<12} {'old (µs)':<12} {'speedup':<10}")
    print("-" * 62)
    for m_splits, k, label in _MOE_BENCH_CONFIGS:
        total_m = sum(m_splits)
        x = torch.randn(total_m, k, dtype=torch.bfloat16, device="cuda")
        group_quantize_nvfp4_pertoken_swizzled(x, m_splits)  # warm up

        us_new = _time_fn(group_quantize_nvfp4_pertoken_swizzled, x, m_splits)
        try:
            us_old = _time_fn(_old_path, x, m_splits)
            print(
                f"  {label:<26} {us_new:<12.1f} {us_old:<12.1f} "
                f"{us_old / us_new:<10.1f}x"
            )
        except Exception:
            print(f"  {label:<26} {us_new:<12.1f} {'N/A':<12} {'N/A':<10}")


# ---------------------------------------------------------------------------
# Tests for group_quantize_nvfp4_tensor_global_packed (weight / tensor-global)
# ---------------------------------------------------------------------------

# (num_experts, n_per_expert, k)
_WEIGHT_SHAPES = [
    (4,  128, 256),
    (8,  128, 1024),
    (64, 128, 4096),
    (4,  256, 512),
]

try:
    import cutlass as _cutlass_mod  # noqa: F401
    _cutlass_available = True
except ImportError:
    _cutlass_available = False

_skip_weight = pytest.mark.skipif(
    group_quantize_nvfp4_tensor_global_packed is None or not _cutlass_available,
    reason=(
        "group_quantize_nvfp4_tensor_global_packed not importable"
        if group_quantize_nvfp4_tensor_global_packed is None
        else "cutlass Python module not installed"
    ),
)


@_skip_weight
@pytest.mark.parametrize("num_experts,n,k", _WEIGHT_SHAPES)
def test_weight_packed_global_scale_matches_per_expert(num_experts: int, n: int, k: int):
    """global_scale[g] must match quantize_weight_nvfp4_swizzled._alpha_b_cached per expert."""
    torch.manual_seed(42)
    x = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
    gw = group_quantize_nvfp4_tensor_global_packed(x, n)

    for g in range(num_experts):
        ref = quantize_weight_nvfp4_swizzled(x[g])
        packed_gs = gw.global_scale[g].item()
        ref_gs    = ref._alpha_b_cached
        assert abs(packed_gs - ref_gs) < 1e-5, (
            f"expert {g}: global_scale mismatch packed={packed_gs:.8f} ref={ref_gs:.8f}"
        )


@_skip_weight
@pytest.mark.parametrize("num_experts,n,k", _WEIGHT_SHAPES)
def test_weight_packed_fp4_data_matches_per_expert(num_experts: int, n: int, k: int):
    """FP4 data bytes must be bit-exact vs quantize_weight_nvfp4_swizzled per expert."""
    torch.manual_seed(42)
    x = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
    gw = group_quantize_nvfp4_tensor_global_packed(x, n)

    for g in range(num_experts):
        ref         = quantize_weight_nvfp4_swizzled(x[g])
        packed_data = gw.data[g * n:(g + 1) * n]
        assert torch.equal(packed_data, ref.data_uint8), (
            f"expert {g}: FP4 data mismatch  "
            f"max_diff={( packed_data.int() - ref.data_uint8.int()).abs().max().item()}  "
            f"nonzero={(packed_data.int() - ref.data_uint8.int()).abs().nonzero().shape[0]}"
        )


@_skip_weight
@pytest.mark.parametrize("num_experts,n,k", _WEIGHT_SHAPES)
def test_weight_packed_scale_view_matches_per_expert(num_experts: int, n: int, k: int):
    """scale_views[g] content must be bit-exact vs quantize_weight_nvfp4_swizzled per expert."""
    torch.manual_seed(42)
    x = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
    gw = group_quantize_nvfp4_tensor_global_packed(x, n)
    scale_views = _get_scale_views(gw)

    for g in range(num_experts):
        ref = quantize_weight_nvfp4_swizzled(x[g])
        sv  = scale_views[g].contiguous()
        rv  = ref._cute_sfb.contiguous()
        assert sv.shape == rv.shape, (
            f"expert {g}: scale_view shape mismatch {sv.shape} vs {rv.shape}"
        )
        assert torch.equal(sv.view(torch.uint8), rv.view(torch.uint8)), (
            f"expert {g}: scale_view content mismatch  "
            f"max_diff={(sv.view(torch.uint8).int() - rv.view(torch.uint8).int()).abs().max().item()}"
        )


@_skip_weight
@pytest.mark.parametrize("num_experts,n,k", _WEIGHT_SHAPES)
def test_weight_packed_cute_fp4_slice_matches_per_expert(num_experts: int, n: int, k: int):
    """cute_fp4_for(g) zero-copy slice must match _make_fp4_tensor_for_cutedsl output."""
    if PackedWeightNvfp4Tensor is None or _make_fp4_tensor_for_cutedsl is None:
        pytest.skip("grouped_gemm_nvfp4_impl not importable")
    torch.manual_seed(42)
    x      = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
    gw     = group_quantize_nvfp4_tensor_global_packed(x, n)
    packed = PackedWeightNvfp4Tensor(
        cute_data=gw.cute_data,
        scale_flat=gw.scale_flat,
        global_scale=gw.global_scale,
        scale_views=_get_scale_views(gw),
        scale_offsets=gw.scale_offsets_gpu,
        row_offsets=gw.row_offsets_gpu,
        n_per_expert=n,
        k=k,
    )

    for g in range(num_experts):
        ref        = quantize_weight_nvfp4_swizzled(x[g])
        ref_fp4    = _make_fp4_tensor_for_cutedsl(ref.data_uint8, n, k, x.device)
        packed_fp4 = packed.cute_fp4_for(g)
        # Compare only the first n*k//2 bytes (actual FP4 data).
        # The second half is zero-padding in cute_data but uninitialized in
        # _make_fp4_tensor_for_cutedsl (torch.empty), so they legitimately differ.
        half = n * k // 2
        p_fp4 = packed_fp4.reshape(-1)[:half]
        r_fp4 = ref_fp4.reshape(-1)[:half]
        assert torch.equal(p_fp4, r_fp4), (
            f"expert {g}: cute_fp4 FP4-data mismatch  "
            f"max_diff={(p_fp4.int() - r_fp4.int()).abs().max().item()}"
        )
