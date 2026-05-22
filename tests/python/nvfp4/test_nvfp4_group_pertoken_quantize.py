"""
Precision and speed tests for group_quantize_nvfp4_pertoken_swizzled.

Precision methodology
---------------------
1. Exact match: compare grouped-kernel output against per-group reference
   (fp4_gemm.rtn_fp4_pertoken called once per group).  FP4 data and global
   scale must be bit-for-bit identical; scale_views must match after unswizzling.

2. Dequant quality: effective bit-width > 3.3 on Gaussian inputs.

Speed methodology
-----------------
Compare wall-clock time of:
  A) new grouped kernel  (one CUDA launch, all groups)
  B) old path            (per-group rtn_fp4_pertoken Python loop  +  per-group
                          _make_cute_sf CPU round-trip)

Skip conditions: fp4_gemm not built, CUDA unavailable, GPU < sm_100.
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

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")

# ---------------------------------------------------------------------------
# Import the new operator
# ---------------------------------------------------------------------------

try:
    from miniTransformer.ops.nvfp4_quantization import (
        GroupQuantizeSwizzledResult,
        group_quantize_nvfp4_pertoken_swizzled,
    )
except ImportError as _e:
    pytestmark = pytest.mark.skip(reason=f"nvfp4_quantization not importable: {_e}")
    group_quantize_nvfp4_pertoken_swizzled = None  # type: ignore[assignment]

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
    # Inverse-permute (3,4,1,5,2,0)^-1 = (5,2,4,0,3,1) to get back (1,nMT,nKT,32,4,4)
    # Simpler: just flatten to the raw swizzled buffer and unswizzle.
    flat = scale_view.permute(5, 2, 0, 4, 1, 3).reshape(-1).to(torch.float32)
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
    """Reference: call fp4_gemm.rtn_fp4_pertoken once per group.

    Scale buffer is zero-initialized and padded to ceil(m_i/128)*128 * k//16
    to avoid buffer overflow from the swizzle addressing formula.
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
        fp4_gemm.rtn_fp4_pertoken(
            data_ref[offset:offset + m_i],
            scale_i,          # padded buffer: kernel writes up to sf_size-1
            gs_ref[offset:offset + m_i],
            x[offset:offset + m_i].contiguous(),
            1.0,
        )
        scales_ref.append(scale_i)
        offset += m_i
    return data_ref, gs_ref, scales_ref


# ---------------------------------------------------------------------------
# Typical MoE shapes used in production
# ---------------------------------------------------------------------------

# (num_groups, m_per_group, k)  — m_per_group can be non-multiple-of-128
TYPICAL_MOE_SHAPES = [
    (8,  128, 1024),    # small, clean multiple of 128
    (16, 64,  2048),    # m < 128
    (64, 192, 4096),    # m > 128, not a clean multiple
    (62, 128, 768),     # realistic: 62 non-empty experts, m=128 per expert
]

# Irregular m_splits like RL training (each expert gets different token count)
_RL_M_SPLITS = [
    128, 0, 96, 192, 64, 128, 0, 48, 128, 160,
    128, 80, 0, 128, 96, 64, 128, 0, 128, 112,
    96, 128, 128, 0, 64, 128, 80, 128, 0, 128,
    128, 96, 64, 128, 128, 0, 80, 128, 64, 128,
    0, 128, 112, 96, 128, 0, 128, 64, 128, 96,
    128, 80, 128, 0, 96, 128, 64, 128, 128, 0,
    128, 112,
]  # 62 groups, mixed empty/non-empty


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

    for g, (sv, sr) in enumerate(zip(res.scale_views, scales_ref)):
        if sv is None:
            continue
        m_i  = m_splits[g]
        m_pad = ((m_i + _MMA_TILE_ROWS - 1) // _MMA_TILE_ROWS) * _MMA_TILE_ROWS
        # Extract flat swizzled data from scale_view and compare to reference
        flat_ours = sv.permute(5, 2, 0, 4, 1, 3).reshape(-1)
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

    offset = 0
    all_bits: List[float] = []
    for g, m_i in enumerate(m_splits):
        if m_i == 0 or res.scale_views[g] is None:
            offset += m_i
            continue
        xi     = x[offset:offset + m_i]
        di     = res.data[offset:offset + m_i]
        gsi    = res.global_scale[offset:offset + m_i]
        xi_dq  = _dequant_swizzled_group(di, res.scale_views[g], gsi, m_i, k)
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

    offset = 0
    all_bits: List[float] = []
    for g, m_i in enumerate(m_splits):
        if m_i == 0 or res.scale_views[g] is None:
            offset += m_i
            continue
        xi     = x[offset:offset + m_i]
        di     = res.data[offset:offset + m_i]
        gsi    = res.global_scale[offset:offset + m_i]
        xi_dq  = _dequant_swizzled_group(di, res.scale_views[g], gsi, m_i, k)
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
    m_splits = [128, 0, 64, 0, 128]
    total_m  = sum(m_splits)
    x = torch.randn(total_m, k, dtype=torch.bfloat16, device="cuda")

    res = group_quantize_nvfp4_pertoken_swizzled(x, m_splits)

    assert res.scale_views[1] is None, "group 1 (m=0) must have None scale_view"
    assert res.scale_views[3] is None, "group 3 (m=0) must have None scale_view"
    assert res.scale_views[0] is not None
    assert res.scale_views[2] is not None
    assert res.scale_views[4] is not None
    assert res.data.shape    == (total_m, k // 2)
    assert res.global_scale.shape == (total_m,)


def test_all_zeros_stable():
    """All-zero input must not produce NaN or Inf."""
    k = 256
    m_splits = [128] * 4
    total_m  = sum(m_splits)
    x = torch.zeros(total_m, k, dtype=torch.bfloat16, device="cuda")

    res = group_quantize_nvfp4_pertoken_swizzled(x, m_splits)
    assert torch.isfinite(res.global_scale).all(), "NaN/Inf in global_scale"
    for sv in res.scale_views:
        if sv is not None:
            assert torch.isfinite(sv.to(torch.float32)).all(), "NaN/Inf in scale_view"


def test_scale_view_shape():
    """scale_views must have the shape (32,4,nMT,4,nKT,1) matching _make_cute_sf output."""
    k = 1024
    m_splits = [128, 256, 192, 64]
    total_m  = sum(m_splits)
    x = torch.randn(total_m, k, dtype=torch.bfloat16, device="cuda")

    res = group_quantize_nvfp4_pertoken_swizzled(x, m_splits)

    nKT = k // 64
    for g, m_i in enumerate(m_splits):
        sv  = res.scale_views[g]
        nMT = ((m_i + _MMA_TILE_ROWS - 1) // _MMA_TILE_ROWS)
        expected = (32, 4, nMT, 4, nKT, 1)
        assert sv.shape == expected, (
            f"group {g}: scale_view shape {sv.shape} != expected {expected}"
        )


def test_scale_offsets_row_offsets_correct():
    """scale_offsets and row_offsets must reflect the actual padded allocation."""
    k = 512
    m_splits = [100, 200, 50, 128]
    total_m  = sum(m_splits)
    x = torch.randn(total_m, k, dtype=torch.bfloat16, device="cuda")

    res = group_quantize_nvfp4_pertoken_swizzled(x, m_splits)

    # row_offsets
    assert res.row_offsets[0] == 0
    expected_row = 0
    for i, m_i in enumerate(m_splits):
        expected_row += m_i
        assert res.row_offsets[i + 1] == expected_row

    # scale_offsets: each group occupies ceil(m_i/128)*128 * k//16
    assert res.scale_offsets[0] == 0
    expected_sf = 0
    for i, m_i in enumerate(m_splits):
        m_pad = ((m_i + _MMA_TILE_ROWS - 1) // _MMA_TILE_ROWS) * _MMA_TILE_ROWS
        expected_sf += m_pad * k // 16
        assert res.scale_offsets[i + 1] == expected_sf


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


@pytest.mark.slow
@pytest.mark.parametrize("m_splits,k,label", _MOE_BENCH_CONFIGS)
def test_bench_grouped_vs_old_path(m_splits, k, label):
    """
    Benchmark new grouped kernel vs old per-group D2H+CPU+H2D path.
    Prints µs and speedup; always passes.
    """
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


@pytest.mark.slow
@pytest.mark.parametrize("m_splits,k,label", _MOE_BENCH_CONFIGS)
def test_bench_grouped_kernel_only(m_splits, k, label):
    """Standalone benchmark: just the new grouped kernel.  Always passes."""
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
        data_ref, gs_ref, _ = _ref_per_group(x, m_splits)

        data_match = torch.equal(res.data, data_ref)
        gs_match   = torch.equal(res.global_scale, gs_ref)

        offset = 0
        bits_list: List[float] = []
        for g, m_i in enumerate(m_splits):
            if m_i == 0 or res.scale_views[g] is None:
                offset += m_i
                continue
            xi    = x[offset:offset + m_i]
            dq    = _dequant_swizzled_group(
                res.data[offset:offset + m_i],
                res.scale_views[g],
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
