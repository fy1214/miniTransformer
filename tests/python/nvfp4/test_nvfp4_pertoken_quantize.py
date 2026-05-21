"""
Speed and precision tests for per-token NVFP4 quantization kernels:
  quantize_nvfp4_pertoken         – row-major scale layout (RTN and Round-4-or-6)
  quantize_nvfp4_pertoken_swizzled – tcgen05 swizzled scale layout

Precision methodology
---------------------
Dequantize the packed output with the recovered per-group FP8 scales and
per-token global scale, then compute effective bit-width:
    eff_bits = -log2(mean_relative_L2_error) / 2
NV FP4 E2M1 theoretical bit-width ≈ 4 bits; effective bit-width > 3.3 is a
reasonable sanity bar for Gaussian inputs.

Skip conditions: fp4_gemm extension not built, CUDA unavailable, or GPU < sm_100.
"""

from __future__ import annotations

import time

import pytest
import torch

try:
    import fp4_gemm
except ImportError:
    fp4_gemm = None

from miniTransformer.ops.nvfp4_quantization import (
    quantize_nvfp4_pertoken,
    quantize_nvfp4_pertoken_swizzled,
)

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
            f"per-token FP4 kernels require sm_100+ (got capability {_maj}.{_min})"
        )

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")

# ---------------------------------------------------------------------------
# FP4 E2M1 lookup table and dequantization helpers
# ---------------------------------------------------------------------------

_FP4_GRID = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 tensor [m, n//2] → int32 tensor [m, n] (nibble indices 0–15)."""
    packed_i32 = packed.to(torch.int32)
    lo = packed_i32 & 0x0F
    hi = (packed_i32 >> 4) & 0x0F
    return torch.stack([lo, hi], dim=-1).flatten(start_dim=-2)


def dequantize_rowmajor(
    data: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Dequantize row-major layout output back to BF16.

    data         : uint8 [m, n//2]
    scale        : float8_e4m3fn [m, n//16]  (one per 16-element group, row-major)
    global_scale : float32 [m]
    """
    grid = _FP4_GRID.to(data.device)
    indices = _unpack_fp4(data)                        # [m, n]
    fp4_vals = grid[indices].to(torch.float32)         # [m, n]

    # scale: [m, n//16] → broadcast to [m, n] by repeating each scale 16 times
    scale_f32 = scale.to(torch.float32)                # [m, n//16]
    scale_expanded = scale_f32.repeat_interleave(16, dim=1)  # [m, n]

    gs = global_scale.to(torch.float32).unsqueeze(1)   # [m, 1]
    return (fp4_vals * scale_expanded * gs).to(torch.bfloat16)


def _unswizzle_scale(scale_flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """
    Undo the tcgen05 MMA swizzled layout to recover plain row-major [m, n//16].

    The swizzled tensor is laid out as:
        [numMTiles * numKTiles * 32 * 4 * 4]  (flat)
    where numMTiles = rows // 128, numKTiles = (cols // 16) // 4.
    We undo the same permutation used during dequant in test_nvfp4_quartet_quant.py.
    """
    n_row_tiles = rows // 128
    n_col_tiles = (cols // 16) // 4
    # reshape to [numMTiles * numKTiles, 32, 4, 4]
    s = scale_flat.reshape(n_row_tiles * n_col_tiles, 32, 4, 4)
    # permute: (tile, col_group4, row32, subgroup4)
    s = s.permute(0, 2, 1, 3).reshape(n_row_tiles, n_col_tiles, 128, 4)
    s = s.permute(0, 2, 1, 3).reshape(rows, n_col_tiles * 4)
    return s


def dequantize_swizzled(
    data: torch.Tensor,
    scale_flat: torch.Tensor,
    global_scale: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Dequantize swizzled-layout output back to BF16."""
    scale_rowmajor = _unswizzle_scale(scale_flat.to(torch.float32), rows, cols)
    grid = _FP4_GRID.to(data.device)
    indices = _unpack_fp4(data)                               # [m, n]
    fp4_vals = grid[indices].to(torch.float32)
    scale_expanded = scale_rowmajor.repeat_interleave(16, dim=1)
    gs = global_scale.to(torch.float32).unsqueeze(1)
    return (fp4_vals * scale_expanded * gs).to(torch.bfloat16)


def _eff_bitwidth(x: torch.Tensor, x_dq: torch.Tensor) -> float:
    """Effective bit-width from per-row relative L2 error."""
    x_f = x.to(torch.float32)
    dq_f = x_dq.to(torch.float32)
    row_rel_err = (x_f - dq_f).pow(2).sum(dim=-1) / x_f.pow(2).sum(dim=-1).clamp(min=1e-9)
    mean_err = row_rel_err.mean()
    return (-torch.log2(mean_err) / 2).item()


# ---------------------------------------------------------------------------
# Precision tests
# ---------------------------------------------------------------------------

PRECISION_SHAPES = [(512, 128), (1024, 512), (1024, 4096), (4096, 2048)]
MIN_EFF_BITS = 3.3  # empirical lower bound for Gaussian inputs


@pytest.mark.parametrize("m,n", PRECISION_SHAPES)
@pytest.mark.parametrize("four_six", [False, True])
def test_pertoken_rowmajor_precision(m: int, n: int, four_six: bool):
    """quantize_nvfp4_pertoken: dequantized output must meet min effective bit-width."""
    torch.manual_seed(42)
    x = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")

    result = quantize_nvfp4_pertoken(x, four_six=four_six)

    assert result.data.shape == (m, n // 2), "data shape mismatch"
    assert result.scale.shape == (m, n // 16), "scale shape mismatch"
    assert result.tensor_scale is not None
    assert result.tensor_scale.shape == (m,), "global_scale shape mismatch"
    assert result.tensor_scale.dtype == torch.float32

    x_dq = dequantize_rowmajor(result.data, result.scale, result.tensor_scale)
    eff_bits = _eff_bitwidth(x, x_dq)
    assert eff_bits > MIN_EFF_BITS, (
        f"effective bitwidth {eff_bits:.2f} < {MIN_EFF_BITS} "
        f"(shape={m}x{n}, four_six={four_six})"
    )


@pytest.mark.parametrize("m,n", PRECISION_SHAPES)
@pytest.mark.parametrize("four_six", [False, True])
def test_pertoken_swizzled_precision(m: int, n: int, four_six: bool):
    """quantize_nvfp4_pertoken_swizzled: dequantized output must meet min effective bit-width."""
    torch.manual_seed(42)
    x = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")

    result = quantize_nvfp4_pertoken_swizzled(x, four_six=four_six)

    assert result.data.shape == (m, n // 2), "data shape mismatch"
    assert result.scale.shape == (m * n // 16,), "scale shape mismatch (expected flat)"
    assert result.tensor_scale is not None
    assert result.tensor_scale.shape == (m,), "global_scale shape mismatch"

    x_dq = dequantize_swizzled(result.data, result.scale, result.tensor_scale, m, n)
    eff_bits = _eff_bitwidth(x, x_dq)
    assert eff_bits > MIN_EFF_BITS, (
        f"effective bitwidth {eff_bits:.2f} < {MIN_EFF_BITS} "
        f"(shape={m}x{n}, four_six={four_six})"
    )


@pytest.mark.parametrize("m,n", PRECISION_SHAPES)
def test_rowmajor_and_swizzled_same_data(m: int, n: int):
    """RTN row-major and RTN swizzled must produce identical FP4 data and global scale."""
    torch.manual_seed(99)
    x = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")

    r_rm = quantize_nvfp4_pertoken(x, four_six=False)
    r_sw = quantize_nvfp4_pertoken_swizzled(x, four_six=False)

    torch.testing.assert_close(r_rm.data, r_sw.data, atol=0, rtol=0)
    torch.testing.assert_close(r_rm.tensor_scale, r_sw.tensor_scale, atol=0, rtol=0)

    # Scales differ in layout; check values match after unswizzling.
    scale_sw_rm = _unswizzle_scale(r_sw.scale.to(torch.float32), m, n)
    torch.testing.assert_close(
        r_rm.scale.to(torch.float32),
        scale_sw_rm,
        atol=0,
        rtol=0,
        msg="row-major and unswizzled swizzled scales differ",
    )


def test_pertoken_all_zeros_stable():
    """All-zero input must not produce NaN or Inf in scale or global_scale."""
    x = torch.zeros(128, 256, dtype=torch.bfloat16, device="cuda")
    for fn in [quantize_nvfp4_pertoken, quantize_nvfp4_pertoken_swizzled]:
        r = fn(x)
        assert torch.isfinite(r.scale.to(torch.float32)).all(), "NaN/Inf in scale"
        assert r.tensor_scale is not None
        assert torch.isfinite(r.tensor_scale).all(), "NaN/Inf in global_scale"


def test_pertoken_global_scale_per_row():
    """Verify global_scale is per-row: each row scaled independently."""
    m, n = 4, 128
    x = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")
    # Set each row to a different magnitude.
    row_mags = [1.0, 2.0, 4.0, 8.0]
    for i, mag in enumerate(row_mags):
        x[i] = mag
    r = quantize_nvfp4_pertoken(x)
    assert r.tensor_scale is not None
    gs = r.tensor_scale  # [m]
    # Ratios between consecutive global scales should reflect magnitude ratios.
    for i in range(1, m):
        ratio = gs[i].item() / gs[i - 1].item()
        assert abs(ratio - 2.0) < 0.1, (
            f"global_scale ratio row{i}/row{i-1} = {ratio:.3f}, expected ~2.0"
        )


@pytest.mark.parametrize("four_six", [False, True])
def test_four_six_better_than_rtn_on_bimodal(four_six: bool):
    """
    On a bimodal distribution round-4-or-6 (four_six=True) should achieve
    >= effective bit-width of RTN (four_six=False). We just verify four_six
    does not regress vs RTN on the same input.
    """
    torch.manual_seed(7)
    m, n = 512, 512
    # bimodal: mix of small (near 0) and large (near ±4) values
    x = torch.where(
        torch.rand(m, n) > 0.5,
        torch.randn(m, n) * 0.1,
        torch.randn(m, n) + 4.0,
    ).to(torch.bfloat16).cuda()

    r_rtn = quantize_nvfp4_pertoken(x, four_six=False)
    r_46 = quantize_nvfp4_pertoken(x, four_six=True)

    dq_rtn = dequantize_rowmajor(r_rtn.data, r_rtn.scale, r_rtn.tensor_scale)
    dq_46 = dequantize_rowmajor(r_46.data, r_46.scale, r_46.tensor_scale)

    bits_rtn = _eff_bitwidth(x, dq_rtn)
    bits_46 = _eff_bitwidth(x, dq_46)
    assert bits_46 >= bits_rtn - 0.05, (
        f"four_six ({bits_46:.3f} bits) significantly worse than RTN ({bits_rtn:.3f} bits)"
    )


# ---------------------------------------------------------------------------
# Speed benchmarks (run with: pytest -s -k bench or python <this_file>)
# ---------------------------------------------------------------------------

BENCH_SHAPES = [
    (1, 128),
    (32, 512),
    (128, 1024),
    (512, 2048),
    (1024, 4096),
    (4096, 4096),
]
WARMUP_ITERS = 20
BENCH_ITERS = 200


def _bench_fn(fn, x: torch.Tensor) -> float:
    """Return average kernel time in milliseconds over BENCH_ITERS iterations."""
    for _ in range(WARMUP_ITERS):
        fn(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(BENCH_ITERS):
        fn(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / BENCH_ITERS * 1000.0


@pytest.mark.slow
@pytest.mark.parametrize("m,n", BENCH_SHAPES)
def test_bench_pertoken_rowmajor(m: int, n: int):
    """Benchmark: quantize_nvfp4_pertoken (row-major). Prints ms; always passes."""
    x = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
    ms = _bench_fn(lambda t: quantize_nvfp4_pertoken(t), x)
    gb_s = (m * n * 2) / (ms * 1e-3) / 1e9  # BF16 read bandwidth
    print(f"\n[rowmajor] {m}x{n}: {ms:.3f} ms | ~{gb_s:.1f} GB/s")


@pytest.mark.slow
@pytest.mark.parametrize("m,n", BENCH_SHAPES)
def test_bench_pertoken_swizzled(m: int, n: int):
    """Benchmark: quantize_nvfp4_pertoken_swizzled (MMA layout). Prints ms; always passes."""
    x = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
    ms = _bench_fn(lambda t: quantize_nvfp4_pertoken_swizzled(t), x)
    gb_s = (m * n * 2) / (ms * 1e-3) / 1e9
    print(f"\n[swizzled] {m}x{n}: {ms:.3f} ms | ~{gb_s:.1f} GB/s")


@pytest.mark.slow
@pytest.mark.parametrize("m,n", BENCH_SHAPES)
def test_bench_four_six_vs_rtn(m: int, n: int):
    """Benchmark: RTN vs Round-4-or-6 overhead (row-major layout)."""
    x = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
    ms_rtn = _bench_fn(lambda t: quantize_nvfp4_pertoken(t, four_six=False), x)
    ms_46 = _bench_fn(lambda t: quantize_nvfp4_pertoken(t, four_six=True), x)
    overhead = (ms_46 / ms_rtn - 1.0) * 100
    print(f"\n[RTN vs 4/6] {m}x{n}: RTN={ms_rtn:.3f} ms, 4/6={ms_46:.3f} ms, overhead={overhead:+.1f}%")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if _SKIP_REASON:
        print(f"Skipped: {_SKIP_REASON}")
    else:
        print("=" * 70)
        print("Precision summary")
        print("=" * 70)
        for m, n in [(512, 128), (1024, 4096), (4096, 2048)]:
            torch.manual_seed(42)
            x = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
            for four_six in [False, True]:
                r = quantize_nvfp4_pertoken(x, four_six=four_six)
                dq = dequantize_rowmajor(r.data, r.scale, r.tensor_scale)
                bits = _eff_bitwidth(x, dq)
                label = "4/6" if four_six else "RTN"
                print(f"  [{label}] {m}x{n}: {bits:.3f} eff-bits")

        print("\n" + "=" * 70)
        print("Speed benchmark")
        print("=" * 70)
        print(f"{'Shape':<16} {'rowmajor (ms)':<18} {'swizzled (ms)':<18} {'RTN (ms)':<14} {'4/6 (ms)':<14}")
        print("-" * 70)
        for m, n in BENCH_SHAPES:
            if n < 128 or n % 128 != 0:
                continue
            x = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
            ms_rm = _bench_fn(lambda t: quantize_nvfp4_pertoken(t), x)
            ms_sw = _bench_fn(lambda t: quantize_nvfp4_pertoken_swizzled(t), x)
            ms_rt = _bench_fn(lambda t: quantize_nvfp4_pertoken(t, four_six=False), x)
            ms_46 = _bench_fn(lambda t: quantize_nvfp4_pertoken(t, four_six=True), x)
            print(f"  {f'{m}x{n}':<14} {ms_rm:<18.3f} {ms_sw:<18.3f} {ms_rt:<14.3f} {ms_46:<14.3f}")
