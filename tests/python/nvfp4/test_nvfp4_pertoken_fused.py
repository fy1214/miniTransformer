"""
Precision and performance tests for the fused per-token NVFP4 quantization kernel.

Fused kernel:   fp4_pertoken_fused  (amax computed inside, single kernel launch)
Two-pass ref:   fp4_pertoken        (external amax_per_row + separate kernel)

Precision methodology
---------------------
The two paths compute identical math; results must be bit-exact:
  - packed FP4 data  (uint8)
  - per-group FP8 scale
  - per-token global scale (float32)

Performance methodology
-----------------------
Wall-clock time measured with CUDA events for kernel-level precision.
Reports latency (µs), throughput (GB/s read BF16 + write FP4), and
speedup of fused over two-pass.

Run all tests:
    pytest tests/python/nvfp4/test_nvfp4_pertoken_fused.py -v

Run benchmarks only (prints tables):
    python tests/python/nvfp4/test_nvfp4_pertoken_fused.py

Skip conditions: fp4_gemm extension not built, CUDA unavailable, or GPU < sm_100.
"""

from __future__ import annotations

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
        _SKIP_REASON = f"per-token FP4 kernels require sm_100+ (got {_maj}.{_min})"

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")

# ---------------------------------------------------------------------------
# Kernel wrappers
# ---------------------------------------------------------------------------

nvfp4_output_t = torch.uint8
nvfp4_scale_t  = torch.float8_e4m3fn


def _alloc(rows: int, cols: int, device: str):
    out          = torch.empty((rows, cols // 2),      dtype=nvfp4_output_t, device=device)
    scale        = torch.empty((rows, cols // 16),     dtype=nvfp4_scale_t,  device=device)
    global_scale = torch.empty(rows,                   dtype=torch.float32,  device=device)
    return out, scale, global_scale


def _alloc_swizzled(rows: int, cols: int, device: str):
    out          = torch.empty((rows, cols // 2),      dtype=nvfp4_output_t, device=device)
    scale        = torch.empty(rows * cols // 16,      dtype=nvfp4_scale_t,  device=device)
    global_scale = torch.empty(rows,                   dtype=torch.float32,  device=device)
    return out, scale, global_scale


def run_fused_linear(x: torch.Tensor, four_six: bool = False):
    """Single-launch fused kernel, row-major scale layout."""
    rows, cols = x.shape
    out, scale, gs = _alloc(rows, cols, x.device)
    fn = fp4_gemm.four_six_fp4_pertoken_linear_fused if four_six else fp4_gemm.rtn_fp4_pertoken_linear_fused
    fn(out, scale, gs, x.contiguous(), 1.0)
    return out, scale, gs


def run_twopass_linear(x: torch.Tensor, four_six: bool = False):
    """Two-pass: external amax_per_row + separate kernel, row-major scale layout."""
    rows, cols = x.shape
    x_c = x.contiguous()
    out, scale, gs = _alloc(rows, cols, x.device)
    amax = x_c.abs().amax(dim=1).to(torch.float32)
    fn = fp4_gemm.four_six_fp4_pertoken_linear if four_six else fp4_gemm.rtn_fp4_pertoken_linear
    fn(out, scale, gs, x_c, amax, 1.0)
    return out, scale, gs


def run_fused_swizzled(x: torch.Tensor, four_six: bool = False):
    """Single-launch fused kernel, swizzled scale layout."""
    rows, cols = x.shape
    out, scale, gs = _alloc_swizzled(rows, cols, x.device)
    fn = fp4_gemm.four_six_fp4_pertoken_fused if four_six else fp4_gemm.rtn_fp4_pertoken_fused
    fn(out, scale, gs, x.contiguous(), 1.0)
    return out, scale, gs


def run_twopass_swizzled(x: torch.Tensor, four_six: bool = False):
    """Two-pass: external amax_per_row + separate kernel, swizzled scale layout."""
    rows, cols = x.shape
    x_c = x.contiguous()
    out, scale, gs = _alloc_swizzled(rows, cols, x.device)
    amax = x_c.abs().amax(dim=1).to(torch.float32)
    fn = fp4_gemm.four_six_fp4_pertoken if four_six else fp4_gemm.rtn_fp4_pertoken
    fn(out, scale, gs, x_c, amax, 1.0)
    return out, scale, gs


# ---------------------------------------------------------------------------
# Precision tests — fused must be bit-exact vs two-pass
# ---------------------------------------------------------------------------
# Both paths compute the same per-row max (tree max = sequential max for
# finite values), same scale formula, same quantize call → bit-exact.

PRECISION_SHAPES = [
    # (rows, cols)
    # small rows — exercises grid-stride loop
    (1,    128),
    (7,    512),
    (16,   768),
    (63,   1024),
    (128,  2048),
    # typical training shapes
    (1024, 512),
    (1024, 768),
    (1024, 2048),
    (4096, 4096),
    # large rows
    (16384, 1024),
    # near grid boundary
    (1025,  256),
    (2047,  512),
]


@pytest.mark.parametrize("rows,cols", PRECISION_SHAPES)
@pytest.mark.parametrize("four_six", [False, True])
def test_fused_linear_bit_exact(rows: int, cols: int, four_six: bool):
    """Fused row-major must produce identical output to two-pass."""
    torch.manual_seed(42)
    x = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")

    f_data, f_scale, f_gs = run_fused_linear(x, four_six)
    r_data, r_scale, r_gs = run_twopass_linear(x, four_six)

    torch.testing.assert_close(f_data,  r_data,  atol=0, rtol=0,
                                msg="FP4 data mismatch (linear)")
    torch.testing.assert_close(f_scale.view(torch.uint8), r_scale.view(torch.uint8),
                                atol=0, rtol=0,
                                msg="FP8 scale mismatch (linear)")
    torch.testing.assert_close(f_gs, r_gs, atol=0, rtol=0,
                                msg="global_scale mismatch (linear)")


@pytest.mark.parametrize("rows,cols", PRECISION_SHAPES)
@pytest.mark.parametrize("four_six", [False, True])
def test_fused_swizzled_bit_exact(rows: int, cols: int, four_six: bool):
    """Fused swizzled must produce identical output to two-pass."""
    torch.manual_seed(42)
    x = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")

    f_data, f_scale, f_gs = run_fused_swizzled(x, four_six)
    r_data, r_scale, r_gs = run_twopass_swizzled(x, four_six)

    torch.testing.assert_close(f_data,  r_data,  atol=0, rtol=0,
                                msg="FP4 data mismatch (swizzled)")
    torch.testing.assert_close(f_scale.view(torch.uint8), r_scale.view(torch.uint8),
                                atol=0, rtol=0,
                                msg="FP8 scale mismatch (swizzled)")
    torch.testing.assert_close(f_gs, r_gs, atol=0, rtol=0,
                                msg="global_scale mismatch (swizzled)")


def test_fused_all_zeros():
    """All-zero input must not produce NaN/Inf; global_scale must be 1.0."""
    x = torch.zeros(128, 256, dtype=torch.bfloat16, device="cuda")
    for fn in [run_fused_linear, run_fused_swizzled]:
        _, scale, gs = fn(x)
        assert torch.isfinite(scale.to(torch.float32)).all(), "NaN/Inf in FP8 scale"
        assert torch.isfinite(gs).all(), "NaN/Inf in global_scale"
        assert (gs == 1.0).all(), "global_scale for all-zero input must be 1.0"


def test_fused_single_row():
    """rows=1 edge case: grid-stride loop runs exactly one iteration."""
    torch.manual_seed(7)
    x = torch.randn(1, 512, dtype=torch.bfloat16, device="cuda")
    f_data, f_scale, f_gs = run_fused_linear(x)
    r_data, r_scale, r_gs = run_twopass_linear(x)
    torch.testing.assert_close(f_data, r_data, atol=0, rtol=0)
    torch.testing.assert_close(f_gs,   r_gs,   atol=0, rtol=0)


def test_fused_global_scale_per_row():
    """Each row must have an independent global_scale proportional to its magnitude."""
    rows, cols = 8, 256
    x = torch.zeros(rows, cols, dtype=torch.bfloat16, device="cuda")
    mags = [float(2 ** i) for i in range(rows)]
    for i, mag in enumerate(mags):
        x[i] = mag
    _, _, gs = run_fused_linear(x)
    for i in range(1, rows):
        ratio = gs[i].item() / gs[i - 1].item()
        assert abs(ratio - 2.0) < 0.05, (
            f"global_scale ratio row{i}/row{i-1} = {ratio:.4f}, expected 2.0"
        )


# ---------------------------------------------------------------------------
# Performance benchmarks
# ---------------------------------------------------------------------------

WARMUP  = 50
REPEATS = 500

BENCH_SHAPES = [
    # small rows — SM under-utilization without grid-stride
    (1,     512),
    (8,     768),
    (16,    768),
    (64,    768),
    (128,   768),
    # typical MoE / training shapes
    (1024,  768),
    (4096,  768),
    (16384, 768),
    # various hidden dims
    (1024,  512),
    (1024,  1024),
    (1024,  2048),
    (1024,  4096),
    (1024,  8192),
]


def _cuda_bench(fn, *args, warmup: int = WARMUP, repeats: int = REPEATS) -> float:
    """Return median kernel time in microseconds using CUDA events."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(repeats):
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3)  # ms → µs

    times.sort()
    # median
    return times[repeats // 2]


def _throughput_gbs(rows: int, cols: int, us: float) -> float:
    """Effective memory bandwidth: BF16 read (2 B/elem) + FP4 write (0.5 B/elem)."""
    bytes_rw = rows * cols * (2 + 0.5)
    return bytes_rw / (us * 1e-6) / 1e9


@pytest.mark.slow
@pytest.mark.parametrize("rows,cols", BENCH_SHAPES)
def test_bench_fused_vs_twopass(rows: int, cols: int):
    """Fused must not be slower than two-pass (with 5% tolerance)."""
    x = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
    x_c = x.contiguous()
    amax = x_c.abs().amax(dim=1).to(torch.float32)

    out_f, scale_f, gs_f = _alloc(rows, cols, "cuda")
    out_r, scale_r, gs_r = _alloc(rows, cols, "cuda")

    us_fused = _cuda_bench(
        fp4_gemm.rtn_fp4_pertoken_linear_fused,
        out_f, scale_f, gs_f, x_c, 1.0)

    us_twopass_kernel = _cuda_bench(
        fp4_gemm.rtn_fp4_pertoken_linear,
        out_r, scale_r, gs_r, x_c, amax, 1.0)

    # Include amax cost in two-pass total (measured separately to avoid fusion by torch)
    us_amax = _cuda_bench(lambda: x_c.abs().amax(dim=1))
    us_twopass = us_twopass_kernel + us_amax

    speedup = us_twopass / us_fused
    print(
        f"\n[{rows:>6}x{cols:<5}] "
        f"fused={us_fused:6.1f}µs  "
        f"twopass={us_twopass:6.1f}µs (kernel={us_twopass_kernel:.1f}+amax={us_amax:.1f})  "
        f"speedup={speedup:.2f}x  "
        f"BW={_throughput_gbs(rows, cols, us_fused):.1f}GB/s"
    )
    assert us_fused <= us_twopass * 1.05, (
        f"fused ({us_fused:.1f}µs) is >5% slower than two-pass ({us_twopass:.1f}µs)"
    )


# ---------------------------------------------------------------------------
# Standalone runner — prints full tables
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if _SKIP_REASON:
        print(f"Skipped: {_SKIP_REASON}")
        raise SystemExit(0)

    device = "cuda"
    sep = "-" * 82

    # ---- Precision summary -------------------------------------------------
    print("=" * 82)
    print("Precision: fused vs two-pass (all must be bit-exact)")
    print("=" * 82)
    precision_shapes = [
        (1, 128), (7, 512), (16, 768), (63, 1024),
        (128, 2048), (1024, 768), (4096, 4096),
    ]
    all_pass = True
    for rows, cols in precision_shapes:
        torch.manual_seed(42)
        x = torch.randn(rows, cols, dtype=torch.bfloat16, device=device)
        for four_six in [False, True]:
            f_data, f_scale, f_gs = run_fused_linear(x, four_six)
            r_data, r_scale, r_gs = run_twopass_linear(x, four_six)
            ok = (
                torch.equal(f_data, r_data)
                and torch.equal(f_scale.view(torch.uint8), r_scale.view(torch.uint8))
                and torch.equal(f_gs, r_gs)
            )
            label = "4/6" if four_six else "RTN"
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"  [{status}] {rows:>6}x{cols:<5} {label}")
    print(sep)
    print("All precision checks:", "PASSED" if all_pass else "FAILED")

    # ---- Performance table -------------------------------------------------
    print()
    print("=" * 82)
    print("Performance: fused vs two-pass (median over 500 iters, CUDA events)")
    print("=" * 82)
    hdr = f"{'Shape':<14} {'fused(µs)':>10} {'twopass(µs)':>12} {'speedup':>9} {'BW(GB/s)':>10}"
    print(hdr)
    print(sep)

    bench_shapes = [
        (1,     512), (8,    768), (16,   768), (64,   768),
        (128,   768), (512,  768), (1024, 768), (4096, 768),
        (16384, 768), (1024, 512), (1024, 1024), (1024, 2048),
        (1024,  4096), (1024, 8192),
    ]
    for rows, cols in bench_shapes:
        x   = torch.randn(rows, cols, dtype=torch.bfloat16, device=device)
        x_c = x.contiguous()
        amax = x_c.abs().amax(dim=1).to(torch.float32)

        out_f, scale_f, gs_f = _alloc(rows, cols, device)
        out_r, scale_r, gs_r = _alloc(rows, cols, device)

        us_f  = _cuda_bench(fp4_gemm.rtn_fp4_pertoken_linear_fused,
                             out_f, scale_f, gs_f, x_c, 1.0)
        us_rk = _cuda_bench(fp4_gemm.rtn_fp4_pertoken_linear,
                             out_r, scale_r, gs_r, x_c, amax, 1.0)
        us_am = _cuda_bench(lambda: x_c.abs().amax(dim=1))
        us_r  = us_rk + us_am

        speedup = us_r / us_f
        bw      = _throughput_gbs(rows, cols, us_f)
        shape   = f"{rows}x{cols}"
        print(f"  {shape:<12} {us_f:>10.1f} {us_r:>12.1f} {speedup:>9.2f}x {bw:>9.1f}")

    print(sep)
