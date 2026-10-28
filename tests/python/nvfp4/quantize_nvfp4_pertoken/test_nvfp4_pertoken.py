"""Single-matrix per-token NVFP4 quant (QAT 6 foundation / Cursor2Ref STE).

What:  ``quantize_nvfp4_pertoken`` and ``quantize_nvfp4_pertoken_swizzled``.
Where: QAT 6 ``Nvfp4Cursor2QuantizeOpRef``; building block for grouped MoE X quant.
Compare: Python golden (``quantize_nvfp4_ref_pertoken_amax``), fused vs two-pass
         bit-exact, row-major vs unswizzled scales.

Markers: ``production`` (correctness), ``bench``+``slow`` (micro-benches).
Requires: fp4_gemm, CUDA sm_100+.
"""

from __future__ import annotations

import time

import pytest
import torch

try:
    import fp4_gemm
except ImportError:
    fp4_gemm = None

from miniTransformer.ops.nvfp4_quantize import (
    quantize_nvfp4_pertoken,
    quantize_nvfp4_pertoken_swizzled,
)
from miniTransformer.ops.nvfp4_ref_quantize import quantize_nvfp4_ref_pertoken_amax

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

pytestmark = [
    pytest.mark.production,
    pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or ""),
]

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
    scale        : float8_e4m3fn [m, n//16]
    global_scale : float32 [m]
    """
    grid = _FP4_GRID.to(data.device)
    indices = _unpack_fp4(data)
    fp4_vals = grid[indices].to(torch.float32)
    scale_expanded = scale.to(torch.float32).repeat_interleave(16, dim=1)
    gs = global_scale.to(torch.float32).unsqueeze(1)
    return (fp4_vals * scale_expanded * gs).to(torch.bfloat16)


def _unswizzle_scale(scale_flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Undo tcgen05 MMA swizzled layout to recover plain row-major [m, n//16]."""
    n_row_tiles = rows // 128
    n_col_tiles = (cols // 16) // 4
    s = scale_flat.reshape(n_row_tiles * n_col_tiles, 32, 4, 4)
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
    indices = _unpack_fp4(data)
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
# Kernel wrappers for fused / two-pass tests
# ---------------------------------------------------------------------------

_nvfp4_output_t = torch.uint8
_nvfp4_scale_t  = torch.float8_e4m3fn


def _alloc(rows: int, cols: int, device: str):
    out          = torch.empty((rows, cols // 2),  dtype=_nvfp4_output_t, device=device)
    scale        = torch.empty((rows, cols // 16), dtype=_nvfp4_scale_t,  device=device)
    global_scale = torch.empty(rows,               dtype=torch.float32,   device=device)
    return out, scale, global_scale


def _alloc_swizzled(rows: int, cols: int, device: str):
    out          = torch.empty((rows, cols // 2), dtype=_nvfp4_output_t, device=device)
    scale        = torch.empty(rows * cols // 16, dtype=_nvfp4_scale_t,  device=device)
    global_scale = torch.empty(rows,              dtype=torch.float32,   device=device)
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
# Per-token kernel precision tests
# ---------------------------------------------------------------------------

PERTOKEN_PRECISION_SHAPES = [(512, 128), (1024, 512), (1024, 4096), (4096, 2048)]
MIN_EFF_BITS = 3.3


@pytest.mark.parametrize("m,n", PERTOKEN_PRECISION_SHAPES)
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


@pytest.mark.parametrize("m,n", PERTOKEN_PRECISION_SHAPES)
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


@pytest.mark.parametrize("m,n", PERTOKEN_PRECISION_SHAPES)
def test_rowmajor_and_swizzled_same_data(m: int, n: int):
    """RTN row-major and RTN swizzled must produce identical FP4 data and global scale."""
    torch.manual_seed(99)
    x = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")

    r_rm = quantize_nvfp4_pertoken(x, four_six=False)
    r_sw = quantize_nvfp4_pertoken_swizzled(x, four_six=False)

    torch.testing.assert_close(r_rm.data, r_sw.data, atol=0, rtol=0)
    torch.testing.assert_close(r_rm.tensor_scale, r_sw.tensor_scale, atol=0, rtol=0)

    scale_sw_rm = _unswizzle_scale(r_sw.scale.to(torch.float32), m, n)
    torch.testing.assert_close(
        r_rm.scale.to(torch.float32), scale_sw_rm, atol=0, rtol=0,
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
    for i, mag in enumerate([1.0, 2.0, 4.0, 8.0]):
        x[i] = mag
    r = quantize_nvfp4_pertoken(x)
    assert r.tensor_scale is not None
    gs = r.tensor_scale
    for i in range(1, m):
        ratio = gs[i].item() / gs[i - 1].item()
        assert abs(ratio - 2.0) < 0.1, (
            f"global_scale ratio row{i}/row{i-1} = {ratio:.3f}, expected ~2.0"
        )


@pytest.mark.parametrize("four_six", [False, True])
def test_four_six_better_than_rtn_on_bimodal(four_six: bool):
    """Round-4-or-6 must not regress vs RTN effective bit-width on bimodal input."""
    torch.manual_seed(7)
    m, n = 512, 512
    x = torch.where(
        torch.rand(m, n) > 0.5,
        torch.randn(m, n) * 0.1,
        torch.randn(m, n) + 4.0,
    ).to(torch.bfloat16).cuda()

    r_rtn = quantize_nvfp4_pertoken(x, four_six=False)
    r_46  = quantize_nvfp4_pertoken(x, four_six=True)

    dq_rtn = dequantize_rowmajor(r_rtn.data, r_rtn.scale, r_rtn.tensor_scale)
    dq_46  = dequantize_rowmajor(r_46.data,  r_46.scale,  r_46.tensor_scale)

    bits_rtn = _eff_bitwidth(x, dq_rtn)
    bits_46  = _eff_bitwidth(x, dq_46)
    assert bits_46 >= bits_rtn - 0.05, (
        f"four_six ({bits_46:.3f} bits) significantly worse than RTN ({bits_rtn:.3f} bits)"
    )


# ---------------------------------------------------------------------------
# Fused kernel precision tests — fused must be bit-exact vs two-pass
# ---------------------------------------------------------------------------

FUSED_PRECISION_SHAPES = [
    (1,     128),
    (7,     512),
    (16,    768),
    (63,   1024),
    (128,  2048),
    (1024,  512),
    (1024,  768),
    (1024, 2048),
    (4096, 4096),
    (16384, 1024),
    (1025,  256),
    (2047,  512),
]


@pytest.mark.parametrize("rows,cols", FUSED_PRECISION_SHAPES)
@pytest.mark.parametrize("four_six", [False, True])
def test_fused_linear_bit_exact(rows: int, cols: int, four_six: bool):
    """Fused row-major must produce identical output to two-pass."""
    torch.manual_seed(42)
    x = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")

    f_data, f_scale, f_gs = run_fused_linear(x, four_six)
    r_data, r_scale, r_gs = run_twopass_linear(x, four_six)

    torch.testing.assert_close(f_data, r_data, atol=0, rtol=0, msg="FP4 data mismatch (linear)")
    torch.testing.assert_close(
        f_scale.view(torch.uint8), r_scale.view(torch.uint8),
        atol=0, rtol=0, msg="FP8 scale mismatch (linear)")
    torch.testing.assert_close(f_gs, r_gs, atol=0, rtol=0, msg="global_scale mismatch (linear)")


@pytest.mark.parametrize("rows,cols", FUSED_PRECISION_SHAPES)
@pytest.mark.parametrize("four_six", [False, True])
def test_fused_swizzled_bit_exact(rows: int, cols: int, four_six: bool):
    """Fused swizzled must produce identical output to two-pass."""
    torch.manual_seed(42)
    x = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")

    f_data, f_scale, f_gs = run_fused_swizzled(x, four_six)
    r_data, r_scale, r_gs = run_twopass_swizzled(x, four_six)

    torch.testing.assert_close(f_data, r_data, atol=0, rtol=0, msg="FP4 data mismatch (swizzled)")
    torch.testing.assert_close(
        f_scale.view(torch.uint8), r_scale.view(torch.uint8),
        atol=0, rtol=0, msg="FP8 scale mismatch (swizzled)")
    torch.testing.assert_close(f_gs, r_gs, atol=0, rtol=0, msg="global_scale mismatch (swizzled)")


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
    for i in range(rows):
        x[i] = float(2 ** i)
    _, _, gs = run_fused_linear(x)
    for i in range(1, rows):
        ratio = gs[i].item() / gs[i - 1].item()
        assert abs(ratio - 2.0) < 0.05, (
            f"global_scale ratio row{i}/row{i-1} = {ratio:.4f}, expected 2.0"
        )


# ---------------------------------------------------------------------------
# Golden comparison tests
# ---------------------------------------------------------------------------

GOLDEN_SHAPES = [
    (16,    128),
    (16,   4096),
    (32,    128),
    (32,   4096),
    (128,   128),
    (128,  4096),
    (512,  4096),
    (1024, 4096),
    (128, 14336),
    (1024, 14336),
    (4096, 4096),
    (4096, 7168),
    (8192, 8192)
]


@pytest.mark.parametrize("m,n", GOLDEN_SHAPES)
def test_pertoken_rowmajor_vs_golden_rtn(m: int, n: int):
    """quantize_nvfp4_pertoken (RTN) must be bit-exact vs Python golden."""
    torch.manual_seed(0)
    x = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")

    cuda_res = quantize_nvfp4_pertoken(x, four_six=False)
    ref_qx, ref_sx, _ = quantize_nvfp4_ref_pertoken_amax(x)

    torch.testing.assert_close(
        cuda_res.data, ref_qx, atol=0, rtol=0,
        msg=f"FP4 data mismatch vs golden ({m}x{n})",
    )
    torch.testing.assert_close(
        cuda_res.scale.view(torch.uint8), ref_sx.view(torch.uint8), atol=0, rtol=0,
        msg=f"FP8 scale mismatch vs golden ({m}x{n})",
    )


@pytest.mark.parametrize("m,n", GOLDEN_SHAPES)
def test_pertoken_rowmajor_vs_golden_foursix(m: int, n: int):
    """quantize_nvfp4_pertoken (4/6): shapes correct, scales finite and non-zero."""
    torch.manual_seed(0)
    x = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")

    cuda_res = quantize_nvfp4_pertoken(x, four_six=True)

    assert cuda_res.scale.shape == (m, n // 16), \
        f"Unexpected scale shape {cuda_res.scale.shape} for ({m}x{n})"
    scale_bytes = cuda_res.scale.view(torch.uint8)
    assert (scale_bytes != 0).all(), f"Zero FP8 scales found in 4/6 output ({m}x{n})"
    assert torch.isfinite(cuda_res.tensor_scale).all(), \
        f"Non-finite global scale in 4/6 output ({m}x{n})"


@pytest.mark.parametrize("rows,cols", GOLDEN_SHAPES)
def test_fused_linear_vs_golden(rows: int, cols: int):
    """Fused row-major kernel must be bit-exact vs Python golden."""
    if cols > 8192:
        pytest.skip(f"fused kernel does not support cols > 8192 (got {cols})")
    torch.manual_seed(0)
    x = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")

    f_data, f_scale, _ = run_fused_linear(x, four_six=False)
    ref_qx, ref_sx, _ = quantize_nvfp4_ref_pertoken_amax(x)

    torch.testing.assert_close(
        f_data, ref_qx, atol=0, rtol=0,
        msg=f"FP4 data mismatch vs golden ({rows}x{cols})",
    )
    torch.testing.assert_close(
        f_scale.view(torch.uint8), ref_sx.view(torch.uint8), atol=0, rtol=0,
        msg=f"FP8 scale mismatch vs golden ({rows}x{cols})",
    )


@pytest.mark.parametrize("rows,cols", GOLDEN_SHAPES)
def test_twopass_linear_vs_golden(rows: int, cols: int):
    """Two-pass row-major kernel must be bit-exact vs Python golden."""
    torch.manual_seed(0)
    x = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")

    r_data, r_scale, _ = run_twopass_linear(x, four_six=False)
    ref_qx, ref_sx, _ = quantize_nvfp4_ref_pertoken_amax(x)

    torch.testing.assert_close(
        r_data, ref_qx, atol=0, rtol=0,
        msg=f"FP4 data mismatch vs golden ({rows}x{cols})",
    )
    torch.testing.assert_close(
        r_scale.view(torch.uint8), ref_sx.view(torch.uint8), atol=0, rtol=0,
        msg=f"FP8 scale mismatch vs golden ({rows}x{cols})",
    )


# ---------------------------------------------------------------------------
# Performance benchmarks
# ---------------------------------------------------------------------------

_WARMUP  = 50
_REPEATS = 500

_WARMUP_ITERS = 20
_BENCH_ITERS  = 200

FUSED_BENCH_SHAPES = [
    (1,     512),
    (8,     768),
    (16,    768),
    (64,    768),
    (128,   768),
    (1024,  768),
    (4096,  768),
    (16384, 768),
    (1024,  512),
    (1024, 1024),
    (1024, 2048),
    (1024, 4096),
    (1024, 8192),
]

PERTOKEN_BENCH_SHAPES = [
    (1,    128),
    (32,   512),
    (128, 1024),
    (512, 2048),
    (1024, 4096),
    (4096, 4096),
]


def _cuda_bench(fn, *args, warmup: int = _WARMUP, repeats: int = _REPEATS) -> float:
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
        times.append(start.elapsed_time(end) * 1e3)

    times.sort()
    return times[repeats // 2]


def _wall_bench(fn, x: torch.Tensor) -> float:
    """Return average kernel time in milliseconds using wall clock."""
    for _ in range(_WARMUP_ITERS):
        fn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(_BENCH_ITERS):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / _BENCH_ITERS * 1000.0


def _throughput_gbs(rows: int, cols: int, us: float) -> float:
    """Effective memory bandwidth: BF16 read (2 B/elem) + FP4 write (0.5 B/elem)."""
    return rows * cols * 2.5 / (us * 1e-6) / 1e9


@pytest.mark.bench
@pytest.mark.slow
@pytest.mark.parametrize("rows,cols", FUSED_BENCH_SHAPES)
def test_bench_fused_vs_twopass(rows: int, cols: int):
    """Bench: fused kernel wall time ≤ two-pass + amax (5% tolerance)."""
    x   = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
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
    us_amax    = _cuda_bench(lambda: x_c.abs().amax(dim=1))
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


@pytest.mark.bench
@pytest.mark.slow
@pytest.mark.parametrize("m,n", PERTOKEN_BENCH_SHAPES)
def test_bench_pertoken_rowmajor(m: int, n: int):
    """Bench: ``quantize_nvfp4_pertoken`` row-major throughput (prints; always passes)."""
    x    = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
    ms   = _wall_bench(lambda t: quantize_nvfp4_pertoken(t), x)
    gb_s = (m * n * 2) / (ms * 1e-3) / 1e9
    print(f"\n[rowmajor] {m}x{n}: {ms:.3f} ms | ~{gb_s:.1f} GB/s")


@pytest.mark.bench
@pytest.mark.slow
@pytest.mark.parametrize("m,n", PERTOKEN_BENCH_SHAPES)
def test_bench_pertoken_swizzled(m: int, n: int):
    """Bench: ``quantize_nvfp4_pertoken_swizzled`` throughput (prints; always passes)."""
    x    = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
    ms   = _wall_bench(lambda t: quantize_nvfp4_pertoken_swizzled(t), x)
    gb_s = (m * n * 2) / (ms * 1e-3) / 1e9
    print(f"\n[swizzled] {m}x{n}: {ms:.3f} ms | ~{gb_s:.1f} GB/s")


@pytest.mark.bench
@pytest.mark.slow
@pytest.mark.parametrize("m,n", PERTOKEN_BENCH_SHAPES)
def test_bench_four_six_vs_rtn(m: int, n: int):
    """Bench: RTN vs Round-4-or-6 wall-time overhead (row-major)."""
    x      = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
    ms_rtn = _wall_bench(lambda t: quantize_nvfp4_pertoken(t, four_six=False), x)
    ms_46  = _wall_bench(lambda t: quantize_nvfp4_pertoken(t, four_six=True),  x)
    overhead = (ms_46 / ms_rtn - 1.0) * 100
    print(f"\n[RTN vs 4/6] {m}x{n}: RTN={ms_rtn:.3f} ms, 4/6={ms_46:.3f} ms, overhead={overhead:+.1f}%")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if _SKIP_REASON:
        print(f"Skipped: {_SKIP_REASON}")
        raise SystemExit(0)

    device = "cuda"
    sep = "-" * 82

    # ---- Per-token precision -----------------------------------------------
    print("=" * 82)
    print("Per-token precision (effective bit-width)")
    print("=" * 82)
    for m, n in [(512, 128), (1024, 4096), (4096, 2048)]:
        torch.manual_seed(42)
        x = torch.randn(m, n, dtype=torch.bfloat16, device=device)
        for four_six in [False, True]:
            r    = quantize_nvfp4_pertoken(x, four_six=four_six)
            dq   = dequantize_rowmajor(r.data, r.scale, r.tensor_scale)
            bits = _eff_bitwidth(x, dq)
            label = "4/6" if four_six else "RTN"
            print(f"  [{label}] {m}x{n}: {bits:.3f} eff-bits")

    # ---- Fused precision ---------------------------------------------------
    print()
    print("=" * 82)
    print("Fused vs two-pass precision (all must be bit-exact)")
    print("=" * 82)
    all_pass = True
    for rows, cols in [(1, 128), (7, 512), (16, 768), (63, 1024),
                       (128, 2048), (1024, 768), (4096, 4096)]:
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
            label  = "4/6" if four_six else "RTN"
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"  [{status}] {rows:>6}x{cols:<5} {label}")
    print(sep)
    print("All precision checks:", "PASSED" if all_pass else "FAILED")

    # ---- Fused performance -------------------------------------------------
    print()
    print("=" * 82)
    print("Fused vs two-pass performance (median 500 iters, CUDA events)")
    print("=" * 82)
    print(f"{'Shape':<14} {'fused(µs)':>10} {'twopass(µs)':>12} {'speedup':>9} {'BW(GB/s)':>10}")
    print(sep)
    for rows, cols in [(1, 512), (8, 768), (16, 768), (64, 768), (128, 768),
                       (512, 768), (1024, 768), (4096, 768), (16384, 768),
                       (1024, 512), (1024, 1024), (1024, 2048), (1024, 4096), (1024, 8192)]:
        x    = torch.randn(rows, cols, dtype=torch.bfloat16, device=device)
        x_c  = x.contiguous()
        amax = x_c.abs().amax(dim=1).to(torch.float32)
        out_f, scale_f, gs_f = _alloc(rows, cols, device)
        out_r, scale_r, gs_r = _alloc(rows, cols, device)
        us_f  = _cuda_bench(fp4_gemm.rtn_fp4_pertoken_linear_fused, out_f, scale_f, gs_f, x_c, 1.0)
        us_rk = _cuda_bench(fp4_gemm.rtn_fp4_pertoken_linear,       out_r, scale_r, gs_r, x_c, amax, 1.0)
        us_am = _cuda_bench(lambda: x_c.abs().amax(dim=1))
        us_r  = us_rk + us_am
        speedup = us_r / us_f
        shape   = f"{rows}x{cols}"
        print(f"  {shape:<12} {us_f:>10.1f} {us_r:>12.1f} {speedup:>9.2f}x {_throughput_gbs(rows, cols, us_f):>9.1f}")
    print(sep)

    # ---- Per-token speed ---------------------------------------------------
    print()
    print("=" * 82)
    print("Per-token kernel speed (wall clock)")
    print("=" * 82)
    print(f"{'Shape':<16} {'rowmajor(ms)':<16} {'swizzled(ms)':<16} {'RTN(ms)':<12} {'4/6(ms)':<12}")
    print(sep)
    for m, n in PERTOKEN_BENCH_SHAPES:
        if n < 128 or n % 128 != 0:
            continue
        x     = torch.randn(m, n, dtype=torch.bfloat16, device=device)
        ms_rm = _wall_bench(lambda t: quantize_nvfp4_pertoken(t),                   x)
        ms_sw = _wall_bench(lambda t: quantize_nvfp4_pertoken_swizzled(t),          x)
        ms_rt = _wall_bench(lambda t: quantize_nvfp4_pertoken(t, four_six=False),   x)
        ms_46 = _wall_bench(lambda t: quantize_nvfp4_pertoken(t, four_six=True),    x)
        print(f"  {f'{m}x{n}':<14} {ms_rm:<16.3f} {ms_sw:<16.3f} {ms_rt:<12.3f} {ms_46:<12.3f}")
