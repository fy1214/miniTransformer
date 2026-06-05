"""
Correctness tests for mxfp8_grouped_gemm (CUTLASS SM100, zero-overhead).

Reference: dequantize A/B from MXFP8 + E8M0 scale factors to FP32, then
           run fp32 matmul group-by-group. Compare against CUTLASS BF16 output.

Scale factor layout: CUTLASS SfKMajorAtom (swizzled), produced by
mxfp8_grouped_quantize which calls the C++ kernel directly.
"""

import os
import sys

import pytest
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

pytest.importorskip("fp4_gemm")

from miniTransformer.ops.mxfp8_quantize import mxfp8_grouped_quantize, _sf_size_per_group
from miniTransformer.ops.mxfp8_grouped_cutlass_gemm import MxFP8GemmInputs, mxfp8_grouped_gemm


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------

def _e8m0_to_float(raw: torch.Tensor) -> torch.Tensor:
    """Convert uint8 E8M0 byte → float32 via 2^(raw - 127)."""
    return torch.pow(2.0, raw.float() - 127.0)


def _mxfp8_dequant_group(
    fp8_data: torch.Tensor,   # [M, K] uint8 (FP8 E4M3)
    sf_flat:  torch.Tensor,   # flat uint8, swizzled E8M0 SFs for this group
    M: int, K: int,
) -> torch.Tensor:
    """
    Dequantize one group from MXFP8 to FP32.

    The swizzled layout is SfKMajorAtom:
        shape  = (num_m_tiles, num_k_tiles, 32, 16)
                  [128-row tiles, 4-k-group tiles, elements within tile]
        where num_m_tiles = ceil(M/128), num_k_tiles = ceil((K//32)/4)
        and the 32 × 16 inner tile stores 32 rows × (4 k-groups of 4) = 16 sf/row

    For simplicity we dequantize via:
      1. Interpret the raw FP8 bytes as E4M3 via view(float8_e4m3fn)
      2. Un-swizzle the scale factors back to [M_pad, K//32]
      3. Expand scales to [M_pad, K] and multiply element-wise
      4. Slice to [M, K] and cast to float32.
    """
    # Step 1: cast FP8 data to float32 via torch native float8
    data_fp8 = fp8_data.view(torch.float8_e4m3fn).float()  # [M, K] float32

    # Step 2: un-swizzle SFs from (num_m, num_k, 32, 16) → [M_pad, K//32]
    num_m = (M + 127) // 128
    num_k = (K // 32 + 3) // 4
    M_pad = num_m * 128

    # swizzled buffer: (num_m, num_k, 32, 16)
    sf_4d = sf_flat.reshape(num_m, num_k, 32, 16)

    # Inverse of the swizzle: reshape to (num_m, num_k, 32, 4, 4) -> permute(0,3,2,1,4)
    #   -> reshape to (num_m, 4, 32, num_k, 4) -> reshape to (M_pad, K//32)
    sf_5d = sf_4d.reshape(num_m, num_k, 32, 4, 4)        # last 16 = (4, 4)
    sf_orig = sf_5d.permute(0, 3, 2, 1, 4).contiguous()  # (num_m, 4, 32, num_k, 4)
    sf_2d = sf_orig.reshape(M_pad, K // 32)               # [M_pad, K//32]

    # Slice to actual M rows
    sf_2d = sf_2d[:M, :]   # [M, K//32]

    # Step 3: decode E8M0 → float32 and expand to [M, K]
    sf_f32 = _e8m0_to_float(sf_2d)        # [M, K//32]
    sf_exp = sf_f32.repeat_interleave(32, dim=1)  # [M, K]

    # Step 4: multiply and return
    return (data_fp8 * sf_exp).to(torch.float32)


def _mxfp8_ref(
    A_fp8: torch.Tensor,   # [M, K] uint8
    B_fp8: torch.Tensor,   # [N, K] uint8 (B is stored transposed in K-major)
    A_sf:  torch.Tensor,   # flat uint8 swizzled SFs for A
    B_sf:  torch.Tensor,   # flat uint8 swizzled SFs for B
    M: int, N: int, K: int,
) -> torch.Tensor:
    """Reference matmul: dequant both A and B, then do fp32 A @ B^T."""
    A_f32 = _mxfp8_dequant_group(A_fp8, A_sf, M, K)   # [M, K]
    B_f32 = _mxfp8_dequant_group(B_fp8, B_sf, N, K)   # [N, K]
    return torch.matmul(A_f32, B_f32.t()).to(torch.float32)  # [M, N]


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------

def _build_inputs(shapes, seed=42, device="cuda"):
    """
    Build MxFP8GemmInputs for mxfp8_grouped_gemm.

    shapes: list of (M_g, N_g, K) — K must be the same for all groups.

    Returns:
        inputs        : MxFP8GemmInputs ready for mxfp8_grouped_gemm
        per_group_info: list of (m_g, n_g, K, a_sf_g, b_sf_g) for reference checks
    """
    K = shapes[0][2]
    G = len(shapes)

    a_fp32_list, b_fp32_list = [], []
    for i, (m, n, k) in enumerate(shapes):
        assert k == K, "All groups must share the same K"
        torch.manual_seed(seed + i)
        torch.cuda.manual_seed(seed + i)
        a_fp32_list.append(torch.randn(m, K, device=device, dtype=torch.bfloat16))
        b_fp32_list.append(torch.randn(n, K, device=device, dtype=torch.bfloat16))

    m_splits = [s[0] for s in shapes]
    n_splits = [s[1] for s in shapes]

    A_bf16 = torch.cat(a_fp32_list, dim=0).contiguous()
    B_bf16 = torch.cat(b_fp32_list, dim=0).contiguous()

    A, A_sf_flat, a_row_offsets, a_sf_offsets = mxfp8_grouped_quantize(A_bf16, m_splits)
    B, B_sf_flat, b_row_offsets, b_sf_offsets = mxfp8_grouped_quantize(B_bf16, n_splits)

    # CPU problem sizes tensor [G*3]: (M_g, N_g, K) per group
    prob_host = torch.zeros(G * 3, dtype=torch.int32)
    for i, (m, n, k) in enumerate(shapes):
        prob_host[i * 3 + 0] = m
        prob_host[i * 3 + 1] = n
        prob_host[i * 3 + 2] = k

    inputs = MxFP8GemmInputs(
        act_data=A,
        act_sf=A_sf_flat,
        a_rows=a_row_offsets,
        a_sf_offsets=a_sf_offsets,
        wgt_data=B,
        wgt_sf=B_sf_flat,
        b_rows=b_row_offsets,
        b_sf_offsets=b_sf_offsets,
        prob_host=prob_host,
    )

    # Extract per-group SF slices for the reference dequant
    per_group_info = []
    for g in range(G):
        m_g = int((a_row_offsets[g + 1] - a_row_offsets[g]).item())
        n_g = int((b_row_offsets[g + 1] - b_row_offsets[g]).item())
        a_sf_start = int(a_sf_offsets[g].item())
        a_sf_end   = int(a_sf_offsets[g + 1].item())
        b_sf_start = int(b_sf_offsets[g].item())
        b_sf_end   = int(b_sf_offsets[g + 1].item())
        per_group_info.append((m_g, n_g, K, A_sf_flat[a_sf_start:a_sf_end], B_sf_flat[b_sf_start:b_sf_end]))

    return inputs, per_group_info


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    prop = torch.cuda.get_device_properties(0)
    if prop.major < 10:
        pytest.skip(f"Requires SM100 (Blackwell); got SM{prop.major}{prop.minor}")


def _error_metrics(pred: torch.Tensor, ref: torch.Tensor):
    pred_f = pred.float()
    ref_f  = ref.float()
    diff   = (pred_f - ref_f).abs()
    rmse   = torch.sqrt(torch.mean((pred_f - ref_f) ** 2))
    ref_rms = torch.sqrt(torch.mean(ref_f ** 2)).clamp_min(1e-8)
    return {
        "max_abs":   float(diff.max()),
        "mean_abs":  float(diff.mean()),
        "rmse":      float(rmse),
        "norm_rmse": float(rmse / ref_rms),
    }


def _print_metrics(prefix: str, m: dict):
    print(
        f"{prefix}: max_abs={m['max_abs']:.6f}  mean_abs={m['mean_abs']:.6f}"
        f"  rmse={m['rmse']:.6f}  norm_rmse={m['norm_rmse']:.6f}"
    )


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

# (m_splits, N, K)  — K and N shared by all groups; M varies (ragged M)
GROUPED_CASES = [
    pytest.param([128],        128,  128,  id="single_128_128_128"),
    pytest.param([128, 256],   256,  256,  id="two_groups_128_256_N256_K256"),
    pytest.param([128, 128, 128], 512, 256, id="three_groups_equal"),
    pytest.param([256, 512],   256,  512,  id="varied_M_N256_K512"),
    pytest.param([128],       4096, 7168,  id="deepgemm_shape"),
    pytest.param([64, 128, 192], 256, 128, id="ragged_M_three_groups"),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMxfp8GroupedGemmCorrectness:

    @pytest.mark.parametrize("m_splits, n, k", GROUPED_CASES)
    @pytest.mark.parametrize("seed", [42])
    @pytest.mark.parametrize("use_2sm", [False, True])
    def test_correctness_vs_fp32_ref(self, m_splits, n, k, seed, use_2sm):
        """CUTLASS MXFP8 grouped GEMM output must be close to FP32 dequant reference."""
        _require_sm100()
        device = "cuda"
        shapes = [(m, n, k) for m in m_splits]

        inputs, per_group_info = _build_inputs(shapes, seed=seed, device=device)

        D = mxfp8_grouped_gemm(inputs, out_dtype=torch.bfloat16, use_2sm=use_2sm)
        torch.cuda.synchronize()

        a_row_offsets = inputs.a_rows
        b_row_offsets = inputs.b_rows
        A = inputs.act_data
        B = inputs.wgt_data

        for g, (m_g, n_g, K, a_sf_g, b_sf_g) in enumerate(per_group_info):
            row_start = int(a_row_offsets[g].item())
            row_end   = int(a_row_offsets[g + 1].item())
            col_start = int(b_row_offsets[g].item())
            col_end   = int(b_row_offsets[g + 1].item())

            D_g = D[row_start:row_end, :]
            A_g = A[row_start:row_end]
            B_g = B[col_start:col_end]

            ref = _mxfp8_ref(A_g, B_g, a_sf_g, b_sf_g, m_g, n_g, K)
            ref_bf16 = ref.bfloat16()

            assert not torch.isnan(D_g).any(), f"group {g} has NaN (use_2sm={use_2sm})"
            assert not torch.isinf(D_g).any(), f"group {g} has Inf (use_2sm={use_2sm})"
            assert D_g.dtype == torch.bfloat16

            m_metrics = _error_metrics(D_g, ref_bf16)
            print(f"\ngroup {g} M={m_g} N={n_g} K={K} use_2sm={use_2sm}")
            _print_metrics("  cutlass_vs_ref", m_metrics)

            assert m_metrics["norm_rmse"] < 0.05, (
                f"group {g} norm_rmse={m_metrics['norm_rmse']:.6f} >= 0.05"
            )
            assert m_metrics["max_abs"] < 0.5, (
                f"group {g} max_abs={m_metrics['max_abs']:.6f} >= 0.5"
            )

    def test_output_shape_and_dtype(self):
        _require_sm100()
        device = "cuda"
        N, K = 256, 128
        shapes = [(128, N, K), (256, N, K)]

        inputs, _ = _build_inputs(shapes, seed=0, device=device)
        total_M = int(inputs.act_data.size(0))

        D = mxfp8_grouped_gemm(inputs, out_dtype=torch.bfloat16, use_2sm=False)
        torch.cuda.synchronize()

        assert D.dtype == torch.bfloat16
        assert D.shape == (total_M, N)
        assert not torch.isnan(D).any()

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_seed_sweep(self, seed):
        _require_sm100()
        device = "cuda"
        N, K = 256, 256
        shapes = [(128, N, K), (256, N, K), (64, N, K)]

        inputs, per_group_info = _build_inputs(shapes, seed=seed, device=device)

        D = mxfp8_grouped_gemm(inputs, out_dtype=torch.bfloat16, use_2sm=False)
        torch.cuda.synchronize()

        a_row_offsets = inputs.a_rows
        b_row_offsets = inputs.b_rows
        A = inputs.act_data
        B = inputs.wgt_data

        for g, (m_g, n_g, K_g, a_sf_g, b_sf_g) in enumerate(per_group_info):
            row_start = int(a_row_offsets[g].item())
            row_end   = int(a_row_offsets[g + 1].item())
            col_start = int(b_row_offsets[g].item())
            col_end   = int(b_row_offsets[g + 1].item())
            D_g = D[row_start:row_end, :]
            A_g = A[row_start:row_end]
            B_g = B[col_start:col_end]
            ref = _mxfp8_ref(A_g, B_g, a_sf_g, b_sf_g, m_g, n_g, K_g).bfloat16()
            m_metrics = _error_metrics(D_g, ref)
            assert m_metrics["norm_rmse"] < 0.05, (
                f"seed={seed} group {g} norm_rmse={m_metrics['norm_rmse']:.6f}"
            )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

_BENCH_CONFIGS = [
    pytest.param([(128, 4096, 7168)] * 8, id="moe_8x128_4096_7168"),
    pytest.param([(128, 7168, 4096)] * 8, id="moe_8x128_7168_4096"),
    pytest.param([(256, 4096, 7168)] * 8, id="moe_8x256_4096_7168"),
    pytest.param([(128, 128, 128)],        id="single_128_128_128"),
]

_BENCH_WARMUP = 5
_BENCH_ITERS  = 20


def _cuda_time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _tflops(shapes, elapsed_ms: float) -> float:
    flops = sum(2 * m * n * k for m, n, k in shapes)
    return flops / (elapsed_ms * 1e-3) / 1e12


class TestMxfp8GroupedGemmBenchmark:

    @pytest.mark.parametrize("shapes", _BENCH_CONFIGS)
    @pytest.mark.parametrize("use_2sm", [False, True])
    def test_benchmark(self, shapes, use_2sm):
        _require_sm100()
        device = "cuda"

        inputs, _ = _build_inputs(shapes, seed=42, device=device)

        def fn():
            mxfp8_grouped_gemm(inputs, out_dtype=torch.bfloat16, use_2sm=use_2sm)

        # warmup + first run to trigger CUTLASS initialize
        fn()
        torch.cuda.synchronize()

        elapsed_ms = _cuda_time_ms(fn, warmup=_BENCH_WARMUP, iters=_BENCH_ITERS)
        tf = _tflops(shapes, elapsed_ms)
        sm = "2SM" if use_2sm else "1SM"
        G = len(shapes)
        print(
            f"\n  [{sm}] groups={G}  shapes={shapes[0]}{'...' if G > 1 else ''}"
            f"  latency={elapsed_ms * 1e3:.1f} µs  TFLOPS={tf:.2f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
