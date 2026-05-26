"""
Correctness and absolute-accuracy tests for grouped_nvfp4_cutlass_gemm_per_token_scale.

Checks two things:
1. Kernel consistency: CUTLASS grouped FP4 output matches the quantized reference path.
2. Absolute precision: CUTLASS grouped FP4 output is compared with FP32 matmul ground truth.
"""

import os
import sys

import pytest
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

pytest.importorskip("fp4_gemm")

from miniTransformer.quantiztion import quantization
from miniTransformer.quantiztion.nvfp4_quantize_op_ref import Nvfp4TiledQuantizeRefOp
from miniTransformer.ops.nvfp4_grouped_cutlass_gemm import (
    grouped_nvfp4_cutlass_gemm_per_token_scale,
)
from miniTransformer.ops.nvfp4_qgemm import swizzle_scale


_QPARAMS = quantization.QParams(
    scaling_dtype=torch.float8_e4m3fn,
    output_dtype=torch.uint8,
    eps=0.0,
    pow_2_scales=False,
    quant_tile_shape=(1, 16),
    with_2d_quantization=False,
    with_rht=False,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    prop = torch.cuda.get_device_properties(0)
    if prop.major < 10:
        pytest.skip(f"Requires SM100 (Blackwell); got SM{prop.major}{prop.minor}")


def _error_metrics(pred: torch.Tensor, ref: torch.Tensor):
    pred_f = pred.float()
    ref_f = ref.float()
    diff = (pred_f - ref_f).abs()
    rmse = torch.sqrt(torch.mean((pred_f - ref_f) ** 2))
    ref_rms = torch.sqrt(torch.mean(ref_f ** 2)).clamp_min(1e-8)
    norm_rmse = rmse / ref_rms
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse": float(rmse.item()),
        "norm_rmse": float(norm_rmse.item()),
    }


def _build_group(m: int, n: int, k: int, device: str, seed: int):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    x_fp32 = torch.randn(m, k, device=device, dtype=torch.float32)
    w_fp32 = torch.empty(n, k, device=device, dtype=torch.float32)
    torch.nn.init.normal_(w_fp32, mean=0.0, std=0.023)

    quantize_op = Nvfp4TiledQuantizeRefOp()
    qres_x = quantize_op.quantize(x_fp32, _QPARAMS, return_transpose=False)
    qres_w = quantize_op.quantize(w_fp32, _QPARAMS, return_transpose=False)

    qx = qres_x.data
    qw = qres_w.data
    sx_2d = qres_x.scale.to(torch.float8_e4m3fn)
    sw_2d = qres_w.scale.to(torch.float8_e4m3fn)

    global_amax_x = qres_x.tensor_scale
    global_amax_w = qres_w.tensor_scale_t
    if global_amax_x is None:
        global_amax_x = x_fp32.abs().max().to(torch.float32).view(1)
    if global_amax_w is None:
        global_amax_w = w_fp32.abs().max().to(torch.float32).view(1)
    if global_amax_x.numel() > 1:
        global_amax_x = global_amax_x.max().view(1)
    if global_amax_w.numel() > 1:
        global_amax_w = global_amax_w.max().view(1)

    sx_4d = swizzle_scale(sx_2d.float(), m, k).to(torch.float8_e4m3fn)
    sw_4d = swizzle_scale(sw_2d.float(), n, k).to(torch.float8_e4m3fn)

    return x_fp32, w_fp32, qx, qw, sx_2d, sw_2d, sx_4d, sw_4d, global_amax_x, global_amax_w


def _fp32_ground_truth(
    x_fp32: torch.Tensor,
    w_fp32: torch.Tensor,
    per_token_scale: torch.Tensor,
) -> torch.Tensor:
    y = torch.matmul(x_fp32.float(), w_fp32.float().t())
    return y * per_token_scale.float().unsqueeze(1)


def _quantized_reference(
    qx: torch.Tensor,
    qw: torch.Tensor,
    sx_2d: torch.Tensor,
    sw_2d: torch.Tensor,
    global_amax_x: torch.Tensor,
    global_amax_w: torch.Tensor,
    per_token_scale: torch.Tensor,
) -> torch.Tensor:
    quantize_op = Nvfp4TiledQuantizeRefOp()
    y = quantize_op._ref_qgemm(
        qx=qx,
        qw=qw,
        m_params=None,
        out_dtype=torch.float32,
        sx=sx_2d,
        sw=sw_2d,
        bias=None,
        out=None,
        accumulate=False,
        pow_2_scales=False,
        global_amax_qx=global_amax_x,
        global_amax_qw=global_amax_w,
        qparams_x=_QPARAMS,
        qparams_w=_QPARAMS,
    )
    return y.float() * per_token_scale.float().unsqueeze(1)


def _print_metrics(prefix: str, metrics: dict):
    print(
        f"{prefix}: "
        f"max_abs={metrics['max_abs']:.6f}, "
        f"mean_abs={metrics['mean_abs']:.6f}, "
        f"rmse={metrics['rmse']:.6f}, "
        f"norm_rmse={metrics['norm_rmse']:.6f}"
    )


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------


GROUPED_CASES = [
    pytest.param([128], 128, 128, id="single_group_small"),
    pytest.param([128, 256], 256, 256, id="two_groups"),
    pytest.param([128, 128, 128], 512, 256, id="three_groups"),
    pytest.param([256, 512], 256, 512, id="varied_M"),
    pytest.param([128], 7168, 4096, id="deepgemm_shape"),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGroupedCutlassGemmPerTokenScale:
    @staticmethod
    def _run_one(m_splits, n, k, device, seed, out_dtype=torch.bfloat16):
        a_list, b_list = [], []
        a_sf_list, b_sf_list = [], []
        per_token_scale_list = []
        tensor_scale_a_list, tensor_scale_b_list = [], []
        qref_list, fp32_ref_list = [], []

        for i, m in enumerate(m_splits):
            (
                x_fp32,
                w_fp32,
                qx,
                qw,
                sx_2d,
                sw_2d,
                sx_4d,
                sw_4d,
                global_amax_x,
                global_amax_w,
            ) = _build_group(m, n, k, device=device, seed=seed + i)

            per_token_scale = torch.rand(m, device=device, dtype=torch.float32) * 0.5 + 0.5

            a_list.append(qx)
            b_list.append(qw)
            a_sf_list.append(sx_4d)
            b_sf_list.append(sw_4d)
            per_token_scale_list.append(per_token_scale)
            tensor_scale_a_list.append(global_amax_x)
            tensor_scale_b_list.append(global_amax_w)

            qref_list.append(
                _quantized_reference(
                    qx,
                    qw,
                    sx_2d,
                    sw_2d,
                    global_amax_x,
                    global_amax_w,
                    per_token_scale,
                )
            )
            fp32_ref_list.append(_fp32_ground_truth(x_fp32, w_fp32, per_token_scale))

        d_list = grouped_nvfp4_cutlass_gemm_per_token_scale(
            a_list=a_list,
            b_list=b_list,
            a_sf_list=a_sf_list,
            b_sf_list=b_sf_list,
            per_token_scale_list=per_token_scale_list,
            out_dtype=out_dtype,
            sf_is_swizzled=True,
            tensor_scale_a_list=tensor_scale_a_list,
            tensor_scale_b_list=tensor_scale_b_list,
        )

        return d_list, qref_list, fp32_ref_list

    @pytest.mark.parametrize("m_splits, n, k", GROUPED_CASES)
    @pytest.mark.parametrize("seed", [42])
    @pytest.mark.parametrize("out_dtype", [torch.bfloat16])
    def test_matches_quantized_reference_and_fp32_ground_truth(
        self, m_splits, n, k, seed, out_dtype
    ):
        _require_sm100()
        device = "cuda"
        d_list, qref_list, fp32_ref_list = self._run_one(
            m_splits, n, k, device=device, seed=seed, out_dtype=out_dtype
        )

        for i, (got, qref, fp32_ref) in enumerate(zip(d_list, qref_list, fp32_ref_list)):
            assert got.shape == qref.shape == fp32_ref.shape
            assert got.dtype == out_dtype
            assert not torch.isnan(got).any(), f"group {i} has NaN"
            assert not torch.isinf(got).any(), f"group {i} has Inf"

            kernel_metrics = _error_metrics(got, qref)
            accuracy_metrics = _error_metrics(got, fp32_ref)

            print(f"\ngroup {i} M={m_splits[i]}, N={n}, K={k}")
            _print_metrics("  cutlass_vs_quantized_ref", kernel_metrics)
            _print_metrics("  cutlass_vs_fp32_ground_truth", accuracy_metrics)

            assert kernel_metrics["max_abs"] < 0.15, (
                f"group {i} kernel mismatch max_abs={kernel_metrics['max_abs']:.6f}"
            )
            assert kernel_metrics["mean_abs"] < 0.03, (
                f"group {i} kernel mismatch mean_abs={kernel_metrics['mean_abs']:.6f}"
            )
            assert kernel_metrics["rmse"] < 0.04, (
                f"group {i} kernel mismatch rmse={kernel_metrics['rmse']:.6f}"
            )

            assert accuracy_metrics["norm_rmse"] < 0.20, (
                f"group {i} absolute precision norm_rmse={accuracy_metrics['norm_rmse']:.6f}"
            )
            assert accuracy_metrics["mean_abs"] < 0.12, (
                f"group {i} absolute precision mean_abs={accuracy_metrics['mean_abs']:.6f}"
            )

    @pytest.mark.parametrize("m_splits, n, k", GROUPED_CASES)
    @pytest.mark.parametrize("seed", [7])
    def test_output_shape_and_dtype(self, m_splits, n, k, seed):
        _require_sm100()
        device = "cuda"
        a_list, b_list, a_sf_list, b_sf_list, pts_list = [], [], [], [], []
        ts_a_list, ts_b_list = [], []

        for i, m in enumerate(m_splits):
            (
                _,
                _,
                qx,
                qw,
                _,
                _,
                sx_4d,
                sw_4d,
                global_amax_x,
                global_amax_w,
            ) = _build_group(m, n, k, device=device, seed=seed + i)
            a_list.append(qx)
            b_list.append(qw)
            a_sf_list.append(sx_4d)
            b_sf_list.append(sw_4d)
            pts_list.append(torch.ones(m, device=device, dtype=torch.float32))
            ts_a_list.append(global_amax_x)
            ts_b_list.append(global_amax_w)

        d_list = grouped_nvfp4_cutlass_gemm_per_token_scale(
            a_list,
            b_list,
            a_sf_list,
            b_sf_list,
            pts_list,
            out_dtype=torch.bfloat16,
            tensor_scale_a_list=ts_a_list,
            tensor_scale_b_list=ts_b_list,
        )
        assert len(d_list) == len(m_splits)
        for i, (d, m) in enumerate(zip(d_list, m_splits)):
            assert d.shape == (m, n), f"group {i}: expected ({m},{n}), got {d.shape}"
            assert d.dtype == torch.bfloat16, f"group {i} dtype mismatch"

    def test_empty_group_list(self):
        result = grouped_nvfp4_cutlass_gemm_per_token_scale(
            a_list=[],
            b_list=[],
            a_sf_list=[],
            b_sf_list=[],
            per_token_scale_list=[],
        )
        assert result == []

    def test_per_token_scale_ones_matches_quantized_reference(self):
        _require_sm100()
        device = "cuda"
        m, n, k = 128, 128, 256
        seed = 99

        (
            _,
            _,
            qx,
            qw,
            sx_2d,
            sw_2d,
            sx_4d,
            sw_4d,
            global_amax_x,
            global_amax_w,
        ) = _build_group(m, n, k, device=device, seed=seed)
        pts_ones = torch.ones(m, device=device, dtype=torch.float32)

        got = grouped_nvfp4_cutlass_gemm_per_token_scale(
            a_list=[qx],
            b_list=[qw],
            a_sf_list=[sx_4d],
            b_sf_list=[sw_4d],
            per_token_scale_list=[pts_ones],
            out_dtype=torch.bfloat16,
            tensor_scale_a_list=[global_amax_x],
            tensor_scale_b_list=[global_amax_w],
        )[0]

        ref = _quantized_reference(
            qx,
            qw,
            sx_2d,
            sw_2d,
            global_amax_x,
            global_amax_w,
            pts_ones,
        )

        metrics = _error_metrics(got, ref)
        assert metrics["max_abs"] < 0.15
        assert metrics["mean_abs"] < 0.03

    def test_preallocated_output(self):
        _require_sm100()
        device = "cuda"
        m, n, k = 128, 256, 128
        seed = 55

        (
            _,
            _,
            qx,
            qw,
            _,
            _,
            sx_4d,
            sw_4d,
            global_amax_x,
            global_amax_w,
        ) = _build_group(m, n, k, device=device, seed=seed)
        pts = torch.rand(m, device=device, dtype=torch.float32)
        out = torch.zeros(m, n, device=device, dtype=torch.bfloat16)

        d_list = grouped_nvfp4_cutlass_gemm_per_token_scale(
            a_list=[qx],
            b_list=[qw],
            a_sf_list=[sx_4d],
            b_sf_list=[sw_4d],
            per_token_scale_list=[pts],
            out_dtype=torch.bfloat16,
            out_list=[out],
            tensor_scale_a_list=[global_amax_x],
            tensor_scale_b_list=[global_amax_w],
        )

        assert d_list[0].data_ptr() == out.data_ptr(), "output tensor not reused"
        assert not torch.all(out == 0), "output was not written"

    @pytest.mark.parametrize(
        "m_splits, n, k",
        [pytest.param([128, 256, 128], 256, 256, id="three_groups_seed_sweep")],
    )
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_seed_sweep(self, m_splits, n, k, seed):
        _require_sm100()
        device = "cuda"
        d_list, qref_list, fp32_ref_list = self._run_one(
            m_splits, n, k, device=device, seed=seed, out_dtype=torch.bfloat16
        )
        for i, (got, qref, fp32_ref) in enumerate(zip(d_list, qref_list, fp32_ref_list)):
            kernel_metrics = _error_metrics(got, qref)
            accuracy_metrics = _error_metrics(got, fp32_ref)
            assert kernel_metrics["max_abs"] < 0.15, (
                f"seed={seed} group {i}: max_abs={kernel_metrics['max_abs']:.6f}"
            )
            assert kernel_metrics["mean_abs"] < 0.03, (
                f"seed={seed} group {i}: mean_abs={kernel_metrics['mean_abs']:.6f}"
            )
            assert accuracy_metrics["norm_rmse"] < 0.20, (
                f"seed={seed} group {i}: norm_rmse={accuracy_metrics['norm_rmse']:.6f}"
            )

_BENCH_CONFIGS = [
    pytest.param([(128, 4096, 7168)] * 8, id="moe_8x128_4096_7168"),
    pytest.param([(128, 7168, 4096)] * 8, id="moe_8x128_7168_4096"),
    pytest.param([(256, 4096, 7168)] * 8, id="moe_8x256_4096_7168"),
    pytest.param([(512, 4096, 7168)] * 4, id="moe_4x512_4096_7168"),
    pytest.param([(128, 128, 128)],        id="single_128_128_128"),
    pytest.param([(128, 7168, 4096)],      id="single_128_7168_4096"),
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


class TestGroupedCutlassGemmBenchmark:
    @staticmethod
    def _build_inputs(shapes, seed=42):
        device = "cuda"
        quantize_op = Nvfp4TiledQuantizeRefOp()
        a_list, b_list, a_sf_list, b_sf_list, pts_list, ts_a_list, ts_b_list = [], [], [], [], [], [], []

        for i, (m, n, k) in enumerate(shapes):
            torch.manual_seed(seed + i)
            torch.cuda.manual_seed(seed + i)

            x = torch.randn(m, k, device=device, dtype=torch.float32)
            w = torch.empty(n, k, device=device, dtype=torch.float32)
            torch.nn.init.normal_(w, mean=0.0, std=0.023)

            qres_x = quantize_op.quantize(x, _QPARAMS, return_transpose=False)
            qres_w = quantize_op.quantize(w, _QPARAMS, return_transpose=False)

            sx_4d = swizzle_scale(qres_x.scale.float(), m, k).to(torch.float8_e4m3fn)
            sw_4d = swizzle_scale(qres_w.scale.float(), n, k).to(torch.float8_e4m3fn)

            ts_a = qres_x.tensor_scale
            ts_b = qres_w.tensor_scale_t
            if ts_a is None:
                ts_a = x.abs().max().float().view(1)
            if ts_b is None:
                ts_b = w.abs().max().float().view(1)
            if ts_a.numel() > 1:
                ts_a = ts_a.max().view(1)
            if ts_b.numel() > 1:
                ts_b = ts_b.max().view(1)

            a_list.append(qres_x.data)
            b_list.append(qres_w.data)
            a_sf_list.append(sx_4d)
            b_sf_list.append(sw_4d)
            pts_list.append(torch.ones(m, device=device, dtype=torch.float32))
            ts_a_list.append(ts_a)
            ts_b_list.append(ts_b)

        return a_list, b_list, a_sf_list, b_sf_list, pts_list, ts_a_list, ts_b_list

    @pytest.mark.parametrize("shapes", _BENCH_CONFIGS)
    def test_benchmark(self, shapes):
        _require_sm100()
        a_list, b_list, a_sf_list, b_sf_list, pts_list, ts_a_list, ts_b_list = (
            self._build_inputs(shapes)
        )

        def fn():
            grouped_nvfp4_cutlass_gemm_per_token_scale(
                a_list=a_list,
                b_list=b_list,
                a_sf_list=a_sf_list,
                b_sf_list=b_sf_list,
                per_token_scale_list=pts_list,
                out_dtype=torch.bfloat16,
                sf_is_swizzled=True,
                tensor_scale_a_list=ts_a_list,
                tensor_scale_b_list=ts_b_list,
            )

        elapsed_ms = _cuda_time_ms(fn, warmup=_BENCH_WARMUP, iters=_BENCH_ITERS)
        tf = _tflops(shapes, elapsed_ms)
        print(
            f"\n  groups={len(shapes)}  shapes={shapes[0]}{'...' if len(shapes) > 1 else ''}"
            f"  latency={elapsed_ms*1e3:.1f} µs  TFLOPS={tf:.2f}"
        )


# ---------------------------------------------------------------------------
# Benchmark vs TE general_grouped_gemm
# ---------------------------------------------------------------------------


def _make_te_quantizer(columnwise=False):
    from transformer_engine.pytorch import NVFP4Quantizer
    return NVFP4Quantizer(
        rowwise=True,
        columnwise=columnwise,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
    )


def _build_te_inputs(shapes, seed=42):
    """Build TE-quantized inputs for general_grouped_gemm (layout='TN': A=weight, B=act)."""
    q_x = _make_te_quantizer(columnwise=False)
    q_w = _make_te_quantizer(columnwise=False)
    a_list, b_list, out_list, m_splits = [], [], [], []
    for i, (m, n, k) in enumerate(shapes):
        torch.manual_seed(seed + i)
        torch.cuda.manual_seed(seed + i)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.empty(n, k, device="cuda", dtype=torch.bfloat16)
        torch.nn.init.normal_(w, std=0.023)
        x_t = q_x.make_empty(x.shape, dtype=x.dtype, device=x.device, requires_grad=False)
        w_t = q_w.make_empty(w.shape, dtype=w.dtype, device=w.device, requires_grad=False)
        a_list.append(q_w.update_quantized(w, w_t))   # A = weight (transposed in TN layout)
        b_list.append(q_x.update_quantized(x, x_t))   # B = activation
        out_list.append(torch.empty(m, n, dtype=torch.bfloat16, device="cuda"))
        m_splits.append(m)
    return a_list, b_list, out_list, m_splits


class TestGroupedCutlassGemmVsTE:
    """Compare latency of CUTLASS C++ kernel vs TE general_grouped_gemm."""

    @pytest.mark.parametrize("shapes", _BENCH_CONFIGS)
    def test_benchmark_vs_te(self, shapes):
        _require_sm100()
        pytest.importorskip("transformer_engine")
        from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm

        # ---- CUTLASS C++ inputs ----
        cutlass_inputs = TestGroupedCutlassGemmBenchmark._build_inputs(shapes)
        a_list, b_list, a_sf_list, b_sf_list, pts_list, ts_a_list, ts_b_list = cutlass_inputs

        def fn_cutlass():
            grouped_nvfp4_cutlass_gemm_per_token_scale(
                a_list=a_list,
                b_list=b_list,
                a_sf_list=a_sf_list,
                b_sf_list=b_sf_list,
                per_token_scale_list=pts_list,
                out_dtype=torch.bfloat16,
                sf_is_swizzled=True,
                tensor_scale_a_list=ts_a_list,
                tensor_scale_b_list=ts_b_list,
            )

        # ---- TE inputs ----
        te_a, te_b, te_out, m_splits = _build_te_inputs(shapes)

        def fn_te():
            general_grouped_gemm(
                A=te_a,
                B=te_b,
                out=te_out,
                out_dtype=torch.bfloat16,
                layout="TN",
                m_splits=m_splits,
                single_output=False,
                grad=False,
                bias=None,
                use_bias=False,
                use_split_accumulator=False,
                accumulate=False,
            )

        fn_cutlass()
        fn_te()

        ms_cutlass = _cuda_time_ms(fn_cutlass, warmup=_BENCH_WARMUP, iters=_BENCH_ITERS)
        ms_te      = _cuda_time_ms(fn_te,      warmup=_BENCH_WARMUP, iters=_BENCH_ITERS)

        tf_cutlass = _tflops(shapes, ms_cutlass)
        tf_te      = _tflops(shapes, ms_te)
        speedup    = ms_te / ms_cutlass

        print(
            f"\n  groups={len(shapes)}  shapes={shapes[0]}{'...' if len(shapes) > 1 else ''}\n"
            f"  {'':20s} {'Latency µs':>12}  {'TFLOPS':>9}\n"
            f"  {'CUTLASS C++':20s} {ms_cutlass*1e3:>12.1f}  {tf_cutlass:>9.2f}\n"
            f"  {'TE general_grouped':20s} {ms_te*1e3:>12.1f}  {tf_te:>9.2f}\n"
            f"  speedup (cutlass/te): {speedup:.2f}x"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
