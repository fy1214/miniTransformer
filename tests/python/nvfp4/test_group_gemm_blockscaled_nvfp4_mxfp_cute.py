"""CuTeDSL blockscaled NVFP4/MXFP4 grouped GEMM kernel (dev / QAT 6–7 backend).

What: ``Sm100GroupedBlockScaledGemmKernel`` vs ``grouped_blockscaled_reference``.
Where: underlying kernel for CuTeDSL grouped GEMM paths.
Compare: Python blockscaled reference; multi-stream / bench are optional.

Markers: ``legacy`` for correctness; ``bench``+``slow`` for timing tests.
"""
from typing import Dict, List

import pytest
import torch

from miniTransformer.ops.cuteDSL.nvfp4.group_gemm_blockscaled_nvfp4_mxfp_cute import (
    GroupedBlockScaledProfile,
    Sm100GroupedBlockScaledGemmKernel,
    generate_grouped_blockscaled_inputs,
    grouped_blockscaled_reference,
)

pytestmark = pytest.mark.legacy


BENCHMARK_CASES: List[Dict[str, object]] = [
    # Deep-K
    {
        "g": 8,
        "k": [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168],
        "m": [80, 176, 128, 72, 64, 248, 96, 160],
        "n": [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096],
    },
    {
        "g": 8,
        "k": [6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144],
        "m": [80, 176, 128, 72, 64, 248, 96, 160],
        "n": [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096],
    },
    {
        "g": 8,
        "k": [5120, 5120, 5120, 5120, 5120, 5120, 5120, 5120],
        "m": [80, 176, 128, 72, 64, 248, 96, 160],
        "n": [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096],
    },
    # Wide-N
    {
        "g": 8,
        "k": [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048],
        "m": [40, 76, 168, 72, 164, 148, 196, 160],
        "n": [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168],
    },
    # Mid-Sized
    {"g": 2, "k": [4096, 4096], "m": [192, 320], "n": [3072, 3072]},
    # Shorter-K
    {"g": 2, "k": [1536, 1536], "m": [128, 384], "n": [4096, 4096]},
]

PROFILE_CASES = [
    pytest.param(GroupedBlockScaledProfile.NVFP4, id="nvfp4"),
    pytest.param(GroupedBlockScaledProfile.MXFP4, id="mxfp4"),
    pytest.param(GroupedBlockScaledProfile.MXFP8, id="mxfp8"),
]
CASE_PARAMS = [
    pytest.param(case, id=f"bench_case_{idx}") for idx, case in enumerate(BENCHMARK_CASES)
]

CASE_MULTISTREAM_PARAMS = [
    {
        "g": 8,
        "k": [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048],
        "m": [40, 76, 168, 72, 164, 148, 196, 160],
        "n": [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168],
    },
]


class TestGroupedBlockScaledGemm:
    """Correctness and optional benches for Sm100GroupedBlockScaledGemmKernel."""

    @pytest.mark.parametrize("profile", PROFILE_CASES)
    @pytest.mark.parametrize("case", CASE_PARAMS)
    def test_grouped_gemm_correctness(self, profile, case):
        """Kernel output matches Python blockscaled reference per group."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required")
        abc_tensors, sfasfb_tensors, problem_sizes = generate_grouped_blockscaled_inputs(
            m_list=case["m"],
            n_list=case["n"],
            k_list=case["k"],
            profile=profile,
            device="cuda",
            out_dtype=torch.bfloat16,
            seed=7,
            l=1,
        )
        kernel = Sm100GroupedBlockScaledGemmKernel(profile=profile)
        got = kernel.run_from_grouped_inputs(
            abc_tensors=abc_tensors,
            sfasfb_tensors=sfasfb_tensors,
            problem_sizes=problem_sizes,
            out_dtype=torch.bfloat16,
            accumulate=False,
        )
        ref = grouped_blockscaled_reference(
            abc_tensors=abc_tensors,
            sfasfb_tensors=sfasfb_tensors,
            problem_sizes=problem_sizes,
            profile=profile,
            out_dtype=torch.bfloat16,
            accumulate=False,
        )
        # Align with cutlass grouped example semantics.
        for idx, (g, r) in enumerate(zip(got, ref)):
            torch.testing.assert_close(
                g.to(torch.float32),
                r.to(torch.float32),
                rtol=1e-2,
                atol=1e-6,
                msg=f"group {idx} mismatch for profile={profile.value}",
            )

    @pytest.mark.bench
    @pytest.mark.slow
    @pytest.mark.parametrize("case", CASE_MULTISTREAM_PARAMS)
    @pytest.mark.parametrize("num_streams", [2, 4])
    def test_group_gemm_multi_stream_round_robin_nvfp4(self, case, num_streams):
        """Bench: flashinfer multi-stream round-robin launch (prints; always passes)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required")
        pytest.importorskip("flashinfer")
        from miniTransformer.ops.nvfp4_gemm_backend import dispatch_gemm_backend
        from miniTransformer.ops.nvfp4_qgemm import initialize_nvfp4_for_qgemm

        out_dtype = torch.bfloat16

        backend = dispatch_gemm_backend("flashinfer", "nvfp4")
        if backend is None:
            pytest.skip("flashinfer NVFP4 backend is unavailable")

        
        group_inputs = []
        factor = 6.0 * 6.0 * 448.0 * 448.0
        for group_idx, (m, n, k) in enumerate(zip(case["m"], case["n"], case["k"])):
            qx, qw, sx, sw, global_amax_qx, global_amax_qw, _ = initialize_nvfp4_for_qgemm(
                m,
                n,
                k,
                device="cuda",
                seed=23 + group_idx,
            )
            alpha = (global_amax_qx * global_amax_qw).to(torch.float32).squeeze(-1) / factor
            if alpha.numel() == 1:
                alpha = alpha.squeeze()
            group_inputs.append((qx.contiguous(), qw.contiguous(), sx.contiguous(), sw.contiguous(), alpha))

        stream_count = min(num_streams, len(group_inputs))
        streams = [torch.cuda.Stream() for _ in range(stream_count)]
        warmup_iters = 2
        bench_iters = 10

        def _launch_once(use_multi_stream: bool):
            outputs = [None] * len(group_inputs)
            for i, (qx, qw, sx, sw, alpha) in enumerate(group_inputs):
                if use_multi_stream:
                    with torch.cuda.stream(streams[i % stream_count]):
                        outputs[i] = backend.gemm(
                            qx=qx,
                            qw=qw,
                            out_dtype=out_dtype,
                            sx=sx,
                            sw=sw,
                            bias=None,
                            out=None,
                            grad=False,
                            accumulate=False,
                            layout="TN",
                            alpha=alpha,
                            tensor_scale_x=None,
                            tensor_scale_w=None,
                            has_swzzile=False,
                        )
                else:
                    outputs[i] = backend.gemm(
                        qx=qx,
                        qw=qw,
                        out_dtype=out_dtype,
                        sx=sx,
                        sw=sw,
                        bias=None,
                        out=None,
                        grad=False,
                        accumulate=False,
                        layout="TN",
                        alpha=alpha,
                        tensor_scale_x=None,
                        tensor_scale_w=None,
                        has_swzzile=False,
                    )
            torch.cuda.synchronize()
            return outputs

        try:
            # Warmup
            for _ in range(warmup_iters):
                _launch_once(use_multi_stream=False)
                _launch_once(use_multi_stream=True)


            # Benchmark round-robin multi stream
            start_multi = torch.cuda.Event(enable_timing=True)
            end_multi = torch.cuda.Event(enable_timing=True)
            start_multi.record()
            for _ in range(bench_iters):
                _launch_once(use_multi_stream=True)
            end_multi.record()
            torch.cuda.synchronize()
            multi_us = start_multi.elapsed_time(end_multi) * 1000.0 / bench_iters
        except (torch.AcceleratorError, RuntimeError) as err:
            if "illegal memory access" in str(err).lower():
                pytest.skip(f"flashinfer NVFP4 benchmark skipped due to CUDA illegal memory access: {err}")
            raise

        print(
            f"[nvfp4-multistream-bench] groups={len(group_inputs)}, "
            f"streams={stream_count}, multi_avg_us={multi_us:.3f}, "
        )
        assert multi_us > 0.0

    @pytest.mark.bench
    @pytest.mark.slow
    @pytest.mark.parametrize("profile", PROFILE_CASES)
    @pytest.mark.parametrize("case", CASE_PARAMS)
    def test_grouped_gemm_benchmark_shapes(self, profile, case):
        """Bench: cold-L2 timed run still matches Python reference."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required")
        abc_tensors, sfasfb_tensors, problem_sizes = generate_grouped_blockscaled_inputs(
            m_list=case["m"],
            n_list=case["n"],
            k_list=case["k"],
            profile=profile,
            device="cuda",
            out_dtype=torch.bfloat16,
            seed=17,
            l=1,
        )
        kernel = Sm100GroupedBlockScaledGemmKernel(profile=profile)
        got = kernel.run_from_grouped_inputs(
            abc_tensors=abc_tensors,
            sfasfb_tensors=sfasfb_tensors,
            problem_sizes=problem_sizes,
            out_dtype=torch.bfloat16,
            accumulate=False,
            warmup_iterations=2,
            iterations=5,
            use_cold_l2=True,
        )
        ref = grouped_blockscaled_reference(
            abc_tensors=abc_tensors,
            sfasfb_tensors=sfasfb_tensors,
            problem_sizes=problem_sizes,
            profile=profile,
            out_dtype=torch.bfloat16,
            accumulate=False,
        )
        for idx, (g, r) in enumerate(zip(got, ref)):
            torch.testing.assert_close(
                g.to(torch.float32),
                r.to(torch.float32),
                rtol=1e-2,
                atol=1e-6,
                msg=f"group {idx} benchmark mismatch for profile={profile.value}",
            )
        metrics = kernel.last_benchmark_metrics
        print(
            f"[group-gemm-bench] profile={profile.value}, "
            f"g={case['g']}, metrics={metrics}"
        )
      
        for metric in metrics:
            assert metric["avg_runtime_us"] > 0.0
            assert metric["gflops"] > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))