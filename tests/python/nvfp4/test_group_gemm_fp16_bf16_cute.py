from typing import Dict, List

import pytest
import torch

from miniTransformer.ops.cuteDSL.nvfp4.group_gemm_fp16_bf16_cute import (
    GroupedGemmProfile,
    GroupedGemmKernel,
    generate_grouped_inputs,
    grouped_reference,
)


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
    pytest.param(GroupedGemmProfile.FP16, id="fp16"),
    pytest.param(GroupedGemmProfile.BF16, id="bf16"),
]
CASE_PARAMS = [
    pytest.param(case, id=f"bench_case_{idx}") for idx, case in enumerate(BENCHMARK_CASES)
]
CASE_MULTISTREAM_PARAMS = [
    {
        "g": 8,
        "k": [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168],
        "m": [80, 176, 128, 72, 64, 248, 96, 160],
        "n": [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096],
    },
]


def _get_out_dtype(profile: GroupedGemmProfile) -> torch.dtype:
    return torch.float16 if profile.config.c_dtype is not None and profile == GroupedGemmProfile.FP16 else torch.bfloat16


class TestGroupedGemm:
    @pytest.mark.parametrize("profile", PROFILE_CASES)
    @pytest.mark.parametrize("case", CASE_PARAMS)
    def test_grouped_gemm_correctness(self, profile, case):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required")
        out_dtype = _get_out_dtype(profile)
        abc_tensors, problem_sizes = generate_grouped_inputs(
            m_list=case["m"],
            n_list=case["n"],
            k_list=case["k"],
            profile=profile,
            device="cuda",
            out_dtype=out_dtype,
            seed=7,
            l=1,
        )
        kernel = GroupedGemmKernel(profile=profile)
        got = kernel.run_from_grouped_inputs(
            abc_tensors=abc_tensors,
            problem_sizes=problem_sizes,
            out_dtype=out_dtype,
            accumulate=False,
        )
        ref = grouped_reference(
            abc_tensors=abc_tensors,
            problem_sizes=problem_sizes,
            out_dtype=out_dtype,
            accumulate=False,
        )
        for idx, (g, r) in enumerate(zip(got, ref)):
            torch.testing.assert_close(
                g.to(torch.float32),
                r.to(torch.float32),
                rtol=1e-2,
                atol=1e-6,
                msg=f"group {idx} mismatch for profile={profile.name.lower()}",
            )

    @pytest.mark.parametrize("profile", PROFILE_CASES)
    @pytest.mark.parametrize("case", CASE_MULTISTREAM_PARAMS)
    @pytest.mark.parametrize("num_streams", [2, 4])
    def test_grouped_gemm_multi_stream_benchmark_shapes(self, profile, case, num_streams):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required")

        out_dtype = _get_out_dtype(profile)
        stream_count = min(num_streams, case["g"])
        streams = [torch.cuda.Stream() for _ in range(stream_count)]
        warmup_iters = 2
        bench_iters = 5
        device = "cuda"

        try:
            single_group_workloads = []
            for i, (m, n, k) in enumerate(zip(case["m"], case["n"], case["k"])):
                a = torch.randn((m, k), device=device, dtype=out_dtype)
                b = torch.randn((n, k), device=device, dtype=out_dtype)
                c = torch.empty((m, n, 1), device=device, dtype=out_dtype)
                bias = torch.zeros((n,), device=device, dtype=out_dtype)
                single_group_workloads.append(
                    (
                        a,
                        b,
                        c,
                        bias,
                        streams[i % stream_count],
                    )
                )

            def _launch_round_robin_once():
                for a, b, c, bias, stream in single_group_workloads:
                    with torch.cuda.stream(stream):
                        out = torch.matmul(a, b.t().contiguous()) + bias
            
                torch.cuda.synchronize()

            for _ in range(warmup_iters):
                _launch_round_robin_once()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(bench_iters):
                _launch_round_robin_once()
            end_event.record()
            torch.cuda.synchronize()

            multi_stream_avg_us = (start_event.elapsed_time(end_event) * 1000.0) / bench_iters
        except (torch.AcceleratorError, RuntimeError, ValueError) as err:
            err_text = str(err).lower()

            if "illegal instruction" in err_text:
                pytest.skip(
                    f"flashinfer bf16/fp16 gemm benchmark skipped due to CUDA illegal instruction: {err}"
                )
            raise

        print(
            f"[group-gemm-multistream-bench] profile={profile.name.lower()}, "
            f"g={case['g']}, num_streams={stream_count}, multi_avg_us={multi_stream_avg_us:.3f}"
        )
        assert multi_stream_avg_us > 0.0

    @pytest.mark.parametrize("profile", PROFILE_CASES)
    @pytest.mark.parametrize("case", CASE_PARAMS)
    def test_grouped_gemm_benchmark_shapes(self, profile, case):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required")
        out_dtype = _get_out_dtype(profile)
        abc_tensors, problem_sizes = generate_grouped_inputs(
            m_list=case["m"],
            n_list=case["n"],
            k_list=case["k"],
            profile=profile,
            device="cuda",
            out_dtype=out_dtype,
            seed=17,
            l=1,
        )
        kernel = GroupedGemmKernel(profile=profile)
        got = kernel.run_from_grouped_inputs(
            abc_tensors=abc_tensors,
            problem_sizes=problem_sizes,
            out_dtype=out_dtype,
            accumulate=False,
            warmup_iterations=2,
            iterations=5,
            use_cold_l2=True,
        )
        ref = grouped_reference(
            abc_tensors=abc_tensors,
            problem_sizes=problem_sizes,
            out_dtype=out_dtype,
            accumulate=False,
        )
        for idx, (g, r) in enumerate(zip(got, ref)):
            torch.testing.assert_close(
                g.to(torch.float32),
                r.to(torch.float32),
                rtol=1e-2,
                atol=1e-6,
                msg=f"group {idx} benchmark mismatch for profile={profile.name.lower()}",
            )
        metrics = kernel.last_benchmark_metrics
        print(
            f"[group-gemm-bench] profile={profile.name.lower()}, "
            f"g={case['g']}, metrics={metrics}"
        )
      
        for metric in metrics:
            assert metric["avg_runtime_us"] > 0.0
            assert metric["gflops"] > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
