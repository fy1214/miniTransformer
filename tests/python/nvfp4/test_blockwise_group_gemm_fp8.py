from typing import Dict, List

import pytest
import torch

from miniTransformer.ops.cuteDSL.nvfp4.blockwise_group_gemm_fp8 import (
    BlockwiseGroupedFp8Profile,
    BlockwiseContiguousGroupedGemmKernel,
    generate_grouped_blockwise_fp8_inputs,
    grouped_blockwise_fp8_reference,
)


BENCHMARK_CASES: List[Dict[str, object]] = [
    {
        "g": 8,
        "k": [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168],
        "m": [80, 176, 128, 72, 64, 248, 96, 160],
        "n": [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096],
    },
    {
        "g": 8,
        "k": [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048],
        "m": [40, 76, 168, 72, 164, 148, 196, 160],
        "n": [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168],
    },
    {"g": 2, "k": [4096, 4096], "m": [192, 320], "n": [3072, 3072]},
    {"g": 2, "k": [1536, 1536], "m": [128, 384], "n": [4096, 4096]},
]

PROFILE_CASES = [
    pytest.param(BlockwiseGroupedFp8Profile.FP8, id="fp8"),
]
CASE_PARAMS = [
    pytest.param(case, id=f"bench_case_{idx}") for idx, case in enumerate(BENCHMARK_CASES)
]


class TestBlockwiseGroupedFp8Gemm:
    @pytest.mark.parametrize("profile", PROFILE_CASES)
    @pytest.mark.parametrize("case", CASE_PARAMS)
    def test_grouped_gemm_correctness(self, profile, case):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required")
        abc_tensors, scale_tensors, problem_sizes = generate_grouped_blockwise_fp8_inputs(
            m_list=case["m"],
            n_list=case["n"],
            k_list=case["k"],
            profile=profile,
            device="cuda",
            out_dtype=torch.bfloat16,
            seed=7,
            l=1,
        )
        kernel = BlockwiseContiguousGroupedGemmKernel(profile=profile)
        got = kernel.run_from_grouped_inputs(
            abc_tensors=abc_tensors,
            scale_tensors=scale_tensors,
            problem_sizes=problem_sizes,
            out_dtype=torch.bfloat16,
            accumulate=False,
        )
        ref = grouped_blockwise_fp8_reference(
            abc_tensors=abc_tensors,
            scale_tensors=scale_tensors,
            problem_sizes=problem_sizes,
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
                msg=f"group {idx} mismatch for profile={profile.name.lower()}",
            )

    @pytest.mark.parametrize("profile", PROFILE_CASES)
    @pytest.mark.parametrize("case", CASE_PARAMS)
    def test_grouped_gemm_benchmark_shapes(self, profile, case):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required")
        abc_tensors, scale_tensors, problem_sizes = generate_grouped_blockwise_fp8_inputs(
            m_list=case["m"],
            n_list=case["n"],
            k_list=case["k"],
            profile=profile,
            device="cuda",
            out_dtype=torch.bfloat16,
            seed=17,
            l=1,
        )
        kernel = BlockwiseContiguousGroupedGemmKernel(profile=profile)
        got = kernel.run_from_grouped_inputs(
            abc_tensors=abc_tensors,
            scale_tensors=scale_tensors,
            problem_sizes=problem_sizes,
            out_dtype=torch.bfloat16,
            accumulate=False,
            warmup_iterations=2,
            iterations=5,
            use_cold_l2=True,
        )
        ref = grouped_blockwise_fp8_reference(
            abc_tensors=abc_tensors,
            scale_tensors=scale_tensors,
            problem_sizes=problem_sizes,
            out_dtype=torch.bfloat16,
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