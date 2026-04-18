from typing import Dict, List

import pytest
import torch

from miniTransformer.ops.cuteDSL.nvfp4.group_gemm_cute import (
    GroupedBlockScaledProfile,
    Sm100GroupedBlockScaledGemmKernel,
    generate_grouped_blockscaled_inputs,
    grouped_blockscaled_reference,
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
    pytest.param(GroupedBlockScaledProfile.NVFP4, id="nvfp4"),
    pytest.param(GroupedBlockScaledProfile.MXFP4, id="mxfp4"),
    pytest.param(GroupedBlockScaledProfile.MXFP8, id="mxfp8"),
]
CASE_PARAMS = [
    pytest.param(case, id=f"bench_case_{idx}") for idx, case in enumerate(BENCHMARK_CASES)
]


def _benchmark_kernel_call(kernel_call, warmup: int = 2, iters: int = 5) -> float:
    for _ in range(warmup):
        _ = kernel_call()
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(iters):
        _ = kernel_call()
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt) * 1000.0 / iters


class TestGroupedBlockScaledGemm:
    @pytest.mark.parametrize("profile", PROFILE_CASES)
    @pytest.mark.parametrize("case", CASE_PARAMS)
    def test_grouped_gemm_correctness(self, profile, case):
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
        got = kernel(
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

    @pytest.mark.parametrize("profile", PROFILE_CASES)
    @pytest.mark.parametrize("case", CASE_PARAMS)
    def test_grouped_gemm_benchmark_shapes(self, profile, case):
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
        elapsed_us = _benchmark_kernel_call(
            lambda: kernel(
                abc_tensors=abc_tensors,
                sfasfb_tensors=sfasfb_tensors,
                problem_sizes=problem_sizes,
                out_dtype=torch.bfloat16,
                accumulate=False,
            )
        )
        print(
            f"[group-gemm-bench] profile={profile.value}, "
            f"g={case['g']}, elapsed={elapsed_us:.3f} us"
        )
        assert elapsed_us > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))