import os
import sys
import time
import pytest
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from miniTransformer.quantiztion.nvfp4_quantize_op_ref import Nvfp4TiledQuantizeRefOp
from miniTransformer.ops.nvfp4_qgemm import (
    initialize_nvfp4_for_qgemm,
    swizzle_scale,
)

import fp4_gemm

GEMM_WORKLOAD_PARAMS = [
    pytest.param(2048, 5120, 10240, id="linear_qkv"),
    pytest.param(8192, 2048, 1280, id="linear_proj"),
    pytest.param(2048, 5120, 51200, id="linear_fc1"),
    pytest.param(8192, 6400, 1280, id="linear_fc2"),
   
    pytest.param(256, 256, 256, id="function_test0"),
    pytest.param(8192, 8192, 8192, id="function_test1"),
]

class TestCutlassNvfp4Gemm:

    @pytest.mark.parametrize(
        "M, K, N",
        GEMM_WORKLOAD_PARAMS,
    )
    @pytest.mark.parametrize("seed", [1234])
    @pytest.mark.parametrize("out_dtype", [torch.bfloat16])
    def test_cutlass_nvfp4_matches_ref(self, M, K, N, seed, out_dtype):
        warmup = 10
        bench_iters = 100

        device = "cuda"

        qx, qw, sx, sw, global_amax_qx, global_amax_qw, qparams = (
            initialize_nvfp4_for_qgemm(M, N, K, device=device, seed=seed)
        )

        quantize_op = Nvfp4TiledQuantizeRefOp()

        y_ref = quantize_op._ref_qgemm(
            qx=qx,
            qw=qw,
            m_params=None,
            out_dtype=out_dtype,
            sx=sx,
            sw=sw,
            bias=None,
            out=None,
            accumulate=False,
            pow_2_scales=False,
            global_amax_qx=global_amax_qx,
            global_amax_qw=global_amax_qw,
            qparams_x=qparams,
            qparams_w=qparams,
        )

        assert y_ref.shape == (M, N)
        assert y_ref.dtype == out_dtype

        sx_4d = swizzle_scale(sx, M, K)  # (M//128, K//64, 32, 16)
        sw_4d = swizzle_scale(sw, N, K)  # (N//128, K//64, 32, 16)

        D_cutlass = torch.empty((M, N), device=device, dtype=out_dtype)

        factor = 6.0 * 6.0 * 448.0 * 448.0
        alpha = (global_amax_qx * global_amax_qw).to(torch.float32).squeeze(-1).item() / factor
        alpha = torch.tensor(alpha, device=device, dtype=torch.float32)

        for _ in range(warmup):
            fp4_gemm.cutlass_gemm(
                D_cutlass,
                qx,
                qw,
                sx_4d,
                sw_4d,
                alpha,
            )
            if device == "cuda":
                torch.cuda.synchronize()

        start = time.time()
        for _ in range(bench_iters):
            fp4_gemm.cutlass_gemm(
                D_cutlass,  # output
                qx,         # A：uint8 packed NVFP4
                qw,         # B：uint8 packed NVFP4
                sx_4d,      # A_sf：block‑scaled scale for A
                sw_4d,      # B_sf：block‑scaled scale for B
                alpha,      # alpha
            )
        if device == "cuda":
            torch.cuda.synchronize()
        cutlass_time = (time.time() - start) / bench_iters

        assert D_cutlass.shape == (M, N)
        assert D_cutlass.dtype == out_dtype
        assert not torch.isnan(D_cutlass).any()
        assert not torch.isinf(D_cutlass).any()

        y_ref_f32 = y_ref.to(torch.float32)
        y_cutlass_f32 = D_cutlass.to(torch.float32)

        abs_diff = (y_ref_f32 - y_cutlass_f32).abs()
        rel_diff = abs_diff / (y_ref_f32.abs() + 1e-8)

        print(f"\nCutlass NVFP4 GEMM M={M}, K={K}, N={N}, seed={seed}")
        print(
            f"  Cutlass time (avg over {bench_iters} iters, "
            f"warmup={warmup}): {cutlass_time * 1e6:.3f} us"
        )
        print(f"  Max abs diff: {abs_diff.max().item():.6f}")
        print(f"  Mean abs diff: {abs_diff.mean().item():.6f}")
        print(f"  Max rel diff: {rel_diff.max().item():.6f}")
        print(f"  Mean rel diff: {rel_diff.mean().item():.6f}")

        torch.testing.assert_close(
            y_ref_f32,
            y_cutlass_f32,
            rtol=1e-6,
            atol=1e-6,
            msg=f"Cutlass NVFP4 GEMM mismatch for M={M}, K={K}, N={N}, seed={seed}",
        )

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
