import os
import sys
import time
import pytest
import torch

from miniTransformer.ops.nvfp4_gemm_backend import dispatch_gemm_backend

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from miniTransformer.quantiztion.nvfp4_quantize_op_ref import Nvfp4TiledQuantizeRefOp
from miniTransformer.ops.nvfp4_qgemm import (
    initialize_nvfp4_for_qgemm,
    swizzle_scale,
)

import fp4_gemm


class TestCutlassNvfp4Gemm:

    @pytest.mark.parametrize(
        "M, K, N, seed",
        [
            (256, 256, 256, 10),
            (256, 256, 256, 1234),    
            (8192, 8192, 8192, 22),
            (8192, 8192, 8192, 1234),
        ],
    )
    def test_cutlass_nvfp4_matches_ref(self, M, K, N, seed):
        device = "cuda"
        out_dtype = torch.bfloat16

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

        factor = 6.0 * 6.0 * 448.0 * 448.0
        alpha = (global_amax_qx * global_amax_qw).to(torch.float32).squeeze(-1).item() / factor
        alpha = torch.tensor(alpha, device=device, dtype=torch.float32)

        start = time.time()

        backend = dispatch_gemm_backend("cutlass", "nvfp4")
        D_cutlass = backend.gemm(
            qx, qw, out_dtype, sx, sw,
            None, None, False, False,
            "TN", alpha, None, None, False
        )

        backend = dispatch_gemm_backend("cublas", "nvfp4")
        ref = backend.gemm(
            qx, qw, out_dtype, sx, sw,
            None, None, False, False,
            "TN", alpha, None, None, False
        )

        print(f'\ny_ref:{y_ref}')
        print(f'D_cutlass:{D_cutlass}')
        print(f'ref:{ref}')

        if device == "cuda":
            torch.cuda.synchronize()
        cutlass_time = time.time() - start

        assert D_cutlass.shape == (M, N)
        assert D_cutlass.dtype == out_dtype
        assert not torch.isnan(D_cutlass).any()
        assert not torch.isinf(D_cutlass).any()

        y_ref_f32 = y_ref.to(torch.float32)
        y_cutlass_f32 = D_cutlass.to(torch.float32)

        abs_diff = (y_ref_f32 - y_cutlass_f32).abs()
        rel_diff = abs_diff / (y_ref_f32.abs() + 1e-8)

        print(f"\nCutlass NVFP4 GEMM M={M}, K={K}, N={N}, seed={seed}")
        print(f"  Cutlass time: {cutlass_time * 1000:.3f} ms")
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
