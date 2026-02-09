import torch
import pytest
import time
from miniTransformer.ops.nvfp4_quantize_op_ref import Nvfp4TiledQuantizeRefOp
from miniTransformer.ops.nvfp4_qgemm_op import (
    qgemm_cmp_triton_real,
    initialize_nvfp4_for_qgemm,
)


class TestRefQgemmComparison:
    """Test cases comparing reference QGEMM implementations."""

    @pytest.mark.parametrize(
        "M, K, N, pow_2_scales, seed",
        [
            (256, 256, 256, False, 10),  # NVFP4 mode, block_length=16
            (256, 256, 256, False, 1234),  # NVFP4 mode, block_length=16
            (8192, 8192, 8192, False, 10),  # Larger matrix  
            (8192, 8192, 8192, False, 1234),  # Larger matrix  
        ],
    )
    def test_ref_qgemm_vs_triton(
        self, M, K, N, pow_2_scales, seed
    ):
        """
        Test comparing _ref_qgemm and _ref_qgemm_with_triton.
        
        Args:
            M: Number of rows in input
            K: Inner dimension
            N: Number of columns in weight
            pow_2_scales: Whether to use power-of-2 scaling
        """
        device = "cuda"
        out_dtype = torch.bfloat16
 
        qx, qw, sx, sw, global_amax_qx, global_amax_qw, qparams = initialize_nvfp4_for_qgemm(
            M, N, K, device=device, seed=seed
        )

        quantize_op = Nvfp4TiledQuantizeRefOp()

        # Warmup runs
        warmup_runs = 3
        for _ in range(warmup_runs):
            _ = quantize_op._ref_qgemm(
                qx=qx, qw=qw, m_params=None, out_dtype=out_dtype,
                sx=sx, sw=sw, bias=None, out=None, accumulate=False,
                pow_2_scales=pow_2_scales, global_amax_qx=global_amax_qx,
                global_amax_qw=global_amax_qw, qparams_x=qparams, qparams_w=qparams,
            )
            _ = qgemm_cmp_triton_real(
                qx=qx, qw=qw, m_params=None, out_dtype=out_dtype,
                sx=sx, sw=sw, bias=None, out=None, accumulate=False,
                pow_2_scales=pow_2_scales, global_amax_qx=global_amax_qx,
                global_amax_qw=global_amax_qw, qparams_x=qparams, qparams_w=qparams
            )
        
        # Synchronize before timing
        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark _ref_qgemm
        num_runs = 10
        start_time = time.time()
        for _ in range(num_runs):
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
                pow_2_scales=pow_2_scales,
                global_amax_qx=global_amax_qx,
                global_amax_qw=global_amax_qw,
                qparams_x=qparams,
                qparams_w=qparams,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        ref_time = (time.time() - start_time) / num_runs

        # Benchmark qgemm_cmp_triton_real
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_runs):
            y_triton = qgemm_cmp_triton_real(
                qx=qx,
                qw=qw,
                m_params=None,
                out_dtype=out_dtype,
                sx=sx,
                sw=sw,
                bias=None,
                out=None,
                accumulate=False,
                pow_2_scales=pow_2_scales,
                global_amax_qx=global_amax_qx,
                global_amax_qw=global_amax_qw,
                qparams_x=qparams,
                qparams_w=qparams
            )
        if device == "cuda":
            torch.cuda.synchronize()
        triton_time = (time.time() - start_time) / num_runs

        # Print comparison info
        print(f"\n{'='*80}")
        print(f"Comparison for M={M}, K={K}, N={N}, pow_2_scales={pow_2_scales}:")
        print(f"{'='*80}")
        
        # Performance comparison
        print(f"\nPerformance Comparison (average over {num_runs} runs):")
        print(f"  _ref_qgemm:                {ref_time*1000:.4f} ms")
        print(f"  _ref_qgemm_with_triton:    {triton_time*1000:.4f} ms")
        speedup = ref_time / triton_time if triton_time > 0 else float('inf')
        if speedup > 1:
            print(f"  Speedup:                   {speedup:.2f}x (Triton is faster)")
        else:
            print(f"  Slowdown:                  {1/speedup:.2f}x (Triton is slower)")
        
        # Output shape and range
        print(f"\nOutput Information:")
        print(f"  _ref_qgemm output shape: {y_ref.shape}, dtype: {y_ref.dtype}")
        print(f"  _ref_qgemm output range: [{y_ref.min().item():.4f}, {y_ref.max().item():.4f}]")
        print(f"  _ref_qgemm_with_triton output shape: {y_triton.shape}, dtype: {y_triton.dtype}")
        print(f"  _ref_qgemm_with_triton output range: [{y_triton.min().item():.4f}, {y_triton.max().item():.4f}]")
        
        # Calculate difference metrics
        abs_diff = torch.abs(y_ref.to(torch.float32) - y_triton.to(torch.float32))
        rel_diff = abs_diff / (torch.abs(y_ref.to(torch.float32)) + 1e-8)
        print(f"\nNumerical Accuracy:")
        print(f"  Max absolute difference:   {abs_diff.max().item():.6f}")
        print(f"  Mean absolute difference:  {abs_diff.mean().item():.6f}")
        print(f"  Max relative difference:   {rel_diff.max().item():.6f}")
        print(f"  Mean relative difference:  {rel_diff.mean().item():.6f}")

        # Verify both outputs
        assert y_ref.shape == (M, N), f"Expected shape ({M}, {N}), got {y_ref.shape}"
        assert y_ref.dtype == out_dtype, f"Expected dtype {out_dtype}, got {y_ref.dtype}"
        assert not torch.isnan(y_ref).any(), "_ref_qgemm output contains NaN values"
        assert not torch.isinf(y_ref).any(), "_ref_qgemm output contains Inf values"
        
        assert y_triton.shape == (M, N), f"Expected shape ({M}, {N}), got {y_triton.shape}"
        assert y_triton.dtype == out_dtype, f"Expected dtype {out_dtype}, got {y_triton.dtype}"
        assert not torch.isnan(y_triton).any(), "_ref_qgemm_with_triton output contains NaN values"
        assert not torch.isinf(y_triton).any(), "_ref_qgemm_with_triton output contains Inf values"

        # Compare results with appropriate tolerance
        # Note: The tolerance may need adjustment based on the implementation differences
        print(f"(pass M={M}, K={K}, N={N}, pow_2_scales={pow_2_scales})")
        print(f"_ref_qgemm output: {y_ref}")
        print(f"_ref_qgemm_with_triton output: {y_triton}")

        torch.testing.assert_close(
            y_ref.to(torch.float32),
            y_triton.to(torch.float32),
            rtol=8e-3, 
            atol=8e-3, 
            msg=f"_ref_qgemm and _ref_qgemm_with_triton outputs differ significantly for M={M}, K={K}, N={N}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
