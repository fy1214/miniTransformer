import torch
import pytest
import time
from miniTransformer.ops.nvfp4_quantize_op_ref import Nvfp4TiledQuantizeRefOp
from miniTransformer.ops.nvfp4_qgemm_op import (
    qgemm_cmp_cublass_real,
    initialize_nvfp4_for_qgemm,
)


class TestRefQgemmCublasComparison:
    """Test cases comparing reference QGEMM implementations with cuBLAS block-scaled matmul."""

    @pytest.mark.parametrize(
        "M, K, N, pow_2_scales",
        [
            (256, 256, 256, False),  # NVFP4 mode, block_length=16
            (512, 512, 512, False),  # Medium matrix
            (8192, 8192, 8192, False),  # Larger matrix  
        ],
    )
    def test_ref_qgemm_vs_cublas(
        self, M, K, N, pow_2_scales
    ):
        """
        Test comparing _ref_qgemm and qgemm_cmp_cublass_real (cuBLAS block-scaled matmul).
        
        Args:
            M: Number of rows in input
            K: Inner dimension
            N: Number of columns in weight
            pow_2_scales: Whether to use power-of-2 scaling
        """
        device = "cuda"
        out_dtype = torch.bfloat16

        # Check if CUDA is available
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")

        # Check if compute capability supports block-scaled matmul (requires compute capability 10.0+)
        if torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability()
            if compute_capability[0] < 10:
                pytest.skip(f"Block-scaled matmul requires compute capability 10.0+, got {compute_capability[0]}.{compute_capability[1]}")

        qx, qw, sx, sw, global_amax_qx, global_amax_qw, qparams = initialize_nvfp4_for_qgemm(
            M, N, K, device=device, seed=10
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
            _ = qgemm_cmp_cublass_real(
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

        # Benchmark qgemm_cmp_cublass_real
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_runs):
            y_cublas = qgemm_cmp_cublass_real(
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
        cublas_time = (time.time() - start_time) / num_runs

        # Print comparison info
        print(f"\n{'='*80}")
        print(f"Comparison for M={M}, K={K}, N={N}, pow_2_scales={pow_2_scales}:")
        print(f"{'='*80}")
        
        # Performance comparison
        print(f"\nPerformance Comparison (average over {num_runs} runs):")
        print(f"  _ref_qgemm:                {ref_time*1000:.4f} ms")
        print(f"  qgemm_cmp_cublass_real:   {cublas_time*1000:.4f} ms")
        speedup = ref_time / cublas_time if cublas_time > 0 else float('inf')
        if speedup > 1:
            print(f"  Speedup:                   {speedup:.2f}x (cuBLAS is faster)")
        else:
            print(f"  Slowdown:                  {1/speedup:.2f}x (cuBLAS is slower)")
        
        # Output shape and range
        print(f"\nOutput Information:")
        print(f"  _ref_qgemm output shape: {y_ref.shape}, dtype: {y_ref.dtype}")
        print(f"  _ref_qgemm output range: [{y_ref.min().item():.4f}, {y_ref.max().item():.4f}]")
        print(f"  qgemm_cmp_cublass_real output shape: {y_cublas.shape}, dtype: {y_cublas.dtype}")
        print(f"  qgemm_cmp_cublass_real output range: [{y_cublas.min().item():.4f}, {y_cublas.max().item():.4f}]")
        
        # Calculate difference metrics
        abs_diff = torch.abs(y_ref.to(torch.float32) - y_cublas.to(torch.float32))
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
        
        assert y_cublas.shape == (M, N), f"Expected shape ({M}, {N}), got {y_cublas.shape}"
        assert y_cublas.dtype == out_dtype, f"Expected dtype {out_dtype}, got {y_cublas.dtype}"
        assert not torch.isnan(y_cublas).any(), "qgemm_cmp_cublass_real output contains NaN values"
        assert not torch.isinf(y_cublas).any(), "qgemm_cmp_cublass_real output contains Inf values"

        # Compare results with appropriate tolerance
        # Note: cuBLAS block-scaled matmul may have different numerical precision
        print(f"(pass M={M}, K={K}, N={N}, pow_2_scales={pow_2_scales})")

        torch.testing.assert_close(
            y_ref.to(torch.float32),
            y_cublas.to(torch.float32),
            rtol=1e-1,  # 10% relative tolerance
            atol=1e-1,  # 0.1 absolute tolerance
            msg=f"_ref_qgemm and qgemm_cmp_cublass_real outputs differ significantly for M={M}, K={K}, N={N}"
        )

    @pytest.mark.parametrize(
        "M, K, N",
        [
            (256, 256, 256),
            (512, 512, 512),
        ],
    )
    def test_cublas_accumulate(
        self, M, K, N
    ):
        """
        Test accumulation mode for qgemm_cmp_cublass_real.
        
        Args:
            M: Number of rows in input
            K: Inner dimension
            N: Number of columns in weight
        """
        device = "cuda"
        out_dtype = torch.bfloat16

        # Check if CUDA is available
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")

        # Check if compute capability supports block-scaled matmul
        if torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability()
            if compute_capability[0] < 10:
                pytest.skip(f"Block-scaled matmul requires compute capability 10.0+, got {compute_capability[0]}.{compute_capability[1]}")

        qx, qw, sx, sw, global_amax_qx, global_amax_qw, qparams = initialize_nvfp4_for_qgemm(
            M, N, K, device=device, seed=10
        )
        
        quantize_op = Nvfp4TiledQuantizeRefOp()
        pow_2_scales = False

        # Create output tensor for accumulation
        out = torch.randn((M, N), dtype=out_dtype, device=device)

        # Test reference implementation
        y_ref = quantize_op._ref_qgemm(
            qx=qx, qw=qw, m_params=None, out_dtype=out_dtype,
            sx=sx, sw=sw, bias=None, out=out.clone(), accumulate=True,
            pow_2_scales=pow_2_scales, global_amax_qx=global_amax_qx,
            global_amax_qw=global_amax_qw, qparams_x=qparams, qparams_w=qparams,
        )

        # Test cuBLAS implementation
        y_cublas = qgemm_cmp_cublass_real(
            qx=qx, qw=qw, m_params=None, out_dtype=out_dtype,
            sx=sx, sw=sw, bias=None, out=out.clone(), accumulate=True,
            pow_2_scales=pow_2_scales, global_amax_qx=global_amax_qx,
            global_amax_qw=global_amax_qw, qparams_x=qparams, qparams_w=qparams
        )

        # Verify outputs
        assert y_ref.shape == (M, N), f"Expected shape ({M}, {N}), got {y_ref.shape}"
        assert y_cublas.shape == (M, N), f"Expected shape ({M}, {N}), got {y_cublas.shape}"
        assert not torch.isnan(y_ref).any(), "_ref_qgemm output contains NaN values"
        assert not torch.isnan(y_cublas).any(), "qgemm_cmp_cublass_real output contains NaN values"

        # Compare results
        torch.testing.assert_close(
            y_ref.to(torch.float32),
            y_cublas.to(torch.float32),
            rtol=1e-1,
            atol=1e-1,
            msg=f"Accumulation mode outputs differ for M={M}, K={K}, N={N}"
        )

        print(f"✅ Accumulation test passed for M={M}, K={K}, N={N}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
