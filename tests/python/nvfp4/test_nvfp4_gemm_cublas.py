import torch
import pytest
import time
from triton.tools.mxfp import MXFP4Tensor
from miniTransformer.ops import quantization
from miniTransformer.ops.nvfp4_quantize_op_ref import Nvfp4TiledQuantizeRefOp
from miniTransformer.ops.nvfp4_qgemm_op import (
    qgemm_cmp_cublass_real
)


def initialize_nvfp4_for_qgemm(M, N, K, device="cuda", seed=10):
    """
    Initialize NVFP4 format inputs for _ref_qgemm and _ref_qgemm_with_triton testing.
    
    Similar to initialize_block_scaled, but returns format compatible with qgemm interface:
    - qx, qw: Packed uint8 tensors of shape (M, K//2) and (N, K//2)
    - sx, sw: Scale tensors of shape (M, K//16) and (N, K//16), dtype float8_e4m3fn
    - global_amax_qx, global_amax_qw: Global amax values
    
    Args:
        M: Number of rows in input matrix
        N: Number of columns in weight matrix
        K: Inner dimension
        device: Device to create tensors on (default: "cuda")
        seed: Random seed for reproducibility (default: 10)
        
    Returns:
        Tuple of (qx, qw, sx, sw, global_amax_qx, global_amax_qw, qparams):
        - qx: Packed FP4 input tensor (M, K//2), uint8
        - qw: Packed FP4 weight tensor (N, K//2), uint8
        - sx: Scale tensor for x (M, K//16), float8_e4m3fn
        - sw: Scale tensor for w (N, K//16), float8_e4m3fn
        - global_amax_qx: Global amax for x, shape (1,)
        - global_amax_qw: Global amax for w, shape (1,)
        - qparams: Quantization parameters
    """
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
    
    VEC_SIZE = 16  
    
    x_ref = MXFP4Tensor(size=(M, K), device=device).random()
    w_ref = MXFP4Tensor(size=(N, K), device=device).random()
    
    # Pack two fp4 elements per byte along K
    qx = x_ref.to_packed_tensor(dim=1)  # (M, K//2), uint8
    qw = w_ref.to_packed_tensor(dim=1)  # (N, K//2), uint8
    
    x_fp32 = x_ref.to(torch.float32)
    w_fp32 = w_ref.to(torch.float32)
    global_amax_qx = torch.max(torch.abs(x_fp32)).to(torch.float32).view(1)
    global_amax_qw = torch.max(torch.abs(w_fp32)).to(torch.float32).view(1)
    
    epsilon = 1e-8
    a_scale_2d = torch.rand([M, K // VEC_SIZE], device=device) + epsilon
    b_scale_2d = torch.rand([N, K // VEC_SIZE], device=device) + epsilon

    a_scale_2d = a_scale_2d.to(torch.float8_e4m3fn)
    b_scale_2d = b_scale_2d.to(torch.float8_e4m3fn)

    a_scale_shape_4d_shape = [M // 128, K // VEC_SIZE // 4, 32, 16]  # [M//128, K//64, 32, 16]
    b_scale_shape_4d_shape = [N // 128, K // VEC_SIZE // 4, 32, 16]  # [N//128, K//64, 32, 16]

    a_scale_4d = a_scale_2d.reshape(a_scale_shape_4d_shape)
    b_scale_4d = b_scale_2d.reshape(b_scale_shape_4d_shape)

    def pack_scale_from_4d(scale_4d, target_M, target_K):
        M_chunk, K_chunk = scale_4d.shape[0], scale_4d.shape[1]
        scale_4d_fp32 = scale_4d.to(torch.float32)
        scale_5d = scale_4d_fp32.reshape(M_chunk, K_chunk, 32, 4, 4)
        scale_5d = scale_5d.permute(0, 3, 2, 1, 4).contiguous()
        scale_2d = scale_5d.reshape(target_M, target_K // 16)
        return scale_2d
    
    sx = pack_scale_from_4d(a_scale_4d, M, K).to(torch.float8_e4m3fn)
    sw = pack_scale_from_4d(b_scale_4d, N, K).to(torch.float8_e4m3fn)
    
    # create QParams
    qparams = quantization.QParams(
        scaling_dtype=torch.float8_e4m3fn,
        output_dtype=torch.uint8,
        eps=0.0,
        pow_2_scales=False,
        quant_tile_shape=(1, 16),
        with_2d_quantization=False,
        with_rht=False,
    )

    return qx, qw, sx, sw, a_scale_2d, b_scale_2d, global_amax_qx, global_amax_qw, qparams


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

        qx, qw, sx, sw, sx_ref, sw_ref, global_amax_qx, global_amax_qw, qparams = initialize_nvfp4_for_qgemm(
            M, N, K, device=device, seed=10
        )
        
        quantize_op = Nvfp4TiledQuantizeRefOp()

        # Warmup runs
        warmup_runs = 3
        for _ in range(warmup_runs):
            _ = quantize_op._ref_qgemm(
                qx=qx, qw=qw, m_params=None, out_dtype=out_dtype,
                sx=sx_ref, sw=sw_ref, bias=None, out=None, accumulate=False,
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
                sx=sx_ref,
                sw=sw_ref,
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
            atol=8e-3,  # 10% relative tolerance
            rtol=8e-3,  # 0.1 absolute tolerance
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

        qx, qw, sx, sw, sx_ref, sw_ref, global_amax_qx, global_amax_qw, qparams = initialize_nvfp4_for_qgemm(
            M, N, K, device=device, seed=10
        )
        
        quantize_op = Nvfp4TiledQuantizeRefOp()
        pow_2_scales = False

        # Create output tensor for accumulation
        out = torch.randn((M, N), dtype=out_dtype, device=device)

        # Test reference implementation
        y_ref = quantize_op._ref_qgemm(
            qx=qx, qw=qw, m_params=None, out_dtype=out_dtype,
            sx=sx_ref, sw=sw_ref, bias=None, out=out.clone(), accumulate=True,
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
            atol=8e-3,
            rtol=8e-3,
            msg=f"Accumulation mode outputs differ for M={M}, K={K}, N={N}"
        )

        print(f"✅ Accumulation test passed for M={M}, K={K}, N={N}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
