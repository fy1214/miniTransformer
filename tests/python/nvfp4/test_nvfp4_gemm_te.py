"""
NVFP4 GEMM Exact Correctness Tests
===================================

This module tests the accuracy of NVFP4 quantized GEMM operations across 
different backends (cuBLAS, CUTLASS, Triton, CuTe DSL, FlashInfer) by 
comparing against reference implementations.

The tests verify that custom NVFP4 GEMM implementations match the behavior
of the native Transformer Engine implementation.
"""
from typing import Tuple

import pytest
import torch
import fp4_gemm
from transformer_engine.pytorch import NVFP4Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.custom_recipes.quantization_nvfp4 import NVFP4QuantizerRef
from transformer_engine.pytorch.custom_recipes import utils
from transformer_engine.pytorch.constants import TE_DType

from miniTransformer.ops.nvfp4_gemm_backend import dispatch_gemm_backend
from miniTransformer.quantiztion import quantization
from miniTransformer.quantiztion.nvfp4_quantize_op_ref import Nvfp4TiledQuantizeRefOp


# ==================== Constants ====================

NVFP4_OUTPUT_DTYPE = torch.uint8
NVFP4_SCALE_DTYPE = torch.float8_e4m3fn

# NVFP4 quantization constants
NVFP4_FP4_DTYPE = tex.DType.kFloat4E2M1
NVFP4_AMAX_DIVISOR = 6.0 * 6.0 * 448.0 * 448.0

# Tolerance for floating point comparison
RTOL = 1e-6
ATOL = 1e-6


# ==================== Helper Functions ====================

def quantize_transpose_nvfp4(
    tensor: torch.Tensor,
    with_2d_quantization: bool,
    return_transpose: bool
) -> Tuple[quantization.QuantizeResult, object]:
    """
    Quantize a tensor to NVFP4 format using Transformer Engine quantizer.
    
    Args:
        tensor: Input tensor to quantize (shape: [M, N])
        with_2d_quantization: Whether to use 2D quantization
        return_transpose: Whether to return transposed quantized data
        
    Returns:
        Tuple containing:
            - QuantizeResult with quantized data, scales, and amax values
            - Native NVFP4 quantized tensor from Transformer Engine
    """
    M, N = tensor.shape
    device = tensor.device
    dtype = tensor.dtype

    # Initialize NVFP4 quantizer
    nvfp4_quantizer = NVFP4Quantizer(
        fp4_dtype=NVFP4_FP4_DTYPE,
        rowwise=True,
        columnwise=return_transpose,
        with_amax_reduction=False,
        with_2d_quantization=with_2d_quantization,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
    )
    
    # Create empty quantized tensor and populate it
    quantized_tensor = nvfp4_quantizer.make_empty(
        (M, N), dtype=dtype, device=device, requires_grad=False
    )
    quantized_tensor = nvfp4_quantizer.update_quantized(tensor, quantized_tensor)
    
    # Extract rowwise quantized data and scales
    assert quantized_tensor._rowwise_data is not None
    rowwise_data = quantized_tensor._rowwise_data.view(dtype=torch.uint8)
    
    assert quantized_tensor._rowwise_scale_inv is not None
    rowwise_scale = quantized_tensor._rowwise_scale_inv
    
    # Extract columnwise data if available
    columnwise_data = (
        quantized_tensor._columnwise_data.view(dtype=torch.uint8)
        if quantized_tensor._columnwise_data is not None
        else None
    )
    columnwise_scale = quantized_tensor._columnwise_scale_inv
    
    # Extract amax values
    amax_rowwise = quantized_tensor._amax_rowwise
    amax_columnwise = quantized_tensor._amax_columnwise

    # Package results
    result = quantization.QuantizeResult(
        tensor, 
        rowwise_scale, 
        columnwise_data, 
        columnwise_scale, 
        amax_rowwise, 
        amax_columnwise
    )
    
    return result, quantized_tensor


def check_nvfp4_gemm_versus_reference(
    x_dtype: torch.dtype,
    w_dtype: torch.dtype,
    out_dtype: torch.dtype,
    M: int,
    K: int,
    N: int,
    accumulate: bool,
    x_columnwise: bool = False,
    w_columnwise: bool = False,
    backend: str = 'cutlass',
) -> None:
    """
    Compare NVFP4 GEMM implementation against reference implementation.
    
    This function performs a quantized matrix multiplication using a specified
    backend and compares the result against the native Transformer Engine 
    implementation.
    
    Args:
        x_dtype: Data type for input matrix X
        w_dtype: Data type for weight matrix W
        out_dtype: Data type for output matrix
        M: Number of rows in X and output
        K: Number of columns in X and rows in W
        N: Number of columns in W and output
        accumulate: Whether to accumulate into existing output tensor
        x_columnwise: Whether X is stored columnwise
        w_columnwise: Whether W is stored columnwise
        backend: Backend to use ('cublas', 'cutlass', 'triton', 'cuteDSL', 'flashinfer')
        
    Raises:
        AssertionError: If the difference between implementations exceeds tolerance
    """
    # ========== Setup ==========
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # ========== Generate Random Tensors ==========
    x = torch.randn((M, K), device=device, dtype=x_dtype)
    w = torch.randn((N, K), device=device, dtype=w_dtype)

    # ========== Quantize Inputs ==========
    x_quantized, x_nvfp4_native = quantize_transpose_nvfp4(
        x, with_2d_quantization=False, return_transpose=True
    )
    w_quantized, w_nvfp4_native = quantize_transpose_nvfp4(
        w, with_2d_quantization=True, return_transpose=True
    )

    # ========== Prepare Output Tensors ==========
    if accumulate:
        out = torch.randn((M, N), dtype=out_dtype, device=device)
    else:
        out = torch.empty((M, N), dtype=out_dtype, device=device)
    
    out_native = out.clone()

    # ========== Test Backend GEMM ==========
    # Calculate scaling factor
    alpha = (
        x_quantized.tensor_scale * w_quantized.tensor_scale
    ) / NVFP4_AMAX_DIVISOR

    # Dispatch to backend
    backend_impl = dispatch_gemm_backend(backend, "nvfp4")
    out_backend = backend_impl.gemm(
        x_quantized.data,
        w_quantized.data,
        out_dtype,
        x_quantized.scale,
        w_quantized.scale,
        None,  # x_scale_inv
        None,  # w_scale_inv
        False,  # use_x_scale_inv
        False,  # use_w_scale_inv
        "TN",  # transpose mode
        alpha,
        None,  # bias
        None,  # out_scale
        False,  # use_out_scale
    )

    # ========== Reference Implementation ==========
    ref_op = Nvfp4TiledQuantizeRefOp()
    
    # Configure quantization parameters for X
    qparam_x = quantization.QParams(
        output_dtype=torch.uint8,
        scaling_dtype=torch.float8_e4m3fn,
        quant_tile_shape=(1, 16),
        with_2d_quantization=False,
        with_rht=True
    )
    
    # Configure quantization parameters for W
    qparam_w = quantization.QParams(
        output_dtype=torch.uint8,
        scaling_dtype=torch.float8_e4m3fn,
        quant_tile_shape=(16, 16),
        with_2d_quantization=True,
        with_rht=False
    )

    # ========== Native TE GEMM (Ground Truth) ==========
    workspace = torch.empty(4, dtype=torch.uint8, device=device)
    
    transa = not w_columnwise
    transb = x_columnwise
    
    out_native = tex.generic_gemm(
        w_nvfp4_native,
        transa,
        x_nvfp4_native,
        transb,
        out_native if accumulate else None,
        None,  # out_quantizer
        TE_DType[out_dtype],
        None,  # bias
        TE_DType[torch.bfloat16],  # bias_dtype
        False,  # use_gelu
        None,  # gelu_input
        False,  # use_grad
        workspace,
        workspace.shape[0],
        accumulate,
        False,  # use_split_accumulator
    )[0]

    # ========== Compare Results ==========
    out_native_f32 = out_native.to(torch.float32)
    out_backend_f32 = out_backend.to(torch.float32)

    abs_diff = (out_native_f32 - out_backend_f32).abs()
    rel_diff = abs_diff / (out_native_f32.abs() + 1e-8)

    # Print statistics
    print(f"\n{'='*60}")
    print(f"Backend: {backend} | NVFP4 GEMM Test")
    print(f"Matrix dimensions: M={M}, K={K}, N={N} | Seed={seed}")
    print(f"{'-'*60}")
    print(f"  Max absolute diff:  {abs_diff.max().item():.6e}")
    print(f"  Mean absolute diff: {abs_diff.mean().item():.6e}")
    print(f"  Max relative diff:  {rel_diff.max().item():.6e}")
    print(f"  Mean relative diff: {rel_diff.mean().item():.6e}")
    print(f"{'='*60}")

    # Assert closeness
    torch.testing.assert_close(
        out_native_f32,
        out_backend_f32,
        rtol=RTOL,
        atol=ATOL,
        msg=f"{backend} NVFP4 GEMM mismatch for M={M}, K={K}, N={N}, seed={seed}",
    )



# ==================== Pytest Fixtures ====================

@pytest.fixture
def device() -> str:
    """Provide CUDA device for testing."""
    return "cuda"


@pytest.fixture
def seed() -> int:
    """Provide random seed for reproducibility."""
    return 0


# ==================== Test Cases ====================

@pytest.mark.parametrize(
    "backend",
    ['cublas', 'cutlass', 'triton', 'cuteDSL', 'flashinfer'],
    ids=lambda b: f"backend={b}"
)
@pytest.mark.parametrize(
    "M,K,N",
    [(1024, 1024, 1024)],
    ids=lambda dims: f"M={dims[0]},K={dims[1]},N={dims[2]}"
)
@pytest.mark.parametrize(
    "accumulate",
    [False],
    ids=lambda acc: f"accumulate={acc}"
)
def test_nvfp4_gemm_exact(
    backend: str,
    M: int,
    K: int,
    N: int,
    accumulate: bool,
    device: str,
    seed: int
) -> None:
    """
    Test NVFP4 GEMM exactness across different backends.
    
    This test verifies that the NVFP4 quantized GEMM implementation in the
    specified backend produces results that match the reference Transformer
    Engine implementation within specified tolerances.
    
    Args:
        backend: Backend implementation to test
                 ('cublas', 'cutlass', 'triton', 'cuteDSL', 'flashinfer')
        M: Matrix dimension - number of rows in X and output
        K: Matrix dimension - number of columns in X and rows in W
        N: Matrix dimension - number of columns in W and output
        accumulate: Whether to accumulate into output tensor
        device: CUDA device (from fixture)
        seed: Random seed for reproducibility (from fixture)
        
    The test performs:
        1. Random tensor generation with fixed seed
        2. NVFP4 quantization of input tensors
        3. GEMM using specified backend
        4. GEMM using reference implementation
        5. Comparison within tolerance (rtol=1e-6, atol=1e-6)
    """
    check_nvfp4_gemm_versus_reference(
        x_dtype=torch.bfloat16,
        w_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        M=M,
        K=K,
        N=N,
        accumulate=accumulate,
        x_columnwise=False,
        w_columnwise=False,
        backend=backend
    )


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    # Allow running with: python test_nvfp4_gemm_te.py
    pytest.main([__file__, "-v", "-s"])