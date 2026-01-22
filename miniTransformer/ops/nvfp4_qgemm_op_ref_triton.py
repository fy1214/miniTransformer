import torch
import pytest
import time
from typing import Optional, Tuple
from miniTransformer.utils import utils
from miniTransformer.ops import quantization

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.tools.mxfp import MXFP4Tensor
from miniTransformer.ops.nvfp4_quantize_op_ref import cast_from_fp4x2
from miniTransformer.ops.nvfp4_quantize_op_ref import Nvfp4TiledQuantizeRefOp


# triton kernel
def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    kernel_name = kernel.name
    if "ELEM_PER_BYTE_A" and "ELEM_PER_BYTE_B" and "VEC_SIZE" in args:
        if args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 1:
            kernel_name += "_mxfp8"
        elif args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 2:
            kernel_name += "_mixed"
        elif args["ELEM_PER_BYTE_A"] == 2 and args["ELEM_PER_BYTE_B"] == 2:
            if args["VEC_SIZE"] == 16:
                kernel_name += "_nvfp4"
            elif args["VEC_SIZE"] == 32:
                kernel_name += "_mxfp4"
    ret["name"] = f"{kernel_name} [M={M}, N={N}, K={K}]"
    ret["flops"] = 2.0 * M * N * K
    return ret

@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_matmul_kernel(  
        a_desc,  
        a_scale_desc,  
        b_desc,  
        b_scale_desc,  
        c_desc,  
        M: tl.constexpr,  
        N: tl.constexpr,  
        K: tl.constexpr,  
        output_type: tl.constexpr,  
        ELEM_PER_BYTE_A: tl.constexpr,  
        ELEM_PER_BYTE_B: tl.constexpr,  
        VEC_SIZE: tl.constexpr,  
        BLOCK_M: tl.constexpr,  
        BLOCK_N: tl.constexpr,  
        BLOCK_K: tl.constexpr,  
        rep_m: tl.constexpr,  
        rep_n: tl.constexpr,  
        rep_k: tl.constexpr, 
        NUM_STAGES: tl.constexpr,  
):  #
    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.float16
    elif output_type == 2:
        output_dtype = tl.float8e4nv

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k_a = 0
    offs_k_b = 0
    offs_scale_m = pid_m * rep_m
    offs_scale_n = pid_n * rep_n
    offs_scale_k = 0

    MIXED_PREC: tl.constexpr = ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = a_desc.load([offs_am, offs_k_a])
        b = b_desc.load([offs_bn, offs_k_b])
        scale_a = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])
        scale_b = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])

        scale_a = scale_a.reshape(rep_m, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        scale_b = scale_b.reshape(rep_n, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)

        if MIXED_PREC:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e2m1", accumulator)
        elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
            accumulator = tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator)
        else:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator)

        offs_k_a += BLOCK_K // ELEM_PER_BYTE_A
        offs_k_b += BLOCK_K // ELEM_PER_BYTE_B
        offs_scale_k += rep_k

    c_desc.store([offs_am, offs_bn], accumulator.to(output_dtype))

# triton kernel wrapper
def block_scaled_matmul(a_desc, a_scale_desc, b_desc, b_scale_desc, dtype_dst, M, N, K, rep_m, rep_n, rep_k, configs):
    output = torch.empty((M, N), dtype=dtype_dst, device="cuda")
    if dtype_dst == torch.float32:
        dtype_dst = 0
    elif dtype_dst == torch.float16:
        dtype_dst = 1
    elif dtype_dst == torch.float8_e4m3fn:
        dtype_dst = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype_dst}")

    BLOCK_M = configs["BLOCK_SIZE_M"]
    BLOCK_N = configs["BLOCK_SIZE_N"]
    c_desc = TensorDescriptor.from_tensor(output, [BLOCK_M, BLOCK_N])

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    block_scaled_matmul_kernel[grid](
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        M,
        N,
        K,
        dtype_dst,
        configs["ELEM_PER_BYTE_A"],
        configs["ELEM_PER_BYTE_B"],
        configs["VEC_SIZE"],
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_N"],
        configs["BLOCK_SIZE_K"],
        rep_m,
        rep_n,
        rep_k,
        configs["num_stages"],
    )
    return output

def qgemm_cmp_triton_ref(
    qx: torch.Tensor,
    qw: torch.Tensor,
    m_params: quantization.MMParams,  
    out_dtype: torch.dtype,
    sx: torch.Tensor,
    sw: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
    pow_2_scales: bool = False,
    global_amax_qx: torch.Tensor | None = None,
    global_amax_qw: torch.Tensor | None = None,
    qparams_x: Optional[quantization.QParams] = None,
    qparams_w: Optional[quantization.QParams] = None,
) -> torch.Tensor:
    """Alternative simplified QGEMM implementation for verification.
    
    This implementation uses element-wise scaling instead of block-wise computation.
    It serves as an alternative reference to verify the correctness of _ref_qgemm.
    """
    assert bias is None, "Bias is not implemented for triton reference."

    M, K_packed = qx.shape
    N, K_w_packed = qw.shape
    
    # Dequantize from FP4x2 to high precision
    high_precision_x = cast_from_fp4x2(qx, torch.float32)
    high_precision_w = cast_from_fp4x2(qw, torch.float32)
    
    M, K = high_precision_x.shape
    N, K_w = high_precision_w.shape
    assert K == K_w, "K dimension mismatch between qx and qw"

    block_length = 32 if pow_2_scales else 16
    assert K % block_length == 0, f"K dimension must be divisible by {block_length}"
    assert N % 8 == 0, "N dimension must be divisible by 8"

    # Convert scales to float32
    sx_fp32 = sx.to(torch.float32)
    sw_fp32 = sw.to(torch.float32)
    
    # Expand scales to match tensor dimensions
    # sx shape: (M, K//block_length) -> (M, K)
    # sw shape: (N, K//block_length) -> (N, K)
    sx_expanded = sx_fp32.repeat_interleave(block_length, dim=1)
    sw_expanded = sw_fp32.repeat_interleave(block_length, dim=1)
    
    # Apply element-wise scaling
    x_scaled = high_precision_x * sx_expanded  # (M, K)
    w_scaled = high_precision_w * sw_expanded  # (N, K)
    
    # Perform GEMM: x_scaled @ w_scaled.T
    y = torch.matmul(x_scaled, w_scaled.T)  # (M, N)
    
    # Apply global scaling for NVFP4 mode
    if not pow_2_scales and K > 0:
        assert global_amax_qx is not None
        assert global_amax_qw is not None
        factor = 6.0 * 6.0 * 448.0 * 448.0
        partial_alpha = global_amax_qx * global_amax_qw
        alpha = torch.div(partial_alpha, factor).squeeze(-1)
        y = alpha * y

    # Handle accumulation
    if accumulate:
        assert out is not None, "Output tensor must be provided for accumulation."
        y = (out.to(torch.float32) + y).to(out_dtype)
        return y
    else:
        y = y.to(out_dtype)
        return y


def qgemm_cmp_triton_real(
    qx: torch.Tensor,
    qw: torch.Tensor,
    m_params: quantization.MMParams,
    out_dtype: torch.dtype,
    sx: torch.Tensor,
    sw: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
    pow_2_scales: bool = False,
    global_amax_qx: torch.Tensor | None = None,
    global_amax_qw: torch.Tensor | None = None,
    qparams_x: Optional[quantization.QParams] = None,
    qparams_w: Optional[quantization.QParams] = None,
) -> torch.Tensor:
    """
    Triton block-scaled QGEMM implementation using block_scaled_matmul.
    """
    assert bias is None, "Bias is not implemented for triton block-scaled version."

    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256
    VEC_SIZE = 16  
    ELEM_PER_BYTE_A = 2
    ELEM_PER_BYTE_B = 2

    device = "cuda"
    a = qx.to(device)
    b = qw.to(device)

    M, K_bytes_a = a.shape
    N, K_bytes_b = b.shape
    K = 2 * K_bytes_a 
    assert K_bytes_a == K_bytes_b, f"K dimension mismatch: {K_bytes_a} vs {K_bytes_b}"

    assert M % 128 == 0, f"M ({M}) must be divisible by 128"
    assert N % 128 == 0, f"N ({N}) must be divisible by 128"
    assert K % 64 == 0, f"K ({K}) must be divisible by 64"

    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K // ELEM_PER_BYTE_A])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K // ELEM_PER_BYTE_B])

    sx_M, sx_K = sx.shape
    sw_N, sw_K = sw.shape
    
    assert sx_M == M, f"sx first dimension mismatch: {sx_M} vs {M}"
    assert sw_N == N, f"sw first dimension mismatch: {sw_N} vs {N}"
    assert sx_K == K // 16, f"sx second dimension mismatch: {sx_K} vs {K//16}, K={K}"
    assert sw_K == K // 16, f"sw second dimension mismatch: {sw_K} vs {K//16}, K={K}"

    a_scale_shape = [M // 128, K // VEC_SIZE // 4, 32, 16]  # [M//128, K//64, 32, 16]
    b_scale_shape = [N // 128, K // VEC_SIZE // 4, 32, 16]  # [N//128, K//64, 32, 16]


    sx_fp32 = sx.to(torch.float32).to(device) if sx.dtype != torch.float32 else sx.to(device)
    sw_fp32 = sw.to(torch.float32).to(device) if sw.dtype != torch.float32 else sw.to(device)

    sx_reshaped = sx_fp32.reshape(M // 128, 4, 32, K // 64, 4)
    sx_reshaped = sx_reshaped.permute(0, 3, 2, 1, 4).contiguous()
    sx_reshaped = sx_reshaped.reshape(a_scale_shape)

    sw_reshaped = sw_fp32.reshape(N // 128, 4, 32, K // 64, 4)
    sw_reshaped = sw_reshaped.permute(0, 3, 2, 1, 4).contiguous()
    sw_reshaped = sw_reshaped.reshape(b_scale_shape)

    sx_reshaped = sx_reshaped.to(torch.float8_e4m3fn)
    sw_reshaped = sw_reshaped.to(torch.float8_e4m3fn)

    rep_m = BLOCK_M // 128
    rep_n = BLOCK_N // 128
    rep_k = BLOCK_K // VEC_SIZE // 4

    a_scale_block_shape = [1, rep_m, rep_k, 2, 256]
    b_scale_block_shape = [1, rep_n, rep_k, 2, 256]

    sx_5d = sx_reshaped.reshape(1, sx_reshaped.shape[0], sx_reshaped.shape[1], 2, 256)
    sw_5d = sw_reshaped.reshape(1, sw_reshaped.shape[0], sw_reshaped.shape[1], 2, 256)

    sx_5d = sx_5d.to(device)
    sw_5d = sw_5d.to(device)

    a_scale_desc = TensorDescriptor.from_tensor(sx_5d, block_shape=a_scale_block_shape)
    b_scale_desc = TensorDescriptor.from_tensor(sw_5d, block_shape=b_scale_block_shape)

    configs = {
        "BLOCK_SIZE_M": BLOCK_M,
        "BLOCK_SIZE_N": BLOCK_N,
        "BLOCK_SIZE_K": BLOCK_K,
        "num_stages": 4,
        "ELEM_PER_BYTE_A": ELEM_PER_BYTE_A,
        "ELEM_PER_BYTE_B": ELEM_PER_BYTE_B,
        "VEC_SIZE": VEC_SIZE,
    }


    output_dtype_for_kernel = torch.float16 
    output = block_scaled_matmul(
        a_desc, a_scale_desc, b_desc, b_scale_desc,
        output_dtype_for_kernel, M, N, K, rep_m, rep_n, rep_k, configs
    )

    # Apply global scaling for NVFP4 mode
    if not pow_2_scales and K > 0:
        assert global_amax_qx is not None
        assert global_amax_qw is not None
        factor = 6.0 * 6.0 * 448.0 * 448.0
        partial_alpha = global_amax_qx * global_amax_qw
        alpha = torch.div(partial_alpha, factor).squeeze(-1)
        output = alpha * output

    # Handle accumulation
    if accumulate:
        assert out is not None, "Output tensor must be provided for accumulation."
        output = (out.to(torch.float32) + output.to(torch.float32)).to(out_dtype)
        return output
    else:
        output = output.to(out_dtype)
        return output

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
    

    a_scale_shape_4d = [M // 128, K // VEC_SIZE // 4, 32, 16]  # [M//128, K//64, 32, 16]
    b_scale_shape_4d = [N // 128, K // VEC_SIZE // 4, 32, 16]  # [N//128, K//64, 32, 16]
    
    epsilon = 1e-8
    a_scale_4d = torch.rand(a_scale_shape_4d, device=device) + epsilon
    b_scale_4d = torch.rand(b_scale_shape_4d, device=device) + epsilon
    
    a_scale_4d = a_scale_4d.to(torch.float8_e4m3fn)
    b_scale_4d = b_scale_4d.to(torch.float8_e4m3fn)
    
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
    
    return qx, qw, sx, sw, global_amax_qx, global_amax_qw, qparams

