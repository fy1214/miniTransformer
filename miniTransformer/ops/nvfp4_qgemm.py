import functools

import torch
import pytest
import cutlass
import fp4_gemm
import cutlass.cute as cute
import cutlass.torch as cutlass_torch

from typing import Optional

from miniTransformer.ops.cuteDSL.nvfp4.scale_factor_swizzle import cvt_sf_MKL_to_M32x4xrm_K4xrk_L
from miniTransformer.quantiztion import quantization
from miniTransformer.quantiztion.nvfp4_quantize_op_ref import cast_from_fp4x2
from miniTransformer.quantiztion.nvfp4_quantize_op_ref import Nvfp4TiledQuantizeRefOp
from triton.tools.tensor_descriptor import TensorDescriptor
from miniTransformer.ops.triton.gemm_tl import block_scaled_matmul
from cutlass.cute.runtime import make_ptr

from miniTransformer.ops.cuteDSL.nvfp4.gemm_cute_opts import (
    OptimizedBlockScaledDenseGemmKernel,
    ab_dtype,
    sf_dtype,
    sf_vec_size,
)

def ceil_div(a, b):
    return (a + b - 1) // b


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


def qgemm_tl(
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

    # sx, sw: 4D (M//128, K//64, 32, 16) and (N//128, K//64, 32, 16) from init; use directly, no permute
    a_scale_shape = [M // 128, K // VEC_SIZE // 4, 32, 16]  # [M//128, K//64, 32, 16]
    b_scale_shape = [N // 128, K // VEC_SIZE // 4, 32, 16]
    
  # Convert 2D -> 4D for triton/cuBLAS
    sx_4d = swizzle_scale(sx, M, K).to(torch.float8_e4m3fn)
    sw_4d = swizzle_scale(sw, N, K).to(torch.float8_e4m3fn)

    sx_reshaped = sx_4d.to(torch.float8_e4m3fn).to(device)
    sw_reshaped = sw_4d.to(torch.float8_e4m3fn).to(device)

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

    # Global alpha for NVFP4: (global_amax_qx * global_amax_qw) / (6*6*448*448). Use 1.0 if pow_2_scales or K==0.
    if not pow_2_scales and K > 0 and global_amax_qx is not None and global_amax_qw is not None:
        factor = 6.0 * 6.0 * 448.0 * 448.0
        alpha = (global_amax_qx * global_amax_qw).to(torch.float32).squeeze(-1).item() / factor
    else:
        alpha = 1.0

    output_dtype_for_kernel = torch.float16
    output = block_scaled_matmul(
        a_desc, a_scale_desc, b_desc, b_scale_desc,
        alpha,
        output_dtype_for_kernel, M, N, K, rep_m, rep_n, rep_k, configs
    )

    # Handle accumulation
    if accumulate:
        assert out is not None, "Output tensor must be provided for accumulation."
        output = (out.to(torch.float32) + output.to(torch.float32)).to(out_dtype)
        return output
    else:
        output = output.to(out_dtype)
        return output


def qgemm_cmp_cublass_real(
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
    cuBLAS block-scaled QGEMM implementation using fp4_gemm.block_scaled_matmul_nvfp4.
    
    This implementation uses cuBLAS's hardware-accelerated block-scaled matmul for NVFP4 format.
    sx, sw must be 4D (M//128, K//64, 32, 16) and (N//128, K//64, 32, 16); use directly, no permute.
    
    Args:
        qx: Packed FP4 input tensor (M, K//2), uint8
        qw: Packed FP4 weight tensor (N, K//2), uint8
        m_params: Matrix multiplication parameters
        out_dtype: Output data type
        sx: Scale tensor for x, 4D (M//128, K//64, 32, 16), float8_e4m3fn
        sw: Scale tensor for w, 4D (N//128, K//64, 32, 16), float8_e4m3fn
        bias: Optional bias tensor (not supported, must be None)
        out: Optional output tensor for accumulation
        accumulate: Whether to accumulate into output tensor
        pow_2_scales: Whether scales are power-of-2 (not used in cuBLAS path)
        global_amax_qx: Global amax for x (used for NVFP4 global scaling)
        global_amax_qw: Global amax for w (used for NVFP4 global scaling)
        qparams_x: Quantization parameters for x (not used)
        qparams_w: Quantization parameters for w (not used)
        
    Returns:
        Output tensor (M, N) in out_dtype
    """
    assert bias is None, "Bias is not implemented for cuBLAS block-scaled version."
    
    device = qx.device
    M, K_bytes_a = qx.shape
    N, K_bytes_b = qw.shape
    K = K_bytes_a * 2  # K in elements (2 elements per byte)
    
    assert K_bytes_a == K_bytes_b, f"K dimension mismatch: {K_bytes_a} vs {K_bytes_b}"
    assert qx.dtype == torch.uint8, f"qx must be uint8, got {qx.dtype}"
    assert qw.dtype == torch.uint8, f"qw must be uint8, got {qw.dtype}"
    
    # sx, sw: 4D (M//128, K//64, 32, 16) and (N//128, K//64, 32, 16) from init; use directly, no permute
    VEC_SIZE = 16
    a_scale_shape = [M // 128, K // 64, 32, 16]
    b_scale_shape = [N // 128, K // 64, 32, 16]
    assert sx.shape == tuple(a_scale_shape), f"sx shape {sx.shape} vs {a_scale_shape}"
    assert sw.shape == tuple(b_scale_shape), f"sw shape {sw.shape} vs {b_scale_shape}"

    sx_4d = sx.to(torch.float8_e4m3fn).to(device)
    sw_4d = sw.to(torch.float8_e4m3fn).to(device)
    sx_cublas = sx_4d.contiguous().flatten()
    sw_cublas = sw_4d.contiguous().flatten()
    
    # Create workspace tensor (32MB as in reference implementation)
    workspace_size = 32 * 1024 * 1024
    workspace_tensor = torch.empty(workspace_size, dtype=torch.uint8, device=device)
    
    # Prepare output tensor (cuBLAS outputs FP16)
    output = torch.empty((M, N), dtype=torch.float16, device=device)
    
    # Call cuBLAS block-scaled matmul (extension expects 8 args: ..., workspace_size, alpha)
    fp4_gemm.block_scaled_matmul_nvfp4(
        qx, qw, output, sx_cublas, sw_cublas, workspace_tensor, workspace_size, 1.0
    )
    
    # Apply global scaling for NVFP4 mode
    # if not pow_2_scales and K > 0:
    #     assert global_amax_qx is not None, "global_amax_qx must be provided for NVFP4 mode"
    #     assert global_amax_qw is not None, "global_amax_qw must be provided for NVFP4 mode"
    #     factor = 6.0 * 6.0 * 448.0 * 448.0
    #     partial_alpha = global_amax_qx * global_amax_qw
    #     alpha = torch.div(partial_alpha, factor).squeeze(-1)
    #     output = alpha * output
    
    # Handle accumulation
    if accumulate:
        assert out is not None, "Output tensor must be provided for accumulation."
        output = (out.to(torch.float32) + output.to(torch.float32)).to(out_dtype)
        return output
    else:
        output = output.to(out_dtype)
        return output

def pack_scale_from_4d(scale_4d, target_M, target_K):
    """
    4D scale (M//128, K//64, 32, 16) -> 2D (M, K//16) for _ref_qgemm.
    Triton/cuBLAS use 4D directly.
    """
    scale_4d_fp32 = scale_4d.to(torch.float32)
    scale_2d = scale_4d_fp32.reshape(target_M, target_K // 16).contiguous()
    return scale_2d


def swizzle_scale(scale_2d, target_M, target_K):
    """
    2D scale (M, K//16) -> 4D (M//128, K//64, 32, 16) for triton/cuBLAS.
    Inverse of pack_scale_from_4d.
    """
    num_chunk_m, num_chunk_k = target_M // 128, (target_K // 16) // 4
    u = scale_2d.reshape(num_chunk_m, 4, 32, num_chunk_k, 4)
    return u.permute(0, 3, 2, 1, 4).reshape(
        num_chunk_m, num_chunk_k, 32, 16
    )


def initialize_nvfp4_for_qgemm(M, N, K, device="cuda", seed=10):
    """
    Initialize NVFP4 format inputs for _ref_qgemm and qgemm_tl testing.

    Same flow as linear.py for data: FP32 tensors + quantize_op.quantize.
    - x: randn (M, K); w: init like linear (normal mean=0, std=0.023).
    - Scales: from Nvfp4TiledQuantizeRefOp.quantize() output (2D), converted to 4D for triton/cuBLAS.
      - sx_4d/sw_4d: 4D (M//128, K//64, 32, 16) for triton/cuBLAS (via swizzle_scale).
      - sx/sw: 2D (M, K//16) for _ref_qgemm (directly from quantize output).

    Returns:
        (qx, qw, sx_4d, sw_4d, sx, sw, global_amax_qx, global_amax_qw, qparams)
        sx_4d/sw_4d: 4D for triton/cuBLAS; sx/sw: 2D for _ref_qgemm.
    """
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    VEC_SIZE = 16

    # Linear-style: FP32 input + weight, then quantize (like linear.py)
    x_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
    w_fp32 = torch.empty(N, K, device=device, dtype=torch.float32)
    torch.nn.init.normal_(w_fp32, mean=0.0, std=0.023)

    x_params = quantization.QParams(
        scaling_dtype=torch.float8_e4m3fn,
        output_dtype=torch.uint8,
        eps=0.0,
        pow_2_scales=False,
        quant_tile_shape=(1, 16),
        with_2d_quantization=False,
        with_rht=False,
    )
    w_params = quantization.QParams(
        scaling_dtype=torch.float8_e4m3fn,
        output_dtype=torch.uint8,
        eps=0.0,
        pow_2_scales=False,
        quant_tile_shape=(1, 16),
        with_2d_quantization=False,
        with_rht=False,
    )

    quantize_op = Nvfp4TiledQuantizeRefOp()
    qresult_x = quantize_op.quantize(x_fp32, x_params, return_transpose=False)
    qresult_w = quantize_op.quantize(w_fp32, w_params, return_transpose=False)

    qx = qresult_x.data
    qw = qresult_w.data
    global_amax_qx = qresult_x.tensor_scale
    global_amax_qw = qresult_w.tensor_scale_t
    if global_amax_qx is None:
        global_amax_qx = torch.max(torch.abs(x_fp32)).to(torch.float32).view(1)
    if global_amax_qw is None:
        global_amax_qw = torch.max(torch.abs(w_fp32)).to(torch.float32).view(1)
    if global_amax_qx.numel() > 1:
        global_amax_qx = global_amax_qx.max().view(1)
    if global_amax_qw.numel() > 1:
        global_amax_qw = global_amax_qw.max().view(1)

    sx_2d = qresult_x.scale  # (M, K//16)
    sw_2d = qresult_w.scale  # (N, K//16)

    sx = sx_2d.to(torch.float8_e4m3fn)
    sw = sw_2d.to(torch.float8_e4m3fn)

    qparams = x_params
    return qx, qw, sx, sw, global_amax_qx, global_amax_qw, qparams



def create_ref_scale_factor_tensor(l, mn, sf_k):
    """
    Create the reference scale factor tensor (mn, sf_k, l) in ref layout.
    Returns float8_e4m3fn tensor on CPU; callers may copy_ their data into it.
    """
    ref_shape = (l, mn, sf_k)
    ref_permute_order = (1, 2, 0)
    ref_f8_random_int = torch.randint(1, 3, ref_shape, dtype=torch.int8)
    ref_f8_torch_tensor_cpu = ref_f8_random_int.to(dtype=torch.float8_e4m3fn)
    ref_f8_torch_tensor_cpu_permuted = ref_f8_torch_tensor_cpu.permute(
        *ref_permute_order
    )
    return ref_f8_torch_tensor_cpu_permuted


def create_cute_scale_factor_tensor(l, mn, sf_k, ref_f8_torch_tensor_cpu_permuted):
    """
    Create the CUTE-format scale factor tensor on CUDA from ref (mn, sf_k, l).
    """
    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        l,
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )
    # ref_f8_torch_tensor_cpu_permuted: (mn//128, sf_k//4, 32, 16, 1)
    mma_permute_order = (3, 4, 1, 5, 2, 0) # (32, 4, mn/128, 4, sf_k/4, 1)
    rand_int_tensor = torch.randint(0, 2, mma_shape, dtype=torch.int8)
    cute_f8_torch_tensor_cpu = rand_int_tensor.to(dtype=torch.float8_e4m3fn)
    cute_f8_torch_tensor_cpu = cute_f8_torch_tensor_cpu.permute(*mma_permute_order)
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        make_ptr(
            cutlass.Float8E4M3FN,
            ref_f8_torch_tensor_cpu_permuted.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        ),
        make_ptr(
            cutlass.Float8E4M3FN,
            cute_f8_torch_tensor_cpu.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        ),
        mn,
        sf_k,
        l,
        mma_shape,
    )
    return cute_f8_torch_tensor_cpu.cuda()


def cutedsl_nvfp4_gemm_optimized(
    qx,
    qw,
    sx,
    sw,
    M,
    N,
    K,
    alpha,
    out_dtype,
    device,
    out=None,
    accumulate=False,
    has_swizzle=True,
    do_benchmark=False,
    warmup_iterations=10,
    iterations=100,
    use_cold_l2=False,
):
    """Run CuTeDSL NVFP4 block-scaled GEMM and return D = alpha * GEMM(qx, qw, sx, sw) in out_dtype."""

    alpha_value = float(alpha.item())
    l = 1
    sf_k = ceil_div(K, sf_vec_size)

    a_ref_f4 = qx.view(torch.float4_e2m1fn_x2).to(device)
    b_ref_f4 = qw.view(torch.float4_e2m1fn_x2).to(device)
    if accumulate:
        assert out is not None, "out must be provided when accumulate=True"
        c_tensor = out.unsqueeze(-1).contiguous().to(out_dtype).to(device)
    else:
        c_tensor = torch.empty((M, N, l), dtype=out_dtype, device=device)

    def _ref_2d_to_cute_sf(data_2d, mn, has_swizzle=True):
        ref = data_2d.unsqueeze(-1).to(dtype=torch.float8_e4m3fn)
        if has_swizzle:
            return create_cute_scale_factor_tensor(l, mn, sf_k, ref)
        else:
            return ref

    sfa_tensor = _ref_2d_to_cute_sf(sx, M, has_swizzle)
    sfb_tensor = _ref_2d_to_cute_sf(sw, N, has_swizzle)

    a_ptr = make_ptr(ab_dtype, a_ref_f4.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(ab_dtype, b_ref_f4.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    c_cutlass_dtype = cutlass.BFloat16 if c_tensor.dtype == torch.bfloat16 else cutlass.Float32
    c_ptr = make_ptr(c_cutlass_dtype, c_tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)

    sfa_ptr = make_ptr(sf_dtype, sfa_tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    sfb_ptr = make_ptr(sf_dtype, sfb_tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)

    gemm = OptimizedBlockScaledDenseGemmKernel()
    stream = cutlass_torch.default_stream()
    compiled = cute.compile(
        gemm,
        a_ptr,
        b_ptr,
        sfa_ptr,
        sfb_ptr,
        c_ptr,
        (M, N, K, l),
        stream,
        epilogue_op=lambda x: x * alpha_value,
        accumulate=accumulate,
    )

    if do_benchmark:
        def generate_tensors():
            args = cute.testing.JitArguments(
                a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (M, N, K, l), stream
            )
            args.add_to_scope([a_ref_f4, b_ref_f4, sfa_tensor, sfb_tensor, c_tensor])
            return args

        workspace_count = 1
        if use_cold_l2:
            one_workspace_bytes = (
                a_ref_f4.numel() * a_ref_f4.element_size()
                + b_ref_f4.numel() * b_ref_f4.element_size()
                + sfa_tensor.numel() * sfa_tensor.element_size()
                + sfb_tensor.numel() * sfb_tensor.element_size()
                + c_tensor.numel() * c_tensor.element_size()
            )
            workspace_count = cute.testing.get_workspace_count(
                one_workspace_bytes, warmup_iterations, iterations
            )

        # benchmark() returns kernel time in microseconds.
        elapsed_us = cute.testing.benchmark(
            compiled,
            workspace_generator=generate_tensors,
            workspace_count=workspace_count,
            stream=stream,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
        )

        print(f"CuTeDSL benchmark time: {elapsed_us:.3f} us")
        peta_flops = (4 * M * N * K * l) / (elapsed_us * 1e-6) / 1e9 / 1000000
        print(f"CuTeDSL FLOPS: {peta_flops:.6f} PFLOPS")
        bytes_transfer = (
            2 * M * K / 2 * l * a_ref_f4.element_size()
            + 2 * N * K / 2 * l * b_ref_f4.element_size()
            + 2 * M * N * l * c_tensor.element_size()
            + 2 * M * sf_k * l * sfa_tensor.element_size()
            + 2 * N * sf_k * l * sfb_tensor.element_size()
        )
        print(f"CuTeDSL bytes: {bytes_transfer} Bytes")
        bandwidth = bytes_transfer / elapsed_us * 1e-3
        print(f"CuTeDSL BW: {bandwidth:.3f} GB/s")

    compiled(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (M, N, K, l), stream)
    return c_tensor.squeeze(-1).float().to(out_dtype)
