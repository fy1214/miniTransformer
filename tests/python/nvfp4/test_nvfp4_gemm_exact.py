import torch
import fp4_gemm

from transformer_engine.pytorch import NVFP4Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.custom_recipes.quantization_nvfp4 import NVFP4QuantizerRef
from transformer_engine.pytorch.custom_recipes import utils

nvfp4_output_t = torch.uint8
nvfp4_scale_t = torch.float8_e4m3fn

def quantize_transpose_nvfp4(
    x: torch.Tensor,
    with_2d_quantization: bool,
    return_transpose: bool
):
    M, N = x.shape
    device = x.device
    x_dtype = x.dtype

    nvfp4_quantizer = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=return_transpose,
        with_amax_reduction=False,
        with_2d_quantization=with_2d_quantization,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
    )
    x_nvfp4_sut = nvfp4_quantizer.make_empty(
        (M, N), dtype=x_dtype, device=device, requires_grad=False
    )
    x_nvfp4_sut = nvfp4_quantizer.update_quantized(x, x_nvfp4_sut)
    assert x_nvfp4_sut._rowwise_data is not None
    qx_te: torch.Tensor = x_nvfp4_sut._rowwise_data.view(dtype=torch.uint8)
    assert x_nvfp4_sut._rowwise_scale_inv is not None
    sx_te: torch.Tensor = x_nvfp4_sut._rowwise_scale_inv
    qx_t_te = (
        x_nvfp4_sut._columnwise_data.view(dtype=torch.uint8)
        if x_nvfp4_sut._columnwise_data is not None
        else None
    )
    sx_t_te = x_nvfp4_sut._columnwise_scale_inv
    qx_amax_te = x_nvfp4_sut._amax_rowwise


    return qx_te, sx_te, qx_t_te, sx_t_te, qx_amax_te


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
):
    
    # Setup device and random seed
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    x = torch.randn((M, K), device="cuda").to(x_dtype)
    w = torch.randn((N, K), device="cuda").to(w_dtype)

    qx, sx, qx_t, sx_t, qmax = quantize_transpose_nvfp4(x, False, True)
    qw, sw, qw_t, sw_t, wmax = quantize_transpose_nvfp4(w, True, True)

    # Setup out tensor if accumulate is True
    if accumulate:
        out = torch.randn((M, N), dtype=out_dtype, device=device)
    else:
        out = torch.empty((M, N), dtype=out_dtype, device=device)

    # Native TE GEMM using tex.generic_gemm (cuBLAS GEMM)
    # Allocate cuBLAS workspace
    workspace = torch.empty(4, dtype=torch.uint8, device=device)

    out = fp4_gemm.cublas_gemm(
        qx,
        sx,
        qmax.item(),
        True,
        qw,
        sw,
        wmax.item(),
        False,
        1, 0,
        out,
        accumulate,
        workspace,
        workspace.shape[0],
    )


if __name__ == "__main__":
    check_nvfp4_gemm_versus_reference(
        x_dtype=torch.bfloat16,
        w_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        M=1024,
        K=1024,
        N=1024,
        accumulate=False,
        x_columnwise=False,
        w_columnwise=False,
    )