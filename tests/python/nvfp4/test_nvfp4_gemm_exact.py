import torch
import fp4_gemm

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

    qx = torch.empty((M, N // 2), dtype=x_dtype, device=device).to(nvfp4_output_t)
    sx = torch.empty((M, N // 16), dtype=x_dtype, device=device).to(nvfp4_scale_t)
    if return_transpose:
        qx_t = torch.empty((N, M // 2), dtype=x_dtype, device=device).to(nvfp4_output_t)
        sx_t = torch.empty((N, M // 16), dtype=x_dtype, device=device).to(nvfp4_scale_t)
    else:
        qx_t = None
        sx_t = None
    
    fp4_gemm.quantize_transpose_nvfp4(x, qx, sx, qx_t, sx_t, with_2d_quantization, return_transpose)

    return qx, sx, qx_t, sx_t


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
    x_max = torch.max(torch.abs(x)).to(torch.float32).view(1)

    w = torch.randn((N, K), device="cuda").to(w_dtype)
    w_max = torch.max(torch.abs(w)).to(torch.float32).view(1)

    qx, sx, qx_t, sx_t = quantize_transpose_nvfp4(x, False, False)
    qw, sw, qw_t, sw_t = quantize_transpose_nvfp4(w, True, False)

    # Setup out tensor if accumulate is True
    if accumulate:
        out = torch.randn((M, N), dtype=out_dtype, device=device)
    else:
        out = None

    # Native TE GEMM using tex.generic_gemm (cuBLAS GEMM)
    # Allocate cuBLAS workspace
    workspace = torch.empty(4, dtype=torch.uint8, device=device)

    out = fp4_gemm.nvfp4_gemm(
        x,
        sx,
        x_max,
        True,
        w,
        sw,
        w_max,
        False,
        1, 0,
        out,
        accumulate,
        False,
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