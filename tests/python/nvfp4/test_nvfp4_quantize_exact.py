import torch
import fp4_gemm

nvfp4_scale_t = float8_e4m3fn
nvfp4_output_t = torch.unit8

def unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    repeated = x.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F
    repeated[:, 1::2] >>= 4
    return repeated


def check_quantization_nvfp4_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_transpose: bool,
    with_2d_quantization: bool,
) -> None:
    
    # Setup device and random seed
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Input
    x = torch.randn((M, N), dtype=x_dtype, device=device)

    qx = torch.randn((M, N // 2), dtype=nvfp4_output_t, device=device)
    sx = torch.randn((M, N // 16), dtype=nvfp4_scale_t, device=device)
    if return_transpose:
        qx_t = torch.randn((N, M // 2), dtype=nvfp4_output_t, device=device)
        sx_t = torch.randn((N, M // 16), dtype=nvfp4_scale_t, device=device)
    else:
        qx_t = None
        sx_t = None

    fp4_gemm.quantize_transpose_nvfp4(x, qx, sx, qx_t, sx_t, with_2d_quantization, return_transpose)


if __name__ == "__main__":
    check_quantization_nvfp4_versus_reference(
        x_dtype=torch.bfloat16,
        M=1024,
        N=4096,
        return_transpose=True,
        with_2d_quantization=True,
    )
