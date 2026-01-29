import torch
import fp4_gemm

from miniTransformer.ops.nvfp4_quantize_op_ref import Nvfp4TiledQuantizeRefOp
from miniTransformer.ops.quantization import QParams

from transformer_engine.pytorch import NVFP4Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.custom_recipes.quantization_nvfp4 import NVFP4QuantizerRef
from transformer_engine.pytorch.custom_recipes import utils

nvfp4_output_t = torch.uint8
nvfp4_scale_t = torch.float8_e4m3fn

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

    ref_op = Nvfp4TiledQuantizeRefOp()
    param = QParams(
        output_dtype=torch.uint8, 
        scaling_dtype=torch.float8_e4m3fn, 
        quant_tile_shape=(16, 16), 
        with_2d_quantization=True,
        with_rht=False
    )
    result = ref_op.quantize(
        x,
        param,
        True,
        False,
    )
    qx_ref, sx_ref = result.data, result.scale

    torch.testing.assert_close(qx_te, qx_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(sx_te, sx_ref, atol=0.0, rtol=0.0)

if __name__ == "__main__":
    check_quantization_nvfp4_versus_reference(
        x_dtype=torch.bfloat16,
        M=1024,
        N=4096,
        return_transpose=True,
        with_2d_quantization=True,
    )
