from miniTransformer.ops.nvfp4_qgemm import swizzle_scale
from miniTransformer.quantiztion import quantization
from miniTransformer.quantiztion.nvfp4_quantize_op_ref import Nvfp4TiledQuantizeRefOp
import torch
import fp4_gemm

from transformer_engine.pytorch import NVFP4Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.custom_recipes.quantization_nvfp4 import NVFP4QuantizerRef
from transformer_engine.pytorch.custom_recipes import utils
from transformer_engine.pytorch.constants import TE_DType

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
    qx: torch.Tensor = x_nvfp4_sut._rowwise_data.view(dtype=torch.uint8)
    assert x_nvfp4_sut._rowwise_scale_inv is not None
    sx: torch.Tensor = x_nvfp4_sut._rowwise_scale_inv
    qx_t = (
        x_nvfp4_sut._columnwise_data.view(dtype=torch.uint8)
        if x_nvfp4_sut._columnwise_data is not None
        else None
    )
    sx_t = x_nvfp4_sut._columnwise_scale_inv
    qx_amax = x_nvfp4_sut._amax_rowwise
    qx_amax_t = x_nvfp4_sut._amax_columnwise

    return quantization.QuantizeResult(qx, sx, qx_t, sx_t, qx_amax, qx_amax_t), x_nvfp4_sut


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

    x_res, x_nvfp4_native = quantize_transpose_nvfp4(x, False, True)
    w_res, w_nvfp4_native = quantize_transpose_nvfp4(w, True, True)

    # Setup out tensor if accumulate is True
    if accumulate:
        out = torch.randn((M, N), dtype=out_dtype, device=device)
    else:
        out = torch.empty((M, N), dtype=out_dtype, device=device)

    # Native TE GEMM using tex.generic_gemm (cuBLAS GEMM)
    # Allocate cuBLAS workspace
    workspace = torch.empty(4, dtype=torch.uint8, device=device)

    '''
    fp4_gemm.cublas_gemm(
        x_res.data,
        swizzle_scale(x_res.scale, M, K),
        x_res.tensor_scale.item(),
        True,
        w_res.data,
        swizzle_scale(x_res.scale, N, K),
        w_res.tensor_scale.item(),
        False,
        1, 0,
        out,
        accumulate,
        workspace,
        workspace.shape[0],
    )
    '''
    fp4_gemm.block_scaled_matmul_nvfp4(
        x_res.data,
        w_res.data,
        out,
        swizzle_scale(x_res.scale, M, K),
        swizzle_scale(x_res.scale, N, K),
        x_res.tensor_scale.item(),
        w_res.tensor_scale.item(),
        workspace,
        workspace.shape[0]
    )

    ref_op = Nvfp4TiledQuantizeRefOp()
    qparam_x = quantization.QParams(
        output_dtype=torch.uint8, 
        scaling_dtype=torch.float8_e4m3fn, 
        quant_tile_shape=(1, 16), 
        with_2d_quantization=False,
        with_rht=True
    )
    qparam_w = quantization.QParams(
        output_dtype=torch.uint8, 
        scaling_dtype=torch.float8_e4m3fn, 
        quant_tile_shape=(16, 16), 
        with_2d_quantization=True,
        with_rht=False
    )
    x_res_ref = ref_op.quantize(x, qparam_x, True)
    w_res_ref = ref_op.quantize(w, qparam_w, True)
    y_ref = ref_op.qgemm(
        x_res_ref.data,
        w_res_ref.data,
        None,
        torch.bfloat16,
        x_res_ref.scale,
        w_res_ref.scale,
        None,
        out,
        accumulate=accumulate,
        global_amax_qx=x_res_ref.tensor_scale,
        global_amax_qw=w_res_ref.tensor_scale,
        qparams_x=qparam_x,
        qparams_w=qparam_w
    )

    transa = True if not w_columnwise else False
    transb = False if not x_columnwise else True
    out_quantizer = None
    bias = None
    bias_dtype = TE_DType[torch.bfloat16]
    use_gelu = False
    gelu_input = None
    use_grad = False
    use_split_accumulator = False
    
    y_native = tex.generic_gemm(
        w_nvfp4_native,
        transa,
        x_nvfp4_native,
        transb,
        out.clone() if accumulate else None,
        out_quantizer,
        TE_DType[out_dtype],
        bias,
        bias_dtype,
        use_gelu,
        gelu_input,
        use_grad,
        workspace,
        workspace.shape[0],
        accumulate,
        use_split_accumulator,
    )[0]

    # just in case of accumulation, make sure y_ref and y_native are not the same tensor
    assert y_ref is not y_native, "y_ref and y_native should not be the same tensor"
    # Reset nans to zeros because torch.assert_close does not assume nans to be equal
    assert not torch.isnan(y_ref.float()).all(), "All elements are nan"
    y_ref = torch.where(y_ref.isnan(), torch.zeros_like(y_ref), y_ref)
    y_native = torch.where(y_native.isnan(), torch.zeros_like(y_native), y_native)

    # Compare results with some tolerance
    torch.testing.assert_close(y_native, y_ref, atol=8e-3, rtol=8e-3)

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