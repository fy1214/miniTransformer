import math
import torch
import fp4_gemm
from typing import Optional

from miniTransformer.quantiztion import quantization
from miniTransformer.utils import utils
from miniTransformer.ops.nvfp4_qgemm import (
    qgemm_tl,
    swizzle_scale,
)

from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm

nvfp4_output_t = torch.uint8
nvfp4_scale_t = torch.float8_e4m3fn

use_cpp_allocator = False

class Nvfp4TiledQuantizeOp(quantization.QuantizeOpBase):
    def __init__(self):
        super().__init__()

    @property
    def supports_allgather(self) -> bool:
        return False   

    def _quantize(
        self,
        x: torch.Tensor,
        qparams: quantization.QParams,
        return_transpose: bool,
        with_2d_quantization: bool
    ):
        M, N = x.shape
        device = x.device
        x_dtype = x.dtype

        nvfp4_quantizer = NVFP4Quantizer(
            rowwise=True,
            columnwise=return_transpose,
            with_amax_reduction=False,
            amax_reduction_group=None,
            with_rht=qparams.with_rht,
            with_post_rht_amax=qparams.with_rht,
            with_2d_quantization=with_2d_quantization,
            stochastic_rounding=qparams.stochastic_rounding
        )
        if use_cpp_allocator:
            nvfp4_tensor = nvfp4_quantizer(x)
        else:
            nvfp4_tensor = nvfp4_quantizer.make_empty(
                (M, N), dtype=x_dtype, device=device, requires_grad=False
            )
            nvfp4_tensor = nvfp4_quantizer.update_quantized(x, nvfp4_tensor)

        return nvfp4_tensor

    def quantize(
        self,
        x: torch.Tensor,
        qparams: quantization.QParams,
        return_transpose: bool = False,
        reduce_amax: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        input_meta: Optional[quantization.QuantizeInputMetaBase] = None,
    ) -> quantization.QuantizeResult:
        
        assert x.dtype in utils.HIGH_PRECISION_FLOAT_DTYPES, "Unsupported input dtype."

        assert qparams.quant_tile_shape is not None
        assert len(qparams.quant_tile_shape) == 2
        

        nvfp4_tensor = self._quantize(
            x,
            qparams,
            return_transpose=return_transpose,
            with_2d_quantization=qparams.with_2d_quantization,
        )
        return quantization.QuantizeResult(data=nvfp4_tensor, scale=nvfp4_tensor, data_t=nvfp4_tensor, scale_t=nvfp4_tensor)

    def qgemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        m_params: quantization.MMParams,
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        grad: bool = False,
        accumulate: bool = False,
        layout: str = "TN",
        global_amax_qx: torch.Tensor | None = None,
        global_amax_qw: torch.Tensor | None = None,
        qparams_x: Optional[quantization.QParams] = None,
        qparams_w: Optional[quantization.QParams] = None,
        custom_gemm: bool = False,
        custom_gemm_backend: str = "cutlass",  # cutlass, triton, cuteDSL
    ) -> torch.Tensor:

        if custom_gemm:
            assert custom_gemm_backend in [
                "cutlass",
                "triton",
                "cuteDSL",
            ], "Unsupported custom gemm backend."

            device = qx.device
            M, K_bytes_a = qx.shape
            N, K_bytes_b = qw.shape
            assert (
                K_bytes_a == K_bytes_b
            ), f"K dimension mismatch between qx ({K_bytes_a}) and qw ({K_bytes_b})"
            K = 2 * K_bytes_a  # 2 FP4 values per byte

            if custom_gemm_backend == "cutlass":

                sx_4d = swizzle_scale(sx, M, K).to(torch.float8_e4m3fn)
                sw_4d = swizzle_scale(sw, N, K).to(torch.float8_e4m3fn)

                if (
                    not (qparams_x and qparams_x.pow_2_scales)
                    and K > 0
                    and global_amax_qx is not None
                    and global_amax_qw is not None
                ):
                    factor = 6.0 * 6.0 * 448.0 * 448.0
                    partial_alpha = (global_amax_qx * global_amax_qw).to(torch.float32)
                    alpha = partial_alpha.squeeze(-1) / factor
                    alpha = alpha.item()
                else:
                    alpha = 1.0

                out_tensor = torch.empty((M, N), device=device, dtype=out_dtype)
                fp4_gemm.cutlass_gemm(
                    out_tensor,  # output
                    qx,  # A: packed NVFP4
                    qw,  # B: packed NVFP4
                    sx_4d,  # A scales
                    sw_4d,  # B scales
                    torch.tensor(alpha, device=device, dtype=torch.float32),
                )
                gemm_out = out_tensor

            elif custom_gemm_backend == "triton":
                gemm_out = qgemm_tl(
                    qx=qx,
                    qw=qw,
                    m_params=m_params,
                    out_dtype=out_dtype,
                    sx=sx,
                    sw=sw,
                    bias=bias,
                    out=out,
                    accumulate=accumulate,
                    pow_2_scales=qparams_x.pow_2_scales and qparams_w.pow_2_scales
                    if (qparams_x is not None and qparams_w is not None)
                    else False,
                    global_amax_qx=global_amax_qx,
                    global_amax_qw=global_amax_qw,
                    qparams_x=qparams_x,
                    qparams_w=qparams_w,
                )
            elif custom_gemm_backend == "cuteDSL":
                sx_4d = swizzle_scale(sx, M, K).to(torch.float8_e4m3fn)
                sw_4d = swizzle_scale(sw, N, K).to(torch.float8_e4m3fn)
                alpha = 1.0
                out_tensor = torch.empty((M, N), device=device, dtype=out_dtype)
                fp4_gemm.cuteDSL_gemm(
                    out_tensor,
                    qx,
                    qw,
                    sx_4d,
                    sw_4d,
                    torch.tensor(alpha, device=device, dtype=torch.float32),
                )
                gemm_out = out_tensor
            else:
                raise ValueError(f"Unsupported custom gemm backend: {custom_gemm_backend}")
        else:
            gemm_out = general_gemm(
                A=qw,
                B=qx,
                out_dtype=out_dtype,
                quantization_params=None,
                gelu=False,
                gelu_in=None,
                alpha=1.0,
                beta=None,
                accumulate=accumulate,
                layout=layout,
                out=out,
                bias=bias,
                use_split_accumulator=False,
                grad=grad,
                ub=None,
                ub_type=None,
                extra_output=None,
                bulk_overlap=False
            )[0]
        return gemm_out