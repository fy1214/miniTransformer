import math
import torch
import fp4_gemm
from typing import Optional, List, Tuple

from miniTransformer.quantiztion import quantization
from miniTransformer.quantiztion.quantization import GroupQuantizeResult
from miniTransformer.utils import utils

import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch import NVFP4Quantizer, MXFP8Quantizer
from transformer_engine.pytorch.cpp_extensions import split_quantize
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm, general_grouped_gemm
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor, QuantizedTensorStorage


nvfp4_output_t = torch.uint8
nvfp4_scale_t = torch.float8_e4m3fn

use_cpp_allocator = False

def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)


def _grouped_gemm_cutedsl(
    A: list,
    B: list,
    m_splits: list,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    # Kept for backward compatibility with existing imports/tests.
    # Actual implementation is hosted in nvfp4_quartet_op and gated there.
    from miniTransformer.quantiztion.nvfp4_quartet_op import grouped_gemm_cutedsl

    return grouped_gemm_cutedsl(A, B, m_splits, out_dtype)


class Nvfp4TiledQuantizeOp(quantization.QuantizeOpBase):
    def __init__(self):
        super().__init__()
        self.should_update_quantize_weights = True

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

        nvfp4_quantizer = NVFP4Quantizer(  # NVFP4Quantizer
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

        if qparams.param_type == qparams.param_type.X:
            return quantization.QuantizeResult(data=x, scale=x, data_t=nvfp4_tensor, scale_t=nvfp4_tensor)

        if qparams.param_type == qparams.param_type.WEIGHT:
            w_qdq = nvfp4_tensor.dequantize()
            return quantization.QuantizeResult(data=w_qdq, scale=w_qdq, data_t=nvfp4_tensor, scale_t=nvfp4_tensor)

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
        custom_gemm_backend: str = "cutlass",  # cutlass, triton, cuteDSL
    ) -> torch.Tensor:

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


    def group_quantize(
        self,
        x: torch.Tensor,
        qparams: quantization.QParams,
        m_splits: List[int] = None,
        return_transpose: bool = False,
        reduce_amax: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        input_meta: Optional[quantization.QuantizeInputMetaBase] = None,
    ) -> GroupQuantizeResult:
        """
        If param type is weight, the shape is [num_gemms, n, k].
        If param type is X, the shape is [m_sum, k], m_sum should split into m_splits.
        If param type is Y, the shape is [m_sum, n].
        """
        num_gemms = len(m_splits)

        if qparams.param_type == qparams.param_type.WEIGHT:
            # Weight always needs columnwise data for backward dgrad GEMM,
            # regardless of the return_transpose flag.
            # If columnwise=False, _columnwise_data is None, and general_grouped_gemm
            # cannot find the transposed weight needed for dgrad, triggering a swizzle
            # assertion (NVTE_NVFP4_1D_SCALING not supported in that path).
            quantize_tensor_list_forward = []
            quantize_tensor_list_backward = []
            for i in range(num_gemms):
                w_quantizer = NVFP4Quantizer(
                    rowwise=True,
                    columnwise=True,   # always True: backward dgrad needs weight transpose
                    with_amax_reduction=False,
                    amax_reduction_group=None,
                    with_rht=qparams.with_rht,
                    with_post_rht_amax=qparams.with_rht,
                    with_2d_quantization=qparams.with_2d_quantization,
                    stochastic_rounding=qparams.stochastic_rounding
                )
                w_q = w_quantizer.quantize(x[i])
                quantize_tensor_list_forward.append(w_q.dequantize())
                quantize_tensor_list_backward.append(w_q)

                return GroupQuantizeResult(
                    data=quantize_tensor_list_forward,
                    data_t=quantize_tensor_list_backward,
                    scale=quantize_tensor_list_forward,
                    scale_t=quantize_tensor_list_backward,
                    tensor_scale=quantize_tensor_list_forward,
                    tensor_scale_t=quantize_tensor_list_backward,
                )
        elif qparams.param_type == qparams.param_type.X:
            quantize_tensor_list_forward = torch.split(x, m_splits)
            quantize_tensor_list_backward = []
            if return_transpose:
                quantizers = []
                for i in range(num_gemms):
                    x_quantizer = NVFP4Quantizer(
                        rowwise=False,
                        columnwise=True,
                        with_amax_reduction=False,
                        amax_reduction_group=None,
                        with_rht=qparams.with_rht,
                        with_post_rht_amax=qparams.with_rht,
                        with_2d_quantization=qparams.with_2d_quantization,
                        stochastic_rounding=qparams.stochastic_rounding
                    )
                    quantizers.append(x_quantizer)
                quantize_tensor_list_backward = tex.split_quantize(
                    x,
                    m_splits,
                    quantizers,
                )

            return GroupQuantizeResult(
                data=quantize_tensor_list_forward,
                data_t=quantize_tensor_list_backward,
                scale=quantize_tensor_list_forward,
                scale_t=quantize_tensor_list_backward,
                tensor_scale=quantize_tensor_list_forward,
                tensor_scale_t=quantize_tensor_list_backward,
            )
        else:
            # GRAD path: columnwise controlled by return_transpose
            quantizers = []
            for i in range(num_gemms):
                x_quantizer = NVFP4Quantizer(
                    rowwise=True,
                    columnwise=return_transpose,
                    with_amax_reduction=False,
                    amax_reduction_group=None,
                    with_rht=qparams.with_rht,
                    with_post_rht_amax=qparams.with_rht,
                    with_2d_quantization=qparams.with_2d_quantization,
                    stochastic_rounding=qparams.stochastic_rounding
                )
                quantizers.append(x_quantizer)
            quantize_tensor_list = tex.split_quantize(
                x,
                m_splits,
                quantizers,
            )

        return GroupQuantizeResult(
            data=quantize_tensor_list,
            data_t=quantize_tensor_list,
            scale=quantize_tensor_list,
            scale_t=quantize_tensor_list,
            tensor_scale=quantize_tensor_list,
            tensor_scale_t=quantize_tensor_list,
        )


    def grouped_gemm(
        self,
        A: List[torch.Tensor],
        B: List[torch.Tensor],
        sA: List[torch.Tensor],
        sB: List[torch.Tensor],
        global_amax_A: List[torch.Tensor],
        global_amax_B: List[torch.Tensor],
        out: List[torch.Tensor],
        out_dtype: torch.dtype,
        layout: str = "TN",
        m_splits: List[int] = None,
        gelu: bool = False,
        grad=False,
        accumulate: bool = False,
        bias: Optional[List[torch.Tensor]] = None,
        use_bias: bool = False,
        qparams_x: Optional[quantization.QParams] = None,
        qparams_w: Optional[quantization.QParams] = None,
        extra_info_A: Optional[dict] = None,
        extra_info_B: Optional[dict] = None,
    ) -> Tuple[List[torch.Tensor], ...]:
        activation_dtype = out_dtype

        # Determine whether input is a quantized tensor or BF16 plain tensor
        a0 = A[0]
        is_quantized = isinstance(a0, (QuantizedTensor, QuantizedTensorStorage))

        # Optional env-gated bridge for grouped cuteDSL path.
        # Default remains TE grouped_gemm for compatibility on environments
        # without newer cuteDSL.
        from miniTransformer.quantiztion.nvfp4_quartet_op import is_grouped_gemm_cutedsl_enabled

        if (
            is_grouped_gemm_cutedsl_enabled()
            and is_quantized
            and not grad
            and not gelu
            and not accumulate
            and layout == "TN"
            and _is_sm100()
        ):
            try:
                return _grouped_gemm_cutedsl(A, B, m_splits, activation_dtype)
            except (TypeError, NotImplementedError):
                pass

        if grad:
            if layout == "NN":
                if is_quantized:
                    n_cols = B[0]._rowwise_data.size(1) * 2
                else:
                    # BF16: B[i] shape [n, k], dgrad shape [m_sum, k]
                    n_cols = B[0].size(1)
                out = torch.empty(
                    [sum(m_splits), n_cols],
                    dtype=activation_dtype,
                    device=B[0].device,
                )
                out = [out]
        else:
            if is_quantized:
                n_cols = B[0]._rowwise_data.size(0)
            else:
                # BF16 fprop: B[i] shape [n, k], output shape [m_sum, n]
                n_cols = B[0].size(0)
            out = torch.empty(
                [sum(m_splits), n_cols],
                dtype=activation_dtype,
                device=B[0].device,
            )
            out = [out]

        general_grouped_gemm(
            A=B,
            B=A,
            out=out,
            out_dtype=activation_dtype,
            layout=layout,
            m_splits=m_splits,
            single_output=False if (grad and layout == "NT") else True,
            grad=grad,
            bias=bias,
            use_bias=use_bias,
            use_split_accumulator=False,
            accumulate=accumulate
        )

        if grad and layout == "NT":
            return out
        else:
            return out[0]
