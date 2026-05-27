import torch
import fp4_gemm
from typing import Optional, List, Tuple, Union

from miniTransformer.quantiztion import quantization
from miniTransformer.quantiztion.quantization import GroupQuantizeResult
from miniTransformer.utils import utils
from miniTransformer.ops.nvfp4_grouped_cutlass_gemm import (
    V2GemmInputs,
    grouped_cutlass_gemm_v2,
)

import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch import NVFP4Quantizer, MXFP8Quantizer
from transformer_engine.pytorch.cpp_extensions import split_quantize
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm, general_grouped_gemm
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor, QuantizedTensorStorage


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


class Nvfp4Cursor2QuantizeOp(quantization.QuantizeOpBase):
    """
  NVFP4 quantize op for GroupLinear (cursor2 path):

  - Activation: per-token NVFP4 (group_quantize_nvfp4_pertoken_swizzled)
  - Weight: tensor-global NVFP4 (group_quantize_nvfp4_tensor_global_packed)
  - Forward grouped GEMM: CUTLASS v2 grouped FP4 GEMM (SM100+)

  Use with ``disable_backward_quant=True`` in GroupLinear when activation has no
  columnwise TE storage. Weight backward with columnwise still uses TE when
  ``return_transpose=True``.
    """

    def __init__(self):
        super().__init__()
        self.should_update_quantize_weights = True
        # Cached quantizers — rebuilt only when num_gemms changes.
        self._cached_num_gemms: int = 0
        self._w_quantizers_fp8: List = []   # rowwise=False, columnwise=True
        self._x_quantizers: List = []       # rowwise=False, columnwise=True
        self._dy_quantizers: List = []      # rowwise=True,  columnwise=True

    def _refresh_quantizer_cache(self, num_gemms: int) -> None:
        """Rebuild MXFP8 quantizer lists when num_gemms changes."""
        if self._cached_num_gemms == num_gemms:
            return
        self._cached_num_gemms = num_gemms
        fp8_col = dict(rowwise=False, columnwise=True, fp8_dtype=tex.DType.kFloat8E4M3)
        fp8_both = dict(rowwise=True, columnwise=True, fp8_dtype=tex.DType.kFloat8E4M3)
        self._w_quantizers_fp8 = [MXFP8Quantizer(**fp8_col) for _ in range(num_gemms)]
        self._x_quantizers = [MXFP8Quantizer(**fp8_col) for _ in range(num_gemms)]
        self._dy_quantizers = [MXFP8Quantizer(**fp8_both) for _ in range(num_gemms)]

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
        Grouped quantize for ``GroupLinear`` forward (v2 CUTLASS GEMM).

        - Activation (X): per-token NVFP4 → ``extra_info`` with swizzled data/scales
        - Weight: tensor-global NVFP4 → ``extra_info`` with packed data/scales + n/k
        - Weight + ``return_transpose``: also produces TE FP8 columnwise tensors for dgrad
        """
        num_gemms = len(m_splits)
        self._refresh_quantizer_cache(num_gemms)

        if qparams.param_type == qparams.param_type.WEIGHT:
            from miniTransformer.ops.nvfp4_quantize import group_quantize_nvfp4_tensor_global_packed

            gw = group_quantize_nvfp4_tensor_global_packed(x, x.shape[1], skip_cute_data=True)
            device = gw.data.device
            n, k = x.shape[1], x.shape[2]

            if return_transpose:
                w_out_splits = [x[i].shape[0] for i in range(num_gemms)]
                quantize_tensor_list_transpose = tex.split_quantize(
                    x.reshape(-1, x.shape[-1]),
                    w_out_splits,
                    self._w_quantizers_fp8,
                )
            else:
                quantize_tensor_list_transpose = []

            return GroupQuantizeResult(
                data=[gw],
                data_t=quantize_tensor_list_transpose,
                scale=[gw],
                scale_t=quantize_tensor_list_transpose,
                tensor_scale=[gw],
                tensor_scale_t=quantize_tensor_list_transpose,
                extra_info={
                    "data":          gw.data,
                    "scale_flat":    gw.scale_flat,
                    "global_scale":  gw.global_scale,
                    "row_offsets":   torch.tensor(gw.row_offsets,   dtype=torch.int32, device=device),
                    "scale_offsets": torch.tensor(gw.scale_offsets, dtype=torch.int64, device=device),
                    "n": n,
                    "k": k,
                },
            )

        if qparams.param_type == qparams.param_type.X:
            from miniTransformer.ops.nvfp4_quantize import group_quantize_nvfp4_pertoken_swizzled

            gq = group_quantize_nvfp4_pertoken_swizzled(x, m_splits)
            device = gq.data.device

            quantize_tensor_list = tex.split_quantize(x, m_splits, self._x_quantizers)
            return GroupQuantizeResult(
                data=[gq],
                scale=[gq],
                data_t=quantize_tensor_list,
                scale_t=quantize_tensor_list,
                tensor_scale=[gq],
                tensor_scale_t=quantize_tensor_list,
                extra_info={
                    "data":          gq.data,
                    "scale_flat":    gq.scale_flat,
                    "global_scale":  gq.global_scale,
                    "row_offsets":   torch.tensor(gq.row_offsets,   dtype=torch.int32, device=device),
                    "scale_offsets": torch.tensor(gq.scale_offsets, dtype=torch.int64, device=device),
                },
            )

        # Dy: MXFP8 quant for backward
        quantize_tensor_list = tex.split_quantize(
            x,
            m_splits,
            self._dy_quantizers,
        )
        return GroupQuantizeResult(
            data=quantize_tensor_list,
            scale=quantize_tensor_list,
            data_t=quantize_tensor_list,
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

        a0 = A[0]
        is_te_quantized = isinstance(a0, (QuantizedTensor, QuantizedTensorStorage))

        if (
            _is_sm100()
            and not grad
            and not gelu
            and not accumulate
            and layout == "TN"
            and extra_info_A is not None
            and extra_info_B is not None
        ):
            ng = len(m_splits)
            prob_host = torch.zeros(ng * 3, dtype=torch.int32)
            for g in range(ng):
                prob_host[g * 3 + 0] = m_splits[g]
                prob_host[g * 3 + 1] = extra_info_B["n"]
                prob_host[g * 3 + 2] = extra_info_B["k"]
            inputs = V2GemmInputs(
                act_data=extra_info_A["data"],
                act_sf=extra_info_A["scale_flat"],
                act_ts=extra_info_A["global_scale"],
                a_rows=extra_info_A["row_offsets"],
                a_sf_offsets=extra_info_A["scale_offsets"],
                wgt_data=extra_info_B["data"],
                wgt_sf=extra_info_B["scale_flat"],
                wgt_ts=extra_info_B["global_scale"],
                b_rows=extra_info_B["row_offsets"],
                b_sf_offsets=extra_info_B["scale_offsets"],
                prob_host=prob_host,
                m_splits=m_splits,
            )
            return grouped_cutlass_gemm_v2(inputs, activation_dtype)

        if grad:
            if layout == "NN":
                # dgrad = dy @ W → in_features = last dim of _columnwise_data.
                # size(0) is out_features; size(-1) is in_features.
                n_cols = B[0]._columnwise_data.size(-1)
                out = torch.empty(
                    [sum(m_splits), n_cols],
                    dtype=activation_dtype,
                    device=B[0]._columnwise_data.device,
                )
                out = [out]
        else:
            if is_te_quantized:
                n_cols = B[0]._rowwise_data.size(0)
            else:
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
