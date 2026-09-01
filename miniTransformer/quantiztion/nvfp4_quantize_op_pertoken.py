"""NVFP4 per-token QuantizeOp (TE PR #3045).

Forward:
  - ``tex.nvfp4_per_token_group_quantize_bulk`` (rowwise act / weight)
  - ``tex.nvfp4_cutlass_grouped_per_token_gemm`` (TN, plain tensor lists)

Backward:
  - MXFP8 ``split_quantize`` + TE ``general_grouped_gemm``
"""

from __future__ import annotations

import inspect
from typing import List, Optional, Tuple, Union

import torch
from torch.profiler import record_function

from miniTransformer.ops._te_compat import general_gemm, general_grouped_gemm
from miniTransformer.quantiztion import quantization
from miniTransformer.quantiztion.quantization import GroupQuantizeResult
from miniTransformer.utils import utils

import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch import MXFP8Quantizer, NVFP4Quantizer
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor, QuantizedTensorStorage

_NVFP4_QUANTIZER_PARAMS = inspect.signature(NVFP4Quantizer.__init__).parameters
if "per_token" not in _NVFP4_QUANTIZER_PARAMS:
    raise RuntimeError(
        "Nvfp4TiledQuantizePertokenOp requires Transformer Engine PR #3045 "
        "(NVFP4Quantizer(per_token=True))."
    )
_HAS_PER_TOKEN_WEIGHT_2D = "per_token_weight_2d" in _NVFP4_QUANTIZER_PARAMS
_HAS_BULK = hasattr(tex, "nvfp4_per_token_group_quantize_bulk") and hasattr(
    tex, "nvfp4_cutlass_grouped_per_token_gemm"
)


def _make_per_token_quantizer(
    *,
    rowwise: bool,
    columnwise: bool,
    is_weight: bool = False,
) -> NVFP4Quantizer:
    kwargs = dict(
        rowwise=rowwise,
        columnwise=columnwise,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
        per_token=True,
    )
    if _HAS_PER_TOKEN_WEIGHT_2D:
        kwargs["per_token_weight_2d"] = bool(is_weight)
    elif is_weight:
        raise RuntimeError(
            "NVFP4Quantizer is missing per_token_weight_2d; need a newer TE #3045 build."
        )
    q = NVFP4Quantizer(**kwargs)
    q.optimize_for_gemm = True
    return q


def _bulk_rowwise(
    x: torch.Tensor,
    m_splits: List[int],
    *,
    row_amax: Optional[torch.Tensor] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    if not _HAS_BULK:
        raise RuntimeError(
            "Nvfp4TiledQuantizePertokenOp forward requires "
            "tex.nvfp4_per_token_group_quantize_bulk and "
            "tex.nvfp4_cutlass_grouped_per_token_gemm."
        )
    q_row, s_row, amax_row, _, _, _ = tex.nvfp4_per_token_group_quantize_bulk(
        x.contiguous(),
        [int(m) for m in m_splits],
        True,
        False,
        with_swizzle=True,
        do_amax=row_amax is None,
        row_amax=row_amax,
    )
    return q_row, s_row, amax_row


def _is_plain_tensor_list(tensors: List[torch.Tensor]) -> bool:
    if not tensors:
        return False
    t0 = tensors[0]
    return isinstance(t0, torch.Tensor) and not isinstance(
        t0, (QuantizedTensor, QuantizedTensorStorage)
    )


def _out_n_from_weight(wgt_data: List[torch.Tensor], m_splits: List[int]) -> int:
    for g, m in enumerate(m_splits):
        if m > 0:
            return wgt_data[g].shape[0]
    return wgt_data[0].shape[0]


def _direct_grouped_gemm_tn(
    act_data: List[torch.Tensor],
    wgt_data: List[torch.Tensor],
    act_sf: List[torch.Tensor],
    wgt_sf: List[torch.Tensor],
    act_alpha: List[torch.Tensor],
    wgt_alpha: List[torch.Tensor],
    m_splits: List[int],
    out: torch.Tensor,
) -> torch.Tensor:
    views: List[torch.Tensor] = []
    start = 0
    for size in m_splits:
        views.append(out[start : start + size])
        start += size

    a_data, b_data, a_sf, b_sf = [], [], [], []
    alpha_a, alpha_b, d = [], [], []
    for g, m in enumerate(m_splits):
        if m == 0:
            if views[g].numel() > 0:
                views[g].zero_()
            continue
        ka, kb = act_data[g], wgt_data[g]
        m_rows, n_cols = ka.shape[0], kb.shape[0]
        a_data.append(ka)
        b_data.append(kb)
        a_sf.append(act_sf[g])
        b_sf.append(wgt_sf[g])
        alpha_a.append(act_alpha[g])
        alpha_b.append(wgt_alpha[g])
        d.append(views[g].view(m_rows, n_cols))

    if a_data:
        tex.nvfp4_cutlass_grouped_per_token_gemm(
            a_data,
            b_data,
            a_sf,
            b_sf,
            alpha_a,
            alpha_b,
            d,
            True,
            True,
            False,
            [],
            "default",
        )
    return out


def _te_grouped_gemm(
    A: List[torch.Tensor],
    B: List[torch.Tensor],
    out,
    *,
    layout: str,
    m_splits: List[int],
    out_dtype: torch.dtype,
    grad: bool,
    accumulate: bool,
    bias,
    use_bias: bool,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    with record_function(f"gg_te_grouped_gemm_{layout}_grad{grad}"):
        general_grouped_gemm(
            A=B,
            B=A,
            out=out,
            out_dtype=out_dtype,
            layout=layout,
            m_splits=m_splits,
            single_output=False if (grad and layout == "NT") else True,
            grad=grad,
            bias=bias,
            use_bias=use_bias,
            use_split_accumulator=False,
            accumulate=accumulate,
        )
    if grad and layout == "NT":
        return out
    return out[0]


class Nvfp4TiledQuantizePertokenOp(quantization.QuantizeOpBase):
    def __init__(self):
        super().__init__()
        self.should_update_quantize_weights = True
        self._cached_num_gemms: int = 0
        self._w_quantizers_fp8: List = []
        self._x_quantizers_fp8: List = []
        self._dy_quantizers: List = []

    def _refresh_cache(self, num_gemms: int) -> None:
        if self._cached_num_gemms == num_gemms and len(self._dy_quantizers) == num_gemms:
            return
        self._cached_num_gemms = num_gemms
        fp8_col = dict(rowwise=False, columnwise=True, fp8_dtype=tex.DType.kFloat8E4M3)
        fp8_both = dict(rowwise=True, columnwise=True, fp8_dtype=tex.DType.kFloat8E4M3)
        self._w_quantizers_fp8 = [MXFP8Quantizer(**fp8_col) for _ in range(num_gemms)]
        self._x_quantizers_fp8 = [MXFP8Quantizer(**fp8_col) for _ in range(num_gemms)]
        self._dy_quantizers = [MXFP8Quantizer(**fp8_both) for _ in range(num_gemms)]

    @property
    def supports_allgather(self) -> bool:
        return False

    def quantize(
        self,
        x: torch.Tensor,
        qparams: quantization.QParams,
        return_transpose: bool = False,
        reduce_amax: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        input_meta: Optional[quantization.QuantizeInputMetaBase] = None,
    ) -> quantization.QuantizeResult:
        del reduce_amax, tp_group, input_meta
        assert x.dtype in utils.HIGH_PRECISION_FLOAT_DTYPES
        is_weight = qparams.param_type == qparams.param_type.WEIGHT
        q = _make_per_token_quantizer(
            rowwise=True,
            columnwise=True if is_weight else return_transpose,
            is_weight=is_weight,
        )
        t = q.make_empty(x.shape, dtype=x.dtype, device=x.device, requires_grad=False)
        t = q.update_quantized(x, t)
        return quantization.QuantizeResult(data=t, scale=t, data_t=t, scale_t=t)

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
        custom_gemm_backend: str = "cutlass",
    ) -> torch.Tensor:
        del m_params, sx, sw, global_amax_qx, global_amax_qw, qparams_x, qparams_w, custom_gemm_backend
        return general_gemm(
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
            bulk_overlap=False,
        )[0]

    def group_quantize(
        self,
        x: torch.Tensor,
        qparams: quantization.QParams,
        m_splits: List[int] = None,
        return_transpose: bool = False,
        reduce_amax: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        input_meta: Optional[quantization.QuantizeInputMetaBase] = None,
        row_amax: Optional[torch.Tensor] = None,
    ) -> GroupQuantizeResult:
        del reduce_amax, tp_group, input_meta
        num_gemms = len(m_splits)
        self._refresh_cache(num_gemms)

        if qparams.param_type == qparams.param_type.Y:
            with record_function("sq_dy_split_quantize_fp8"):
                mx = tex.split_quantize(x, m_splits, self._dy_quantizers)
            return GroupQuantizeResult(
                data=mx, scale=mx, data_t=mx, scale_t=mx,
                tensor_scale=mx, tensor_scale_t=mx,
            )

        if qparams.param_type == qparams.param_type.WEIGHT:
            n, k = x.shape[1], x.shape[2]
            w_splits = [n] * num_gemms
            with record_function("sq_w_nvfp4_bulk"):
                data, scale, tensor_scale = _bulk_rowwise(
                    x.reshape(-1, k), w_splits,
                )
            if return_transpose:
                with record_function("sq_w_split_quantize_fp8"):
                    data_t = tex.split_quantize(
                        x.reshape(-1, k), w_splits, self._w_quantizers_fp8,
                    )
            else:
                data_t = []
            return GroupQuantizeResult(
                data=data,
                scale=scale,
                tensor_scale=tensor_scale,
                data_t=data_t,
                scale_t=data_t,
                tensor_scale_t=data_t,
                extra_info={"n": n, "k": k},
            )

        skip_k1 = row_amax if row_amax is not None else None
        with record_function("sq_x_nvfp4_bulk"):
            data, scale, tensor_scale = _bulk_rowwise(x, m_splits, row_amax=skip_k1)
        if return_transpose:
            with record_function("sq_x_split_quantize_fp8"):
                data_t = tex.split_quantize(x, m_splits, self._x_quantizers_fp8)
        else:
            data_t = []
        return GroupQuantizeResult(
            data=data,
            scale=scale,
            tensor_scale=tensor_scale,
            data_t=data_t,
            scale_t=data_t,
            tensor_scale_t=data_t,
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
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        del extra_info_A, extra_info_B, qparams_x, qparams_w

        if (
            not grad
            and layout == "TN"
            and not gelu
            and not accumulate
            and not use_bias
            and _is_plain_tensor_list(A)
            and _is_plain_tensor_list(B)
        ):
            if out is None or (isinstance(out, list) and len(out) == 0):
                n_cols = _out_n_from_weight(B, m_splits)
                out_t = torch.empty(
                    [sum(m_splits), n_cols], dtype=out_dtype, device=B[0].device,
                )
            elif isinstance(out, list):
                out_t = out[0]
            else:
                out_t = out
            with record_function("grouped_gemm_nvfp4_direct_tn"):
                return _direct_grouped_gemm_tn(
                    A, B, sA, sB, global_amax_A, global_amax_B, m_splits, out_t,
                )

        a0 = A[0]
        is_quantized = isinstance(a0, (QuantizedTensor, QuantizedTensorStorage))

        if not grad:
            if is_quantized:
                n_cols = B[0]._rowwise_data.size(0)
                device = B[0]._rowwise_data.device
            else:
                n_cols = B[0].size(0)
                device = B[0].device
            out = [torch.empty([sum(m_splits), n_cols], dtype=out_dtype, device=device)]
            return _te_grouped_gemm(
                A, B, out,
                layout=layout, m_splits=m_splits, out_dtype=out_dtype,
                grad=False, accumulate=accumulate, bias=bias, use_bias=use_bias,
            )

        del sA, sB, global_amax_A, global_amax_B
        if layout == "NN":
            if is_quantized:
                n_cols = B[0]._columnwise_data.size(-1)
                device = B[0]._columnwise_data.device
            else:
                n_cols = B[0].size(1)
                device = B[0].device
            out = [torch.empty([sum(m_splits), n_cols], dtype=out_dtype, device=device)]
        elif layout == "NT" and (out is None or (isinstance(out, list) and out[0] is None)):
            if is_quantized:
                n_out = A[0]._columnwise_data.size(-1)
                k_in = B[0]._columnwise_data.size(-1)
                device = A[0]._columnwise_data.device
            else:
                n_out = A[0].size(1)
                k_in = B[0].size(1)
                device = A[0].device
            out = [
                torch.empty([n_out, k_in], dtype=out_dtype, device=device)
                for _ in range(len(m_splits))
            ]

        return _te_grouped_gemm(
            A, B, out,
            layout=layout, m_splits=m_splits, out_dtype=out_dtype,
            grad=True, accumulate=accumulate, bias=bias, use_bias=use_bias,
        )
