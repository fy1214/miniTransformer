"""NVFP4 per-token QuantizeOp (TE PR #3045).

Forward:
  - TE ``nvfp4_per_token_group_quantize_bulk[_dense]`` (rowwise act / weight)
  - TE ``nvfp4_cutlass_grouped_per_token_gemm[_dense]`` (TN)

Backward:
  - MXFP8 ``split_quantize`` + TE ``general_grouped_gemm``
"""

from __future__ import annotations

import inspect
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.profiler import record_function

from miniTransformer.ops._te_compat import general_gemm, general_grouped_gemm
from miniTransformer.quantiztion import quantization
from miniTransformer.quantiztion.quantization import GroupQuantizeResult
from miniTransformer.utils import utils

import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch import MXFP8Quantizer, NVFP4Quantizer
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor, QuantizedTensorStorage

_NVFP4_QUANTIZER_INIT_PARAMS = inspect.signature(NVFP4Quantizer.__init__).parameters
if "per_token" not in _NVFP4_QUANTIZER_INIT_PARAMS:
    raise RuntimeError(
        "Nvfp4TiledQuantizePertokenOp requires Transformer Engine PR #3045 "
        "(NVFP4Quantizer(per_token=True))."
    )

_SUPPORTS_PER_TOKEN_WEIGHT_2D = "per_token_weight_2d" in _NVFP4_QUANTIZER_INIT_PARAMS
_HAS_LIST_QUANT_AND_GEMM = hasattr(tex, "nvfp4_per_token_group_quantize_bulk") and hasattr(
    tex, "nvfp4_cutlass_grouped_per_token_gemm"
)
_HAS_DENSE_QUANT = hasattr(tex, "nvfp4_per_token_group_quantize_bulk_dense")
_HAS_DENSE_GEMM = hasattr(tex, "nvfp4_cutlass_grouped_per_token_gemm_dense")


def is_dense_nvfp4_path_enabled() -> bool:
    """Whether to use dense (concatenated) TE packaging APIs.

    Controlled by ``MINITE_NVFP4_PATH``:
      - ``auto`` (default): dense when TE exposes ``*_dense`` bindings
      - ``dense``: require dense APIs (raises if missing)
      - ``list`` / ``legacy``: force per-expert list packaging
    """
    mode = os.environ.get("MINITE_NVFP4_PATH", "auto").strip().lower()
    if mode in ("list", "legacy", "0", "false", "off"):
        return False
    if mode in ("dense", "1", "true", "on"):
        if not (_HAS_DENSE_QUANT and _HAS_DENSE_GEMM):
            raise RuntimeError(
                "MINITE_NVFP4_PATH=dense but TE dense APIs are missing "
                "(nvfp4_per_token_group_quantize_bulk_dense / "
                "nvfp4_cutlass_grouped_per_token_gemm_dense)."
            )
        return True
    return bool(_HAS_DENSE_QUANT and _HAS_DENSE_GEMM)


def make_nvfp4_per_token_quantizer(
    *,
    rowwise: bool,
    columnwise: bool,
    is_weight: bool = False,
) -> NVFP4Quantizer:
    kwargs: Dict[str, Any] = dict(
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
    if _SUPPORTS_PER_TOKEN_WEIGHT_2D:
        kwargs["per_token_weight_2d"] = bool(is_weight)
    elif is_weight:
        raise RuntimeError(
            "NVFP4Quantizer is missing per_token_weight_2d; need a newer TE #3045 build."
        )
    quantizer = NVFP4Quantizer(**kwargs)
    quantizer.optimize_for_gemm = True
    return quantizer


def quantize_nvfp4_rowwise_per_group_list(
    x: torch.Tensor,
    m_splits: List[int],
    *,
    precomputed_row_amax: Optional[torch.Tensor] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Per-expert list packaging via ``nvfp4_per_token_group_quantize_bulk``."""
    if not _HAS_LIST_QUANT_AND_GEMM:
        raise RuntimeError(
            "Nvfp4TiledQuantizePertokenOp forward requires "
            "tex.nvfp4_per_token_group_quantize_bulk and "
            "tex.nvfp4_cutlass_grouped_per_token_gemm."
        )
    quantized_rows, scale_factors, row_amaxes, _, _, _ = tex.nvfp4_per_token_group_quantize_bulk(
        x.contiguous(),
        [int(m) for m in m_splits],
        True,  # rowwise
        False,  # columnwise
        with_swizzle=True,
        do_amax=precomputed_row_amax is None,
        row_amax=precomputed_row_amax,
    )
    return quantized_rows, scale_factors, row_amaxes


def quantize_nvfp4_rowwise_per_group_dense(
    x: torch.Tensor,
    m_splits: List[int],
    *,
    precomputed_row_amax: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense concatenated packaging via ``nvfp4_per_token_group_quantize_bulk_dense``."""
    if not _HAS_DENSE_QUANT:
        raise RuntimeError("tex.nvfp4_per_token_group_quantize_bulk_dense is required")
    quantized_rows, scale_factors, row_amaxes, _, _, _ = (
        tex.nvfp4_per_token_group_quantize_bulk_dense(
            x.contiguous(),
            [int(m) for m in m_splits],
            True,
            False,
            with_swizzle=True,
            do_amax=precomputed_row_amax is None,
            row_amax=precomputed_row_amax,
        )
    )
    return quantized_rows, scale_factors, row_amaxes


def pack_nvfp4_rowwise_group_result(
    x: torch.Tensor,
    m_splits: List[int],
    *,
    precomputed_row_amax: Optional[torch.Tensor] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], Optional[Dict[str, Any]]]:
    """Shared dense/list packaging used by activation and weight group_quantize."""
    if is_dense_nvfp4_path_enabled():
        packed_data, packed_scale, packed_tensor_scale = quantize_nvfp4_rowwise_per_group_dense(
            x, m_splits, precomputed_row_amax=precomputed_row_amax
        )
        data = [packed_data]
        scale = [packed_scale]
        tensor_scale = [packed_tensor_scale]
        packed_extra: Dict[str, Any] = {"dense_rowwise": True, "m_splits": list(m_splits)}
        if extra:
            packed_extra.update(extra)
        return data, scale, tensor_scale, packed_extra

    data, scale, tensor_scale = quantize_nvfp4_rowwise_per_group_list(
        x, m_splits, precomputed_row_amax=precomputed_row_amax
    )
    return data, scale, tensor_scale, extra


def _is_plain_torch_tensor_list(tensors: List[torch.Tensor]) -> bool:
    if not tensors:
        return False
    first = tensors[0]
    return isinstance(first, torch.Tensor) and not isinstance(
        first, (QuantizedTensor, QuantizedTensorStorage)
    )


def _output_n_from_weight_list(
    weight_data: List[torch.Tensor], m_splits: List[int]
) -> int:
    for expert_idx, num_tokens in enumerate(m_splits):
        if num_tokens > 0:
            return weight_data[expert_idx].shape[0]
    return weight_data[0].shape[0]


def _allocate_contiguous_output(
    out: Union[torch.Tensor, List[torch.Tensor], None],
    *,
    total_tokens: int,
    n_cols: int,
    out_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if out is None or (isinstance(out, list) and len(out) == 0):
        return torch.empty([total_tokens, n_cols], dtype=out_dtype, device=device)
    if isinstance(out, list):
        return out[0]
    return out


def resolve_nvfp4_grouped_gemm_kind(name: Optional[str]) -> str:
    """Map MoE layer name / ub_name to TE NVFP4 grouped tile family.

    TE selects fc1 (2SM N256) vs fc2 (2SM N128) tiles only when
    ``gemm_kind`` is explicitly ``fc1`` / ``fc2``; it is not inferred from K.
    """
    kind = str(name or "default").strip().lower()
    if kind in ("fc1", "fc2", "default"):
        return kind
    # Megatron often passes tp_comm_buffer_name / ub_name like "fc1".
    if kind.endswith("fc1") or kind == "linear_fc1":
        return "fc1"
    if kind.endswith("fc2") or kind == "linear_fc2":
        return "fc2"
    return "default"


def nvfp4_grouped_gemm_tn_list(
    activation_data: List[torch.Tensor],
    weight_data: List[torch.Tensor],
    activation_scale: List[torch.Tensor],
    weight_scale: List[torch.Tensor],
    activation_tensor_scale: List[torch.Tensor],
    weight_tensor_scale: List[torch.Tensor],
    m_splits: List[int],
    out: torch.Tensor,
    gemm_kind: str = "default",
) -> torch.Tensor:
    """TN grouped GEMM with per-expert list tensors."""
    gemm_kind = resolve_nvfp4_grouped_gemm_kind(gemm_kind)
    output_views: List[torch.Tensor] = []
    row_offset = 0
    for num_tokens in m_splits:
        output_views.append(out[row_offset : row_offset + num_tokens])
        row_offset += num_tokens

    a_data, b_data, a_scale, b_scale = [], [], [], []
    a_tensor_scale, b_tensor_scale, d_views = [], [], []
    for expert_idx, num_tokens in enumerate(m_splits):
        if num_tokens == 0:
            if output_views[expert_idx].numel() > 0:
                output_views[expert_idx].zero_()
            continue
        act = activation_data[expert_idx]
        wgt = weight_data[expert_idx]
        m_rows, n_cols = act.shape[0], wgt.shape[0]
        a_data.append(act)
        b_data.append(wgt)
        a_scale.append(activation_scale[expert_idx])
        b_scale.append(weight_scale[expert_idx])
        a_tensor_scale.append(activation_tensor_scale[expert_idx])
        b_tensor_scale.append(weight_tensor_scale[expert_idx])
        d_views.append(output_views[expert_idx].view(m_rows, n_cols))

    if a_data:
        tex.nvfp4_cutlass_grouped_per_token_gemm(
            a_data,
            b_data,
            a_scale,
            b_scale,
            a_tensor_scale,
            b_tensor_scale,
            d_views,
            True,
            True,
            False,
            [],
            gemm_kind,
        )
    return out


def nvfp4_grouped_gemm_tn_dense(
    activation_data: torch.Tensor,
    weight_data: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    activation_tensor_scale: torch.Tensor,
    weight_tensor_scale: torch.Tensor,
    m_splits: List[int],
    n_cols: int,
    out: torch.Tensor,
    gemm_kind: str = "default",
) -> torch.Tensor:
    """TN grouped GEMM with dense concatenated tensors.

    Prefix offset tables are omitted so TE fills them on device from
    ``m_splits`` (uniform N from packed B/D, ``k_sf=K/16``). Empty experts
    (``m_splits[i] == 0``) are allowed; TE skips those groups while keeping
    the ``G*N`` weight pack intact.
    """
    gemm_kind = resolve_nvfp4_grouped_gemm_kind(gemm_kind)
    if sum(int(m) for m in m_splits) == 0:
        out.zero_()
        return out
    if int(out.size(1)) != int(n_cols):
        raise RuntimeError(
            f"dense grouped GEMM out N={out.size(1)} != n_cols={n_cols}"
        )

    tex.nvfp4_cutlass_grouped_per_token_gemm_dense(
        activation_data,
        weight_data,
        activation_scale.reshape(-1) if activation_scale.dim() > 1 else activation_scale,
        weight_scale.reshape(-1) if weight_scale.dim() > 1 else weight_scale,
        activation_tensor_scale,
        weight_tensor_scale,
        out,
        None,
        None,
        None,
        None,
        [int(m) for m in m_splits],
        False,
        gemm_kind,
    )
    return out


def te_general_grouped_gemm(
    lhs: List[torch.Tensor],
    rhs: List[torch.Tensor],
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
    """Call TE ``general_grouped_gemm`` with miniTE (lhs,rhs) → TE (A=rhs, B=lhs) swap."""
    with record_function(f"gg_te_grouped_gemm_{layout}_grad{grad}"):
        general_grouped_gemm(
            A=rhs,
            B=lhs,
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
        self._cached_num_experts: int = 0
        self._weight_mxfp8_columnwise_quantizers: List[MXFP8Quantizer] = []
        self._activation_mxfp8_columnwise_quantizers: List[MXFP8Quantizer] = []
        self._grad_output_mxfp8_quantizers: List[MXFP8Quantizer] = []

    def _ensure_mxfp8_backward_quantizers(self, num_experts: int) -> None:
        if (
            self._cached_num_experts == num_experts
            and len(self._grad_output_mxfp8_quantizers) == num_experts
        ):
            return
        self._cached_num_experts = num_experts
        columnwise_only = dict(
            rowwise=False, columnwise=True, fp8_dtype=tex.DType.kFloat8E4M3
        )
        rowwise_and_columnwise = dict(
            rowwise=True, columnwise=True, fp8_dtype=tex.DType.kFloat8E4M3
        )
        self._weight_mxfp8_columnwise_quantizers = [
            MXFP8Quantizer(**columnwise_only) for _ in range(num_experts)
        ]
        self._activation_mxfp8_columnwise_quantizers = [
            MXFP8Quantizer(**columnwise_only) for _ in range(num_experts)
        ]
        self._grad_output_mxfp8_quantizers = [
            MXFP8Quantizer(**rowwise_and_columnwise) for _ in range(num_experts)
        ]

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
        quantizer = make_nvfp4_per_token_quantizer(
            rowwise=True,
            columnwise=True if is_weight else return_transpose,
            is_weight=is_weight,
        )
        quantized = quantizer.make_empty(
            x.shape, dtype=x.dtype, device=x.device, requires_grad=False
        )
        quantized = quantizer.update_quantized(x, quantized)
        return quantization.QuantizeResult(
            data=quantized, scale=quantized, data_t=quantized, scale_t=quantized
        )

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
        del (
            m_params,
            sx,
            sw,
            global_amax_qx,
            global_amax_qw,
            qparams_x,
            qparams_w,
            custom_gemm_backend,
        )
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
        assert m_splits is not None
        num_experts = len(m_splits)
        self._ensure_mxfp8_backward_quantizers(num_experts)

        # Grad output (dy): MXFP8 for dgrad/wgrad.
        if qparams.param_type == qparams.param_type.Y:
            with record_function("sq_dy_split_quantize_fp8"):
                mxfp8_dy = tex.split_quantize(x, m_splits, self._grad_output_mxfp8_quantizers)
            return GroupQuantizeResult(
                data=mxfp8_dy,
                scale=mxfp8_dy,
                data_t=mxfp8_dy,
                scale_t=mxfp8_dy,
                tensor_scale=mxfp8_dy,
                tensor_scale_t=mxfp8_dy,
            )

        # Weight: NVFP4 rowwise for forward; MXFP8 columnwise for backward.
        if qparams.param_type == qparams.param_type.WEIGHT:
            out_features, in_features = x.shape[1], x.shape[2]
            weight_row_splits = [out_features] * num_experts
            weight_2d = x.reshape(-1, in_features)
            with record_function(
                "sq_w_nvfp4_bulk_dense"
                if is_dense_nvfp4_path_enabled()
                else "sq_w_nvfp4_bulk"
            ):
                data, scale, tensor_scale, extra = pack_nvfp4_rowwise_group_result(
                    weight_2d,
                    weight_row_splits,
                    extra={"n": out_features, "k": in_features},
                )
            if return_transpose:
                with record_function("sq_w_split_quantize_fp8"):
                    data_t = tex.split_quantize(
                        weight_2d,
                        weight_row_splits,
                        self._weight_mxfp8_columnwise_quantizers,
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
                extra_info=extra,
            )

        # Activation X: NVFP4 rowwise for forward; MXFP8 columnwise for backward.
        # ``row_amax`` skips the K1 amax pass when upstream already fused it.
        with record_function(
            "sq_x_nvfp4_bulk_dense" if is_dense_nvfp4_path_enabled() else "sq_x_nvfp4_bulk"
        ):
            data, scale, tensor_scale, extra = pack_nvfp4_rowwise_group_result(
                x, m_splits, precomputed_row_amax=row_amax
            )
        if return_transpose:
            with record_function("sq_x_split_quantize_fp8"):
                data_t = tex.split_quantize(
                    x, m_splits, self._activation_mxfp8_columnwise_quantizers
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
            extra_info=extra,
        )

    def _can_use_dense_nvfp4_forward_gemm(
        self,
        *,
        grad: bool,
        layout: str,
        gelu: bool,
        accumulate: bool,
        use_bias: bool,
        extra_info_A: Optional[dict],
        extra_info_B: Optional[dict],
    ) -> bool:
        return (
            not grad
            and layout == "TN"
            and not gelu
            and not accumulate
            and not use_bias
            and _HAS_DENSE_GEMM
            and isinstance(extra_info_A, dict)
            and isinstance(extra_info_B, dict)
            and bool(extra_info_A.get("dense_rowwise"))
            and bool(extra_info_B.get("dense_rowwise"))
        )

    def _can_use_list_nvfp4_forward_gemm(
        self,
        *,
        grad: bool,
        layout: str,
        gelu: bool,
        accumulate: bool,
        use_bias: bool,
        activation_data: List[torch.Tensor],
        weight_data: List[torch.Tensor],
        extra_info_A: Optional[dict],
    ) -> bool:
        return (
            not grad
            and layout == "TN"
            and not gelu
            and not accumulate
            and not use_bias
            and _is_plain_torch_tensor_list(activation_data)
            and _is_plain_torch_tensor_list(weight_data)
            and not (isinstance(extra_info_A, dict) and extra_info_A.get("dense_rowwise"))
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
        del qparams_x, qparams_w
        assert m_splits is not None
        # Module passes TE tile family via extra_info_A["gemm_kind"] (fc1/fc2/default).
        gemm_kind = "default"
        if isinstance(extra_info_A, dict) and extra_info_A.get("gemm_kind") is not None:
            gemm_kind = resolve_nvfp4_grouped_gemm_kind(extra_info_A.get("gemm_kind"))

        # Forward TN: dense concatenated NVFP4 path.
        if self._can_use_dense_nvfp4_forward_gemm(
            grad=grad,
            layout=layout,
            gelu=gelu,
            accumulate=accumulate,
            use_bias=use_bias,
            extra_info_A=extra_info_A,
            extra_info_B=extra_info_B,
        ):
            n_cols = int(
                extra_info_B.get("n", B[0].shape[0] // max(len(m_splits), 1))
            )
            out_tensor = _allocate_contiguous_output(
                out,
                total_tokens=sum(m_splits),
                n_cols=n_cols,
                out_dtype=out_dtype,
                device=B[0].device,
            )
            with record_function("grouped_gemm_nvfp4_direct_tn_dense"):
                return nvfp4_grouped_gemm_tn_dense(
                    A[0],
                    B[0],
                    sA[0],
                    sB[0],
                    global_amax_A[0],
                    global_amax_B[0],
                    m_splits,
                    n_cols,
                    out_tensor,
                    gemm_kind=gemm_kind,
                )

        # Forward TN: per-expert list NVFP4 path.
        if self._can_use_list_nvfp4_forward_gemm(
            grad=grad,
            layout=layout,
            gelu=gelu,
            accumulate=accumulate,
            use_bias=use_bias,
            activation_data=A,
            weight_data=B,
            extra_info_A=extra_info_A,
        ):
            n_cols = _output_n_from_weight_list(B, m_splits)
            out_tensor = _allocate_contiguous_output(
                out,
                total_tokens=sum(m_splits),
                n_cols=n_cols,
                out_dtype=out_dtype,
                device=B[0].device,
            )
            with record_function("grouped_gemm_nvfp4_direct_tn"):
                return nvfp4_grouped_gemm_tn_list(
                    A,
                    B,
                    sA,
                    sB,
                    global_amax_A,
                    global_amax_B,
                    m_splits,
                    out_tensor,
                    gemm_kind=gemm_kind,
                )

        first_lhs = A[0]
        is_quantized = isinstance(first_lhs, (QuantizedTensor, QuantizedTensorStorage))

        # Forward fallback: TE general_grouped_gemm on QuantizedTensor / plain tensors.
        if not grad:
            if is_quantized:
                n_cols = B[0]._rowwise_data.size(0)
                device = B[0]._rowwise_data.device
            else:
                n_cols = B[0].size(0)
                device = B[0].device
            out = [
                torch.empty(
                    [sum(m_splits), n_cols], dtype=out_dtype, device=device
                )
            ]
            return te_general_grouped_gemm(
                A,
                B,
                out,
                layout=layout,
                m_splits=m_splits,
                out_dtype=out_dtype,
                grad=False,
                accumulate=accumulate,
                bias=bias,
                use_bias=use_bias,
            )

        # Backward: MXFP8 / BF16 tensors via TE general_grouped_gemm (NN dgrad, NT wgrad).
        del sA, sB, global_amax_A, global_amax_B
        if layout == "NN":
            if is_quantized:
                n_cols = B[0]._columnwise_data.size(-1)
                device = B[0]._columnwise_data.device
            else:
                n_cols = B[0].size(1)
                device = B[0].device
            out = [
                torch.empty(
                    [sum(m_splits), n_cols], dtype=out_dtype, device=device
                )
            ]
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

        return te_general_grouped_gemm(
            A,
            B,
            out,
            layout=layout,
            m_splits=m_splits,
            out_dtype=out_dtype,
            grad=True,
            accumulate=accumulate,
            bias=bias,
            use_bias=use_bias,
        )
