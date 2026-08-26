"""NVFP4 QuantizeOp using TE PR #3045 per-token Quantizer / Tensor.

Activation and gradient: ``NVFP4Quantizer(per_token=True)`` → ``NVFP4Tensor``
with ``_per_token=True`` (per-row outer amax + 1x16 inner SF).

Weight: ``NVFP4Quantizer(per_token=True, per_token_weight_2d=True)`` so the
weight is a 16x16 / scalar-amax cast dressed in per-token layout (fwd
rowwise and dgrad columnwise share one tensor).

``qgemm`` / ``grouped_gemm`` call TE ``general_gemm`` /
``general_grouped_gemm``, which dispatch to
``nvfp4_cutlass_per_token_gemm`` / ``nvfp4_cutlass_grouped_per_token_gemm``
when both operands have ``_per_token=True``.
"""

from __future__ import annotations

import inspect
import logging
from typing import List, Optional, Tuple

import torch

from miniTransformer.ops._te_compat import general_gemm, general_grouped_gemm
from miniTransformer.quantiztion import quantization
from miniTransformer.quantiztion.quantization import GroupQuantizeResult
from miniTransformer.utils import utils

import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor, QuantizedTensorStorage

_NVFP4_QUANTIZER_PARAMS = inspect.signature(NVFP4Quantizer.__init__).parameters
if "per_token" not in _NVFP4_QUANTIZER_PARAMS:
    raise RuntimeError(
        "Nvfp4TiledQuantizePertokenOp requires Transformer Engine PR #3045 "
        "(NVFP4Quantizer(per_token=True) and NVFP4Tensor._per_token). "
        "The installed TE has no per_token argument."
    )
_HAS_PER_TOKEN_WEIGHT_2D = "per_token_weight_2d" in _NVFP4_QUANTIZER_PARAMS
_SPLIT_QUANTIZE_HAS_ROW_AMAX = False
try:
    _SPLIT_QUANTIZE_HAS_ROW_AMAX = "row_amax" in inspect.signature(tex.split_quantize).parameters
except (TypeError, ValueError):
    _SPLIT_QUANTIZE_HAS_ROW_AMAX = False
_warned_missing_split_quantize_row_amax = False
_logger = logging.getLogger(__name__)


def _split_quantize(tensor, m_splits, quantizers, *, row_amax=None):
    """tex.split_quantize with optional prefused per-token row_amax (skip K1)."""
    global _warned_missing_split_quantize_row_amax
    kwargs = {}
    if row_amax is not None:
        if _SPLIT_QUANTIZE_HAS_ROW_AMAX:
            kwargs["row_amax"] = row_amax
        elif not _warned_missing_split_quantize_row_amax:
            _warned_missing_split_quantize_row_amax = True
            _logger.warning(
                "input_row_amax was provided but tex.split_quantize has no row_amax; "
                "NVFP4 per-token K1 will still run. Need a TE build that accepts "
                "split_quantize(..., row_amax=)."
            )
    return tex.split_quantize(tensor, m_splits, quantizers, **kwargs)


def _make_per_token_quantizer(
    *,
    rowwise: bool,
    columnwise: bool,
    is_weight: bool = False,
) -> NVFP4Quantizer:
    """Build a PR #3045 per-token NVFP4Quantizer.

    Per-token is mutually exclusive with 2D block scaling, row_scaled_nvfp4,
    4over6, and amax reduction. RHT / SR are forced off (no per-token kernel
    yet), matching ``NVFP4PerTokenBlockScaling``.
    """
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
    quantizer = NVFP4Quantizer(**kwargs)
    # Ask TE to emit GEMM-ready swizzled rowwise scales during K2 so the
    # per-token GEMM path can skip a separate swizzle launch.
    quantizer.optimize_for_gemm = True
    return quantizer


class Nvfp4TiledQuantizePertokenOp(quantization.QuantizeOpBase):
    def __init__(self):
        super().__init__()
        self.should_update_quantize_weights = True
        # Cached quantizers — rebuilt only when num_gemms changes.
        self._cached_num_gemms: int = 0
        self._w_quantizers: List = []       # per_token_weight_2d, rowwise+columnwise
        self._x_quantizers: List = []       # act, rowwise only
        self._x_quantizers_col: List = []   # act/grad, rowwise+columnwise

    def _refresh_quantizer_cache(self, num_gemms: int) -> None:
        """Rebuild per-token NVFP4Quantizer lists when num_gemms changes."""
        if self._cached_num_gemms == num_gemms:
            return
        self._cached_num_gemms = num_gemms
        self._w_quantizers = [
            _make_per_token_quantizer(rowwise=True, columnwise=True, is_weight=True)
            for _ in range(num_gemms)
        ]
        self._x_quantizers = [
            _make_per_token_quantizer(rowwise=True, columnwise=False, is_weight=False)
            for _ in range(num_gemms)
        ]
        self._x_quantizers_col = [
            _make_per_token_quantizer(rowwise=True, columnwise=True, is_weight=False)
            for _ in range(num_gemms)
        ]

    @property
    def supports_allgather(self) -> bool:
        return False

    def _quantize(
        self,
        x: torch.Tensor,
        qparams: quantization.QParams,
        return_transpose: bool,
        with_2d_quantization: bool,
    ):
        del with_2d_quantization  # mutex with per_token
        is_weight = qparams.param_type == qparams.param_type.WEIGHT
        # Weight always keeps columnwise for dgrad; activation follows return_transpose.
        columnwise = True if is_weight else return_transpose
        nvfp4_quantizer = _make_per_token_quantizer(
            rowwise=True,
            columnwise=columnwise,
            is_weight=is_weight,
        )
        nvfp4_tensor = nvfp4_quantizer.make_empty(
            x.shape, dtype=x.dtype, device=x.device, requires_grad=False
        )
        return nvfp4_quantizer.update_quantized(x, nvfp4_tensor)

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
        return quantization.QuantizeResult(
            data=nvfp4_tensor,
            scale=nvfp4_tensor,
            data_t=nvfp4_tensor,
            scale_t=nvfp4_tensor,
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
        # TE general_gemm sees NVFP4Tensor._per_token and routes to
        # tex.nvfp4_cutlass_per_token_gemm.
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
        """
        If param type is weight, the shape is [num_gemms, n, k].
        If param type is X, the shape is [m_sum, k], m_sum should split into m_splits.
        If param type is Y, the shape is [m_sum, n].

        row_amax: optional prefused per-token row abs-max ``(sum_M,)`` fp32,
            aligned with activation rows (Fuse1 sort/pad, Fuse2 SwiGLU).
            When set and TE ``split_quantize`` accepts it, K1 is skipped.
        """
        num_gemms = len(m_splits)
        self._refresh_quantizer_cache(num_gemms)

        if qparams.param_type == qparams.param_type.WEIGHT:
            # per_token_weight_2d: 16x16 inner + scalar outer amax, stored in
            # per-token layout so fwd (rowwise) and dgrad (columnwise) share it.
            quantize_tensor_list = [
                self._w_quantizers[i].quantize(x[i]) for i in range(num_gemms)
            ]
        else:
            # ACT / GRAD: per-token 1D (per-row outer amax). columnwise for bwd.
            columnwise = True if qparams.param_type == qparams.param_type.Y else return_transpose
            quantizers = self._x_quantizers_col if columnwise else self._x_quantizers
            quantize_tensor_list = _split_quantize(
                x,
                m_splits,
                quantizers,
                row_amax=row_amax,
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
        a0 = A[0]
        is_quantized = isinstance(a0, (QuantizedTensor, QuantizedTensorStorage))

        if grad:
            if layout == "NN":
                if is_quantized:
                    n_cols = B[0]._rowwise_data.size(1) * 2
                else:
                    n_cols = B[0].size(1)
                out = [
                    torch.empty(
                        [sum(m_splits), n_cols],
                        dtype=activation_dtype,
                        device=B[0].device,
                    )
                ]
        else:
            if is_quantized:
                n_cols = B[0]._rowwise_data.size(0)
            else:
                n_cols = B[0].size(0)
            out = [
                torch.empty(
                    [sum(m_splits), n_cols],
                    dtype=activation_dtype,
                    device=B[0].device,
                )
            ]

        # TE general_grouped_gemm sees NVFP4Tensor._per_token and routes to
        # tex.nvfp4_cutlass_grouped_per_token_gemm.
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
            accumulate=accumulate,
        )

        if grad and layout == "NT":
            return out
        return out[0]
