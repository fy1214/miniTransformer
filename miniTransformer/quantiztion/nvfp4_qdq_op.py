"""NVFP4 QDQ op — X forward uses quant_utils QDQ simulation; all other paths use NVFP4.

X forward applies quant_dequant_block or quant_dequant_pertoken (controlled by `mode`),
making forward GEMMs simulate FP4 activation precision while weight quantization,
backward, and grouped GEMM remain on the standard NVFP4 hardware path.
"""

import torch
from typing import Callable, List, Literal, Optional, Tuple

import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm, general_grouped_gemm
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor, QuantizedTensorStorage

from miniTransformer.ops.exp.quant_utils import (
    ScaleType,
    _quantize_e2m1,
    default_norm_max,
    fp_format_max,
    quant_dequant_block,
    quant_dequant_pertoken,
)
from miniTransformer.quantiztion import quantization
from miniTransformer.quantiztion.quantization import GroupQuantizeResult
from miniTransformer.utils import utils


# Default NVFP4 E2M1 parameters
_FP4_E2M1_MAX = fp_format_max(2, 1)   # 6.0
_BLOCK_SIZE   = 16
_SCALE_TYPE   = ScaleType.E4M3
_QUANTIZE_FN  = _quantize_e2m1
_NORM_MAX     = default_norm_max(_SCALE_TYPE)


class Nvfp4QdqOp(quantization.QuantizeOpBase):
    """NVFP4 op where X forward uses Python QDQ simulation; all other paths are NVFP4.

    Args:
        mode:        "block" or "pertoken" – selects the X-forward QDQ function.
        block_size:  Quantization block size (default 16).
        scale_type:  Scale format for the block scale.
        quantize_fn: Lookup quantizer (default: _quantize_e2m1).
        norm_max:    Normalisation constant for pertoken mode.
    """

    should_update_quantize_weights = True

    def __init__(
        self,
        mode: Literal["block", "pertoken"] = "pertoken",
        block_size: int = _BLOCK_SIZE,
        scale_type: ScaleType = _SCALE_TYPE,
        quantize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        norm_max: float = _NORM_MAX,
    ) -> None:
        super().__init__()
        if mode not in ("block", "pertoken"):
            raise ValueError(f"mode must be 'block' or 'pertoken', got {mode!r}")

        self.mode        = mode
        self.block_size  = block_size
        self.scale_type  = ScaleType(scale_type)
        self.quantize_fn = quantize_fn if quantize_fn is not None else _QUANTIZE_FN
        self.fp_max      = _FP4_E2M1_MAX
        self.norm_max    = norm_max

    @property
    def supports_allgather(self) -> bool:
        return False

    # ── X-forward QDQ ────────────────────────────────────────────────────────

    def _qdq(self, x: torch.Tensor) -> torch.Tensor:
        """Apply QDQ to a 2D tensor and return BF16 dequantized result."""
        assert x.dim() == 2, f"_qdq expects 2D input, got shape {x.shape}"
        if self.mode == "block":
            return quant_dequant_block(
                x,
                block_size=self.block_size,
                fp_max=self.fp_max,
                quantize_fn=self.quantize_fn,
                scale_type=self.scale_type,
            )
        else:
            return quant_dequant_pertoken(
                x,
                block_size=self.block_size,
                fp_max=self.fp_max,
                quantize_fn=self.quantize_fn,
                scale_type=self.scale_type,
                norm_max=self.norm_max,
            )

    # ── NVFP4 helper (weight / grad / X backward) ────────────────────────────

    def _quantize_nvfp4(
        self,
        x: torch.Tensor,
        qparams: quantization.QParams,
        return_transpose: bool,
    ):
        M, N = x.shape
        quantizer = NVFP4Quantizer(
            rowwise=True,
            columnwise=return_transpose,
            with_amax_reduction=False,
            amax_reduction_group=None,
            with_rht=qparams.with_rht,
            with_post_rht_amax=qparams.with_rht,
            with_2d_quantization=qparams.with_2d_quantization,
            stochastic_rounding=qparams.stochastic_rounding,
        )
        nvfp4_tensor = quantizer.make_empty(
            (M, N), dtype=x.dtype, device=x.device, requires_grad=False
        )
        return quantizer.update_quantized(x, nvfp4_tensor)

    # ── quantize ──────────────────────────────────────────────────────────────

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

        if qparams.param_type == qparams.param_type.X:
            # Forward: QDQ simulation
            x_qdq = self._qdq(x.view(-1, x.shape[-1])).view(x.shape)
            # Backward: NVFP4 columnwise tensor for dgrad
            nvfp4_tensor = self._quantize_nvfp4(x, qparams, return_transpose=return_transpose)
            return quantization.QuantizeResult(
                data=x_qdq, scale=x_qdq,
                data_t=nvfp4_tensor, scale_t=nvfp4_tensor,
            )

        # WEIGHT and Y: standard NVFP4
        nvfp4_tensor = self._quantize_nvfp4(x, qparams, return_transpose=return_transpose)

        if qparams.param_type == qparams.param_type.WEIGHT:
            w_qdq = nvfp4_tensor.dequantize()
            return quantization.QuantizeResult(
                data=w_qdq, scale=w_qdq,
                data_t=nvfp4_tensor, scale_t=nvfp4_tensor,
            )

        return quantization.QuantizeResult(
            data=nvfp4_tensor, scale=nvfp4_tensor,
            data_t=nvfp4_tensor, scale_t=nvfp4_tensor,
        )

    # ── qgemm ─────────────────────────────────────────────────────────────────

    def qgemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        m_params: quantization.MMParams,
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        grad: bool = False,
        accumulate: bool = False,
        layout: str = "TN",
        global_amax_qx: Optional[torch.Tensor] = None,
        global_amax_qw: Optional[torch.Tensor] = None,
        qparams_x: Optional[quantization.QParams] = None,
        qparams_w: Optional[quantization.QParams] = None,
        custom_gemm_backend: str = "cutlass",
    ) -> torch.Tensor:
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

    # ── group_quantize ────────────────────────────────────────────────────────

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
        num_gemms = len(m_splits)

        if qparams.param_type == qparams.param_type.WEIGHT:
            forward_list, backward_list = [], []
            for i in range(num_gemms):
                w_quantizer = NVFP4Quantizer(
                    rowwise=True,
                    columnwise=True,
                    with_amax_reduction=False,
                    amax_reduction_group=None,
                    with_rht=qparams.with_rht,
                    with_post_rht_amax=qparams.with_rht,
                    with_2d_quantization=qparams.with_2d_quantization,
                    stochastic_rounding=qparams.stochastic_rounding,
                )
                w_q = w_quantizer.quantize(x[i])
                forward_list.append(w_q.dequantize())
                backward_list.append(w_q)
            return GroupQuantizeResult(
                data=forward_list, data_t=backward_list,
                scale=forward_list, scale_t=backward_list,
                tensor_scale=forward_list, tensor_scale_t=backward_list,
            )

        elif qparams.param_type == qparams.param_type.X:
            # Forward: QDQ per shard
            shards = list(torch.split(x, m_splits))
            forward_list = [
                self._qdq(xi.view(-1, xi.shape[-1])).view(xi.shape)
                for xi in shards
            ]
            # Backward: NVFP4 columnwise via split_quantize
            backward_list = []
            if return_transpose:
                quantizers = [
                    NVFP4Quantizer(
                        rowwise=False, columnwise=True,
                        with_amax_reduction=False, amax_reduction_group=None,
                        with_rht=qparams.with_rht, with_post_rht_amax=qparams.with_rht,
                        with_2d_quantization=qparams.with_2d_quantization,
                        stochastic_rounding=qparams.stochastic_rounding,
                    )
                    for _ in range(num_gemms)
                ]
                backward_list = tex.split_quantize(x, m_splits, quantizers)
            return GroupQuantizeResult(
                data=forward_list, data_t=backward_list,
                scale=forward_list, scale_t=backward_list,
                tensor_scale=forward_list, tensor_scale_t=backward_list,
            )

        else:
            # Y / grad: NVFP4 via split_quantize
            quantizers = [
                NVFP4Quantizer(
                    rowwise=True, columnwise=return_transpose,
                    with_amax_reduction=False, amax_reduction_group=None,
                    with_rht=qparams.with_rht, with_post_rht_amax=qparams.with_rht,
                    with_2d_quantization=qparams.with_2d_quantization,
                    stochastic_rounding=qparams.stochastic_rounding,
                )
                for _ in range(num_gemms)
            ]
            quantize_tensor_list = tex.split_quantize(x, m_splits, quantizers)
            return GroupQuantizeResult(
                data=quantize_tensor_list, data_t=quantize_tensor_list,
                scale=quantize_tensor_list, scale_t=quantize_tensor_list,
                tensor_scale=quantize_tensor_list, tensor_scale_t=quantize_tensor_list,
            )

    # ── grouped_gemm ──────────────────────────────────────────────────────────

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
        grad: bool = False,
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
                n_cols = B[0]._rowwise_data.size(1) * 2 if is_quantized else B[0].size(1)
                out = [torch.empty(
                    [sum(m_splits), n_cols], dtype=activation_dtype, device=B[0].device
                )]
        else:
            n_cols = B[0]._rowwise_data.size(0) if is_quantized else B[0].size(0)
            out = [torch.empty(
                [sum(m_splits), n_cols], dtype=activation_dtype, device=B[0].device
            )]

        general_grouped_gemm(
            A=B, B=A, out=out, out_dtype=activation_dtype,
            layout=layout, m_splits=m_splits,
            single_output=False if (grad and layout == "NT") else True,
            grad=grad, bias=bias, use_bias=use_bias,
            use_split_accumulator=False, accumulate=accumulate,
        )

        if grad and layout == "NT":
            return out
        else:
            return out[0]
