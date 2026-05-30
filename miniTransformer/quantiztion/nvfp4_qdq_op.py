"""NVFP4 QDQ quantize op backed by quant_utils pure-Python simulation.

This op performs quantize-dequantize (QDQ) simulation using the lookup-table
quantizers from miniTransformer.ops.exp.quant_utils.  It supports two modes
selected at construction time:

  mode="block"    – quant_dequant_block   (single-level per-block scaling)
  mode="pertoken" – quant_dequant_pertoken (two-level per-token + per-block)

The GEMM itself runs in BF16 (torch.matmul) after QDQ, making this a W4A4
simulation rather than a hardware FP4 path.  It is intended for precision
experiments and ablations, not production throughput.
"""

import torch
from typing import Callable, List, Literal, Optional, Tuple

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
    """QDQ-simulation quantize op using quant_utils lookup quantizers.

    Args:
        mode:        "block" or "pertoken" – selects the QDQ function.
        block_size:  Number of elements per quantization block (default 16).
        scale_type:  Scale format for the block scale, "e4m3" or "e8m0".
        quantize_fn: Lookup quantizer to use (default: _quantize_e2m1).
                     Pass any function with signature (x: Tensor) -> Tensor.
        norm_max:    Normalisation constant for pertoken mode (default 448.0).
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

        self.mode       = mode
        self.block_size = block_size
        self.scale_type = ScaleType(scale_type)
        self.quantize_fn = quantize_fn if quantize_fn is not None else _QUANTIZE_FN
        self.fp_max     = _FP4_E2M1_MAX
        self.norm_max   = norm_max

    @property
    def supports_allgather(self) -> bool:
        return False

    # ── QDQ dispatch ──────────────────────────────────────────────────────────

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
        assert x.dtype in utils.HIGH_PRECISION_FLOAT_DTYPES, \
            f"Unsupported input dtype: {x.dtype}"

        orig_shape = x.shape
        x2d = x.view(-1, orig_shape[-1])

        x_qdq = self._qdq(x2d).view(orig_shape)

        # Scale and transposed tensors are unused by the BF16 GEMM path; fill
        # with empty sentinels so callers that read .scale / .data_t don't crash.
        empty = torch.empty(0, device=x.device)

        if return_transpose:
            x_qdq_t = self._qdq(x2d.t().contiguous()).view(orig_shape[-1], -1)
            return quantization.QuantizeResult(
                data=x_qdq, scale=empty,
                data_t=x_qdq_t, scale_t=empty,
                tensor_scale=empty, tensor_scale_t=empty,
            )

        return quantization.QuantizeResult(
            data=x_qdq, scale=empty,
            data_t=None, scale_t=None,
            tensor_scale=empty, tensor_scale_t=empty,
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
        custom_gemm_backend: str = "cublas",
    ) -> torch.Tensor:
        # qx and qw are already QDQ BF16 tensors from quantize().
        # Run a plain BF16 matmul; layout "TN" means qw is [N, K].
        qx_f = qx.to(out_dtype)
        qw_f = qw.to(out_dtype)

        if layout == "TN":
            y = torch.nn.functional.linear(qx_f, qw_f, bias=bias.to(out_dtype) if bias is not None else None)
        elif layout == "NN":
            # dgrad: qx is [M, N], qw is [N, K] -> out [M, K]
            y = torch.matmul(qx_f, qw_f)
            if bias is not None:
                y = y + bias.to(out_dtype)
        elif layout == "NT":
            # wgrad: qx is [K, M], qw is [K, N] -> out [M, N] = qx.T @ qw
            y = torch.matmul(qx_f.t(), qw_f)
            if bias is not None:
                y = y + bias.to(out_dtype)
        else:
            raise ValueError(f"Unsupported layout: {layout!r}")

        if accumulate:
            assert out is not None, "out must be provided when accumulate=True"
            out.add_(y)
            return out
        return y

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
        """
        Split x and apply QDQ to each shard.

        param_type == WEIGHT : x is [num_gemms, N, K]; each x[i] is [N, K].
        param_type == X / Y  : x is [M_sum, K]; split by m_splits.
        """
        if qparams.param_type == qparams.param_type.WEIGHT:
            shards = [x[i] for i in range(x.shape[0])]
        else:
            shards = list(torch.split(x, m_splits))

        forward_list: List[torch.Tensor] = []
        backward_list: List[torch.Tensor] = []
        empty = torch.empty(0, device=x.device)
        empty_list = [empty] * len(shards)

        for xi in shards:
            xi_qdq = self._qdq(xi.view(-1, xi.shape[-1])).view(xi.shape)
            forward_list.append(xi_qdq)
            if return_transpose:
                xi_qdq_t = self._qdq(xi.view(-1, xi.shape[-1]).t().contiguous())
                backward_list.append(xi_qdq_t)
            else:
                backward_list.append(xi_qdq)

        return GroupQuantizeResult(
            data=forward_list,
            data_t=backward_list,
            scale=empty_list,
            scale_t=empty_list,
            tensor_scale=empty_list,
            tensor_scale_t=empty_list,
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
    ) -> torch.Tensor:
        results = []
        for i in range(len(m_splits)):
            b = bias[i] if (use_bias and bias is not None) else None
            o = out[i] if (accumulate and out is not None) else None
            results.append(
                self.qgemm(
                    A[i], B[i], None, out_dtype,
                    sA[i], sB[i], b, o,
                    grad, accumulate, layout,
                    global_amax_A[i], global_amax_B[i],
                    qparams_x, qparams_w,
                )
            )
        return torch.cat(results, dim=0)
