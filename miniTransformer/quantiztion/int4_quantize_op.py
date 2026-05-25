"""
INT4 fake-quantization op with Straight-Through Estimator (STE).

Applies per-group symmetric INT4 fake quantization to weights, retaining results
as BF16 (no actual INT4 packing). Intended for QAT (quantization-aware training).

Quantization spec:
    - Format: symmetric INT4, range [-7, 7] (q_max=7, one negative value reserved for symmetry)
    - Granularity: per-group along the K dimension; group_size from qparams.quant_tile_shape[1]
    - Scale: scale = amax(|x_group|) / 7, clamped to a minimum of 1e-5

STE (Straight-Through Estimator):
    forward:  x_fq = dequant(quant(x)) = round(x/scale).clamp(-7,7) * scale
    backward: dx = dy (gradient passes through unchanged)

Backward path:
    group_linear.py sets disable_backward_quant=True, so backward dy and x bypass
    INT4 quantization and use BF16 grouped GEMM directly. Only the forward quantization
    path (weight + activation) is implemented here.

Interface compatibility:
    Matches Nvfp4TiledQuantizeOp (group_quantize / grouped_gemm / qgemm).
    Because fake-quantized tensors remain BF16, grouped_gemm uses the BF16 plain tensor
    path — no QuantizedTensor / NVFP4Quantizer needed.
"""

import dataclasses
import torch
from typing import List, Optional, Tuple

from miniTransformer.quantiztion import quantization
from miniTransformer.utils import utils
from miniTransformer.quantiztion.quantization import GroupQuantizeResult, ParamType


def _get_te_gemm():
    from transformer_engine.pytorch.cpp_extensions.gemm import (
        general_gemm,
        general_grouped_gemm,
    )
    return general_gemm, general_grouped_gemm


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


class _FakeInt4QuantizeSTE(torch.autograd.Function):
    """
    Per-group symmetric INT4 fake quantization with STE.

    Group shape: [1, group_size] (grouped along K dimension, each row independently)
    Quantization range: [-7, 7] (q_max=7, symmetric)

    forward: x_fq = round(x / scale).clamp(-7, 7) * scale
    backward: dx = dy (Straight-Through Estimator, gradient passes through)

    Ported from Megatron's _FakeInt4QuantizationSTE; supports non-aligned sizes via pad-then-crop.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group_size: int) -> torch.Tensor:
        m, n = x.shape

        # Pad N to a multiple of group_size
        n_padded = _ceil_div(n, group_size) * group_size
        if n_padded != n:
            x_padded = torch.zeros(
                (m, n_padded), dtype=x.dtype, device=x.device
            )
            x_padded[:m, :n] = x
        else:
            x_padded = x

        num_groups = n_padded // group_size
        x_view = x_padded.view(m, num_groups, group_size)

        q_max = 7
        x_max = x_view.abs().float().amax(dim=-1, keepdim=True)   # [m, num_groups, 1]
        x_scale = (x_max / q_max).clamp(min=1e-5)                 # [m, num_groups, 1]

        x_div   = x_view.float() / x_scale
        x_round = torch.round(x_div)
        x_q_clamped  = x_round.clamp(-q_max, q_max)
        x_dequant = (x_q_clamped * x_scale).to(x.dtype)           # [m, num_groups, group_size]

        x_out = x_dequant.view(m, n_padded)
        if n_padded != n:
            x_out = x_out[:, :n].contiguous()

        return x_out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def fake_int4_quantize_ste(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Apply per-group INT4 fake quantization (STE) to x. Preserves main_grad attribute."""
    x_out = _FakeInt4QuantizeSTE.apply(x, group_size)
    if hasattr(x, "main_grad"):
        x_out.main_grad = x.main_grad
    return x_out


class Int4TiledQuantizeOp(quantization.QuantizeOpBase):
    """
    INT4 per-group fake quantization operator implementing the QuantizeOpBase interface.

    - Fake-quantized tensors remain BF16; all GEMMs use the BF16 plain tensor path.
    - group_quantize returns a GroupQuantizeResult where data/scale are BF16 tensor lists.
    - grouped_gemm delegates to TE general_grouped_gemm (BF16 path).
    - Backward is controlled by group_linear.py's disable_backward_quant=True;
      backward GEMMs use BF16 directly without invoking this op's quantization logic.

    Interface is identical to Nvfp4TiledQuantizeOp and can be used as a drop-in replacement.
    """

    def __init__(self):
        super().__init__()

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
        """
        Apply per-group INT4 fake quantization to a single tensor.

        Args:
            x: Input tensor, shape [M, K], must be high-precision float.
            qparams: Quantization params; group_size = qparams.quant_tile_shape[1].
            return_transpose: Whether to also return the fake-quantized transpose.

        Returns:
            QuantizeResult where data is BF16 fake-quantized; scale is a placeholder (same as data).
        """
        assert x.dtype in utils.HIGH_PRECISION_FLOAT_DTYPES, (
            f"Int4TiledQuantizeOp.quantize: unsupported input dtype {x.dtype}"
        )
        assert qparams.quant_tile_shape is not None
        assert len(qparams.quant_tile_shape) == 2

        group_size = qparams.quant_tile_shape[1]
        x_fq = fake_int4_quantize_ste(x, group_size)

        x_fq_t = x_fq.T.contiguous() if return_transpose else None

        return quantization.QuantizeResult(
            data=x_fq,
            scale=x_fq,        # no separate scale for fake quant; use data as placeholder
            data_t=x_fq_t,
            scale_t=x_fq_t,
            tensor_scale=None,
            tensor_scale_t=None,
        )

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
        Batched per-group INT4 fake quantization for grouped GEMM.

        Input shape convention (consistent with nvfp4_quantize_op.py):
            WEIGHT: x.shape = [num_gemms, n, k]; each expert's [n, k] is quantized independently.
            ACT/GRAD: x.shape = [m_sum, k]; split by m_splits before quantization.

        Args:
            x: Input tensor.
            qparams: Quantization params; group_size = qparams.quant_tile_shape[1].
            m_splits: Per-expert token counts; len(m_splits) == num_gemms.
            return_transpose: Whether to also return each tensor's transpose.

        Returns:
            GroupQuantizeResult where data/scale are BF16 fake-quantized tensor lists.
        """
        assert m_splits is not None, "group_quantize requires m_splits"
        assert qparams.quant_tile_shape is not None

        group_size = qparams.quant_tile_shape[1]
        num_gemms = len(m_splits)

        if qparams.param_type == ParamType.WEIGHT:
            # x: [num_gemms, n, k] — fake-quantize each [n, k] slab independently
            assert x.dim() == 3, (
                f"WEIGHT group_quantize expects 3D input [num_gemms, n, k], got {x.shape}"
            )
            fq_list = [
                fake_int4_quantize_ste(x[i], group_size)
                for i in range(num_gemms)
            ]
        else:
            # ACT/GRAD: pass through without quantization
            assert x.dim() == 2, (
                f"ACT/GRAD group_quantize expects 2D input [m_sum, k], got {x.shape}"
            )
            splits = list(torch.split(x, m_splits, dim=0))
            fq_list = splits

        if return_transpose:
            fq_t_list = [f.T.contiguous() for f in fq_list]
        else:
            fq_t_list = [None] * num_gemms

        # scale uses data as placeholder — fake quant has no separate scale tensor
        return GroupQuantizeResult(
            data=fq_list,
            scale=fq_list,
            data_t=fq_t_list,
            scale_t=fq_t_list,
            tensor_scale=fq_list,
            tensor_scale_t=fq_t_list,
        )

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
        """
        BF16 GEMM (fake-quantized tensors remain BF16; delegates to TE general_gemm).

        sx/sw are placeholders and are not used.
        """
        general_gemm, _ = _get_te_gemm()
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
            bulk_overlap=False,
        )[0]
        return gemm_out

    def grouped_gemm(
        self,
        A: List[torch.Tensor],
        B: List[torch.Tensor],
        sA: Optional[List[torch.Tensor]],
        sB: Optional[List[torch.Tensor]],
        global_amax_A: Optional[List[torch.Tensor]],
        global_amax_B: Optional[List[torch.Tensor]],
        out: Optional[List[torch.Tensor]],
        out_dtype: torch.dtype,
        layout: str = "TN",
        m_splits: Optional[List[int]] = None,
        gelu: bool = False,
        grad: bool = False,
        accumulate: bool = False,
        bias: Optional[List[torch.Tensor]] = None,
        use_bias: bool = False,
        qparams_x: Optional[quantization.QParams] = None,
        qparams_w: Optional[quantization.QParams] = None,
    ) -> torch.Tensor:
        """
        BF16 grouped GEMM corresponding to group_linear.py's quantize_op.grouped_gemm call.

        A and B are BF16 fake-quantized tensor lists (plain torch.Tensor).
        sA/sB/global_amax_A/global_amax_B are None (no separate scale for fake quant).

        Layout convention (consistent with nvfp4_quantize_op.py):
            fprop (layout="TN"):  Y = A @ B.T, A=x_splits, B=weights
            dgrad (layout="NN"):  dX = dy @ W, A=dy_splits, B=weights
            wgrad (layout="NT"):  dW = dy.T @ x, A=dy_splits, B=x_splits

        Note: TE general_grouped_gemm's A/B convention is swapped vs. intuition;
        we match nvfp4_quantize_op.py by passing A and B transposed.
        """
        assert m_splits is not None, "grouped_gemm requires m_splits"

        _, general_grouped_gemm = _get_te_gemm()

        if grad:
            if layout == "NN":
                # dgrad: A=dy_splits [m_i, n], B=weights [n, k] -> [m_sum, k]
                n_cols = B[0].size(1)
                out = torch.empty(
                    [sum(m_splits), n_cols],
                    dtype=out_dtype,
                    device=B[0].device,
                )
                out = [out]
            # wgrad (layout="NT") out is provided by caller (main_grad buffer)
        else:
            # fprop: B[i] shape [n, k] -> output shape [m_sum, n]
            n_cols = B[0].size(0)
            out = torch.empty(
                [sum(m_splits), n_cols],
                dtype=out_dtype,
                device=B[0].device,
            )
            out = [out]

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
        else:
            return out[0]
