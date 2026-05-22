"""
int4_quantize_op.py
===================
作者: mingfa
日期: 2026-05-09

INT4 Fake-Quantization Op with Straight-Through Estimator (STE).

功能:
    对权重做 per-group symmetric INT4 fake quantization，量化结果保留为 BF16
    （fake quant），不做真正的 INT4 packing。适用于 QAT（量化感知训练）。

量化规格:
    - 格式: symmetric INT4，量化范围 [-7, 7]（q_max=7，保留一个负数位做对称性）
    - 粒度: per-group，group_size 来自 qparams.quant_tile_shape[1]（通常 128）
    - group 方向: 沿 K 维（输入维）做分组，每 group_size 个元素共享一个 scale
    - Scale 计算: scale = amax(|x_group|) / 7，最小值 clamp 到 1e-5

STE (Straight-Through Estimator):
    forward:  x_fq = dequant(quant(x)) = round(x/scale).clamp(-7,7) * scale
    backward: dx = dy（梯度直通，不阻断梯度流）

Backward 路径:
    group_linear.py 中设置 disable_backward_quant=True，
    因此 backward 的 dy 和 x 不经过 INT4 量化，直接走 BF16 grouped GEMM。
    本文件只需实现 forward 路径的量化（weight + activation）。

与 nvfp4_quantize_op.py 的关系:
    接口完全一致（group_quantize / grouped_gemm / qgemm），
    但由于 fake quant 后 tensor 仍是 BF16，grouped_gemm 走 BF16 plain tensor 路径，
    无需 QuantizedTensor / NVFP4Quantizer。

参考:
    - /root/Megatron-LM/megatron/core/extensions/transformer_engine.py
      _FakeInt4QuantizationSTE 和 fake_int4_quantization_ste
    - miniTransformer/quantiztion/nvfp4_quantize_op.py（接口参考）
    - miniTransformer/module/group_linear.py（调用方）
"""

import dataclasses
import torch
from typing import List, Optional, Tuple

from miniTransformer.quantiztion import quantization
from miniTransformer.utils import utils
from miniTransformer.quantiztion.quantization import GroupQuantizeResult, ParamType


# ── TE 延迟导入（避免循环 import）────────────────────────────────────────

def _get_te_gemm():
    from transformer_engine.pytorch.cpp_extensions.gemm import (
        general_gemm,
        general_grouped_gemm,
    )
    return general_gemm, general_grouped_gemm


# ── 辅助函数 ─────────────────────────────────────────────────────────────

def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


# ── INT4 Fake Quant STE ───────────────────────────────────────────────────

class _FakeInt4QuantizeSTE(torch.autograd.Function):
    """
    Per-group symmetric INT4 fake quantization with STE.

    Group shape: [1, group_size]（沿 K 维分组，每行独立）
    Quantization range: [-7, 7]（q_max=7，symmetric）

    forward: x_fq = round(x / scale).clamp(-7, 7) * scale
    backward: dx = dy（Straight-Through Estimator，梯度直通）

    完全移植自 Megatron 的 _FakeInt4QuantizationSTE，支持非对齐尺寸（pad-then-crop）。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group_size: int) -> torch.Tensor:
        m, n = x.shape

        # ── 1. pad N 到 group_size 的整数倍 ──────────────────────────────
        n_padded = _ceil_div(n, group_size) * group_size
        if n_padded != n:
            x_padded = torch.zeros(
                (m, n_padded), dtype=x.dtype, device=x.device
            )
            x_padded[:m, :n] = x
        else:
            x_padded = x

        # ── 2. reshape 成 [m, num_groups, group_size] ─────────────────────
        num_groups = n_padded // group_size
        x_view = x_padded.view(m, num_groups, group_size)

        # ── 3. per-group scale = amax / q_max ─────────────────────────────
        q_max = 7
        x_max = x_view.abs().float().amax(dim=-1, keepdim=True)   # [m, num_groups, 1]
        x_scale = (x_max / q_max).clamp(min=1e-5)                 # [m, num_groups, 1]

        # ── 4. fake quant: quantize + dequantize ──────────────────────────
        x_div   = x_view.float() / x_scale
        x_round = torch.round(x_div)
        x_q_clamped  = x_round.clamp(-q_max, q_max)
        x_dequant = (x_q_clamped * x_scale).to(x.dtype)           # [m, num_groups, group_size]

        # ── 5. 还原原始 shape，截取有效部分 ─────────────────────────────
        x_out = x_dequant.view(m, n_padded)
        if n_padded != n:
            x_out = x_out[:, :n].contiguous()

        return x_out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # STE：梯度直通，不阻断
        return grad_output, None


def fake_int4_quantize_ste(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """
    对外接口：对 x 做 per-group INT4 fake quantization（STE）。
    保留 main_grad 属性（Megatron 兼容性）。
    """
    x_out = _FakeInt4QuantizeSTE.apply(x, group_size)
    if hasattr(x, "main_grad"):
        x_out.main_grad = x.main_grad
    return x_out


# ── Int4TiledQuantizeOp ───────────────────────────────────────────────────

class Int4TiledQuantizeOp(quantization.QuantizeOpBase):
    """
    INT4 per-group fake quantization operator，实现 QuantizeOpBase 接口。

    设计原则：
    - fake quant 后 tensor 仍是 BF16，所有 GEMM 走 BF16 plain tensor 路径
    - group_quantize 返回的 GroupQuantizeResult 中 data/scale 均为 BF16 tensor list
    - grouped_gemm 调用 TE 的 general_grouped_gemm（BF16 路径）
    - backward 由 group_linear.py 的 disable_backward_quant=True 控制，
      直接走 BF16 GEMM，不调用本 Op 的量化逻辑

    接口与 Nvfp4TiledQuantizeOp 完全一致，可直接替换使用。
    """

    def __init__(self):
        super().__init__()

    @property
    def supports_allgather(self) -> bool:
        return False

    # ── quantize（单 tensor 接口，兼容 QuantizeOpBase）────────────────────

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
        对单个 tensor 做 per-group INT4 fake quantization。

        Args:
            x: 输入 tensor，shape [M, K]，dtype 必须是高精度浮点
            qparams: 量化参数，group_size = qparams.quant_tile_shape[1]
            return_transpose: 是否同时返回转置后的 fake-quantized tensor

        Returns:
            QuantizeResult，data 是 BF16 fake-quantized tensor，scale 同样为 data（占位）
        """
        assert x.dtype in utils.HIGH_PRECISION_FLOAT_DTYPES, (
            f"Int4TiledQuantizeOp.quantize: unsupported input dtype {x.dtype}"
        )
        assert qparams.quant_tile_shape is not None
        assert len(qparams.quant_tile_shape) == 2

        group_size = qparams.quant_tile_shape[1]
        x_fq = fake_int4_quantize_ste(x, group_size)              # BF16 fake-quantized

        x_fq_t = x_fq.T.contiguous() if return_transpose else None

        return quantization.QuantizeResult(
            data=x_fq,
            scale=x_fq,        # fake quant 无独立 scale，用 data 占位
            data_t=x_fq_t,
            scale_t=x_fq_t,
            tensor_scale=None,
            tensor_scale_t=None,
        )

    # ── group_quantize（批量接口，供 _QuantizedGroupedLinear 调用）─────────

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
        批量 per-group INT4 fake quantization，适配 grouped GEMM 场景。

        输入 shape 约定（与 nvfp4_quantize_op.py 一致）：
            WEIGHT: x.shape = [num_gemms, n, k]，对每个 expert 的 [n, k] 分别量化
            ACT/GRAD: x.shape = [m_sum, k]，按 m_splits split 后分别量化

        Args:
            x: 输入 tensor
            qparams: 量化参数，group_size = qparams.quant_tile_shape[1]
            m_splits: token 按 expert 的分割列表，len(m_splits) == num_gemms
            return_transpose: 是否同时返回每个 tensor 的转置

        Returns:
            GroupQuantizeResult，data/scale 均为 BF16 fake-quantized tensor list
        """
        assert m_splits is not None, "group_quantize requires m_splits"
        assert qparams.quant_tile_shape is not None

        group_size = qparams.quant_tile_shape[1]
        num_gemms = len(m_splits)

        if qparams.param_type == ParamType.WEIGHT:
            # x: [num_gemms, n, k] → 对每个 [n, k] 分别做 fake quant
            assert x.dim() == 3, (
                f"WEIGHT group_quantize expects 3D input [num_gemms, n, k], got {x.shape}"
            )
            fq_list = [
                fake_int4_quantize_ste(x[i], group_size)
                for i in range(num_gemms)
            ]
        else:
            # ACT/GRAD: 不量化
            assert x.dim() == 2, (
                f"ACT/GRAD group_quantize expects 2D input [m_sum, k], got {x.shape}"
            )
            splits = list(torch.split(x, m_splits, dim=0))
            fq_list = splits

        if return_transpose:
            fq_t_list = [f.T.contiguous() for f in fq_list]
        else:
            fq_t_list = [None] * num_gemms

        # fake quant 后 tensor 即是最终结果，scale 用 data 占位（无独立 scale tensor）
        return GroupQuantizeResult(
            data=fq_list,
            scale=fq_list,
            data_t=fq_t_list,
            scale_t=fq_t_list,
            tensor_scale=fq_list,
            tensor_scale_t=fq_t_list,
        )

    # ── qgemm（单次 BF16 GEMM）───────────────────────────────────────────

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
        单次 BF16 GEMM（fake quant 后 tensor 仍是 BF16，直接调用 TE general_gemm）。

        fake quant 后 qx/qw 是 BF16 plain tensor，sx/sw 为占位（不使用）。
        调用约定与 nvfp4_quantize_op.py 的 qgemm 保持一致。
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

    # ── grouped_gemm（BF16 grouped GEMM）────────────────────────────────

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
        BF16 grouped GEMM，对应 group_linear.py 中 quantize_op.grouped_gemm 的调用。

        A, B 均为 BF16 fake-quantized tensor list（plain torch.Tensor），
        sA/sB/global_amax_A/global_amax_B 均为 None（fake quant 无独立 scale）。

        调用约定与 nvfp4_quantize_op.py 的 grouped_gemm 保持一致：
            - fprop (layout="TN"):  Y = A @ B.T，A=x_splits，B=weights
            - dgrad (layout="NN"):  dX = dy @ W，A=dy_splits，B=weights
            - wgrad (layout="NT"):  dW = dy.T @ x，A=dy_splits，B=x_splits

        注意：TE general_grouped_gemm 的 A/B 参数约定与直觉相反，
        这里与 nvfp4_quantize_op.py 保持一致，传入时对调 A 和 B。

        Args:
            A: 左矩阵 tensor list（通常是 x 或 dy）
            B: 右矩阵 tensor list（通常是 weight 或 x_t）
            out_dtype: 输出 dtype
            layout: "TN"（fprop）/ "NN"（dgrad）/ "NT"（wgrad）
            m_splits: 每个 expert 的 token 数列表
            grad: 是否为 backward 路径
            accumulate: 是否累加到 out
        """
        assert m_splits is not None, "grouped_gemm requires m_splits"

        _, general_grouped_gemm = _get_te_gemm()

        # ── 计算 out tensor ──────────────────────────────────────────────
        if grad:
            if layout == "NN":
                # dgrad: A=dy_splits [m_i, n], B=weights [n, k] → [m_sum, k]
                n_cols = B[0].size(1)
                out = torch.empty(
                    [sum(m_splits), n_cols],
                    dtype=out_dtype,
                    device=B[0].device,
                )
                out = [out]
            # wgrad (layout="NT") 的 out 由调用方传入（main_grad buffer）
        else:
            # fprop: B[i] shape [n, k] → output shape [m_sum, n]
            n_cols = B[0].size(0)
            out = torch.empty(
                [sum(m_splits), n_cols],
                dtype=out_dtype,
                device=B[0].device,
            )
            out = [out]

        # ── 调用 TE general_grouped_gemm（A/B 对调，与 nvfp4 保持一致）────
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
