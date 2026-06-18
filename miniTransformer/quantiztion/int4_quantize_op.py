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
import functools
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


@dataclasses.dataclass
class MarlinInt4Weight:
    """Packed SGLang Marlin W4 weight and decode metadata for one expert."""

    qweight: torch.Tensor
    scale: torch.Tensor
    weight_zp: torch.Tensor
    g_idx: torch.Tensor
    g_idx_sort_indices: torch.Tensor
    workspace: torch.Tensor
    wtype: object
    in_features: int
    out_features: int
    group_size: int


@functools.lru_cache(maxsize=1)
def _get_sglang_marlin_ops():
    try:
        from sglang.jit_kernel.gptq_marlin import gptq_marlin_gemm
        from sglang.srt.layers.quantization.marlin_utils import (
            GPTQ_MARLIN_TILE,
            marlin_make_empty_g_idx,
            marlin_make_workspace,
            marlin_permute_scales,
        )
        from sglang.srt.layers.quantization.utils import get_scalar_types
    except ImportError as exc:
        raise ImportError(
            "Int4MarlinForwardQuantizeOp requires SGLang GPTQ Marlin in PYTHONPATH."
        ) from exc

    _, scalar_types = get_scalar_types()
    return (
        gptq_marlin_gemm,
        GPTQ_MARLIN_TILE,
        marlin_make_empty_g_idx,
        marlin_make_workspace,
        marlin_permute_scales,
        scalar_types.uint4b8,
    )


def _device_cache_key(device: torch.device) -> Tuple[str, Optional[int]]:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return device.type, torch.cuda.current_device()
    return device.type, device.index


@functools.lru_cache(maxsize=None)
def _get_sglang_weight_perm_cached(
    num_bits: int,
    device_type: str,
    device_index: Optional[int],
) -> torch.Tensor:
    import numpy as np

    perm_list: list[int] = []
    for i in range(32):
        perm1: list[int] = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])

    perm = np.array(perm_list)
    if num_bits == 4:
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    else:
        raise ValueError(f"Unsupported Marlin num_bits={num_bits}")
    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    device = torch.device(device_type, device_index)
    return torch.from_numpy(perm).to(device=device, dtype=torch.long)


def _get_sglang_weight_perm(num_bits: int, device: torch.device) -> torch.Tensor:
    device_type, device_index = _device_cache_key(device)
    return _get_sglang_weight_perm_cached(num_bits, device_type, device_index)


@functools.lru_cache(maxsize=None)
def _get_pack_shifts(
    pack_factor: int,
    device_type: str,
    device_index: Optional[int],
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    return torch.arange(pack_factor, device=device, dtype=torch.int32) * 4


@functools.lru_cache(maxsize=None)
def _get_empty_marlin_g_idx(
    device_type: str,
    device_index: Optional[int],
) -> torch.Tensor:
    _, _, marlin_make_empty_g_idx, _, _, _ = _get_sglang_marlin_ops()
    device = torch.device(device_type, device_index)
    return marlin_make_empty_g_idx(device)


def _pack_uint4_to_int32(q_uint4: torch.Tensor) -> torch.Tensor:
    """Pack eight uint4 values into one int32 word along the last dimension."""

    q_i32 = q_uint4.to(torch.int32)
    pack_factor = 8
    if q_i32.shape[-1] % pack_factor != 0:
        pad = pack_factor - (q_i32.shape[-1] % pack_factor)
        q_i32 = torch.nn.functional.pad(q_i32, (0, pad))
    q_i32 = q_i32.view(*q_i32.shape[:-1], q_i32.shape[-1] // pack_factor, pack_factor)
    device_type, device_index = _device_cache_key(q_i32.device)
    shifts = _get_pack_shifts(pack_factor, device_type, device_index)
    return (q_i32 << shifts).sum(dim=-1, dtype=torch.int32).contiguous()


@torch.no_grad()
def _quantize_weight_to_marlin(
    weight: torch.Tensor,
    group_size: int,
    symmetric: bool,
    workspace_m: int,
) -> MarlinInt4Weight:
    """Quantize a PyTorch linear weight [N, K] into SGLang GPTQ-Marlin tensors."""

    if not symmetric:
        raise NotImplementedError(
            "INT4_REAL_SYMMETRIC=0 is not supported by this training forward path yet."
        )
    if weight.dim() != 2:
        raise ValueError(f"Marlin INT4 weight must be 2D [N, K], got {tuple(weight.shape)}")
    if weight.dtype not in utils.HIGH_PRECISION_FLOAT_DTYPES:
        raise TypeError(f"Marlin INT4 weight must be fp16/bf16/fp32, got {weight.dtype}")
    if not weight.is_cuda:
        raise ValueError("SGLang Marlin INT4 real forward requires CUDA weights.")

    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
    if group_size == -1:
        group_size = in_features
    if group_size <= 0 or in_features % group_size != 0:
        raise ValueError(
            f"Invalid INT4 group_size={group_size} for in_features={in_features}."
        )
    if in_features % 16 != 0 or out_features % 16 != 0:
        raise ValueError(
            "SGLang Marlin INT4 requires in_features and out_features to be divisible by 16; "
            f"got in_features={in_features}, out_features={out_features}."
        )

    (
        _gptq_marlin_gemm,
        marlin_tile,
        _marlin_make_empty_g_idx,
        marlin_make_workspace,
        marlin_permute_scales,
        wtype,
    ) = _get_sglang_marlin_ops()

    # Match SGLang's test_marlin_utils.marlin_quantize layout: uint4b8
    # dequant semantics are (q - 8) * scale, then q is permuted into Marlin tiles.
    w_kn = weight.t().contiguous().float()
    num_groups = in_features // group_size
    w_grouped = w_kn.view(num_groups, group_size, out_features)
    scale = (w_grouped.abs().amax(dim=1) / 7.0).clamp(min=1e-5)
    q_signed = torch.round(w_grouped / scale[:, None, :]).clamp(-7, 7).to(torch.int32)
    q_uint4 = (q_signed + 8).to(torch.uint8).view(in_features, out_features).contiguous()

    perm = _get_sglang_weight_perm(4, weight.device)
    q_tiled = q_uint4.view(in_features // marlin_tile, marlin_tile, out_features // marlin_tile, marlin_tile)
    q_tiled = q_tiled.permute(0, 2, 1, 3).reshape(in_features // marlin_tile, out_features * marlin_tile)
    q_tiled = q_tiled.reshape(-1, perm.numel())[:, perm].reshape(q_tiled.shape)
    q_repacked = _pack_uint4_to_int32(q_tiled)

    device_type, device_index = _device_cache_key(weight.device)
    g_idx = _get_empty_marlin_g_idx(device_type, device_index)
    g_idx_sort_indices = _get_empty_marlin_g_idx(device_type, device_index)
    scale_repacked = marlin_permute_scales(
        scale.to(weight.dtype).contiguous(),
        size_k=in_features,
        size_n=out_features,
        group_size=group_size,
    )
    weight_zp = _get_empty_marlin_g_idx(device_type, device_index)
    workspace = marlin_make_workspace(weight.device)
    return MarlinInt4Weight(
        qweight=q_repacked,
        scale=scale_repacked,
        weight_zp=weight_zp,
        g_idx=g_idx,
        g_idx_sort_indices=g_idx_sort_indices,
        workspace=workspace,
        wtype=wtype,
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
    )


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


class Int4MarlinForwardQuantizeOp(Int4TiledQuantizeOp):
    """
    Real W4A16 forward operator for GroupedLinear.

    Forward packs BF16 expert weights into SGLang GPTQ-Marlin INT4 and calls
    gptq_marlin_gemm per expert. Backward must be kept in BF16 by the
    caller (DISABLE_BACKWARD_QUANT=1), so dgrad/wgrad use the inherited BF16
    grouped GEMM path with the original BF16 weights saved by GroupedLinear.
    """

    def __init__(self, group_size: int = 128, symmetric: bool = True):
        super().__init__()
        self.group_size = group_size
        self.symmetric = symmetric
        self.should_update_quantize_weights = False

    def quantize(
        self,
        x: torch.Tensor,
        qparams: quantization.QParams,
        return_transpose: bool = False,
        reduce_amax: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        input_meta: Optional[quantization.QuantizeInputMetaBase] = None,
    ) -> quantization.QuantizeResult:
        raise NotImplementedError(
            "Int4MarlinForwardQuantizeOp only supports GroupedLinear. "
            "Use INT4_SUB_CHANNEL or keep non-MoE Linear layers in BF16."
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
        raise NotImplementedError(
            "Int4MarlinForwardQuantizeOp only supports grouped_gemm for GroupedLinear."
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
        assert m_splits is not None, "group_quantize requires m_splits"
        if return_transpose:
            raise ValueError(
                "Int4MarlinForwardQuantizeOp requires DISABLE_BACKWARD_QUANT=1; "
                "SGLang Marlin W4A16 forward does not provide transpose quantized tensors for backward."
            )

        num_gemms = len(m_splits)
        if qparams.param_type == ParamType.WEIGHT:
            if x.dim() != 3:
                raise ValueError(
                    f"WEIGHT group_quantize expects [num_gemms, n, k], got {tuple(x.shape)}"
                )
            packed = [
                _quantize_weight_to_marlin(
                    x[i].contiguous(),
                    self.group_size,
                    self.symmetric,
                    max(m_splits) if m_splits else 1,
                )
                for i in range(num_gemms)
            ]
            return GroupQuantizeResult(
                data=packed,
                scale=packed,
                data_t=[None] * num_gemms,
                scale_t=[None] * num_gemms,
                tensor_scale=packed,
                tensor_scale_t=[None] * num_gemms,
            )

        if x.dim() != 2:
            raise ValueError(f"ACT/GRAD group_quantize expects [m_sum, k], got {tuple(x.shape)}")
        splits = list(torch.split(x, m_splits, dim=0))
        return GroupQuantizeResult(
            data=splits,
            scale=splits,
            data_t=[None] * num_gemms,
            scale_t=[None] * num_gemms,
            tensor_scale=splits,
            tensor_scale_t=[None] * num_gemms,
        )

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
        extra_info_A: Optional[dict] = None,
        extra_info_B: Optional[dict] = None,
    ) -> torch.Tensor:
        if grad or layout != "TN":
            return super().grouped_gemm(
                A=A,
                B=B,
                sA=sA,
                sB=sB,
                global_amax_A=global_amax_A,
                global_amax_B=global_amax_B,
                out=out,
                out_dtype=out_dtype,
                layout=layout,
                m_splits=m_splits,
                gelu=gelu,
                grad=grad,
                accumulate=accumulate,
                bias=bias,
                use_bias=use_bias,
                qparams_x=qparams_x,
                qparams_w=qparams_w,
            )

        if m_splits is None:
            raise ValueError("grouped_gemm requires m_splits")
        if gelu or accumulate or bias is not None or use_bias:
            raise NotImplementedError("Marlin INT4 GroupedLinear forward only supports plain GEMM.")
        if not B or not isinstance(B[0], MarlinInt4Weight):
            raise TypeError("Marlin INT4 forward expects packed MarlinInt4Weight objects as B.")

        gptq_marlin_gemm, _, _, _, _, _ = _get_sglang_marlin_ops()
        total_m = sum(m_splits)
        out_features = B[0].out_features
        device = A[0].device if A else B[0].qweight.device
        y = torch.empty((total_m, out_features), dtype=out_dtype, device=device)

        offset = 0
        for a, b, m in zip(A, B, m_splits):
            if m == 0:
                continue
            if a.shape[-1] != b.in_features:
                raise ValueError(
                    f"Marlin INT4 GEMM K mismatch: A has {a.shape[-1]}, weight has {b.in_features}."
                )
            c = y[offset: offset + m]
            a_contiguous = a.contiguous()
            gemm_out = gptq_marlin_gemm(
                a_contiguous,
                c if c.dtype == a_contiguous.dtype else None,
                b.qweight,
                b.scale,
                None,
                b.weight_zp,
                b.g_idx,
                b.g_idx_sort_indices,
                b.workspace,
                b.wtype,
                size_m=m,
                size_n=b.out_features,
                size_k=b.in_features,
                is_k_full=True,
                use_atomic_add=False,
                use_fp32_reduce=False,
                is_zp_float=False,
            )
            if gemm_out is not c:
                c.copy_(gemm_out.to(dtype=out_dtype))
            offset += m
        return y


