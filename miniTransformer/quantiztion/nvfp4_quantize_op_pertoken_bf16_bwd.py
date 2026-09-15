"""NVFP4 per-token forward + TE-style BF16 backward QuantizeOp.

Forward matches ``Nvfp4TiledQuantizePertokenOp`` (NVFP4 rowwise ``data`` +
NVFP4 grouped GEMM).

Backward aligns with Transformer Engine BF16 ``GroupedLinear``:
  - no MXFP8 columnwise / dy quant
  - ``data_t`` holds BF16 row splits (transpose is via GEMM layout ``NN``/``NT``)
  - ``grouped_gemm(grad=True)`` uses TE ``general_grouped_gemm`` on plain BF16

Enable with ``QAT_PARAMS=16`` (recipe ``FP4_SUB_CHANNEL_PERTOKEN_BF16_BWD``).
Does not require ``DISABLE_BACKWARD_QUANT``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from torch.profiler import record_function

from miniTransformer.quantiztion import quantization
from miniTransformer.quantiztion.nvfp4_quantize_op_pertoken import (
    Nvfp4TiledQuantizePertokenOp,
    is_dense_nvfp4_path_enabled,
    pack_nvfp4_rowwise_group_result,
)
from miniTransformer.quantiztion.quantization import GroupQuantizeResult
from miniTransformer.utils import utils


def split_bf16_rows(x_2d: torch.Tensor, row_splits: List[int]) -> List[torch.Tensor]:
    """Cast to BF16 and split along dim0 (no physical transpose)."""
    assert x_2d.dim() == 2, f"expected 2D tensor, got {tuple(x_2d.shape)}"
    if x_2d.dtype != torch.bfloat16:
        x_2d = x_2d.to(dtype=torch.bfloat16)
    return list(torch.split(x_2d.contiguous(), row_splits, dim=0))


def _bf16_backward_save(
    x_2d: torch.Tensor,
    row_splits: List[int],
    *,
    return_transpose: bool,
    profiler_name: str,
) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
    """Build ``data_t`` / ``scale_t`` for BF16 STE backward.

    ``scale_t`` is an empty list (not ``None``) when saving, because
    ``GroupedLinear`` asserts ``ctx.sw_t is not None``; scales are unused.
    """
    if not return_transpose:
        return [], None
    with record_function(profiler_name):
        data_t = split_bf16_rows(x_2d, row_splits)
    return data_t, []


def _nvfp4_forward_pack(
    x_2d: torch.Tensor,
    row_splits: List[int],
    *,
    profiler_dense: str,
    profiler_list: str,
    precomputed_row_amax: Optional[torch.Tensor] = None,
    extra: Optional[dict] = None,
):
    with record_function(
        profiler_dense if is_dense_nvfp4_path_enabled() else profiler_list
    ):
        return pack_nvfp4_rowwise_group_result(
            x_2d,
            row_splits,
            precomputed_row_amax=precomputed_row_amax,
            extra=extra,
        )


class Nvfp4PertokenBf16BackwardOp(Nvfp4TiledQuantizePertokenOp):
    """NVFP4 forward (QAT15) + BF16 dgrad/wgrad like TE non-FP8 GroupedLinear."""

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
        assert x.dtype in utils.HIGH_PRECISION_FLOAT_DTYPES
        num_experts = len(m_splits)

        # Grad output: BF16 splits only.
        if qparams.param_type == qparams.param_type.Y:
            with record_function("sq_dy_bf16_splits"):
                dy_bf16 = split_bf16_rows(x, m_splits)
            return GroupQuantizeResult(
                data=dy_bf16,
                scale=[],
                data_t=dy_bf16,
                scale_t=None,
                tensor_scale=None,
                tensor_scale_t=None,
            )

        # Weight: NVFP4 rowwise forward + BF16 rows for backward.
        if qparams.param_type == qparams.param_type.WEIGHT:
            out_features, in_features = x.shape[1], x.shape[2]
            weight_row_splits = [out_features] * num_experts
            weight_2d = x.reshape(-1, in_features)
            data, scale, tensor_scale, extra = _nvfp4_forward_pack(
                weight_2d,
                weight_row_splits,
                profiler_dense="sq_w_nvfp4_bulk_dense",
                profiler_list="sq_w_nvfp4_bulk",
                extra={"n": out_features, "k": in_features},
            )
            data_t, scale_t = _bf16_backward_save(
                weight_2d,
                weight_row_splits,
                return_transpose=return_transpose,
                profiler_name="sq_w_bf16_splits",
            )
            return GroupQuantizeResult(
                data=data,
                scale=scale,
                tensor_scale=tensor_scale,
                data_t=data_t,
                scale_t=scale_t,
                tensor_scale_t=None,
                extra_info=extra,
            )

        # Activation: NVFP4 rowwise forward + BF16 token splits for backward.
        data, scale, tensor_scale, extra = _nvfp4_forward_pack(
            x,
            m_splits,
            profiler_dense="sq_x_nvfp4_bulk_dense",
            profiler_list="sq_x_nvfp4_bulk",
            precomputed_row_amax=row_amax,
        )
        data_t, scale_t = _bf16_backward_save(
            x,
            m_splits,
            return_transpose=return_transpose,
            profiler_name="sq_x_bf16_splits",
        )
        return GroupQuantizeResult(
            data=data,
            scale=scale,
            tensor_scale=tensor_scale,
            data_t=data_t,
            scale_t=scale_t,
            tensor_scale_t=None,
            extra_info=extra,
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
        if not grad:
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
                grad=False,
                accumulate=accumulate,
                bias=bias,
                use_bias=use_bias,
                qparams_x=qparams_x,
                qparams_w=qparams_w,
                extra_info_A=extra_info_A,
                extra_info_B=extra_info_B,
            )

        # BF16 STE bwd: plain tensors + NN (dgrad) / NT (wgrad). Scales unused.
        del sA, sB, global_amax_A, global_amax_B, qparams_x, qparams_w
        del extra_info_A, extra_info_B, gelu
        assert layout in ("NN", "NT"), f"BF16 bwd expects NN/NT, got {layout!r}"
        assert isinstance(A, list) and isinstance(B, list)
        assert all(torch.is_tensor(t) for t in A + B)

        return super().grouped_gemm(
            A=A,
            B=B,
            sA=None,
            sB=None,
            global_amax_A=None,
            global_amax_B=None,
            out=out,
            out_dtype=out_dtype,
            layout=layout,
            m_splits=m_splits,
            gelu=False,
            grad=True,
            accumulate=accumulate,
            bias=bias,
            use_bias=use_bias,
        )
