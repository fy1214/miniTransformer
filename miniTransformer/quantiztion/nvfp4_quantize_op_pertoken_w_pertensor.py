"""QAT16: NVFP4 per-token activation + per-tensor (per-expert) weight outer amax.

Temporary ablation vs QAT15: same TE bulk encode + grouped GEMM, but weight
``row_amax`` is ``expert_amax.repeat(N)`` instead of per-row amax.

Enable with ``QAT_PARAMS=16`` (recipe ``FP4_SUB_CHANNEL_PERTOKEN_W_PERTENSOR``).

Note: former QAT16 (``PERTOKEN_BF16_BWD``) moved to ``QAT_PARAMS=17``.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch.profiler import record_function

import transformer_engine.pytorch.cpp_extensions as tex

from miniTransformer.quantiztion import quantization
from miniTransformer.quantiztion.quantization import GroupQuantizeResult
from miniTransformer.quantiztion.nvfp4_quantize_op_pertoken import (
    Nvfp4TiledQuantizePertokenOp,
    is_dense_nvfp4_path_enabled,
    pack_nvfp4_rowwise_group_result,
)


def _expert_tensor_amax_as_row_amax(
    weight_2d: torch.Tensor,
    out_features: int,
    num_experts: int,
) -> torch.Tensor:
    """One amax per expert, broadcast onto all N weight rows (QAT8-like pertensor)."""
    if int(weight_2d.shape[0]) != int(out_features) * int(num_experts):
        raise RuntimeError(
            f"weight rows {weight_2d.shape[0]} != out_features*num_experts "
            f"{out_features}*{num_experts}"
        )
    w3 = weight_2d.view(num_experts, out_features, weight_2d.shape[1])
    expert_amax = w3.abs().amax(dim=(1, 2)).to(torch.float32)
    return expert_amax.repeat_interleave(out_features).contiguous()


class Nvfp4PertokenWeightPertensorOp(Nvfp4TiledQuantizePertokenOp):
    """QAT15 forward path with weight outer scale forced to per-expert tensor amax."""

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
        if qparams.param_type != qparams.param_type.WEIGHT:
            return super().group_quantize(
                x,
                qparams,
                m_splits=m_splits,
                return_transpose=return_transpose,
                reduce_amax=reduce_amax,
                tp_group=tp_group,
                input_meta=input_meta,
                row_amax=row_amax,
            )

        del reduce_amax, tp_group, input_meta, row_amax
        assert m_splits is not None
        num_experts = len(m_splits)
        self._ensure_mxfp8_backward_quantizers(num_experts)

        out_features, in_features = x.shape[1], x.shape[2]
        weight_row_splits = [out_features] * num_experts
        weight_2d = x.reshape(-1, in_features)
        weight_row_amax = _expert_tensor_amax_as_row_amax(
            weight_2d, out_features, num_experts
        )
        with record_function(
            "sq_w_nvfp4_bulk_dense_pertensor"
            if is_dense_nvfp4_path_enabled()
            else "sq_w_nvfp4_bulk_pertensor"
        ):
            data, scale, tensor_scale, extra = pack_nvfp4_rowwise_group_result(
                weight_2d,
                weight_row_splits,
                precomputed_row_amax=weight_row_amax,
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
