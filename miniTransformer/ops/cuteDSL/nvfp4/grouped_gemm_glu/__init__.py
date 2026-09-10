# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
"""Vendored SM100 MoE block-scaled grouped GEMM + GLU (from cudnn-frontend).

Local tree for miniTE editing/debugging. Depends on CUTLASS CuteDSL + SM100 GPU;
does **not** require the cudnn Python pybind module.

Public entry::

    from miniTransformer.ops.cuteDSL.nvfp4.grouped_gemm_glu import (
        grouped_gemm_glu_wrapper_sm100,
        GroupedGemmGluSm100,
    )
"""

from .api import GroupedGemmGluSm100, grouped_gemm_glu_wrapper_sm100
from .moe_blockscaled_grouped_gemm_glu_bias import BlockScaledMoEGroupedGemmGluBiasKernel
from .moe_utils import MoEWeightMode

__all__ = [
    "GroupedGemmGluSm100",
    "grouped_gemm_glu_wrapper_sm100",
    "BlockScaledMoEGroupedGemmGluBiasKernel",
    "MoEWeightMode",
]
