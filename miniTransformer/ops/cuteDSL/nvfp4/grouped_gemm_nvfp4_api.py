"""
Public NVFP4 grouped GEMM API (CuTeDSL path).

Import from here instead of ``quantiztion/nvfp4_quartet_op``.
"""

from miniTransformer.ops.cuteDSL.nvfp4.blockwise_group_gemm_cute_alpha_epilogue import (
    GroupedBlockScaledProfile,
    Sm100GroupedBlockScaledGemmKernel,
    apply_grouped_gemm_alphas,
)
from miniTransformer.ops.cuteDSL.nvfp4.grouped_gemm_nvfp4_impl import (
    PertokenNvfp4Tensor,
    TensorGlobalNvfp4Tensor,
    block_scale_to_sf_fp8,
    build_group_alpha_pertoken,
    dequant_pertoken_nvfp4,
    dequant_tensor_global_nvfp4,
    grouped_gemm_cutedsl_pertoken,
    grouped_gemm_cutedsl_te,
    is_sm100,
    pack_fp4_grid_to_uint8,
    pertoken_scales_to_sfa_2d,
    quantize_activation_nvfp4_pertoken,
    quantize_weight_nvfp4_tensor_global,
    reference_grouped_gemm_pertoken,
)

# Backward-compatible alias: TE tensor-global path
grouped_gemm_cutedsl = grouped_gemm_cutedsl_te

__all__ = [
    "GroupedBlockScaledProfile",
    "PertokenNvfp4Tensor",
    "Sm100GroupedBlockScaledGemmKernel",
    "TensorGlobalNvfp4Tensor",
    "apply_grouped_gemm_alphas",
    "block_scale_to_sf_fp8",
    "build_group_alpha_pertoken",
    "dequant_pertoken_nvfp4",
    "dequant_tensor_global_nvfp4",
    "grouped_gemm_cutedsl",
    "grouped_gemm_cutedsl_te",
    "grouped_gemm_cutedsl_pertoken",
    "is_sm100",
    "pack_fp4_grid_to_uint8",
    "pertoken_scales_to_sfa_2d",
    "quantize_activation_nvfp4_pertoken",
    "quantize_weight_nvfp4_tensor_global",
    "reference_grouped_gemm_pertoken",
]
