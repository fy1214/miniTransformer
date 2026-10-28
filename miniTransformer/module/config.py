import dataclasses
import enum
import logging
import os
import torch
from typing import Dict, Optional, Mapping, Sequence, Type, Union

from miniTransformer.quantiztion.quantization import QParams, MMParams, ParamType
from miniTransformer.quantiztion import quantization
import miniTransformer.utils.env as env
from miniTransformer.recipes import describe_recipe, get_recipe, get_recipe_from_env, recipe_summary

logger = logging.getLogger(__name__)

@enum.unique
class QuantizeType(enum.Enum):
    """Quantization format for linear layers."""

    NONE = "none"
    FP8 = "fp8"
    MXFP8 = "mxfp8"
    INT4 = "int4"
    NVFP4 = "nvfp4"


@dataclasses.dataclass()
class QLinearParams:
    """Quantization parameters of linear layer.

    quantize: Whether to quantize the linear layer.
    x_params: Quantization parameters for activation tensor x (and x^t).
    w_params: Quantization parameters for weight tensor w (and w^t).
    g_params: Quantization parameters for gradient tensor g (and g^t).
    mm_fprop: Matrix multiplication parameters for fwd GEMM.
    mm_dgrad: Matrix multiplication parameters for dgrad bwd GEMM.
    mm_wgrad: Matrix multiplication parameters for wgrad bwd GEMM.

    allgather_fp8: Whether to perform allgather communication in FP8, default is False.
    memory_saving: Whether to use memory saving mode, default is False.
        If enabled, we will not save the full qx^t in the fwd pass to save memory. Instead,
        we only save a shard of the qx^t and AllGather the full qx^t in the bwd pass.

    quantize_op: Custom quantize op implementation. If not provided, automatically determine QuantizeOp based on scaling_type.
    """

    quantize: bool = False  # fp8 / int4 / fp4
    quantize_type: QuantizeType = None

    x_params: QParams = QParams()
    w_params: QParams = QParams()
    g_params: QParams = QParams()

    mm_fprop: MMParams = MMParams()
    mm_dgrad: MMParams = MMParams()
    mm_wgrad: MMParams = MMParams()

    allgather_quantize: bool = False
    memory_saving: bool = False

    quantize_op: Optional[quantization.QuantizeOpBase] = None
    # NOTE: We currently use ub_name for layer name in linear.
    # There is not much variation so we only know type of linear:
    # i.e. qkv, proj, fc1, fc2
    # With real layer names, a Mapping may need to turn into a
    # predicate to be useful.
    overrides_for_layer_name: Mapping[str, "QLinearParams"] = dataclasses.field(
        default_factory=dict
    )

    def __post_init__(self):
        if self.quantize_op is not None:
            for v in self.overrides_for_layer_name.values():
                assert (
                    len(v.overrides_for_layer_name) == 0
                ), "Only top level QLinearParams should have layer_name overrides."

    def customize_for_layer_name(self, layer_name: str) -> "QLinearParams":
        """
        Returns an edited qlinear params if there are any different behaviors than standard
        based on the layer name as communicated by a megatron user buffer name.
        """
        if layer_name in self.overrides_for_layer_name:
            return self.overrides_for_layer_name[layer_name]
        return self


@enum.unique
class QuantizeRecipe(enum.Enum):
    """Pre-defined quantization recipes for linear layers."""

    NON_QUANTIZE = "non_quantize"
    BF16_REF = "bf16_ref"
    INT4_SUB_CHANNEL = "int4_subchannel_scaling"
    INT4_MARLIN_FORWARD = "int4_marlin_forward"
    FP4_SUB_CHANNEL = "fp4_subchannel_scaling"
    FP4_SUB_CHANNEL_PERTOKEN = "fp4_subchannel_scaling_pertoken"
    FP4_SUB_CHANNEL_PERTOKEN_W_PERTENSOR = "fp4_subchannel_scaling_pertoken_w_pertensor"
    FP4_SUB_CHANNEL_PERTOKEN_BF16_BWD = "fp4_subchannel_scaling_pertoken_bf16_bwd"
    FP4_SUB_CHANNEL_QUARTET = "fp4_subchannel_scaling_quartet"
    FP4_SUB_CHANNEL_REF = "fp4_subchannel_scaling_ref"
    FP4_SUB_CHANNEL_CURSOR2 = "fp4_subchannel_scaling_cursor2"
    FP4_SUB_CHANNEL_CURSOR2_REF = "fp4_subchannel_scaling_cursor2_ref"
    FP4_SUB_CHANNEL_CURSOR2_CUTLASS = "fp4_subchannel_scaling_cursor2_cutlass"
    FP4_SUB_CHANNEL_CURSOR2_CUTLASS_BWD = "fp4_subchannel_scaling_cursor2_cutlass_bwd"
    FP4_SUB_CHANNEL_CURSOR2_CUTLASS_V2 = "fp4_subchannel_scaling_cursor2_cutlass_v2"
    FP4_SUB_CHANNEL_W4A16 = "fp4_subchannel_scaling_w4a16"
    FP4_QDQ_BLOCK = "fp4_qdq_block"
    FP4_QDQ_PERTOKEN = "fp4_qdq_pertoken"


def get_qlinear_params_from_predefined(recipe: QuantizeRecipe) -> QLinearParams:
    """Get quantization parameters for linear layer based on recipe."""
    if recipe == QuantizeRecipe.NON_QUANTIZE:
        return QLinearParams(quantize=False, quantize_type=QuantizeType.NONE)
    elif recipe == QuantizeRecipe.BF16_REF:
        from miniTransformer.quantiztion.bf16_ref_op import Bfloat16RefOp
        return QLinearParams(quantize=True, quantize_type=QuantizeType.NONE, quantize_op=Bfloat16RefOp())
    elif recipe == QuantizeRecipe.INT4_SUB_CHANNEL:
        from miniTransformer.quantiztion.int4_quantize_op import Int4TiledQuantizeOp
        group_size = env.get_int4_fake_group_size()
        return QLinearParams(
            quantize=True,
            quantize_type=QuantizeType.INT4,
            x_params=QParams(
                output_dtype=torch.bfloat16,
                quant_tile_shape=(1, group_size),
                param_type=ParamType.X
            ),
            w_params=QParams(
                output_dtype=torch.bfloat16,
                quant_tile_shape=(1, group_size),
                param_type=ParamType.WEIGHT
            ),
            g_params=QParams(
                output_dtype=torch.bfloat16,
                quant_tile_shape=(1, group_size),
                param_type=ParamType.Y
            ),
            quantize_op=Int4TiledQuantizeOp())
    elif recipe == QuantizeRecipe.INT4_MARLIN_FORWARD:
        from miniTransformer.quantiztion.int4_quantize_op import Int4MarlinForwardQuantizeOp
        group_size = env.get_int4_real_group_size()
        return QLinearParams(
            quantize=True,
            quantize_type=QuantizeType.INT4,
            x_params=QParams(
                output_dtype=torch.bfloat16,
                quant_tile_shape=(1, group_size),
                param_type=ParamType.X,
            ),
            w_params=QParams(
                output_dtype=torch.int32,
                quant_tile_shape=(1, group_size),
                param_type=ParamType.WEIGHT,
            ),
            g_params=QParams(
                output_dtype=torch.bfloat16,
                quant_tile_shape=(1, group_size),
                param_type=ParamType.Y,
            ),
            quantize_op=Int4MarlinForwardQuantizeOp(
                group_size=group_size,
                symmetric=env.should_use_int4_real_symmetric(),
            ),
        )
    elif recipe == QuantizeRecipe.FP4_SUB_CHANNEL_REF:
        from miniTransformer.quantiztion.nvfp4_quantize_op_ref import Nvfp4TiledQuantizeRefOp
        return QLinearParams(
            quantize=True,
            quantize_type=QuantizeType.NVFP4,
            x_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=not env.should_disable_rht(),
                param_type=ParamType.X
            ),
            w_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(16, 16),
                with_2d_quantization=not env.should_disable_2d_quantization(),
                with_rht=False,
                param_type=ParamType.WEIGHT
            ),
            g_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=not env.should_disable_rht(),
                stochastic_rounding=not env.should_disable_stochastic_rounding(),
                param_type=ParamType.Y
            ),
            quantize_op=Nvfp4TiledQuantizeRefOp(),
            allgather_quantize=False,
        )
    elif recipe == QuantizeRecipe.FP4_SUB_CHANNEL:
        # Split accumulator for all gemms.
        # fmt: off
        from miniTransformer.quantiztion.nvfp4_quantize_op import Nvfp4TiledQuantizeOp
        return QLinearParams(
            quantize=True,
            quantize_type=QuantizeType.NVFP4,
            x_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=not env.should_disable_rht(),
                stochastic_rounding=False,
                param_type=ParamType.X
            ),
            w_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(16, 16),
                with_2d_quantization=not env.should_disable_2d_quantization(),
                with_rht=False,
                stochastic_rounding = False,
                param_type=ParamType.WEIGHT
            ),
            g_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=not env.should_disable_rht(),
                stochastic_rounding=not env.should_disable_stochastic_rounding(),
                param_type=ParamType.Y
            ),
            quantize_op=Nvfp4TiledQuantizeOp(),
            allgather_quantize=False,
        )
        # fmt: on
    elif recipe == QuantizeRecipe.FP4_SUB_CHANNEL_PERTOKEN:
        # TE PR #3045 per-token NVFP4 (rowwise act + per_token_weight_2d weight).
        from miniTransformer.quantiztion.nvfp4_quantize_op_pertoken import Nvfp4TiledQuantizePertokenOp
        return QLinearParams(
            quantize=True,
            quantize_type=QuantizeType.NVFP4,
            x_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=False,
                stochastic_rounding=False,
                param_type=ParamType.X,
            ),
            w_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(16, 16),
                with_2d_quantization=False,
                with_rht=False,
                stochastic_rounding=False,
                param_type=ParamType.WEIGHT,
            ),
            g_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=False,
                stochastic_rounding=False,
                param_type=ParamType.Y,
            ),
            quantize_op=Nvfp4TiledQuantizePertokenOp(),
            allgather_quantize=False,
        )
    elif recipe == QuantizeRecipe.FP4_SUB_CHANNEL_PERTOKEN_W_PERTENSOR:
        # Temporary ablation (QAT16): QAT15 act per-token + weight per-expert
        # tensor amax broadcast into TE bulk (QAT8-like outer scale).
        from miniTransformer.quantiztion.nvfp4_quantize_op_pertoken_w_pertensor import Nvfp4PertokenWeightPertensorOp
        return QLinearParams(
            quantize=True,
            quantize_type=QuantizeType.NVFP4,
            x_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=False,
                stochastic_rounding=False,
                param_type=ParamType.X,
            ),
            w_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(16, 16),
                with_2d_quantization=False,
                with_rht=False,
                stochastic_rounding=False,
                param_type=ParamType.WEIGHT,
            ),
            g_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=False,
                stochastic_rounding=False,
                param_type=ParamType.Y,
            ),
            quantize_op=Nvfp4PertokenWeightPertensorOp(),
            allgather_quantize=False,
        )
    elif recipe == QuantizeRecipe.FP4_SUB_CHANNEL_PERTOKEN_BF16_BWD:
        # NVFP4 per-token forward (same as QAT15) + TE-style BF16 dgrad/wgrad.
        # data_t stores BF16 splits (no physical transpose); bwd uses layout NN/NT.
        from miniTransformer.quantiztion.nvfp4_quantize_op_pertoken_bf16_bwd import Nvfp4PertokenBf16BackwardOp
        return QLinearParams(
            quantize=True,
            quantize_type=QuantizeType.NVFP4,
            x_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=False,
                stochastic_rounding=False,
                param_type=ParamType.X,
            ),
            w_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(16, 16),
                with_2d_quantization=False,
                with_rht=False,
                stochastic_rounding=False,
                param_type=ParamType.WEIGHT,
            ),
            g_params=QParams(
                output_dtype=torch.bfloat16,
                scaling_dtype=torch.bfloat16,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=False,
                stochastic_rounding=False,
                param_type=ParamType.Y,
            ),
            quantize_op=Nvfp4PertokenBf16BackwardOp(),
            allgather_quantize=False,
        )
    elif recipe == QuantizeRecipe.FP4_SUB_CHANNEL_CURSOR2:
        # Pertoken activation (Triton) + tensor-global weight + CuTeDSL grouped GEMM (SM100).
        from miniTransformer.quantiztion.nvfp4_cursor2_op import Nvfp4Cursor2QuantizeOp
        return QLinearParams(
            quantize=True,
            quantize_type=QuantizeType.NVFP4,
            x_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=not env.should_disable_rht(),
                stochastic_rounding=False,
                param_type=ParamType.X,
            ),
            w_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(16, 16),
                with_2d_quantization=False,
                with_rht=False,
                stochastic_rounding=False,
                param_type=ParamType.WEIGHT,
            ),
            g_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=not env.should_disable_rht(),
                stochastic_rounding=not env.should_disable_stochastic_rounding(),
                param_type=ParamType.Y,
            ),
            quantize_op=Nvfp4Cursor2QuantizeOp(),
            allgather_quantize=False,
        )
    elif recipe == QuantizeRecipe.FP4_SUB_CHANNEL_CURSOR2_REF:
        # Pertoken activation + tensor-global weight + CuTeDSL grouped GEMM (SM100).
        from miniTransformer.quantiztion.nvfp4_cursor2_op_ref import Nvfp4Cursor2QuantizeOpRef
        return QLinearParams(
            quantize=True,
            quantize_type=QuantizeType.NVFP4,
            x_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=not env.should_disable_rht(),
                stochastic_rounding=False,
                param_type=ParamType.X,
            ),
            w_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(16, 16),
                with_2d_quantization=False,
                with_rht=False,
                stochastic_rounding=False,
                param_type=ParamType.WEIGHT,
            ),
            g_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=not env.should_disable_rht(),
                stochastic_rounding=not env.should_disable_stochastic_rounding(),
                param_type=ParamType.Y,
            ),
            quantize_op=Nvfp4Cursor2QuantizeOpRef(),
            allgather_quantize=False,
        )
    elif recipe == QuantizeRecipe.FP4_SUB_CHANNEL_CURSOR2_CUTLASS:
        # Pertoken activation + tensor-global weight + CUTLASS v2 grouped FP4 GEMM (SM100).
        from miniTransformer.quantiztion.nvfp4_cursor2_op_cutlass import Nvfp4Cursor2CutlassQuantizeOp
        return QLinearParams(
            quantize=True,
            quantize_type=QuantizeType.NVFP4,
            x_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=not env.should_disable_rht(),
                stochastic_rounding=False,
                param_type=ParamType.X,
            ),
            w_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(16, 16),
                with_2d_quantization=False,
                with_rht=False,
                stochastic_rounding=False,
                param_type=ParamType.WEIGHT,
            ),
            g_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=not env.should_disable_rht(),
                stochastic_rounding=not env.should_disable_stochastic_rounding(),
                param_type=ParamType.Y,
            ),
            quantize_op=Nvfp4Cursor2CutlassQuantizeOp(),
            allgather_quantize=False,
        )
    elif recipe == QuantizeRecipe.FP4_SUB_CHANNEL_CURSOR2_CUTLASS_BWD:
        # CUTLASS v2 FP4 forward GEMM + MXFP8 CUTLASS backward dgrad (SM100).
        from miniTransformer.quantiztion.nvfp4_cursor2_op_cutlass_backward import Nvfp4Cursor2CutlassBackwardQuantizeOp
        return QLinearParams(
            quantize=True,
            quantize_type=QuantizeType.NVFP4,
            x_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=not env.should_disable_rht(),
                stochastic_rounding=False,
                param_type=ParamType.X,
            ),
            w_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(16, 16),
                with_2d_quantization=False,
                with_rht=False,
                stochastic_rounding=False,
                param_type=ParamType.WEIGHT,
            ),
            g_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=not env.should_disable_rht(),
                stochastic_rounding=not env.should_disable_stochastic_rounding(),
                param_type=ParamType.Y,
            ),
            quantize_op=Nvfp4Cursor2CutlassBackwardQuantizeOp(),
            allgather_quantize=False,
        )
    elif recipe == QuantizeRecipe.FP4_SUB_CHANNEL_CURSOR2_CUTLASS_V2:
        # CUTLASS v2 FP4 forward GEMM + MXFP8 CUTLASS backward dgrad (SM100).
        from miniTransformer.quantiztion.nvfp4_cursor2_op_cutlass_v2 import Nvfp4Cursor2CutlassV2QuantizeOp
        return QLinearParams(
            quantize=True,
            quantize_type=QuantizeType.NVFP4,
            x_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=not env.should_disable_rht(),
                stochastic_rounding=False,
                param_type=ParamType.X,
            ),
            w_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(16, 16),
                with_2d_quantization=False,
                with_rht=False,
                stochastic_rounding=False,
                param_type=ParamType.WEIGHT,
            ),
            g_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=not env.should_disable_rht(),
                stochastic_rounding=not env.should_disable_stochastic_rounding(),
                param_type=ParamType.Y,
            ),
            quantize_op=Nvfp4Cursor2CutlassV2QuantizeOp(),
            allgather_quantize=False,
        )
    elif recipe == QuantizeRecipe.FP4_SUB_CHANNEL_W4A16:
        # Origin activation + tensor-global weight.
        from miniTransformer.quantiztion.nvfp4_w4a16_op import Nvfp4W4A16QuantizeOp
        return QLinearParams(
            quantize=True,
            quantize_type=QuantizeType.NVFP4,
            x_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=not env.should_disable_rht(),
                stochastic_rounding=False,
                param_type=ParamType.X,
            ),
            w_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(16, 16),
                with_2d_quantization=False,
                with_rht=False,
                stochastic_rounding=False,
                param_type=ParamType.WEIGHT,
            ),
            g_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=not env.should_disable_rht(),
                stochastic_rounding=not env.should_disable_stochastic_rounding(),
                param_type=ParamType.Y,
            ),
            quantize_op=Nvfp4W4A16QuantizeOp(),
            allgather_quantize=False,
        )
    elif recipe == QuantizeRecipe.FP4_SUB_CHANNEL_QUARTET:
        from miniTransformer.quantiztion.nvfp4_quartet_op import Nvfp4QuartetQuantizeOp
        return QLinearParams(
            quantize=True,
            quantize_type=QuantizeType.NVFP4,
            x_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=True,
                param_type=ParamType.X
            ),
            w_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=True,
                with_rht=False,
                param_type=ParamType.WEIGHT
            ),
            g_params=QParams(
                output_dtype=torch.uint8,
                scaling_dtype=torch.float8_e4m3fn,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=True,
                param_type=ParamType.Y
            ),
            quantize_op=Nvfp4QuartetQuantizeOp(),
            allgather_quantize=False,
        )
    elif recipe == QuantizeRecipe.FP4_QDQ_BLOCK:
        from miniTransformer.quantiztion.nvfp4_qdq_op import Nvfp4QdqOp
        return QLinearParams(
            quantize=True,
            quantize_type=QuantizeType.NVFP4,
            x_params=QParams(
                output_dtype=torch.bfloat16,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=False,
                stochastic_rounding=False,
                param_type=ParamType.X,
            ),
            w_params=QParams(
                output_dtype=torch.bfloat16,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=False,
                stochastic_rounding=False,
                param_type=ParamType.WEIGHT,
            ),
            g_params=QParams(
                output_dtype=torch.bfloat16,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=False,
                stochastic_rounding=False,
                param_type=ParamType.Y,
            ),
            quantize_op=Nvfp4QdqOp(mode="block"),
            allgather_quantize=False,
        )
    elif recipe == QuantizeRecipe.FP4_QDQ_PERTOKEN:
        from miniTransformer.quantiztion.nvfp4_qdq_op import Nvfp4QdqOp
        return QLinearParams(
            quantize=True,
            quantize_type=QuantizeType.NVFP4,
            x_params=QParams(
                output_dtype=torch.bfloat16,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=False,
                stochastic_rounding=False,
                param_type=ParamType.X,
            ),
            w_params=QParams(
                output_dtype=torch.bfloat16,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=False,
                stochastic_rounding=False,
                param_type=ParamType.WEIGHT,
            ),
            g_params=QParams(
                output_dtype=torch.bfloat16,
                quant_tile_shape=(1, 16),
                with_2d_quantization=False,
                with_rht=False,
                stochastic_rounding=False,
                param_type=ParamType.Y,
            ),
            quantize_op=Nvfp4QdqOp(mode="pertoken"),
            allgather_quantize=False,
        )
    else:
        raise ValueError(f"Unsupported quantize recipe: {recipe}")


_logged_recipes = set()


def _log_recipe_selection(spec, source: str) -> None:
    """One concise summary per effective recipe in the main process."""
    if os.environ.get("RANK", "0") != "0":
        return
    description = describe_recipe(spec, source=source)
    key = (spec.name, source, description["backward"], tuple(description["environment"].items()))
    if key in _logged_recipes:
        return
    _logged_recipes.add(key)
    if source == "default":
        logger.warning(
            "MINITE_RECIPE and QAT_PARAMS are unset. Using historical fallback "
            "fp4_subchannel_scaling_cursor2_ref (QAT6). Select a recipe explicitly."
        )
    logger.info("miniTransformer: %s", recipe_summary(description))


def get_qlinear_params(recipe: Union[str, int, QuantizeRecipe]) -> QLinearParams:
    """Create independent parameters/operators from a name, enum, or legacy QAT ID.

    Existing environment tuning flags still apply. This function neither changes
    environment variables nor replaces TE layers.
    """
    spec = get_recipe(recipe)
    params = get_qlinear_params_from_predefined(QuantizeRecipe(spec.enum_value))
    _log_recipe_selection(spec, "explicit")
    return params


def get_qlinear_params_from_env() -> QLinearParams:
    """Resolve MINITE_RECIPE / QAT_PARAMS with the historical QAT6 fallback."""
    spec, source = get_recipe_from_env()
    params = get_qlinear_params_from_predefined(QuantizeRecipe(spec.enum_value))
    _log_recipe_selection(spec, source)
    return params


def get_qlinear_params_from_env_qat_params() -> QLinearParams:
    """Compatibility entry point; also accepts the new MINITE_RECIPE selector."""
    return get_qlinear_params_from_env()
