import dataclasses
import enum
import logging
import os
import torch
from typing import Dict, Optional, Mapping, Sequence, Type

from transformer_engine.utils import ops
from transformer_engine.utils import quantization
from transformer_engine.utils import quantization_subchannel_block_hybrid
from .quantization import QParams, MMParams, ScalingType

logger = logging.getLogger(__name__)


def should_switch_to_hybrid_linear():
    # Switch to kitchen linear if QAT_PARAMS is set
    qat_params_idx = int(os.getenv("QAT_PARAMS", "0"))
    should_switch = qat_params_idx > 0
    return should_switch


def get_keep_in_bf16_layer_numbers():
    keep_in_bf16_layer_numbers = os.getenv("QAT_KEEP_IN_BF16_LAYER_NUMBERS", None)
    if keep_in_bf16_layer_numbers is None:
        return []
    else:
        return [int(s) for s in keep_in_bf16_layer_numbers.split(",")]


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

    quantize: bool = True  # JQ: fp8 or bf16

    x_params: QParams = QParams()
    w_params: QParams = QParams()
    g_params: QParams = QParams()

    mm_fprop: MMParams = MMParams()
    mm_dgrad: MMParams = MMParams()
    mm_wgrad: MMParams = MMParams()

    allgather_fp8: bool = False  # JQ: needed
    memory_saving: bool = False

    quantize_op: Optional[quantization.QuantizeOpBase] = None  # JQ: needed
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
            # If quantize_op is provided, ensure that scaling types are supported.
            supported_scaling_types = self.quantize_op.supported_scaling_types
            assert (
                self.x_params.scaling_type in supported_scaling_types
            ), "Unsupported scaling type."
            assert (
                self.w_params.scaling_type in supported_scaling_types
            ), "Unsupported scaling type."
            assert (
                self.g_params.scaling_type in supported_scaling_types
            ), "Unsupported scaling type"
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

    # Non-quantize
    NON_QUANTIZE = "non_quantize"
    FP4_SUB_CHANNEL_CUTLASS = "fp4_subchannel_scaling"


def get_qlinear_params_from_predefined(recipe: QuantizeRecipe) -> QLinearParams:
    """Get quantization parameters for linear layer based on recipe."""
    if recipe == QuantizeRecipe.NON_QUANTIZE:
        # Non-quantize (bf16)
        return QLinearParams(quantize=False)
    elif recipe == QuantizeRecipe.FP4_SUB_CHANNEL_CUTLASS:
        _scaling_type = ScalingType.VECTOR_TILED_X_AND_G_BLOCK_TILED_W
        # Split accumulator for all gemms.
        # fmt: off
        return QLinearParams(
            x_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type, quant_tile_shape=(1, 16)),
            w_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type, quant_tile_shape=(16, 16)),
            g_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type, quant_tile_shape=(1, 16)),
            quantize_op=quantization_subchannel_block_hybrid.HybridBlockAndVectorTiledQuantizeOp(ops.Backend.CUTLASS),
            allgather_fp8=False,
        )
        # fmt: on
    else:
        raise ValueError(f"Unsupported quantize recipe: {recipe}")


def get_qlinear_params_from_env_qat_params() -> QLinearParams:
    if os.environ.get("QAT_PARAMS"):
        qat_params_idx = int(os.getenv("QAT_PARAMS"))
    else:
        logger.warning("QAT_PARAMS is not set. Use QAT_PARAMS=6, which uses DeepGemm backend")
        qat_params_idx = 6

    if qat_params_idx == 1:
        return get_qlinear_params_from_predefined(QuantizeRecipe.NON_QUANTIZE)
    if qat_params_idx == 4:
        # default pow_2_scales=False
        return get_qlinear_params_from_predefined(
            QuantizeRecipe.FP8_SUB_CHANNEL_CUTLASS
        )
    if qat_params_idx == 5:
		# default pow_2_scales=False
        return get_qlinear_params_from_predefined(
			QuantizeRecipe.FP8_SUB_CHANNEL_CUBLAS
		)
    def recipe_5001():
        recipe = get_qlinear_params_from_predefined(
            QuantizeRecipe.FP8_SUB_CHANNEL_CUBLAS
        )
        # NOTE: Replicates the choice to use power two scales based on DeepSeekV3
        # dataflow diagram in tensors were stored quantized and double quantized
        # to obtain the transpose. Kitchen is not double quantizing, but this
        # recipe can apply the scale factor modification to match.
        proj_and_fc1_recipe = get_qlinear_params_from_predefined(
            QuantizeRecipe.FP8_SUB_CHANNEL_CUBLAS
        )
        proj_and_fc1_recipe.x_params = dataclasses.replace(
            proj_and_fc1_recipe.x_params, pow_2_scales=True
        )
        fc2_recipe = get_qlinear_params_from_predefined(
            QuantizeRecipe.FP8_SUB_CHANNEL_CUBLAS
        )
        fc2_recipe.g_params = dataclasses.replace(
            fc2_recipe.g_params, pow_2_scales=True
        )
        recipe.overrides_for_layer_name = {
            "proj": proj_and_fc1_recipe,
            "fc1": proj_and_fc1_recipe,
            "fc2": fc2_recipe,
        }
        return recipe

    if qat_params_idx == 5001:
        recipe = recipe_5001()
        return recipe
    if qat_params_idx == 5002:
        recipe = recipe_5001()
        # Same as 5001, but with memory saving mode.
        recipe.memory_saving = True
        for _, overrided_recipe in recipe.overrides_for_layer_name.items():
            overrided_recipe.memory_saving = True
        return recipe
    if qat_params_idx == 5003:
        # Same as 5001, but with pow_2_scales on all layers.
        recipe = get_qlinear_params_from_predefined(
            QuantizeRecipe.FP8_SUB_CHANNEL_CUBLAS
        )
        recipe.x_params = dataclasses.replace(recipe.x_params, pow_2_scales=True)
        recipe.w_params = dataclasses.replace(recipe.w_params, pow_2_scales=True)
        recipe.g_params = dataclasses.replace(recipe.g_params, pow_2_scales=True)
        return recipe
    if qat_params_idx == 6:
        # default pow_2_scales=False
        return get_qlinear_params_from_predefined(
            QuantizeRecipe.FP8_SUB_CHANNEL_DEEPGEMM
        )
    if qat_params_idx == 6001:
        recipe = get_qlinear_params_from_predefined(
            QuantizeRecipe.FP8_SUB_CHANNEL_DEEPGEMM
        )
        # NOTE: Replicates the choice to use power two scales based on DeepSeekV3
        # dataflow diagram in tensors were stored quantized and double quantized
        # to obtain the transpose. Kitchen is not double quantizing, but this
        # recipe can apply the scale factor modification to match.
        proj_and_fc1_recipe = get_qlinear_params_from_predefined(
            QuantizeRecipe.FP8_SUB_CHANNEL_DEEPGEMM
        )
        proj_and_fc1_recipe.x_params = dataclasses.replace(
            proj_and_fc1_recipe.x_params, pow_2_scales=True
        )
        fc2_recipe = get_qlinear_params_from_predefined(
            QuantizeRecipe.FP8_SUB_CHANNEL_DEEPGEMM
        )
        fc2_recipe.g_params = dataclasses.replace(
            fc2_recipe.g_params, pow_2_scales=True
        )
        recipe.overrides_for_layer_name = {
            "proj": proj_and_fc1_recipe,
            "fc1": proj_and_fc1_recipe,
            "fc2": fc2_recipe,
        }
        return recipe
    else:
        raise ValueError(f"Unsupported QAT_PARAMS index: {qat_params_idx}")
