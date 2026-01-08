import dataclasses
import enum
import logging
import os
import torch
from typing import Dict, Optional, Mapping, Sequence, Type

from miniTransformer.ops.nvfp4_quantize_op import Nvfp4TiledQuantizeOp
from miniTransformer.ops.nvfp4_quantize_op_ref import Nvfp4TiledQuantizeRefOp
from ..ops.quantization import QParams, MMParams
from miniTransformer.ops import quantization

logger = logging.getLogger(__name__)

@enum.unique
class QuantizeType(enum.Enum):
    """quantization type for linear layers."""

    # Non-quantize
    NONE = "none"
    FP8 = "fp8"
    MXFP8 = "mxfp8"
    INT4 = "int4"
    NVFP4 = "nvfp4"

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

    # Non-quantize
    NON_QUANTIZE = "non_quantize"
    INT4_SUB_CHANNEL = "int4_subchannel_scaling"
    FP4_SUB_CHANNEL = "fp4_subchannel_scaling"
    FP4_SUB_CHANNEL_REF = "fp4_subchannel_scaling_ref"


def get_qlinear_params_from_predefined(recipe: QuantizeRecipe) -> QLinearParams:
    """Get quantization parameters for linear layer based on recipe."""
    if recipe == QuantizeRecipe.NON_QUANTIZE:
        # Non-quantize (bf16)
        return QLinearParams(quantize=False, quantize_type=QuantizeType.NONE)
    elif recipe == QuantizeRecipe.FP4_SUB_CHANNEL:
        # Split accumulator for all gemms.
        # fmt: off
        return QLinearParams(
            quantize=True,
            quantize_type=QuantizeType.NVFP4,
            x_params=QParams(output_dtype=torch.uint8, scaling_dtype=torch.float8_e4m3fn, quant_tile_shape=(1, 16),  with_2d_quantization=False),
            w_params=QParams(output_dtype=torch.uint8, scaling_dtype=torch.float8_e4m3fn, quant_tile_shape=(16, 16), with_2d_quantization=True),
            g_params=QParams(output_dtype=torch.uint8, scaling_dtype=torch.float8_e4m3fn, quant_tile_shape=(1, 16),  with_2d_quantization=False),
            quantize_op=Nvfp4TiledQuantizeOp(),
            allgather_quantize=False,
        )
        # fmt: on
    elif recipe == QuantizeRecipe.FP4_SUB_CHANNEL_REF:
        return QLinearParams(
            quantize=True,
            quantize_type=QuantizeType.NVFP4,
            x_params=QParams(output_dtype=torch.uint8, scaling_dtype=torch.float8_e4m3fn, quant_tile_shape=(1, 16),  with_2d_quantization=False),
            w_params=QParams(output_dtype=torch.uint8, scaling_dtype=torch.float8_e4m3fn, quant_tile_shape=(16, 16), with_2d_quantization=True),
            g_params=QParams(output_dtype=torch.uint8, scaling_dtype=torch.float8_e4m3fn, quant_tile_shape=(1, 16),  with_2d_quantization=False),
            quantize_op=Nvfp4TiledQuantizeRefOp(),
            allgather_quantize=False,
        )
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
    if qat_params_idx == 2:
        # default pow_2_scales=False
        return get_qlinear_params_from_predefined(
            QuantizeRecipe.INT4_SUB_CHANNEL
        )
    if qat_params_idx == 3:
		# default pow_2_scales=False
        return get_qlinear_params_from_predefined(
			QuantizeRecipe.FP4_SUB_CHANNEL_REF
		)

    raise ValueError(f"Unsupported QAT_PARAMS index: {qat_params_idx}")
