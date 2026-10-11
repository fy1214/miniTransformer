"""Recipe names and inspection metadata, with no Torch or CUDA imports.

The numerical settings remain in module.config. This registry is the single
mapping between public names, historical QAT numbers, and QuantizeRecipe values.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from enum import Enum
import os
from typing import Mapping


@dataclass(frozen=True)
class RecipeSpec:
    name: str
    qat_id: int | None
    enum_value: str
    group: str
    description: str
    operator: str | None
    forward: str
    backward: str
    requirements: tuple[str, ...] = ()


_OPS = "miniTransformer.quantiztion."
_BLACKWELL = ("CUDA GPU compatible with the selected Blackwell kernels", "Transformer Engine NVFP4/MXFP8 interfaces")
_PER_TOKEN = (
    "Transformer Engine NVFP4Quantizer(per_token=True, per_token_weight_2d=...)",
    "TE per-token grouped quantization and GEMM bindings (PR #3045 build)",
)

# Keep QAT numbers fixed: they are used by existing experiments and launchers.
RECIPES = (
    RecipeSpec("unquantized", 1, "non_quantize", "common",
               "Quantization disabled; computation dtype is controlled by the layer.",
               None, "Layer dtype", "Layer dtype"),
    RecipeSpec("nvfp4_block", 4, "fp4_subchannel_scaling", "common",
               "Existing NVFP4 block-scaling recipe; RHT and scaling toggles remain available.",
               _OPS + "nvfp4_quantize_op.Nvfp4TiledQuantizeOp", "NVFP4 block scaling", "NVFP4 block scaling", _BLACKWELL),
    RecipeSpec("nvfp4_cursor2_cutlass_v2", 14, "fp4_subchannel_scaling_cursor2_cutlass_v2", "common",
               "Per-token activations, per-expert tensor weight amax, CUTLASS V2 grouped forward GEMM.",
               _OPS + "nvfp4_cursor2_op_cutlass_v2.Nvfp4Cursor2CutlassV2QuantizeOp",
               "NVFP4 / CUTLASS V2", "MXFP8", _BLACKWELL),
    RecipeSpec("nvfp4_per_token", 15, "fp4_subchannel_scaling_pertoken", "common",
               "Per-token NVFP4 forward with TE weight scaling and MXFP8 backward. First-use example.",
               _OPS + "nvfp4_quantize_op_pertoken.Nvfp4TiledQuantizePertokenOp",
               "NVFP4 per-token", "MXFP8", _BLACKWELL + _PER_TOKEN),
    RecipeSpec("nvfp4_per_token_bf16_bwd", 17, "fp4_subchannel_scaling_pertoken_bf16_bwd", "common",
               "Per-token NVFP4 forward with the dedicated TE-style BF16 backward operator.",
               _OPS + "nvfp4_quantize_op_pertoken_bf16_bwd.Nvfp4PertokenBf16BackwardOp",
               "NVFP4 per-token", "BF16 dedicated operator", _BLACKWELL + _PER_TOKEN),
    RecipeSpec("bf16_ref", None, "bf16_ref", "reference",
               "BF16 reference operator; distinct from disabling quantization.",
               _OPS + "bf16_ref_op.Bfloat16RefOp", "BF16 reference", "Operator-defined", ("fp4_gemm extension",)),
    RecipeSpec("fp4_subchannel_scaling_ref", 3, "fp4_subchannel_scaling_ref", "reference",
               "Reference implementation of NVFP4 block scaling.",
               _OPS + "nvfp4_quantize_op_ref.Nvfp4TiledQuantizeRefOp",
               "NVFP4 reference", "Operator-defined", _BLACKWELL),
    RecipeSpec("fp4_subchannel_scaling_cursor2_ref", 6, "fp4_subchannel_scaling_cursor2_ref", "reference",
               "Cursor2 reference path; historical fallback when neither selector is set.",
               _OPS + "nvfp4_cursor2_op_ref.Nvfp4Cursor2QuantizeOpRef",
               "NVFP4 Cursor2 reference", "MXFP8", _BLACKWELL),
    RecipeSpec("int4_subchannel_scaling", 2, "int4_subchannel_scaling", "experimental",
               "INT4 fake quantization; group size is set by INT4_FAKE_GROUP_SIZE (default 16).",
               _OPS + "int4_quantize_op.Int4TiledQuantizeOp",
               "INT4 fake quantization / high-precision GEMM", "Operator-defined"),
    RecipeSpec("fp4_subchannel_scaling_quartet", 5, "fp4_subchannel_scaling_quartet", "experimental",
               "Quartet NVFP4 recipe with its existing rotation and scaling settings.",
               _OPS + "nvfp4_quartet_op.Nvfp4QuartetQuantizeOp",
               "NVFP4 quartet", "Operator-defined", _BLACKWELL),
    RecipeSpec("fp4_subchannel_scaling_cursor2", 7, "fp4_subchannel_scaling_cursor2", "experimental",
               "Cursor2 path: per-token activations and tensor-global weight amax.",
               _OPS + "nvfp4_cursor2_op.Nvfp4Cursor2QuantizeOp",
               "NVFP4 Cursor2", "MXFP8", _BLACKWELL),
    RecipeSpec("fp4_subchannel_scaling_cursor2_cutlass", 8, "fp4_subchannel_scaling_cursor2_cutlass", "experimental",
               "Earlier Cursor2 CUTLASS path; retained for existing experiments.",
               _OPS + "nvfp4_cursor2_op_cutlass.Nvfp4Cursor2CutlassQuantizeOp",
               "NVFP4 Cursor2 / CUTLASS", "Operator-defined", _BLACKWELL),
    RecipeSpec("fp4_subchannel_scaling_w4a16", 9, "fp4_subchannel_scaling_w4a16", "experimental",
               "Existing W4A16 forward path with high-precision activations.",
               _OPS + "nvfp4_w4a16_op.Nvfp4W4A16QuantizeOp",
               "NVFP4 W4A16", "Operator-defined", _BLACKWELL),
    RecipeSpec("fp4_qdq_block", 10, "fp4_qdq_block", "experimental",
               "NVFP4 block quantize/dequantize forward experiment.",
               _OPS + "nvfp4_qdq_op.Nvfp4QdqOp",
               "NVFP4 block QDQ / BF16 GEMM", "Operator-defined", _BLACKWELL),
    RecipeSpec("fp4_qdq_pertoken", 11, "fp4_qdq_pertoken", "experimental",
               "NVFP4 per-token quantize/dequantize forward experiment.",
               _OPS + "nvfp4_qdq_op.Nvfp4QdqOp",
               "NVFP4 per-token QDQ / BF16 GEMM", "Operator-defined", _BLACKWELL),
    RecipeSpec("fp4_subchannel_scaling_cursor2_cutlass_bwd", 12, "fp4_subchannel_scaling_cursor2_cutlass_bwd", "experimental",
               "Cursor2 CUTLASS forward with the MXFP8 CUTLASS dgrad experiment.",
               _OPS + "nvfp4_cursor2_op_cutlass_backward.Nvfp4Cursor2CutlassBackwardQuantizeOp",
               "NVFP4 / CUTLASS", "MXFP8 / CUTLASS dgrad", _BLACKWELL),
    RecipeSpec("int4_marlin_forward", 13, "int4_marlin_forward", "experimental",
               "Packed INT4 Marlin forward; GroupedLinear requires DISABLE_BACKWARD_QUANT=1.",
               _OPS + "int4_quantize_op.Int4MarlinForwardQuantizeOp",
               "INT4 Marlin W4A16", "BF16 STE for GroupedLinear", ("SGLang Marlin interfaces",)),
    RecipeSpec("nvfp4_per_token_w_per_tensor", 16, "fp4_subchannel_scaling_pertoken_w_pertensor", "experimental",
               "QAT15 activation path with per-expert tensor weight amax; ablation recipe.",
               _OPS + "nvfp4_quantize_op_pertoken_w_pertensor.Nvfp4PertokenWeightPertensorOp",
               "NVFP4 per-token / tensor weight amax", "MXFP8", _BLACKWELL + _PER_TOKEN),
)

_BY_ID = {recipe.qat_id: recipe for recipe in RECIPES if recipe.qat_id is not None}
_BY_NAME = {name: recipe for recipe in RECIPES for name in (recipe.name, recipe.enum_value)}
DEFAULT_RECIPE = _BY_ID[6]
GROUPS = ("common", "reference", "experimental")
RECIPE_ENV_VARS = (
    "MINITE_RECIPE", "QAT_PARAMS", "DISABLE_BACKWARD_QUANT",
    "MINITE_NVFP4_PATH", "QAT_KEEP_IN_BF16_LAYER_NUMBERS",
    "NVTE_NVFP4_DISABLE_RHT", "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING",
    "NVTE_NVFP4_DISABLE_2D_QUANTIZATION", "INT4_FAKE_GROUP_SIZE",
    "OPEN_TRAINING_INT4_GROUP_SIZE", "INT4_REAL_GROUP_SIZE", "INT4_REAL_SYMMETRIC",
)


def get_recipe(recipe: str | int | Enum) -> RecipeSpec:
    """Resolve a public name, original enum value, enum instance, or QAT number."""
    if isinstance(recipe, Enum):
        recipe = recipe.value
    if isinstance(recipe, int) and not isinstance(recipe, bool):
        if recipe in _BY_ID:
            return _BY_ID[recipe]
        raise ValueError(f"Unsupported QAT_PARAMS index: {recipe}. Expected 1–17.")
    if isinstance(recipe, str):
        name = recipe.strip().lower()
        if name.isdecimal():
            return get_recipe(int(name))
        if name in _BY_NAME:
            return _BY_NAME[name]
        suggestions = get_close_matches(name, _BY_NAME, n=2)
        hint = f" Did you mean {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"Unknown recipe {recipe!r}.{hint} Run 'python -m miniTransformer recipes --all'.")
    raise ValueError(f"Expected a recipe name, QuantizeRecipe, or QAT integer; got {recipe!r}.")


def get_recipe_from_env(environ: Mapping[str, str] | None = None) -> tuple[RecipeSpec, str]:
    """Resolve environment selectors, rejecting ambiguous selections."""
    environ = os.environ if environ is None else environ
    named = environ.get("MINITE_RECIPE", "").strip()
    legacy = environ.get("QAT_PARAMS", "").strip()
    named_recipe = get_recipe(named) if named else None
    legacy_recipe = None
    if legacy:
        try:
            index = int(legacy)
        except ValueError as exc:
            raise ValueError(f"QAT_PARAMS must be an integer from 1 to 17, got {legacy!r}.") from exc
        legacy_recipe = get_recipe(index)
    if named_recipe and legacy_recipe and named_recipe != legacy_recipe:
        raise ValueError(
            f"Conflicting selectors: MINITE_RECIPE={named!r} selects {named_recipe.name}, "
            f"but QAT_PARAMS={legacy!r} selects {legacy_recipe.name}. "
            "Unset one selector or make them select the same recipe."
        )
    if named_recipe:
        return named_recipe, "MINITE_RECIPE"
    if legacy_recipe:
        return legacy_recipe, "QAT_PARAMS"
    return DEFAULT_RECIPE, "default"


def describe_recipe(recipe: str | int | Enum | RecipeSpec,
                    environ: Mapping[str, str] | None = None,
                    *, source: str = "explicit") -> dict:
    """Describe configured behavior; dependency/layout availability needs doctor."""
    spec = recipe if isinstance(recipe, RecipeSpec) else get_recipe(recipe)
    environ = os.environ if environ is None else environ
    try:
        disable_backward = int(environ.get("DISABLE_BACKWARD_QUANT", "0")) == 1
    except ValueError as exc:
        raise ValueError("DISABLE_BACKWARD_QUANT must be an integer (use 0 or 1).") from exc
    backward = spec.backward
    backward_source = "recipe"
    notes = []
    if disable_backward and spec.qat_id != 1:
        backward = "BF16 STE via layer bypass"
        backward_source = "DISABLE_BACKWARD_QUANT"
        if spec.qat_id == 17:
            notes.append("Unset DISABLE_BACKWARD_QUANT to use QAT17's dedicated backward operator.")
    if spec.qat_id == 13 and not disable_backward:
        notes.append("GroupedLinear requires DISABLE_BACKWARD_QUANT=1 for this recipe.")
    layout = None
    if spec.qat_id in (15, 16, 17):
        layout = environ.get("MINITE_NVFP4_PATH", "auto").strip().lower()
        if layout not in ("auto", "dense", "list", "legacy", "0", "false", "off", "1", "true", "on"):
            # The legacy runtime treats other values as auto; do not change it.
            notes.append(f"Legacy runtime interprets MINITE_NVFP4_PATH={layout!r} as auto; use auto, dense, or list.")
        notes.append("RHT and stochastic rounding are disabled by these per-token recipes; NVTE toggles do not enable them.")
        notes.append("Layout availability is checked by doctor; 'auto' is resolved by TE at runtime.")
    return {
        "name": spec.name, "qat_id": spec.qat_id, "enum_value": spec.enum_value,
        "group": spec.group, "description": spec.description,
        "source": source, "operator": spec.operator,
        "forward": spec.forward, "backward": backward,
        "backward_source": backward_source, "layout_requested": layout,
        "requirements": ["PyTorch with CUDA", "Transformer Engine", *spec.requirements],
        "environment": {name: environ[name] for name in RECIPE_ENV_VARS if name in environ},
        "notes": notes,
    }


def recipe_summary(description: dict) -> str:
    qat = description["qat_id"]
    legacy = f"QAT{qat}" if qat is not None else "no legacy QAT ID"
    layout = f"; layout={description['layout_requested']}" if description["layout_requested"] else ""
    return (f"{description['name']} ({legacy}; source={description['source']}): "
            f"forward={description['forward']}; backward={description['backward']} "
            f"(source={description['backward_source']}){layout}")
