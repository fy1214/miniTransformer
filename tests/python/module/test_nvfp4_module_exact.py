"""Compare miniTransformer Linear (NVFP4 recipes) against TE Linear reference paths."""

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.custom_recipes import quantization_nvfp4
from transformer_engine.pytorch.custom_recipes import utils

from miniTransformer.module.config import QuantizeRecipe, get_qlinear_params_from_predefined
from miniTransformer.module.linear import Linear

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)

_NATIVE_RECIPES = [
    pytest.param(QuantizeRecipe.FP4_SUB_CHANNEL, id="fp4_sub_channel"),
    pytest.param(QuantizeRecipe.FP4_SUB_CHANNEL_REF, id="fp4_sub_channel_ref"),
]


class GetRecipes:
    @staticmethod
    def nvfp4_vanilla():
        nvfp4_recipe = recipe.NVFP4BlockScaling()
        nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams()
        nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams()
        nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams()
        return nvfp4_recipe

    @staticmethod
    def nvfp4_rht_only():
        nvfp4_recipe = recipe.NVFP4BlockScaling()
        nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams(random_hadamard_transform=True)
        nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams(random_hadamard_transform=False)
        nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams(random_hadamard_transform=True)
        return nvfp4_recipe

    @staticmethod
    def nvfp4_2d_quantization_only():
        nvfp4_recipe = recipe.NVFP4BlockScaling()
        nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams(fp4_2d_quantization=False)
        nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams(fp4_2d_quantization=True)
        nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams(fp4_2d_quantization=False)
        return nvfp4_recipe

    @staticmethod
    def nvfp4_rht_and_2d_quantization():
        nvfp4_recipe = recipe.NVFP4BlockScaling()
        nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams(
            random_hadamard_transform=True, fp4_2d_quantization=False
        )
        nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams(
            random_hadamard_transform=False, fp4_2d_quantization=True
        )
        nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams(
            random_hadamard_transform=True, fp4_2d_quantization=False
        )
        return nvfp4_recipe

    @staticmethod
    def nvfp4_recipe_to_test(with_rht: bool = False, with_2d_quantization: bool = False):
        if with_rht and with_2d_quantization:
            return GetRecipes.nvfp4_rht_and_2d_quantization()
        if with_rht:
            return GetRecipes.nvfp4_rht_only()
        if with_2d_quantization:
            return GetRecipes.nvfp4_2d_quantization_only()
        return GetRecipes.nvfp4_vanilla()


def get_nvfp4_quantizer_factory(with_rht: bool = False, with_2d_quantization: bool = False):
    """Return a TE CustomRecipe quantizer factory for the reference Linear path."""

    def factory(role):
        if role == "linear_input":
            return quantization_nvfp4.NVFP4QuantizerRef(
                dtype=utils.Fp4Formats.E2M1,
                quant_tile_shape=(1, 16),
                pow_2_scales=False,
                with_rht=with_rht,
            )
        if role == "linear_weight":
            return quantization_nvfp4.NVFP4QuantizerRef(
                dtype=utils.Fp4Formats.E2M1,
                quant_tile_shape=(16, 16) if with_2d_quantization else (1, 16),
                pow_2_scales=False,
                with_rht=False,
            )
        if role == "linear_grad_output":
            return quantization_nvfp4.NVFP4QuantizerRef(
                dtype=utils.Fp4Formats.E2M1,
                quant_tile_shape=(1, 16),
                pow_2_scales=False,
                with_rht=with_rht,
            )
        return None

    return factory


def reset_rng_states():
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def check_nvfp4_linear_versus_reference(
    in_features: int,
    out_features: int,
    bias: bool,
    x_dtype: torch.dtype,
    native_recipe: QuantizeRecipe,
    num_steps: int = 1,
    with_rht: bool = False,
    with_2d_quantization: bool = False,
):
    """Run mini Linear vs TE Linear and compare output / grads."""
    device = "cuda"
    batch_size = 32
    seq_len = 128

    reset_rng_states()
    qlinear_params = get_qlinear_params_from_predefined(native_recipe)
    native_module = Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        device=device,
        params_dtype=x_dtype,
        qlinear_params=qlinear_params,
    )

    reset_rng_states()
    ref_module = te.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        device=device,
        params_dtype=x_dtype,
    )

    with torch.no_grad():
        ref_module.weight.copy_(native_module.weight)
        if bias:
            ref_module.bias.copy_(native_module.bias)

    nvfp4_recipe = GetRecipes.nvfp4_recipe_to_test(with_rht, with_2d_quantization)
    nvfp4_ref_recipe = recipe.CustomRecipe(
        qfactory=get_nvfp4_quantizer_factory(with_rht, with_2d_quantization)
    )

    native_outputs = []
    ref_outputs = []

    for step in range(num_steps):
        torch.manual_seed(1234 + step)
        torch.cuda.manual_seed(1234 + step)

        x_val = torch.normal(
            mean=0.0,
            std=1.0,
            size=(batch_size, seq_len, in_features),
            dtype=x_dtype,
            device=device,
        )
        x_native = x_val.clone().detach().requires_grad_(True)
        x_ref = x_native.clone().detach().requires_grad_(True)

        grad_output = torch.normal(
            mean=0.0,
            std=1.0,
            size=(batch_size, seq_len, out_features),
            dtype=x_dtype,
            device=device,
        ).detach()

        with te.autocast(enabled=True, recipe=nvfp4_recipe):
            y_native = native_module(x_native, is_first_microbatch=(step == 0))
        y_native.backward(grad_output)

        with te.autocast(enabled=True, recipe=nvfp4_ref_recipe):
            y_ref = ref_module(x_ref)
        y_ref.backward(grad_output)

        native_outputs.append(
            {
                "output": y_native.detach().clone(),
                "input_grad": (
                    x_native.grad.detach().clone() if x_native.grad is not None else None
                ),
                "weight_grad": (
                    native_module.weight.grad.detach().clone()
                    if native_module.weight.grad is not None
                    else None
                ),
                "bias_grad": (
                    native_module.bias.grad.detach().clone()
                    if bias and native_module.bias.grad is not None
                    else None
                ),
            }
        )
        ref_outputs.append(
            {
                "output": y_ref.detach().clone(),
                "input_grad": (x_ref.grad.detach().clone() if x_ref.grad is not None else None),
                "weight_grad": (
                    ref_module.weight.grad.detach().clone()
                    if ref_module.weight.grad is not None
                    else None
                ),
                "bias_grad": (
                    ref_module.bias.grad.detach().clone()
                    if bias and ref_module.bias.grad is not None
                    else None
                ),
            }
        )

    for step in range(num_steps):
        native_out = native_outputs[step]
        ref_out = ref_outputs[step]
        torch.testing.assert_close(
            native_out["output"],
            ref_out["output"],
            atol=1e-6,
            rtol=1e-6,
            msg=f"Output mismatch at step {step}",
        )
        torch.testing.assert_close(
            native_out["input_grad"],
            ref_out["input_grad"],
            atol=1e-6,
            rtol=1e-6,
            msg=f"Input gradient mismatch at step {step}",
        )
        torch.testing.assert_close(
            native_out["weight_grad"],
            ref_out["weight_grad"],
            atol=1e-6,
            rtol=1e-6,
            msg=f"Weight gradient mismatch at step {step}",
        )
        if bias and native_out["bias_grad"] is not None and ref_out["bias_grad"] is not None:
            torch.testing.assert_close(
                native_out["bias_grad"],
                ref_out["bias_grad"],
                atol=1e-6,
                rtol=1e-6,
                msg=f"Bias gradient mismatch at step {step}",
            )


def check_nvfp4_layernorm_linear_versus_reference(
    in_features: int,
    out_features: int,
    bias: bool,
    normalization: str,
    x_dtype: torch.dtype,
    num_steps: int = 1,
    with_rht: bool = False,
    with_2d_quantization: bool = False,
):
    """Compare TE LayerNormLinear NVFP4BlockScaling vs CustomRecipe reference."""
    device = "cuda"
    batch_size = 32
    seq_len = 128

    reset_rng_states()
    native_module = te.LayerNormLinear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        device=device,
        params_dtype=x_dtype,
        normalization=normalization,
        return_layernorm_output=True,
    )

    reset_rng_states()
    ref_module = te.LayerNormLinear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        device=device,
        params_dtype=x_dtype,
        normalization=normalization,
        return_layernorm_output=True,
    )

    with torch.no_grad():
        ref_module.weight.copy_(native_module.weight)
        if bias:
            ref_module.bias.copy_(native_module.bias)
        if native_module.layer_norm_weight is not None and ref_module.layer_norm_weight is not None:
            ref_module.layer_norm_weight.copy_(native_module.layer_norm_weight)
        if native_module.layer_norm_bias is not None and ref_module.layer_norm_bias is not None:
            ref_module.layer_norm_bias.copy_(native_module.layer_norm_bias)

    nvfp4_recipe = GetRecipes.nvfp4_recipe_to_test(with_rht, with_2d_quantization)
    nvfp4_ref_recipe = recipe.CustomRecipe(
        qfactory=get_nvfp4_quantizer_factory(with_rht, with_2d_quantization)
    )

    native_outputs = []
    ref_outputs = []

    for step in range(num_steps):
        torch.manual_seed(1234 + step)
        torch.cuda.manual_seed(1234 + step)

        x_val = torch.normal(
            mean=0.0,
            std=1.0,
            size=(batch_size, seq_len, in_features),
            dtype=x_dtype,
            device=device,
        )
        x_native = x_val.clone().detach().requires_grad_(True)
        x_ref = x_native.clone().detach().requires_grad_(True)
        grad_output = torch.normal(
            mean=0.0,
            std=1.0,
            size=(batch_size, seq_len, out_features),
            dtype=x_dtype,
            device=device,
        ).detach()

        with te.autocast(enabled=True, recipe=nvfp4_recipe):
            y_native, ln_out_native = native_module(x_native, is_first_microbatch=(step == 0))
        y_native.backward(grad_output)

        with te.autocast(enabled=True, recipe=nvfp4_ref_recipe):
            y_ref, ln_out_ref = ref_module(x_ref)
        y_ref.backward(grad_output)

        native_outputs.append(
            {
                "output": y_native.detach().clone(),
                "ln_out": ln_out_native.detach().clone(),
                "input_grad": (
                    x_native.grad.detach().clone() if x_native.grad is not None else None
                ),
                "weight_grad": (
                    native_module.weight.grad.detach().clone()
                    if native_module.weight.grad is not None
                    else None
                ),
                "bias_grad": (
                    native_module.bias.grad.detach().clone()
                    if bias and native_module.bias.grad is not None
                    else None
                ),
            }
        )
        ref_outputs.append(
            {
                "output": y_ref.detach().clone(),
                "ln_out": ln_out_ref.detach().clone(),
                "input_grad": (x_ref.grad.detach().clone() if x_ref.grad is not None else None),
                "weight_grad": (
                    ref_module.weight.grad.detach().clone()
                    if ref_module.weight.grad is not None
                    else None
                ),
                "bias_grad": (
                    ref_module.bias.grad.detach().clone()
                    if bias and ref_module.bias.grad is not None
                    else None
                ),
            }
        )

    for step in range(num_steps):
        n = native_outputs[step]
        r = ref_outputs[step]
        torch.testing.assert_close(
            n["output"], r["output"], atol=1e-6, rtol=1e-6, msg=f"Output mismatch at step {step}"
        )
        torch.testing.assert_close(
            n["ln_out"], r["ln_out"], atol=1e-6, rtol=1e-6, msg=f"LN output mismatch at step {step}"
        )
        torch.testing.assert_close(
            n["input_grad"],
            r["input_grad"],
            atol=1e-6,
            rtol=1e-6,
            msg=f"Input gradient mismatch at step {step}",
        )
        torch.testing.assert_close(
            n["weight_grad"],
            r["weight_grad"],
            atol=1e-6,
            rtol=1e-6,
            msg=f"Weight gradient mismatch at step {step}",
        )
        if bias and n["bias_grad"] is not None and r["bias_grad"] is not None:
            torch.testing.assert_close(
                n["bias_grad"],
                r["bias_grad"],
                atol=1e-6,
                rtol=1e-6,
                msg=f"Bias gradient mismatch at step {step}",
            )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("native_recipe", _NATIVE_RECIPES)
@pytest.mark.parametrize(
    "in_features, out_features",
    [
        (128, 256),
        (256, 128),
        (512, 512),
        (768, 3072),
        (1024, 4096),
    ],
)
@pytest.mark.parametrize("bias", [False], ids=["no_bias"])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("num_steps", [1, 3], ids=["single_step", "multi_step"])
@pytest.mark.parametrize("with_rht", [True, False], ids=["with_rht", "no_rht"])
@pytest.mark.parametrize(
    "with_2d_quantization", [True, False], ids=["with_2d_quantization", "no_2d_quantization"]
)
def test_nvfp4_linear_versus_reference(
    native_recipe: QuantizeRecipe,
    in_features: int,
    out_features: int,
    bias: bool,
    x_dtype: torch.dtype,
    num_steps: int,
    with_rht: bool,
    with_2d_quantization: bool,
):
    if with_rht and x_dtype != torch.bfloat16:
        pytest.skip("RHT is only supported for bfloat16 input")

    check_nvfp4_linear_versus_reference(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        x_dtype=x_dtype,
        native_recipe=native_recipe,
        num_steps=num_steps,
        with_rht=with_rht,
        with_2d_quantization=with_2d_quantization,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "in_features, out_features",
    [
        (128, 256),
        (256, 128),
    ],
)
@pytest.mark.parametrize("bias", [False], ids=["no_bias"])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("num_steps", [1], ids=["single_step"])
@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"], ids=["LayerNorm", "RMSNorm"])
@pytest.mark.parametrize("with_rht", [True, False], ids=["with_rht", "no_rht"])
@pytest.mark.parametrize(
    "with_2d_quantization", [True, False], ids=["with_2d_quantization", "no_2d_quantization"]
)
def test_nvfp4_layernorm_linear_versus_reference(
    in_features: int,
    out_features: int,
    bias: bool,
    normalization: str,
    x_dtype: torch.dtype,
    num_steps: int,
    with_rht: bool,
    with_2d_quantization: bool,
):
    if with_rht and x_dtype != torch.bfloat16:
        pytest.skip("RHT is only supported for bfloat16 input")

    check_nvfp4_layernorm_linear_versus_reference(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        normalization=normalization,
        x_dtype=x_dtype,
        num_steps=num_steps,
        with_rht=with_rht,
        with_2d_quantization=with_2d_quantization,
    )


if __name__ == "__main__":
    test_nvfp4_linear_versus_reference(
        QuantizeRecipe.FP4_SUB_CHANNEL,
        1024,
        4096,
        False,
        torch.bfloat16,
        3,
        True,
        True,
    )
