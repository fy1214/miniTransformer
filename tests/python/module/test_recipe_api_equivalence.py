"""Target Blackwell checks for the public recipe API versus legacy selectors.

Run on the matched CUDA/TE environment:
    pytest tests/python/module/test_recipe_api_equivalence.py -v
CPU-only checks live in tests/test_recipe_configuration.py.
"""

import dataclasses

import pytest
import torch

from miniTransformer.module.config import (
    QuantizeRecipe, get_qlinear_params, get_qlinear_params_from_env_qat_params,
    get_qlinear_params_from_predefined,
)
from miniTransformer.recipes import RECIPES, get_recipe


def _require_blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("requires the target Blackwell CUDA/TE environment")


@pytest.mark.parametrize("spec", RECIPES, ids=lambda spec: spec.name)
def test_named_factory_preserves_parameters_and_independent_instances(spec, monkeypatch):
    _require_blackwell()
    monkeypatch.delenv("DISABLE_BACKWARD_QUANT", raising=False)
    old = get_qlinear_params_from_predefined(QuantizeRecipe(spec.enum_value))
    new = get_qlinear_params(spec.name)
    assert old is not new
    for field in dataclasses.fields(old):
        if field.name == "quantize_op":
            assert type(old.quantize_op) is type(new.quantize_op)
            if old.quantize_op is not None:
                assert old.quantize_op is not new.quantize_op
        else:
            assert getattr(old, field.name) == getattr(new, field.name)
    old.overrides_for_layer_name["test"] = old
    assert not new.overrides_for_layer_name


@pytest.mark.parametrize("qat_id,disable_backward,layer_kind", [
    (4, False, "grouped"), (14, False, "grouped"), (15, False, "grouped"),
    (16, False, "grouped"), (17, False, "grouped"), (15, True, "grouped"),
    (4, False, "dense"), (15, False, "dense"), (15, True, "dense"),
])
def test_forward_backward_selector_equivalence(qat_id, disable_backward, layer_kind, monkeypatch, nvfp4_cuda_device):
    _require_blackwell()
    from miniTransformer.module.group_linear import GroupedLinear
    from miniTransformer.module.linear import Linear

    device = nvfp4_cuda_device
    spec = get_recipe(qat_id)
    monkeypatch.delenv("MINITE_RECIPE", raising=False)
    monkeypatch.delenv("QAT_KEEP_IN_BF16_LAYER_NUMBERS", raising=False)
    monkeypatch.setenv("QAT_PARAMS", str(qat_id))
    monkeypatch.setenv("DISABLE_BACKWARD_QUANT", "1" if disable_backward else "0")
    monkeypatch.setenv("NVTE_NVFP4_DISABLE_RHT", "1")
    monkeypatch.setenv("NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING", "1")
    monkeypatch.setenv("NVTE_NVFP4_DISABLE_2D_QUANTIZATION", "0" if qat_id == 4 else "1")
    monkeypatch.setenv("MINITE_NVFP4_PATH", "auto")

    def layer():
        kwargs = dict(in_features=256, out_features=256, bias=False,
                      params_dtype=torch.bfloat16, device=device, fuse_wgrad_accumulation=False)
        return GroupedLinear(num_gemms=2, **kwargs) if layer_kind == "grouped" else Linear(**kwargs)

    with torch.cuda.device(device):
        torch.manual_seed(123)
        legacy = layer()
        monkeypatch.delenv("QAT_PARAMS")
        monkeypatch.setenv("MINITE_RECIPE", spec.name)
        named = layer()
        named.load_state_dict(legacy.state_dict())
        x = torch.randn(384, 256, device=device, dtype=torch.bfloat16)
        grad = torch.randn(384, 256, device=device, dtype=torch.bfloat16)
        results = []
        for model in (legacy, named):
            inp = x.detach().clone().requires_grad_(True)
            arguments = {"m_splits": [128, 256]} if layer_kind == "grouped" else {}
            out = model(inp, is_first_microbatch=True, **arguments)
            out.backward(grad)
            results.append((out.detach(), inp.grad, [p.grad for p in model.parameters()]))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(results[0][1], results[1][1], rtol=1e-3, atol=1e-3)
        for old_grad, new_grad in zip(results[0][2], results[1][2]):
            assert old_grad is not None and new_grad is not None
            torch.testing.assert_close(old_grad, new_grad, rtol=1e-3, atol=1e-3)


def test_explicit_parameters_bypass_conflicting_environment(monkeypatch):
    _require_blackwell()
    from miniTransformer.module.base import LayerLinearBaseModule

    monkeypatch.setenv("MINITE_RECIPE", "nvfp4_per_token")
    monkeypatch.setenv("QAT_PARAMS", "17")
    monkeypatch.delenv("QAT_KEEP_IN_BF16_LAYER_NUMBERS", raising=False)
    params = get_qlinear_params_from_predefined(QuantizeRecipe.NON_QUANTIZE)
    selected = LayerLinearBaseModule.set_qlinear_params(
        None, qlinear_params=params, layer_number=None, layer_name=None)
    assert selected is params
    with pytest.raises(ValueError, match="Conflicting selectors"):
        get_qlinear_params_from_env_qat_params()
