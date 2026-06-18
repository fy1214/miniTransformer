import copy
import os

import pytest
import torch

from miniTransformer.module.group_linear import GroupedLinear
from miniTransformer.module.config import (
    QLinearParams,
    QuantizeRecipe,
    QuantizeType,
    get_qlinear_params_from_predefined,
)
from miniTransformer.quantiztion.int4_quantize_op import Int4TiledQuantizeOp
from miniTransformer.quantiztion.quantization import ParamType, QParams


def _require_runtime():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required.")
    pytest.importorskip("transformer_engine.pytorch")
    pytest.importorskip("sglang")


def _error_metrics(pred: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    pred_f = pred.float()
    ref_f = ref.float()
    err = (pred_f - ref_f).abs()
    rmse = torch.sqrt(torch.mean((pred_f - ref_f) ** 2))
    return {
        "max_abs": float(err.max().item()),
        "mean_abs": float(err.mean().item()),
        "rmse": float(rmse.item()),
    }


def _print_metrics(label: str, pred: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    metrics = _error_metrics(pred, ref)
    print(
        f"  [{label}] "
        f"max_abs={metrics['max_abs']:.4e} "
        f"mean_abs={metrics['mean_abs']:.4e} "
        f"rmse={metrics['rmse']:.4e}"
    )
    return metrics


def _grouped_matmul_reference(
    inp: torch.Tensor,
    weights: list[torch.Tensor],
    m_splits: list[int],
    *,
    accumulate_dtype: torch.dtype,
) -> torch.Tensor:
    outs = []
    for x_i, w_i in zip(inp.split(m_splits, dim=0), weights):
        outs.append(x_i.to(accumulate_dtype) @ w_i.to(accumulate_dtype).T)
    return torch.cat(outs, dim=0).to(inp.dtype)


def _make_ste_qlinear_params(group_size: int) -> QLinearParams:
    qparams = copy.deepcopy(get_qlinear_params_from_predefined(QuantizeRecipe.INT4_SUB_CHANNEL))
    qparams.quantize = True
    qparams.quantize_type = QuantizeType.INT4
    qparams.x_params = QParams(
        output_dtype=torch.bfloat16,
        quant_tile_shape=(1, group_size),
        param_type=ParamType.X,
    )
    qparams.w_params = QParams(
        output_dtype=torch.bfloat16,
        quant_tile_shape=(1, group_size),
        param_type=ParamType.WEIGHT,
    )
    qparams.g_params = QParams(
        output_dtype=torch.bfloat16,
        quant_tile_shape=(1, group_size),
        param_type=ParamType.Y,
    )
    qparams.quantize_op = Int4TiledQuantizeOp()
    return qparams


def _build_grouped_linear(
    qlinear_params: QLinearParams,
    *,
    num_gemms: int,
    in_features: int,
    out_features: int,
    seed: int,
) -> GroupedLinear:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return GroupedLinear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        sequence_parallel=False,
        fuse_wgrad_accumulation=True,
        tp_group=None,
        tp_size=1,
        get_rng_state_tracker=None,
        rng_tracker_name=None,
        init_method=None,
        bias=False,
        return_bias=False,
        params_dtype=torch.bfloat16,
        parallel_mode=None,
        device="cuda",
        ub_overlap_rs=False,
        ub_overlap_ag=False,
        ub_name="test_int4_marlin_vs_ste",
        layer_number=0,
        qlinear_params=qlinear_params,
    )


def _sync_grouped_weights(src: GroupedLinear, dst: GroupedLinear, num_gemms: int) -> None:
    with torch.no_grad():
        for i in range(num_gemms):
            getattr(dst, f"weight{i}").copy_(getattr(src, f"weight{i}"))


def _collect_grouped_weights(model: GroupedLinear, num_gemms: int) -> list[torch.Tensor]:
    return [getattr(model, f"weight{i}").detach() for i in range(num_gemms)]


def _init_main_grads(model: GroupedLinear, num_gemms: int) -> None:
    for i in range(num_gemms):
        weight = getattr(model, f"weight{i}")
        weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)


@pytest.mark.parametrize(
    "m_splits,in_features,out_features",
    [
        pytest.param([64, 64, 64, 64], 512, 512, id="uniform_4g_m64_k512_n512"),
        pytest.param([32, 96, 48, 80], 512, 768, id="varying_4g_m32_96_48_80_k512_n768"),
    ],
)
def test_grouped_linear_int4_marlin_vs_ste_forward_diff(
    monkeypatch,
    m_splits,
    in_features,
    out_features,
):
    _require_runtime()
    monkeypatch.setenv("DISABLE_BACKWARD_QUANT", "1")
    monkeypatch.setenv("INT4_REAL_GROUP_SIZE", "128")

    seed = 42
    num_gemms = len(m_splits)
    total_m = sum(m_splits)
    group_size = int(os.environ["INT4_REAL_GROUP_SIZE"])

    marlin_params = get_qlinear_params_from_predefined(QuantizeRecipe.INT4_MARLIN_FORWARD)
    ste_params = _make_ste_qlinear_params(group_size)

    marlin_model = _build_grouped_linear(
        marlin_params,
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        seed=seed,
    )
    ste_model = _build_grouped_linear(
        ste_params,
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        seed=seed,
    )
    _sync_grouped_weights(marlin_model, ste_model, num_gemms)

    torch.manual_seed(seed + 100)
    torch.cuda.manual_seed(seed + 100)
    inp = 0.1 * torch.randn(total_m, in_features, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        out_marlin = marlin_model(inp.clone(), m_splits=m_splits, is_first_microbatch=True)
        out_ste = ste_model(inp.clone(), m_splits=m_splits, is_first_microbatch=True)

    weights = _collect_grouped_weights(marlin_model, num_gemms)
    ref_fp32 = _grouped_matmul_reference(
        inp,
        weights,
        m_splits,
        accumulate_dtype=torch.float32,
    )
    ref_bf16 = _grouped_matmul_reference(
        inp,
        weights,
        m_splits,
        accumulate_dtype=torch.bfloat16,
    )

    print(
        "\n[INT4 forward diff] "
        f"m_splits={m_splits} in={in_features} out={out_features} group_size={group_size}"
    )
    marlin_vs_ste = _print_metrics("Marlin vs STE", out_marlin, out_ste)
    marlin_vs_fp32 = _print_metrics("Marlin vs ref_fp32", out_marlin, ref_fp32)
    ste_vs_fp32 = _print_metrics("STE vs ref_fp32", out_ste, ref_fp32)
    _print_metrics("Marlin vs ref_bf16", out_marlin, ref_bf16)
    _print_metrics("STE vs ref_bf16", out_ste, ref_bf16)

    assert torch.isfinite(out_marlin).all()
    assert torch.isfinite(out_ste).all()
    assert marlin_vs_ste["max_abs"] < 1.0
    assert marlin_vs_fp32["max_abs"] < 1.0
    assert ste_vs_fp32["max_abs"] < 1.0


def test_grouped_linear_int4_marlin_vs_ste_backward_regression(monkeypatch):
    _require_runtime()
    monkeypatch.setenv("DISABLE_BACKWARD_QUANT", "1")
    monkeypatch.setenv("INT4_REAL_GROUP_SIZE", "128")

    seed = 123
    m_splits = [64, 64, 64, 64]
    num_gemms = len(m_splits)
    total_m = sum(m_splits)
    in_features = 512
    out_features = 512
    group_size = int(os.environ["INT4_REAL_GROUP_SIZE"])

    marlin_params = get_qlinear_params_from_predefined(QuantizeRecipe.INT4_MARLIN_FORWARD)
    ste_params = _make_ste_qlinear_params(group_size)

    marlin_model = _build_grouped_linear(
        marlin_params,
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        seed=seed,
    )
    ste_model = _build_grouped_linear(
        ste_params,
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        seed=seed,
    )
    _sync_grouped_weights(marlin_model, ste_model, num_gemms)
    _init_main_grads(marlin_model, num_gemms)
    _init_main_grads(ste_model, num_gemms)

    torch.manual_seed(seed + 100)
    torch.cuda.manual_seed(seed + 100)
    inp = 0.1 * torch.randn(total_m, in_features, device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(seed + 200)
    torch.cuda.manual_seed(seed + 200)
    dy = 0.1 * torch.randn(total_m, out_features, device="cuda", dtype=torch.bfloat16)

    inp_marlin = inp.clone().requires_grad_(True)
    out_marlin = marlin_model(inp_marlin, m_splits=m_splits, is_first_microbatch=True)
    out_marlin.backward(dy.clone())
    dgrad_marlin = inp_marlin.grad.detach().clone()
    wgrad_marlin = [
        getattr(marlin_model, f"weight{i}").main_grad.detach().clone()
        for i in range(num_gemms)
    ]

    inp_ste = inp.clone().requires_grad_(True)
    out_ste = ste_model(inp_ste, m_splits=m_splits, is_first_microbatch=True)
    out_ste.backward(dy.clone())
    dgrad_ste = inp_ste.grad.detach().clone()
    wgrad_ste = [
        getattr(ste_model, f"weight{i}").main_grad.detach().clone()
        for i in range(num_gemms)
    ]

    print(
        "\n[INT4 backward regression] "
        f"m_splits={m_splits} in={in_features} out={out_features} group_size={group_size}"
    )
    dgrad_metrics = _print_metrics("dgrad Marlin vs STE", dgrad_marlin, dgrad_ste)
    assert torch.isfinite(dgrad_marlin).all()
    assert torch.isfinite(dgrad_ste).all()
    torch.testing.assert_close(dgrad_marlin, dgrad_ste, rtol=0.0, atol=0.0)
    assert dgrad_metrics["max_abs"] == 0.0

    for i, (grad_marlin, grad_ste) in enumerate(zip(wgrad_marlin, wgrad_ste)):
        metrics = _print_metrics(f"wgrad[{i}] Marlin vs STE", grad_marlin, grad_ste)
        assert torch.isfinite(grad_marlin).all()
        assert torch.isfinite(grad_ste).all()
        torch.testing.assert_close(grad_marlin, grad_ste, rtol=0.0, atol=0.0)
        assert metrics["max_abs"] == 0.0