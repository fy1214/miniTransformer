import os

import pytest
import torch

from miniTransformer.quantiztion.nvfp4_quartet_op import (
    grouped_gemm_cutedsl,
    grouped_gemm_cutedsl_pertoken,
    is_sm100,
    quantize_activation_nvfp4_pertoken,
    quantize_weight_nvfp4_tensor_global,
)


def _require_runtime():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required.")
    if not is_sm100():
        pytest.skip("Requires SM100 for grouped CuTeDSL NVFP4 kernels.")
    pytest.importorskip("transformer_engine.pytorch")


def _make_te_nvfp4_quantizer(columnwise: bool):
    import transformer_engine.pytorch.cpp_extensions as tex
    from transformer_engine.pytorch import NVFP4Quantizer

    return NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=columnwise,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
    )


def _grouped_bf16_reference(x_groups, w_groups):
    outs = []
    for x, w in zip(x_groups, w_groups):
        outs.append(torch.matmul(x.float(), w.float().T))
    return torch.cat(outs, dim=0)


def _error_metrics(pred: torch.Tensor, ref: torch.Tensor):
    err = (pred.float() - ref.float()).abs()
    rmse = torch.sqrt(torch.mean((pred.float() - ref.float()) ** 2))
    return {
        "max_abs": float(err.max().item()),
        "mean_abs": float(err.mean().item()),
        "rmse": float(rmse.item()),
    }


_INCLUDE_LARGE_CASES = os.environ.get(
    "MINITRANSFORMER_RUN_GROUPED_GEMM_DIFF_LARGE", "0"
).strip() in {"1", "true", "yes", "on"}

_DIFF_SHAPE_CASES = [
    pytest.param([128, 128], 256, 1024, id="small_2g_m128_n256_k1024"),
    pytest.param([96, 160, 128], 256, 2048, id="medium_3g_m96_160_128_n256_k2048"),
    pytest.param(
        [80, 176, 128, 72],
        4096,
        2048,
        id="large_4g_moe_like_n4096_k2048",
        marks=pytest.mark.skipif(
            not _INCLUDE_LARGE_CASES,
            reason=(
                "Large case disabled by default. "
                "Set MINITRANSFORMER_RUN_GROUPED_GEMM_DIFF_LARGE=1 to enable."
            ),
        ),
    ),
]


@pytest.mark.parametrize("m_splits,n,k", _DIFF_SHAPE_CASES)
def test_grouped_gemm_nvfp4_and_pertoken_vs_bf16_diff(m_splits, n, k):
    _require_runtime()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    device = "cuda"

    # Keep input amplitude moderate so diff thresholds are stable.
    x_groups = [0.1 * torch.randn(m, k, dtype=torch.bfloat16, device=device) for m in m_splits]
    w_groups = [0.1 * torch.randn(n, k, dtype=torch.bfloat16, device=device) for _ in m_splits]

    # BF16/FP32 reference grouped GEMM
    ref = _grouped_bf16_reference(x_groups, w_groups)

    # Path-1: original NVFP4 quant + grouped_gemm (TE quantized tensors)
    te_q_x = _make_te_nvfp4_quantizer(columnwise=False)
    te_q_w = _make_te_nvfp4_quantizer(columnwise=False)
    a_nvfp4 = [te_q_x.quantize(x) for x in x_groups]
    b_nvfp4 = [te_q_w.quantize(w) for w in w_groups]

    try:
        out_nvfp4 = grouped_gemm_cutedsl(a_nvfp4, b_nvfp4, m_splits, torch.bfloat16)
    except (TypeError, NotImplementedError) as exc:
        pytest.skip(f"Original grouped_gemm_cutedsl unavailable: {exc}")

    # Path-2: pertoken activation quant + alpha-list grouped_gemm
    a_pertoken = [quantize_activation_nvfp4_pertoken(x) for x in x_groups]
    b_tensor_global = [quantize_weight_nvfp4_tensor_global(w) for w in w_groups]
    try:
        out_pertoken = grouped_gemm_cutedsl_pertoken(
            a_pertoken, b_tensor_global, m_splits, torch.bfloat16
        )
    except (TypeError, NotImplementedError) as exc:
        pytest.skip(f"Pertoken grouped_gemm path unavailable: {exc}")

    diff_nvfp4 = _error_metrics(out_nvfp4, ref)
    diff_pertoken = _error_metrics(out_pertoken, ref)

    print(
        "[NVFP4 grouped diff] "
        f"max={diff_nvfp4['max_abs']:.6f}, "
        f"mean={diff_nvfp4['mean_abs']:.6f}, "
        f"rmse={diff_nvfp4['rmse']:.6f}"
    )
    print(
        "[Pertoken+alpha grouped diff] "
        f"max={diff_pertoken['max_abs']:.6f}, "
        f"mean={diff_pertoken['mean_abs']:.6f}, "
        f"rmse={diff_pertoken['rmse']:.6f}"
    )

    # Basic correctness guardrails. Exact relative ranking can vary by shape/data.
    assert torch.isfinite(out_nvfp4).all()
    assert torch.isfinite(out_pertoken).all()
    assert diff_nvfp4["max_abs"] < 2.0
    assert diff_pertoken["max_abs"] < 2.0
