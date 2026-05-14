"""
Tests for the cuteDSL Sm100GroupedBlockScaledGemmKernel NVFP4 path.

Two test classes:

  TestGroupedGemmCuteDSLAdapter
    Kernel-level alignment: cuteDSL vs dequant+matmul reference using mock
    NVFP4 tensors with synthetic true-scale float8_e4m3fn scales.

  TestGroupLinearCuteDSLForward
    End-to-end forward: MiniGroupedLinear (FP4_SUB_CHANNEL) hitting
    _grouped_gemm_cutedsl, verified via mock.patch and compared against
    te.GroupedLinear (NVFP4BlockScaling).
"""
import sys
import pytest
import torch
from unittest import mock

# Ensure the editable TE source is on the path so the pytorch extension loads correctly.
sys.path.insert(0, "/home/admin/yanglin/workspace/TransformerEngine")
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

from miniTransformer.ops.cuteDSL.nvfp4.blockwise_group_gemm_cute import (
    Sm100GroupedBlockScaledGemmKernel,
    GroupedBlockScaledProfile,
)
from miniTransformer.ops.nvfp4_qgemm import create_cute_scale_factor_tensor
from miniTransformer.quantiztion.nvfp4_quantize_op_ref import cast_to_fp4x2, cast_from_fp4x2
from miniTransformer.module.config import QuantizeRecipe, get_qlinear_params_from_predefined
from miniTransformer.module.group_linear import GroupedLinear as MiniGroupedLinear
from miniTransformer.quantiztion.nvfp4_quantize_op import _is_sm100, _grouped_gemm_cutedsl
from miniTransformer.quantiztion.quantization import QParams, ParamType


# ── shared constants ──────────────────────────────────────────────────────────

_FP4_MAX = 6.0
_SF_VEC_SIZE = 16


# ═══════════════════════════════════════════════════════════════════════════════
# Adapter test helpers (kernel-level, mock tensors with true block scales)
# ═══════════════════════════════════════════════════════════════════════════════

class _MockNVFP4Tensor:
    def __init__(self, rowwise_data, rowwise_scale_inv):
        self._rowwise_data = rowwise_data        # (M, K//2) uint8
        self._rowwise_scale_inv = rowwise_scale_inv  # (M, K//16) float8_e4m3fn true scales


def _make_nvfp4(x_fp32: torch.Tensor) -> _MockNVFP4Tensor:
    """Quantize float32 to NVFP4 with true block scales (block_max / FP4_MAX)."""
    M, K = x_fp32.shape
    assert K % _SF_VEC_SIZE == 0
    x_blocks = x_fp32.reshape(M, K // _SF_VEC_SIZE, _SF_VEC_SIZE)
    block_max = x_blocks.abs().amax(dim=-1).clamp(min=1e-5)
    scale = block_max / _FP4_MAX
    scale_exp = scale.repeat_interleave(_SF_VEC_SIZE, dim=1)
    x_scaled = (x_fp32 / scale_exp).clamp(-_FP4_MAX, _FP4_MAX)
    qx = cast_to_fp4x2(x_scaled.cpu()).cuda()
    scale_f8 = scale.to(torch.float8_e4m3fn).cuda()
    return _MockNVFP4Tensor(qx, scale_f8)


def _dequant(t: _MockNVFP4Tensor) -> torch.Tensor:
    device = t._rowwise_data.device
    dq = cast_from_fp4x2(t._rowwise_data.cpu(), torch.float32).to(device)
    scale_fp32 = t._rowwise_scale_inv.to(torch.float32).repeat_interleave(_SF_VEC_SIZE, dim=1)
    return dq * scale_fp32


def _reference_grouped_gemm(A_list, B_list, out_dtype):
    outputs = []
    for A, B in zip(A_list, B_list):
        outputs.append(torch.mm(_dequant(A), _dequant(B).T).to(out_dtype))
    return torch.cat(outputs, dim=0)


def _adapter_make_cute_sf(sf_2d, mn, sf_k, l=1):
    """Convert (mn, sf_k) float8_e4m3fn true-scale tensor to CuTe MMA layout."""
    ref = sf_2d.cpu().unsqueeze(-1)
    return create_cute_scale_factor_tensor(l, mn, sf_k, ref)


def _adapter_grouped_gemm_cutedsl(A, B, m_splits, out_dtype):
    """Kernel-level grouped NVFP4 GEMM using mock tensors with true block scales."""
    device = A[0]._rowwise_data.device
    K = A[0]._rowwise_data.shape[1] * 2
    sf_k = K // _SF_VEC_SIZE

    abc_torch, sfasfb_torch, problem_sizes = [], [], []
    for i, m_i in enumerate(m_splits):
        N_i = B[i]._rowwise_data.shape[0]
        buf_a = torch.empty(m_i * K, dtype=torch.int8, device=device)
        buf_a[: m_i * K // 2].copy_(A[i]._rowwise_data.reshape(-1).view(torch.int8))
        a_fp4 = buf_a.view(m_i, K, 1)
        buf_b = torch.empty(N_i * K, dtype=torch.int8, device=device)
        buf_b[: N_i * K // 2].copy_(B[i]._rowwise_data.reshape(-1).view(torch.int8))
        b_fp4 = buf_b.view(N_i, K, 1)
        c_out = torch.empty((m_i, N_i), dtype=out_dtype, device=device)
        sfa = _adapter_make_cute_sf(A[i]._rowwise_scale_inv, m_i, sf_k)
        sfb = _adapter_make_cute_sf(B[i]._rowwise_scale_inv, N_i, sf_k)
        abc_torch.append([a_fp4, b_fp4, c_out])
        sfasfb_torch.append([sfa, sfb])
        problem_sizes.append((m_i, N_i, K, 1))

    kernel = Sm100GroupedBlockScaledGemmKernel(profile=GroupedBlockScaledProfile.NVFP4)
    results = kernel.run_from_grouped_inputs(
        abc_tensors={"torch": abc_torch},
        sfasfb_tensors={"torch": sfasfb_torch},
        problem_sizes=problem_sizes,
        out_dtype=out_dtype,
        accumulate=False,
    )
    return torch.cat(results, dim=0)


# ── adapter test cases ────────────────────────────────────────────────────────

ADAPTER_CASES = [
    {"m": [128, 256], "N": 4096, "K": 2048},
    {"m": [80, 176, 128], "N": 4096, "K": 7168},
    {"m": [192, 320], "N": 3072, "K": 4096},
    {"m": [128, 384], "N": 4096, "K": 1536},
]
ADAPTER_CASE_PARAMS = [pytest.param(c, id=f"case_{i}") for i, c in enumerate(ADAPTER_CASES)]


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end forward test helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_nvfp4_recipe() -> recipe.NVFP4BlockScaling:
    r = recipe.NVFP4BlockScaling()
    r.fp4_quant_bwd_grad = recipe.QParams(
        random_hadamard_transform=True,
        stochastic_rounding=False,
        fp4_2d_quantization=False,
    )
    return r


def build_te_grouped_linear(
    num_gemms: int,
    in_features: int,
    out_features: int,
    bias: bool = False,
    seed: int = 42,
) -> te.GroupedLinear:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return te.GroupedLinear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        params_dtype=torch.bfloat16,
        device="cuda",
    )


def build_mini_grouped_linear(
    num_gemms: int,
    in_features: int,
    out_features: int,
    bias: bool = False,
    seed: int = 42,
) -> MiniGroupedLinear:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    qlinear_params = get_qlinear_params_from_predefined(QuantizeRecipe.FP4_SUB_CHANNEL)
    qlinear_params.g_params = QParams(
        output_dtype=torch.uint8,
        scaling_dtype=torch.float8_e4m3fn,
        quant_tile_shape=(1, 16),
        with_2d_quantization=False,
        with_rht=True,
        stochastic_rounding=False,
        param_type=ParamType.Y,
    )
    return MiniGroupedLinear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        sequence_parallel=False,
        fuse_wgrad_accumulation=False,
        tp_group=None,
        tp_size=1,
        get_rng_state_tracker=None,
        rng_tracker_name=None,
        init_method=None,
        bias=bias,
        return_bias=False,
        params_dtype=torch.bfloat16,
        parallel_mode=None,
        device="cuda",
        ub_overlap_rs=False,
        ub_overlap_ag=False,
        ub_name="test_cutedsl_forward",
        layer_number=0,
        qlinear_params=qlinear_params,
    )


def sync_weights_te_to_mini(
    te_model: te.GroupedLinear,
    mini_model: MiniGroupedLinear,
    num_gemms: int,
) -> None:
    with torch.no_grad():
        for i in range(num_gemms):
            te_w = getattr(te_model, f"weight{i}").data
            mini_w = getattr(mini_model, f"weight{i}").data
            assert te_w.shape == mini_w.shape, (
                f"weight{i} shape mismatch: TE={te_w.shape} mini={mini_w.shape}"
            )
            mini_w.copy_(te_w)


# ── forward test cases ────────────────────────────────────────────────────────

FORWARD_CASES = [
    pytest.param(4,  512,  512,  128, id="4x512x512_m128"),
    pytest.param(8, 4096, 4096,  128, id="8x4096x4096_m128"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Test classes
# ═══════════════════════════════════════════════════════════════════════════════

class TestGroupedGemmCuteDSLAdapter:
    """Kernel alignment: cuteDSL vs dequant+matmul reference (mock true-scale tensors)."""

    @pytest.mark.parametrize("case", ADAPTER_CASE_PARAMS)
    def test_cutedsl_vs_dequant_reference(self, case):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        if not _is_sm100():
            pytest.skip("Requires SM100 (Blackwell) GPU")

        out_dtype = torch.bfloat16
        m_splits = case["m"]
        N = case["N"]
        K = case["K"]
        num_groups = len(m_splits)

        torch.manual_seed(42)
        x_list = [torch.randn(m, K) * 0.1 for m in m_splits]
        w_list = [torch.randn(N, K) * 0.1 for _ in range(num_groups)]

        A_list = [_make_nvfp4(x) for x in x_list]
        B_list = [_make_nvfp4(w) for w in w_list]

        cutedsl_out = _adapter_grouped_gemm_cutedsl(A_list, B_list, m_splits, out_dtype)
        ref_out = _reference_grouped_gemm(A_list, B_list, out_dtype)

        assert cutedsl_out.shape == ref_out.shape, (
            f"Shape mismatch: {cutedsl_out.shape} vs {ref_out.shape}"
        )
        torch.testing.assert_close(
            cutedsl_out.to(torch.float32),
            ref_out.to(torch.float32),
            rtol=1e-4,
            atol=1e-4,
            msg=f"cuteDSL vs dequant+matmul mismatch for m={m_splits} N={N} K={K}",
        )
        max_diff = (cutedsl_out.float() - ref_out.float()).abs().max().item()
        print(f"\n[PASS] m={m_splits} N={N} K={K}: max_diff={max_diff:.4f}")


class TestGroupLinearCuteDSLForward:
    """End-to-end forward: MiniGroupedLinear cuteDSL dispatch vs te.GroupedLinear."""

    @pytest.mark.parametrize("num_gemms, in_features, out_features, m_per_expert", FORWARD_CASES)
    def test_forward_cutedsl_vs_te(
        self, num_gemms: int, in_features: int, out_features: int, m_per_expert: int
    ):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        if not _is_sm100():
            pytest.skip("Requires SM100 (Blackwell) GPU — _grouped_gemm_cutedsl not triggered")

        m_splits = [m_per_expert] * num_gemms
        total_m = sum(m_splits)
        nvfp4_recipe = make_nvfp4_recipe()

        te_model = build_te_grouped_linear(num_gemms, in_features, out_features)
        mini_model = build_mini_grouped_linear(num_gemms, in_features, out_features)
        sync_weights_te_to_mini(te_model, mini_model, num_gemms)

        torch.manual_seed(142)
        inp = torch.randn(total_m, in_features, device="cuda", dtype=torch.bfloat16)

        # TE forward (reference)
        with torch.no_grad():
            with te.autocast(enabled=True, recipe=nvfp4_recipe):
                out_te = te_model(inp.clone(), m_splits, is_first_microbatch=True)

        # mini forward — wrap _grouped_gemm_cutedsl to assert it is called
        with mock.patch(
            "miniTransformer.quantiztion.nvfp4_quantize_op._grouped_gemm_cutedsl",
            wraps=_grouped_gemm_cutedsl,
        ) as mock_cutedsl:
            with torch.no_grad():
                out_mini = mini_model(inp.clone(), m_splits=m_splits, is_first_microbatch=True)

        assert mock_cutedsl.called, (
            "_grouped_gemm_cutedsl was NOT called — dispatch condition not met "
            "(expected: is_quantized=True, layout=TN, grad=False, gelu=False, accumulate=False, SM100)"
        )

        diff = (out_te - out_mini).abs()
        rel_diff = diff / (out_te.abs() + 1e-6)
        print(
            f"\n  [n={num_gemms} in={in_features} out={out_features} m={m_per_expert}]"
            f"\n  TE   shape={out_te.shape} dtype={out_te.dtype}"
            f"\n  mini shape={out_mini.shape} dtype={out_mini.dtype}"
            f"\n  max abs diff:  {diff.max():.4e}"
            f"\n  mean abs diff: {diff.mean():.4e}"
            f"\n  max rel diff:  {rel_diff.max():.4e}"
        )

        torch.testing.assert_close(
            out_te, out_mini,
            rtol=0.02, atol=0.06,
            msg=(
                f"[cuteDSL forward vs TE] mismatch: "
                f"num_gemms={num_gemms} in={in_features} out={out_features}"
            ),
        )
        print(f"  cuteDSL dispatch confirmed, output matches TE (rtol=0.02, atol=0.06)")
