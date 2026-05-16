"""Grouped NVFP4 GEMM with per-token activation quantization (CuTeDSL path)."""

import pytest
import torch

from miniTransformer.ops.cuteDSL.nvfp4.grouped_gemm_nvfp4_api import (
    apply_grouped_gemm_alphas,
    block_scale_to_sf_fp8,
    build_group_alpha_pertoken,
    dequant_pertoken_nvfp4,
    grouped_gemm_cutedsl_pertoken,
    is_sm100,
    pertoken_scales_to_sfa_2d,
    quantize_activation_nvfp4_pertoken,
    quantize_weight_nvfp4_tensor_global,
    reference_grouped_gemm_pertoken,
)
from miniTransformer.ops.triton.nvfp4_quantization import dequant_nvfp4_pertoken


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def _require_sm100():
    _require_cuda()
    if not is_sm100():
        pytest.skip("Requires SM100 (Blackwell) for CuTeDSL grouped blockscaled GEMM")


PERTOKEN_CASES = [
    pytest.param(
        {"m_splits": [128, 128], "n": 128, "k": 256},
        id="two_groups_m128_n128_k256",
    ),
    pytest.param(
        {"m_splits": [256], "n": 256, "k": 512},
        id="one_group_m256_n256_k512",
    ),
]


class TestPertokenGroupedGemmHelpers:
    def test_block_sfa_plus_alpha_list_matches_merged_sfa(self):
        _require_cuda()
        torch.manual_seed(11)
        m, k, n = 64, 128, 32
        pt = quantize_activation_nvfp4_pertoken(
            torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        )
        merged_sfa = pertoken_scales_to_sfa_2d(pt.block_scale, pt.global_scale, m, k)
        block_sfa = block_scale_to_sf_fp8(pt.block_scale, m, k)

        from miniTransformer.ops.nvfp4_ref_quantize import cast_from_fp4x2

        x_fp4 = cast_from_fp4x2(pt.data_uint8, torch.float32)
        sf_merged = merged_sfa[:m, : k // 16].float().repeat_interleave(16, dim=1)
        sf_block = block_sfa[:m, : k // 16].float().repeat_interleave(16, dim=1)
        # Merged block scale is rounded into FP8; exact equality with float bs*gs is unrealistic.
        torch.testing.assert_close(
            x_fp4 * sf_merged,
            x_fp4 * sf_block * pt.global_scale.view(m, 1).float(),
            rtol=0.08,
            atol=0.02,
        )

        kernel_out = torch.ones(m, n, device="cuda", dtype=torch.bfloat16)
        alpha = build_group_alpha_pertoken(pt, pt, 1.0, n)
        corrected = apply_grouped_gemm_alphas([kernel_out], [alpha])[0]
        ga = pt.global_scale.float().view(m, 1)
        gb = pt.global_scale.float()[:n].view(1, n)
        expected = kernel_out.float() * ga * gb
        torch.testing.assert_close(corrected.float(), expected, rtol=1e-5, atol=1e-5)

    def test_triton_dequant_matches_helper(self):
        _require_cuda()
        from miniTransformer.ops.triton.nvfp4_quantization import (
            quant_nvfp4_pertoken_triton,
        )

        torch.manual_seed(3)
        m, k = 32, 64
        x = torch.randn(m, k, device="cuda", dtype=torch.float32)
        x_q, gs, bs = quant_nvfp4_pertoken_triton(x)
        triton_dq = dequant_nvfp4_pertoken(x_q, gs, bs, block_size=16)
        pt = quantize_activation_nvfp4_pertoken(x)
        helper_dq = dequant_pertoken_nvfp4(pt, torch.float32)
        torch.testing.assert_close(triton_dq, helper_dq, rtol=2e-2, atol=2e-2)


class TestPertokenGroupedGemmCorrectness:
    @pytest.mark.parametrize("case", PERTOKEN_CASES)
    def test_reference_grouped_matmul(self, case):
        _require_cuda()
        torch.manual_seed(7)
        device = "cuda"
        m_splits = case["m_splits"]
        n, k = case["n"], case["k"]

        a_list = []
        b_list = []
        for m in m_splits:
            a_list.append(
                quantize_activation_nvfp4_pertoken(
                    torch.randn(m, k, device=device, dtype=torch.bfloat16)
                )
            )
            b_list.append(
                quantize_weight_nvfp4_tensor_global(
                    torch.randn(n, k, device=device, dtype=torch.bfloat16)
                )
            )

        ref = reference_grouped_gemm_pertoken(a_list, b_list, m_splits, torch.bfloat16)
        assert ref.shape == (sum(m_splits), n)

    @pytest.mark.parametrize("case", PERTOKEN_CASES)
    def test_cutedsl_matches_reference(self, case):
        _require_sm100()
        torch.manual_seed(9)
        device = "cuda"
        m_splits = case["m_splits"]
        n, k = case["n"], case["k"]
        out_dtype = torch.bfloat16

        a_list = []
        b_list = []
        for m in m_splits:
            a_list.append(
                quantize_activation_nvfp4_pertoken(
                    torch.randn(m, k, device=device, dtype=torch.bfloat16)
                )
            )
            b_list.append(
                quantize_weight_nvfp4_tensor_global(
                    torch.randn(n, k, device=device, dtype=torch.bfloat16)
                )
            )

        ref = reference_grouped_gemm_pertoken(a_list, b_list, m_splits, out_dtype)
        try:
            got = grouped_gemm_cutedsl_pertoken(a_list, b_list, m_splits, out_dtype)
        except (TypeError, NotImplementedError) as exc:
            pytest.skip(f"CuTeDSL grouped GEMM unavailable: {exc}")

        torch.testing.assert_close(
            got.to(torch.float32),
            ref.to(torch.float32),
            rtol=0.12,
            atol=0.05,
        )

    @pytest.mark.parametrize("case", PERTOKEN_CASES)
    def test_both_sides_pertoken(self, case):
        _require_sm100()
        torch.manual_seed(13)
        device = "cuda"
        m_splits = case["m_splits"]
        n, k = case["n"], case["k"]
        out_dtype = torch.bfloat16

        a_list = [
            quantize_activation_nvfp4_pertoken(
                torch.randn(m, k, device=device, dtype=torch.bfloat16)
            )
            for m in m_splits
        ]
        b_list = [
            quantize_activation_nvfp4_pertoken(
                torch.randn(n, k, device=device, dtype=torch.bfloat16)
            )
            for _ in m_splits
        ]

        ref = reference_grouped_gemm_pertoken(a_list, b_list, m_splits, out_dtype)
        try:
            got = grouped_gemm_cutedsl_pertoken(a_list, b_list, m_splits, out_dtype)
        except (TypeError, NotImplementedError) as exc:
            pytest.skip(f"CuTeDSL grouped GEMM unavailable: {exc}")

        torch.testing.assert_close(
            got.to(torch.float32),
            ref.to(torch.float32),
            rtol=0.12,
            atol=0.05,
        )
