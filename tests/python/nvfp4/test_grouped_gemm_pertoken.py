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
        {"m_splits": [128, 128], "n": 256, "k": 1024},
        id="two_groups_m128_n256_k1024",
    ),
]


class TestPertokenGroupedGemmHelpers:
    @staticmethod
    def _error_metrics(pred: torch.Tensor, ref: torch.Tensor):
        err = (pred.float() - ref.float()).abs()
        rmse = torch.sqrt(torch.mean((pred.float() - ref.float()) ** 2))
        return float(err.max().item()), float(err.mean().item()), float(rmse.item())

    def test_pack_depack_matches_triton_grid(self):
        _require_cuda()
        from miniTransformer.ops.nvfp4_ref_quantize import cast_from_fp4x2, cast_to_fp4x2
        from miniTransformer.ops.triton.nvfp4_quantization import (
            quant_nvfp4_pertoken_triton,
        )

        torch.manual_seed(17)
        m, k = 32, 128
        x = torch.randn(m, k, device="cuda", dtype=torch.float32)
        x_q, _, _ = quant_nvfp4_pertoken_triton(x)
        packed = cast_to_fp4x2(x_q.float())
        unpacked = cast_from_fp4x2(packed, torch.float32)
        # Ensure pack->unpack is numerically stable and finite on Triton grid output.
        assert torch.isfinite(unpacked).all()
        max_err = (unpacked - x_q.float()).abs().max().item()
        assert max_err <= 0.26

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
        lhs = x_fp4 * sf_merged
        rhs = x_fp4 * sf_block * pt.global_scale.view(m, 1).float()
        max_abs, mean_abs, _ = self._error_metrics(lhs, rhs)
        assert max_abs < 0.20
        assert mean_abs < 0.02

        # 1D alpha [M] = pertoken A global × scalar B scale (host helper sanity check).
        kernel_out = torch.ones(m, n, device="cuda", dtype=torch.bfloat16)
        alpha_b = 0.75
        alpha = build_group_alpha_pertoken(pt, pt, alpha_b, n)
        corrected = apply_grouped_gemm_alphas([kernel_out], [alpha])[0]
        expected = kernel_out.float() * (pt.global_scale.float() * alpha_b).view(m, 1)
        torch.testing.assert_close(corrected.float(), expected, rtol=1e-5, atol=1e-5)

    def test_triton_dequant_matches_helper(self):
        _require_cuda()
        from miniTransformer.ops.triton.nvfp4_quantization import (
            quant_nvfp4_pertoken_triton,
        )

        torch.manual_seed(3)
        m, k = 32, 64
        x = 0.1 * torch.randn(m, k, device="cuda", dtype=torch.float32)
        x_q, gs, bs = quant_nvfp4_pertoken_triton(x)
        bs_fp8 = block_scale_to_sf_fp8(bs.reshape(m, k // 16), m, k).float().reshape(-1)
        triton_dq = dequant_nvfp4_pertoken(x_q, gs, bs_fp8, block_size=16)
        pt = quantize_activation_nvfp4_pertoken(x)
        helper_dq = dequant_pertoken_nvfp4(pt, torch.float32)
        max_abs, mean_abs, rmse = self._error_metrics(triton_dq, helper_dq)
        assert max_abs < 0.15
        assert mean_abs < 0.02
        assert rmse < 0.04


class TestPertokenGroupedGemmCorrectness:
    @staticmethod
    def _error_metrics(pred: torch.Tensor, ref: torch.Tensor):
        err = (pred.float() - ref.float()).abs()
        rmse = torch.sqrt(torch.mean((pred.float() - ref.float()) ** 2))
        return float(err.max().item()), float(err.mean().item()), float(rmse.item())

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
                    0.1 * torch.randn(m, k, device=device, dtype=torch.bfloat16)
                )
            )
            b_list.append(
                quantize_weight_nvfp4_tensor_global(
                    0.1 * torch.randn(n, k, device=device, dtype=torch.bfloat16)
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
                    0.1 * torch.randn(m, k, device=device, dtype=torch.bfloat16)
                )
            )
            b_list.append(
                quantize_weight_nvfp4_tensor_global(
                    0.1 * torch.randn(n, k, device=device, dtype=torch.bfloat16)
                )
            )

        ref = reference_grouped_gemm_pertoken(a_list, b_list, m_splits, out_dtype)
        try:
            got = grouped_gemm_cutedsl_pertoken(a_list, b_list, m_splits, out_dtype)
        except (TypeError, NotImplementedError) as exc:
            pytest.skip(f"CuTeDSL grouped GEMM unavailable: {exc}")

        max_abs, mean_abs, rmse = self._error_metrics(got, ref)
        assert max_abs < 0.12
        assert mean_abs < 0.03
        assert rmse < 0.04

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
                0.1 * torch.randn(m, k, device=device, dtype=torch.bfloat16)
            )
            for m in m_splits
        ]
        b_list = [
            quantize_activation_nvfp4_pertoken(
                0.1 * torch.randn(n, k, device=device, dtype=torch.bfloat16)
            )
            for _ in m_splits
        ]

        ref = reference_grouped_gemm_pertoken(a_list, b_list, m_splits, out_dtype)
        try:
            got = grouped_gemm_cutedsl_pertoken(a_list, b_list, m_splits, out_dtype)
        except (TypeError, NotImplementedError) as exc:
            pytest.skip(f"CuTeDSL grouped GEMM unavailable: {exc}")

        max_abs, mean_abs, rmse = self._error_metrics(got, ref)
        assert max_abs < 0.12
        assert mean_abs < 0.03
        assert rmse < 0.04
