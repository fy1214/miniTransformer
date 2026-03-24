import time
import pytest
import torch

import cutlass
import cutlass.cute as cute
import miniTransformer.ops.nvfp4_qgemm as nvfp4_qgemm

from sympy import true
from cutlass.cute.runtime import make_ptr
from miniTransformer.quantiztion.nvfp4_quantize_op_ref import Nvfp4TiledQuantizeRefOp
from miniTransformer.ops.cuteDSL.nvfp4.scale_factor_swizzle import cvt_sf_MKL_to_M32x4xrm_K4xrk_L

def ceil_div(a, b):
    return (a + b - 1) // b

def to_blocked(input_matrix):
    rows, cols = input_matrix.shape

    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


def _run_cutedsl_nvfp4_gemm_ref(
    qx,
    qw,
    sx,
    sw,
    M,
    N,
    K,
    alpha,
    out_dtype,
    device,
    return_fp32: bool = False,
    has_swizzle: bool = True,
):

    l = 1
    a_ref_f4 = qx.view(torch.float4_e2m1fn_x2) 
    b_ref_f4 = qw.view(torch.float4_e2m1fn_x2) 

    sx_f8 = sx.to(torch.float8_e4m3fn) if sx.dtype != torch.float8_e4m3fn else sx
    sw_f8 = sw.to(torch.float8_e4m3fn) if sw.dtype != torch.float8_e4m3fn else sw
    if has_swizzle:
        scale_a = to_blocked(sx_f8).to(device)
        scale_b = to_blocked(sw_f8).to(device)
    else:
        scale_a = sx_f8.to(device)
        scale_b = sw_f8.to(device)

    res = torch._scaled_mm(
        a_ref_f4,
        b_ref_f4.T,
        scale_a,
        scale_b,
        bias=None,
        out_dtype=torch.float32,
    )
    y_fp32 = res * alpha
    if return_fp32:
        return y_fp32
    return y_fp32.to(out_dtype)

def _run_cutedsl_nvfp4_gemm(qx, qw, sx, sw, M, N, K, alpha, out_dtype, device):
    """Run CuTeDSL NVFP4 block-scaled GEMM and return D = alpha * GEMM(qx, qw, sx, sw) in out_dtype."""
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass
    from cutlass.cute.runtime import make_ptr

    from miniTransformer.ops.cuteDSL.nvfp4.gemm_cute_norm import (
        NormalBlockScaledDenseGemmKernel,
        ab_dtype,
        sf_dtype,
        sf_vec_size,
    )

    alpha_value = float(alpha.item())
    l = 1
    sf_k = ceil_div(K, sf_vec_size)

    a_ref_f4 = qx.view(torch.float4_e2m1fn_x2)
    b_ref_f4 = qw.view(torch.float4_e2m1fn_x2)
    c_tensor = torch.empty((M, N, l), dtype=out_dtype, device=device)

    def _ref_2d_to_cute_sf(data_2d, mn):
        ref = nvfp4_qgemm.create_ref_scale_factor_tensor(l, mn, sf_k)
        ref.copy_(data_2d.unsqueeze(-1).to(ref.dtype).cpu())
        return nvfp4_qgemm.create_cute_scale_factor_tensor(l, mn, sf_k, ref)

    sfa_tensor = _ref_2d_to_cute_sf(sx, M)
    sfb_tensor = _ref_2d_to_cute_sf(sw, N)

    a_ptr = make_ptr(ab_dtype, a_ref_f4.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(ab_dtype, b_ref_f4.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    c_cutlass_dtype = cutlass.BFloat16 if c_tensor.dtype == torch.bfloat16 else cutlass.Float32
    c_ptr = make_ptr(c_cutlass_dtype, c_tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)

    sfa_ptr = make_ptr(sf_dtype, sfa_tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    sfb_ptr = make_ptr(sf_dtype, sfb_tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)

    gemm = NormalBlockScaledDenseGemmKernel()
    stream = cutlass_torch.default_stream()
    compiled = cute.compile(
        gemm,
        a_ptr,
        b_ptr,
        sfa_ptr,
        sfb_ptr,
        c_ptr,
        (M, N, K, l),
        stream,
        epilogue_op=lambda x: x * alpha_value,
    )
    compiled(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (M, N, K, l), stream)
    return c_tensor.squeeze(-1).float().to(out_dtype)


class TestCuTeDSLNvfp4Gemm:

    @pytest.mark.parametrize(
        "M, K, N, seed",
        [
            (256, 256, 256, 10),
            (256, 256, 256, 1234),
            (8192, 8192, 8192, 22),
            (8192, 8192, 8192, 1234),
        ],
    )
    def test_cutedsl_nvfp4_normal(self, M, K, N, seed):
        device = "cuda"
        out_dtype = torch.bfloat16

        qx, qw, sx, sw, global_amax_qx, global_amax_qw, qparams = (
            nvfp4_qgemm.initialize_nvfp4_for_qgemm(M, N, K, device=device, seed=seed)
        )

        factor = 6.0 * 6.0 * 448.0 * 448.0
        alpha = (global_amax_qx * global_amax_qw).to(torch.float32).squeeze(-1).item() / factor
        alpha_t = torch.tensor(alpha, device=device, dtype=torch.float32)

        # Reference implementation: block layout + torch._scaled_mm + alpha
        y_ref = _run_cutedsl_nvfp4_gemm_ref(
            qx, qw, sx, sw, M, N, K, alpha_t, out_dtype, device
        )

        assert y_ref.shape == (M, N)
        assert y_ref.dtype == out_dtype

        start = time.time()
        D_cute = _run_cutedsl_nvfp4_gemm(
            qx, qw, sx, sw, M, N, K, alpha_t, out_dtype, device
        )
        if device == "cuda":
            torch.cuda.synchronize()
        cutlass_time = time.time() - start

        assert D_cute.shape == (M, N)
        assert D_cute.dtype == out_dtype
        assert not torch.isnan(D_cute).any()
        assert not torch.isinf(D_cute).any()

        y_ref_f32 = y_ref.to(torch.float32)
        y_cute_f32 = D_cute.to(torch.float32)

        abs_diff = (y_ref_f32 - y_cute_f32).abs()
        rel_diff = abs_diff / (y_ref_f32.abs() + 1e-8)

        print(f"\nCuTeDSL NVFP4 GEMM M={M}, K={K}, N={N}, seed={seed}")
        print(f"  CuTeDSL time: {cutlass_time * 1000:.3f} ms")
        print(f"  Max abs diff: {abs_diff.max().item():.6f}")
        print(f"  Mean abs diff: {abs_diff.mean().item():.6f}")
        print(f"  Max rel diff: {rel_diff.max().item():.6f}")
        print(f"  Mean rel diff: {rel_diff.mean().item():.6f}")

        torch.testing.assert_close(
            y_ref_f32,
            y_cute_f32,
            rtol=1e-2,
            atol=1e-6,
            msg=f"CuTeDSL NVFP4 GEMM mismatch for M={M}, K={K}, N={N}, seed={seed}",
        )

    @pytest.mark.parametrize(
        "M, K, N, seed",
        [
            (256, 256, 256, 10),
            (256, 256, 256, 1234),
            (8192, 8192, 8192, 10),
            (8192, 8192, 8192, 1234),
        ],
    )
    @pytest.mark.parametrize("accumulate", [True, False])
    @pytest.mark.parametrize("acc_dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("has_swizzle", [True, False])
    def test_cutedsl_nvfp4_optimized(self, M, K, N, seed, accumulate, acc_dtype, has_swizzle):
        device = "cuda"
        out_dtype = acc_dtype
        out_init = torch.randn((M, N), dtype=out_dtype, device=device) if accumulate else None
        out_init_dsl = out_init.clone() if accumulate else None
        
        qx, qw, sx, sw, global_amax_qx, global_amax_qw, qparams = (
            nvfp4_qgemm.initialize_nvfp4_for_qgemm(M, N, K, device=device, seed=seed)
        )

        factor = 6.0 * 6.0 * 448.0 * 448.0
        alpha = (global_amax_qx * global_amax_qw).to(torch.float32).squeeze(-1).item() / factor
        alpha_t = torch.tensor(alpha, device=device, dtype=torch.float32)

        y_ref = _run_cutedsl_nvfp4_gemm_ref(
            qx, qw, sx, sw, M, N, K, alpha_t, out_dtype, device, return_fp32=accumulate, has_swizzle=has_swizzle
        )
        if accumulate:
            y_ref = (y_ref + out_init.float()).to(out_dtype)

        assert y_ref.shape == (M, N)
        assert y_ref.dtype == out_dtype

        start = time.time()
        D_cute = nvfp4_qgemm.cutedsl_nvfp4_gemm_optimized(
            qx, qw, sx, sw, M, N, K, alpha_t, out_dtype, device, out=out_init_dsl, accumulate=accumulate, has_swizzle=has_swizzle
        )

        if device == "cuda":
            torch.cuda.synchronize()
        cutlass_time = time.time() - start

        assert D_cute.shape == (M, N)
        assert D_cute.dtype == out_dtype
        assert not torch.isnan(D_cute).any()
        assert not torch.isinf(D_cute).any()

        y_ref_f32 = y_ref.to(torch.float32)
        if accumulate:
            y_cute_f32 = out_init_dsl.to(torch.float32)
        else:
            y_cute_f32 = D_cute.to(torch.float32)

        abs_diff = (y_ref_f32 - y_cute_f32).abs()
        rel_diff = abs_diff / (y_ref_f32.abs() + 1e-8)

        print(f"\nCuTeDSL NVFP4 GEMM M={M}, K={K}, N={N}, seed={seed}, accumulate={accumulate}")
        print(f"  CuTeDSL time: {cutlass_time * 1000:.3f} ms")
        print(f"  Max abs diff: {abs_diff.max().item():.6f}")
        print(f"  Mean abs diff: {abs_diff.mean().item():.6f}")
        print(f"  Max rel diff: {rel_diff.max().item():.6f}")
        print(f"  Mean rel diff: {rel_diff.mean().item():.6f}")

        torch.testing.assert_close(
            y_ref_f32,
            y_cute_f32,
            rtol=1e-2,
            atol=1e-6,
            msg=f"CuTeDSL NVFP4 GEMM mismatch for M={M}, K={K}, N={N}, seed={seed}, accumulate={accumulate}",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
