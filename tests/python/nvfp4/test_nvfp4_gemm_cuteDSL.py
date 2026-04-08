from re import X
import time
import sys
from pathlib import Path
import pytest
import torch

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import miniTransformer.ops.nvfp4_qgemm as nvfp4_qgemm

from cutlass.cute.runtime import make_ptr

from miniTransformer.ops.cuteDSL.nvfp4.gemm_cute_norm import ab_dtype, sf_dtype, sf_vec_size
from miniTransformer.ops.cuteDSL.nvfp4.gemm_cute_opts import Nvfp4KernelVariant

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

_DEEPGEMM_NK_NORMAL_ONLY = [
    (24576, 1536),
    (32768, 512),
    (7168, 16384),
    (4096, 7168),
    (7168, 2048),
]
_DEEPGEMM_M_FWD = (128, 4096)

def _deepgemm_normal_only_params():
    params = []
    for m in _DEEPGEMM_M_FWD:
        for n, k in _DEEPGEMM_NK_NORMAL_ONLY:
            params.append(
                pytest.param(m, k, n, id=f"deepgemm_m{m}_n{n}_k{k}")
            )
        n, k = 256, 7168
        params.append(
            pytest.param(m, k, n, id=f"deepgemm_m{m}_n{n}_k{k}_bf16_extra")
        )
    for m in _DEEPGEMM_M_FWD:
        n, k = 129280, 7168
        params.append(
            pytest.param(
                m,
                k,
                n,
                id=f"deepgemm_m{m}_n{n}_k{k}_bf16_extra_large",
                marks=pytest.mark.slow,
            )
        )
    return params


GEMM_WORKLOAD_PARAMS_BASE = [
    pytest.param(2048, 5120, 10240, id="linear_qkv"),
    pytest.param(8192, 2048, 1280, id="linear_proj"),
    pytest.param(2048, 5120, 51200, id="linear_fc1"),
    pytest.param(8192, 6400, 1280, id="linear_fc2"),

    pytest.param(256, 256, 256, id="function_test0"),
    pytest.param(8192, 8192, 8192, id="function_test1"),
]

GEMM_WORKLOAD_PARAMS_NORMAL = GEMM_WORKLOAD_PARAMS_BASE + _deepgemm_normal_only_params()
GEMM_WORKLOAD_PARAMS_OPTIMIZED = GEMM_WORKLOAD_PARAMS_BASE
GEMM2_WORKLOAD_PARAMS = [
    pytest.param(256, 256, 256, id="gemm2_function_test0"),
    pytest.param(8192, 8192, 8192, id="gemm2_function_test1"),
]
GEMM345_WORKLOAD_PARAMS = [
    pytest.param(256, 256, 256, id="gemm345_function_test0"),
]

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


def _run_cutedsl_nvfp4_gemm2(
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
):

    try:
        from miniTransformer.ops.cuteDSL.nvfp4.gemm_cute_opts2 import Sm100BlockScaledDenseGemmKernel
    except ModuleNotFoundError as e:
        pytest.skip(f"gemm_cute_opts2 import unavailable: {e}")

    alpha_value = float(alpha.item())
    l = 1
    sf_k = ceil_div(K, sf_vec_size)

    a_ref_f4 = qx.view(torch.float4_e2m1fn_x2)
    b_ref_f4 = qw.view(torch.float4_e2m1fn_x2)
    c_tensor = torch.empty((M, N, l), dtype=out_dtype, device=device)

    def _ref_2d_to_cute_sf(data_2d, mn):
        ref = data_2d.unsqueeze(-1).to(dtype=torch.float8_e4m3fn, device="cpu")
        return nvfp4_qgemm.create_cute_scale_factor_tensor(l, mn, sf_k, ref)

    sfa_tensor = _ref_2d_to_cute_sf(sx, M)
    sfb_tensor = _ref_2d_to_cute_sf(sw, N)

    a_ptr = make_ptr(ab_dtype, a_ref_f4.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(ab_dtype, b_ref_f4.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    if c_tensor.dtype == torch.bfloat16:
        c_cutlass_dtype = cutlass.BFloat16
    elif c_tensor.dtype == torch.float16:
        c_cutlass_dtype = cutlass.Float16
    else:
        c_cutlass_dtype = cutlass.Float32
    c_ptr = make_ptr(c_cutlass_dtype, c_tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)

    sfa_ptr = make_ptr(sf_dtype, sfa_tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    sfb_ptr = make_ptr(sf_dtype, sfb_tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)

    gemm = Sm100BlockScaledDenseGemmKernel()
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
        lambda x: x,
    )
    compiled(
        a_ptr,
        b_ptr,
        sfa_ptr,
        sfb_ptr,
        c_ptr,
        (M, N, K, l),
        stream,
    )
    return c_tensor.squeeze(-1).float().to(out_dtype)


def _run_cutedsl_nvfp4_gemm_optsx(
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
    module_path,
    kernel_cls_name,
    needs_max_active_clusters=False,
):

    try:
        mod = __import__(module_path, fromlist=[kernel_cls_name])
        kernel_cls = getattr(mod, kernel_cls_name)
    except ModuleNotFoundError as e:
        pytest.skip(f"{module_path} import unavailable: {e}")

    l = 1
    sf_k = ceil_div(K, sf_vec_size)
    a_ref_f4 = qx.view(torch.float4_e2m1fn_x2)
    b_ref_f4 = qw.view(torch.float4_e2m1fn_x2)
    c_tensor = torch.empty((M, N, l), dtype=out_dtype, device=device)

    def _ref_2d_to_cute_sf(data_2d, mn):
        ref = data_2d.unsqueeze(-1).to(dtype=torch.float8_e4m3fn, device="cpu")
        return nvfp4_qgemm.create_cute_scale_factor_tensor(l, mn, sf_k, ref)

    sfa_tensor = _ref_2d_to_cute_sf(sx, M)
    sfb_tensor = _ref_2d_to_cute_sf(sw, N)

    a_ptr = make_ptr(ab_dtype, a_ref_f4.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(ab_dtype, b_ref_f4.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(cutlass.Float16, c_tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    sfa_ptr = make_ptr(sf_dtype, sfa_tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    sfb_ptr = make_ptr(sf_dtype, sfb_tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)

    gemm = kernel_cls()
    stream = cutlass_torch.default_stream()
    compile_args = [
        gemm,
        a_ptr,
        b_ptr,
        sfa_ptr,
        sfb_ptr,
        c_ptr,
        (M, N, K, l),
        stream,
    ]
    run_args = [a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (M, N, K, l), stream]
    if needs_max_active_clusters:
        max_active_clusters = 32
        compile_args.append(max_active_clusters)
    compile_args.append(lambda x: x)
    compiled = cute.compile(*compile_args)
    compiled(*run_args)
    return c_tensor.squeeze(-1).float().to(out_dtype)


def _run_cutedsl_nvfp4_gemm3_1(qx, qw, sx, sw, M, N, K, alpha, out_dtype, device):
    return _run_cutedsl_nvfp4_gemm_optsx(
        qx, qw, sx, sw, M, N, K, alpha, out_dtype, device,
        "miniTransformer.ops.cuteDSL.nvfp4.gemm_cute_opts3_1",
        "Sm100BlockScaledDenseGemm3_1Kernel",
    )


def _run_cutedsl_nvfp4_gemm3(qx, qw, sx, sw, M, N, K, alpha, out_dtype, device):
    return _run_cutedsl_nvfp4_gemm_optsx(
        qx, qw, sx, sw, M, N, K, alpha, out_dtype, device,
        "miniTransformer.ops.cuteDSL.nvfp4.gemm_cute_opts3",
        "Sm100BlockScaledDenseGemm3Kernel",
        needs_max_active_clusters=True,
    )


def _run_cutedsl_nvfp4_gemm4(qx, qw, sx, sw, M, N, K, alpha, out_dtype, device):
    return _run_cutedsl_nvfp4_gemm_optsx(
        qx, qw, sx, sw, M, N, K, alpha, out_dtype, device,
        "miniTransformer.ops.cuteDSL.nvfp4.gemm_cute_opts4",
        "Sm100BlockScaledDenseGemm4Kernel",
    )


def _run_cutedsl_nvfp4_gemm5(qx, qw, sx, sw, M, N, K, alpha, out_dtype, device):
    return _run_cutedsl_nvfp4_gemm_optsx(
        qx, qw, sx, sw, M, N, K, alpha, out_dtype, device,
        "miniTransformer.ops.cuteDSL.nvfp4.gemm_cute_opts5",
        "Sm100BlockScaledDenseGemm5Kernel",
    )


def _run_and_validate_opts_variant(runner, M, K, N, out_dtype, device):
    seed = torch.randint(2**31 - 1, ()).item()
    qx, qw, sx, sw, _, _, _ = nvfp4_qgemm.initialize_nvfp4_for_qgemm(
        M, N, K, device=device, seed=seed
    )
    sx = (sx.to(torch.float32) / 256.0).to(torch.float8_e4m3fn)
    sw = (sw.to(torch.float32) / 256.0).to(torch.float8_e4m3fn)
    alpha_t = torch.tensor(1.0, device=device, dtype=torch.float32)

    y_ref = _run_cutedsl_nvfp4_gemm_ref(
        qx, qw, sx, sw, M, N, K, alpha_t, out_dtype, device, has_swizzle=True
    )
    d_cute = runner(qx, qw, sx, sw, M, N, K, alpha_t, out_dtype, device)

    assert d_cute.shape == (M, N)
    assert d_cute.dtype == out_dtype
    assert not torch.isnan(d_cute).any()
    assert not torch.isinf(d_cute).any()

    torch.testing.assert_close(
        y_ref.to(torch.float32),
        d_cute.to(torch.float32),
        rtol=1e-2,
        atol=1e-6,
        msg=f"CuTeDSL NVFP4 opts variant mismatch for M={M}, K={K}, N={N}, seed={seed}",
    )


def _run_and_validate_optimized_path(
    *,
    M,
    K,
    N,
    accumulate,
    out_dtype,
    has_swizzle,
    do_benchmark,
    do_autotune,
    kernel_variant: Nvfp4KernelVariant,
    device="cuda",
):
    seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    out_init = torch.randn((M, N), dtype=out_dtype, device=device) if accumulate else None
    out_init_dsl = out_init.clone() if accumulate else None

    qx, qw, sx, sw, global_amax_qx, global_amax_qw, _ = (
        nvfp4_qgemm.initialize_nvfp4_for_qgemm(M, N, K, device=device, seed=seed)
    )

    if accumulate:
        # accu mode only use in wgrad, aplha = 1.0
        alpha_t = torch.tensor(1.0, device=device, dtype=torch.float32)
    else:
        factor = 6.0 * 6.0 * 448.0 * 448.0
        alpha = (global_amax_qx * global_amax_qw).to(torch.float32).squeeze(-1).item() / factor
        alpha_t = torch.tensor(alpha, device=device, dtype=torch.float32)
  
    y_ref = _run_cutedsl_nvfp4_gemm_ref(
        qx, qw, sx, sw, M, N, K, alpha_t, out_dtype, device,
        return_fp32=accumulate, has_swizzle=has_swizzle
    )
    if accumulate:
        y_ref = (y_ref + out_init.float()).to(out_dtype)

    d_cute = nvfp4_qgemm.cutedsl_nvfp4_gemm_optimized(
        qx, qw, sx, sw, M, N, K, alpha_t, out_dtype, device,
        kernel_variant=kernel_variant,
        out=out_init_dsl,
        accumulate=accumulate,
        has_swizzle=has_swizzle,
        do_benchmark=do_benchmark,
        do_autotune=do_autotune,
        warmup_iterations=10,
        iterations=100,
        use_cold_l2=False,
    )

    assert d_cute.shape == (M, N)
    assert d_cute.dtype == out_dtype
    assert not torch.isnan(d_cute).any()
    assert not torch.isinf(d_cute).any()

    y_ref_f32 = y_ref.to(torch.float32)
    y_cute_f32 = out_init_dsl.to(torch.float32) if accumulate else d_cute.to(torch.float32)

    torch.testing.assert_close(
        y_ref_f32,
        y_cute_f32,
        rtol=1e-2,
        atol=1e-6,
        msg=f"CuTeDSL NVFP4 GEMM mismatch for M={M}, K={K}, N={N}, seed={seed}, accumulate={accumulate}",
    )

class TestCuTeDSLNvfp4Gemm:
    @pytest.mark.parametrize("M, K, N", GEMM2_WORKLOAD_PARAMS)
    @pytest.mark.parametrize("acc_dtype", [torch.float16])
    def test_cutedsl_nvfp4_gemm2_kernel(self, M, K, N, acc_dtype):
        device = "cuda"
        out_dtype = acc_dtype
        seed = torch.randint(2**31 - 1, ()).item()

        qx, qw, sx, sw, global_amax_qx, global_amax_qw, qparams = (
            nvfp4_qgemm.initialize_nvfp4_for_qgemm(M, N, K, device=device, seed=seed)
        )

        # GEMM2 tutorial kernel writes FP16 C; keep scale factors in a stable range.
        sx = (sx.to(torch.float32) / 256.0).to(torch.float8_e4m3fn)
        sw = (sw.to(torch.float32) / 256.0).to(torch.float8_e4m3fn)
        alpha_t = torch.tensor(1.0, device=device, dtype=torch.float32)

        y_ref = _run_cutedsl_nvfp4_gemm_ref(
            qx, qw, sx, sw, M, N, K, alpha_t, out_dtype, device, has_swizzle=True
        )

        assert y_ref.shape == (M, N)
        assert y_ref.dtype == out_dtype

        D_cute = _run_cutedsl_nvfp4_gemm2(
            qx, qw, sx, sw, M, N, K, alpha_t, out_dtype, device
        )

        assert D_cute.shape == (M, N)
        assert D_cute.dtype == out_dtype
        assert not torch.isnan(D_cute).any()
        assert not torch.isinf(D_cute).any()

        y_ref_f32 = y_ref.to(torch.float32)
        y_cute_f32 = D_cute.to(torch.float32)

        torch.testing.assert_close(
            y_ref_f32,
            y_cute_f32,
            rtol=1e-2,
            atol=1e-6,
            msg=f"CuTeDSL NVFP4 GEMM2 mismatch for M={M}, K={K}, N={N}, seed={seed}",
        )

    @pytest.mark.parametrize("M, K, N", GEMM345_WORKLOAD_PARAMS)
    @pytest.mark.parametrize("acc_dtype", [torch.float16])
    def test_cutedsl_nvfp4_gemm3_1_kernel(self, M, K, N, acc_dtype):
        _run_and_validate_opts_variant(
            _run_cutedsl_nvfp4_gemm3_1, M, K, N, acc_dtype, "cuda"
        )

    @pytest.mark.parametrize("M, K, N", GEMM345_WORKLOAD_PARAMS)
    @pytest.mark.parametrize("acc_dtype", [torch.float16])
    def test_cutedsl_nvfp4_gemm3_kernel(self, M, K, N, acc_dtype):
        _run_and_validate_opts_variant(
            _run_cutedsl_nvfp4_gemm3, M, K, N, acc_dtype, "cuda"
        )

    @pytest.mark.parametrize("M, K, N", GEMM345_WORKLOAD_PARAMS)
    @pytest.mark.parametrize("acc_dtype", [torch.float16])
    def test_cutedsl_nvfp4_gemm4_kernel(self, M, K, N, acc_dtype):
        _run_and_validate_opts_variant(
            _run_cutedsl_nvfp4_gemm4, M, K, N, acc_dtype, "cuda"
        )

    @pytest.mark.parametrize("M, K, N", GEMM345_WORKLOAD_PARAMS)
    @pytest.mark.parametrize("acc_dtype", [torch.float16])
    def test_cutedsl_nvfp4_gemm5_kernel(self, M, K, N, acc_dtype):
        _run_and_validate_opts_variant(
            _run_cutedsl_nvfp4_gemm5, M, K, N, acc_dtype, "cuda"
        )


    @pytest.mark.parametrize("M, K, N", GEMM_WORKLOAD_PARAMS_OPTIMIZED)
    @pytest.mark.parametrize("accumulate", [False, True])
    @pytest.mark.parametrize("acc_dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("do_benchmark", [False])
    @pytest.mark.parametrize("do_autotune", [False])
    @pytest.mark.parametrize("kernel_variant", [Nvfp4KernelVariant.OPTS0])
    def test_cutedsl_nvfp4_normal(
        self, M, K, N, accumulate, acc_dtype, do_benchmark, do_autotune, kernel_variant
    ):
        _run_and_validate_optimized_path(
            M=M,
            K=K,
            N=N,
            accumulate=accumulate,
            out_dtype=acc_dtype,
            has_swizzle=False,
            do_benchmark=do_benchmark,
            do_autotune=do_autotune,
            kernel_variant=kernel_variant,
        )

    @pytest.mark.parametrize("M, K, N", GEMM_WORKLOAD_PARAMS_OPTIMIZED)
    @pytest.mark.parametrize("accumulate", [False, True])
    @pytest.mark.parametrize("acc_dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("has_swizzle", [False])
    @pytest.mark.parametrize("do_benchmark", [False, True])
    @pytest.mark.parametrize("do_autotune", [False, True])
    @pytest.mark.parametrize("kernel_variant", [Nvfp4KernelVariant.CURRENT_OPTIMIZED])
    def test_cutedsl_nvfp4_optimized(
        self, M, K, N, accumulate, acc_dtype, has_swizzle, do_benchmark, do_autotune, kernel_variant
    ):
        _run_and_validate_optimized_path(
            M=M,
            K=K,
            N=N,
            accumulate=accumulate,
            out_dtype=acc_dtype,
            has_swizzle=has_swizzle,
            do_benchmark=do_benchmark,
            do_autotune=do_autotune,
            kernel_variant=kernel_variant,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
