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

from miniTransformer.ops.cuteDSL.nvfp4.gemm_cute_opts0 import ab_dtype, sf_dtype, sf_vec_size
from miniTransformer.ops.cuteDSL.nvfp4.gemm_cute_opts1 import Nvfp4KernelVariant

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

DEEPGEMM_NORMAL_ONLY_PARAMS = [
    pytest.param(128, 16384, 7168, id="deepgemm_2"),
    pytest.param(128, 7168, 4096, id="deepgemm_3"),
    pytest.param(4096, 1536, 24576, id="deepgemm_6"),
    pytest.param(4096, 512, 32768, id="deepgemm_7"),
    pytest.param(4096, 16384, 7168, id="deepgemm_8"),
    pytest.param(4096, 7168, 4096, id="deepgemm_9"),
    pytest.param(4096, 2048, 7168, id="deepgemm_10"),
    pytest.param(4096, 7168, 256, id="deepgemm_11"),
    pytest.param(128,7168,129280, id="deepgemm_12"),
    pytest.param(4096,7168,129280, id="deepgemm_13"),
]


GEMM_WORKLOAD_PARAMS_BASE = [
    pytest.param(2048, 5120, 10240, id="linear_qkv"),
    pytest.param(8192, 2048, 1280, id="linear_proj"),
    pytest.param(2048, 5120, 51200, id="linear_fc1"),
    pytest.param(8192, 6400, 1280, id="linear_fc2"),

    pytest.param(256, 256, 256, id="function_test0"),
    pytest.param(8192, 8192, 8192, id="function_test1"),
]

GEMM_WORKLOAD_PARAMS_NORMAL = GEMM_WORKLOAD_PARAMS_BASE + DEEPGEMM_NORMAL_ONLY_PARAMS
GEMM_WORKLOAD_PARAMS_OPTIMIZED = GEMM_WORKLOAD_PARAMS_BASE


def ref_kernel(
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


def make_match_reference(
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
  
    y_ref = ref_kernel(
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
 
    @pytest.mark.parametrize("M, K, N", GEMM_WORKLOAD_PARAMS_OPTIMIZED)
    @pytest.mark.parametrize("accumulate", [False, True])
    @pytest.mark.parametrize("acc_dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("do_benchmark", [False])
    @pytest.mark.parametrize("do_autotune", [False])
    @pytest.mark.parametrize(
        "kernel_variant",
        [
            Nvfp4KernelVariant.OPTS0,
            Nvfp4KernelVariant.OPTS1,
            Nvfp4KernelVariant.OPTS2,
            Nvfp4KernelVariant.OPTS3,
            Nvfp4KernelVariant.OPTS4,
            Nvfp4KernelVariant.OPTS5,
            Nvfp4KernelVariant.OPTS6,
        ],
    )
    def test_cutedsl_nvfp4_all_kernels(
        self, M, K, N, accumulate, acc_dtype, do_benchmark, do_autotune, kernel_variant
    ):
        make_match_reference(
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
    @pytest.mark.parametrize("accumulate", [False])
    @pytest.mark.parametrize("acc_dtype", [torch.bfloat16])
    @pytest.mark.parametrize("has_swizzle", [False])
    @pytest.mark.parametrize("do_benchmark", [True])
    @pytest.mark.parametrize("do_autotune", [False])
    @pytest.mark.parametrize("kernel_variant", [Nvfp4KernelVariant.OPTS2])
    def test_cutedsl_nvfp4_current_kernels(
        self, M, K, N, accumulate, acc_dtype, has_swizzle, do_benchmark, do_autotune, kernel_variant
    ):
        make_match_reference(
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
