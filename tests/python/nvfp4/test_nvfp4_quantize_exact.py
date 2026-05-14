"""
Exact / precision checks for NVFP4 quantization:

1. Transformer Engine ``NVFP4Quantizer`` vs miniTransformer ``Nvfp4TiledQuantizeRefOp``.
2. CUDA ``fp4_gemm.quantize_transpose_nvfp4`` vs TE ``NVFP4Quantizer`` for both
   ``with_2d_quantization=True`` (2D kernel) and ``with_2d_quantization=False`` on BF16
   (TE-aligned ``quantize_transpose_tuned_1D`` port).

The custom CUDA kernel targets Blackwell (sm_100+); TE-vs-CUDA checks are skipped on older GPUs.
"""

from __future__ import annotations

import pytest
import torch

try:
    import fp4_gemm
except ImportError:
    fp4_gemm = None

from miniTransformer.quantiztion.nvfp4_quantize_op_ref import Nvfp4TiledQuantizeRefOp
from miniTransformer.quantiztion import quantization
from miniTransformer.quantiztion.quantization import QParams

try:
    from transformer_engine.pytorch import NVFP4Quantizer
    import transformer_engine_torch as tex
except ImportError:
    NVFP4Quantizer = None
    tex = None

nvfp4_output_t = torch.uint8
nvfp4_scale_t = torch.float8_e4m3fn

_SKIP_CUDA_TE = None
if fp4_gemm is None:
    _SKIP_CUDA_TE = "fp4_gemm extension not built"
elif NVFP4Quantizer is None:
    _SKIP_CUDA_TE = "transformer_engine not installed"
elif not torch.cuda.is_available():
    _SKIP_CUDA_TE = "CUDA not available"
else:
    _maj, _min = torch.cuda.get_device_capability()
    if _maj < 10:
        _SKIP_CUDA_TE = (
            f"quantize_transpose_nvfp4 requires sm_100+ (got capability {_maj}.{_min})"
        )


def unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    repeated = x.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F
    repeated[:, 1::2] >>= 4
    return repeated


def _te_global_amax_1d(te_field: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    """Single float32 element on ``device`` for CUDA global encode (2D kernel)."""
    if te_field is None:
        return torch.tensor([1.0], dtype=torch.float32, device=device)
    t = te_field.detach().to(device=device, dtype=torch.float32).reshape(-1)
    if t.numel() == 0:
        return torch.tensor([1.0], dtype=torch.float32, device=device)
    return t.max().reshape(1).contiguous()


def _alloc_cuda_quant_buffers(
    rows: int, cols: int, return_transpose: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out = torch.empty((rows, cols // 2), dtype=torch.uint8, device="cuda")
    out_s = torch.empty((rows, cols // 16), dtype=torch.float8_e4m3fn, device="cuda")
    if return_transpose:
        out_t = torch.empty((cols, rows // 2), dtype=torch.uint8, device="cuda")
        out_st = torch.empty((cols, rows // 16), dtype=torch.float8_e4m3fn, device="cuda")
    else:
        out_t = torch.empty((1, 1), dtype=torch.uint8, device="cuda")
        out_st = torch.empty((1, 1), dtype=torch.float8_e4m3fn, device="cuda")
    return out, out_s, out_t, out_st


def check_quantization_nvfp4_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_transpose: bool,
    with_2d_quantization: bool,
) -> None:
    device = torch.device("cuda")
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    x = torch.randn((M, N), dtype=x_dtype, device=device)

    nvfp4_quantizer = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=return_transpose,
        with_amax_reduction=False,
        with_2d_quantization=with_2d_quantization,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
    )
    x_nvfp4_sut = nvfp4_quantizer.make_empty(
        (M, N), dtype=x_dtype, device=device, requires_grad=False
    )
    x_nvfp4_sut = nvfp4_quantizer.update_quantized(x, x_nvfp4_sut)
    assert x_nvfp4_sut._rowwise_data is not None
    qx_te: torch.Tensor = x_nvfp4_sut._rowwise_data.view(dtype=torch.uint8)
    assert x_nvfp4_sut._rowwise_scale_inv is not None
    sx_te: torch.Tensor = x_nvfp4_sut._rowwise_scale_inv

    tile_m, tile_n = ((16, 16) if with_2d_quantization else (1, 16))
    ref_op = Nvfp4TiledQuantizeRefOp()
    param = QParams(
        output_dtype=torch.uint8,
        scaling_dtype=torch.float8_e4m3fn,
        quant_tile_shape=(tile_m, tile_n),
        with_2d_quantization=with_2d_quantization,
        with_rht=False,
    )
    result = ref_op.quantize(x, param, return_transpose=return_transpose, reduce_amax=False)
    qx_ref, sx_ref = result.data, result.scale

    torch.testing.assert_close(qx_te, qx_ref.view(dtype=torch.uint8), atol=0.0, rtol=0.0)
    torch.testing.assert_close(sx_te, sx_ref.view(dtype=torch.uint8), atol=0.0, rtol=0.0)


def check_quantization_cuda_versus_te(
    M: int,
    N: int,
    return_transpose: bool,
    with_2d_quantization: bool,
    *,
    bit_exact: bool = True,
) -> None:
    """
    Compare ``fp4_gemm.quantize_transpose_nvfp4`` against TE ``NVFP4Quantizer``.

    **Bit-exact** (``bit_exact=True``): CUDA matches TE when either ``with_2d_quantization=True``
    (2D kernel) or ``with_2d_quantization=False`` on BF16 (tuned 1D kernel, aligned with TE
    ``quantize_transpose_nvfp4_tuned_1D``).
    """
    if M % 32 != 0 or N % 32 != 0:
        raise ValueError("quantize_transpose_nvfp4 requires M and N to be multiples of 32")

    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    x_dtype = torch.bfloat16
    x = torch.randn((M, N), dtype=x_dtype, device=device)

    te_q = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=return_transpose,
        with_amax_reduction=False,
        with_2d_quantization=with_2d_quantization,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
    )
    x_te = te_q.make_empty((M, N), dtype=x_dtype, device=device, requires_grad=False)
    x_te = te_q.update_quantized(x, x_te)

    q_te = x_te._rowwise_data.view(torch.uint8)
    s_te = x_te._rowwise_scale_inv
    qt_te = (
        x_te._columnwise_data.view(torch.uint8)
        if x_te._columnwise_data is not None
        else None
    )
    st_te = x_te._columnwise_scale_inv

    out, out_s, out_t, out_st = _alloc_cuda_quant_buffers(M, N, return_transpose)
    amax_row = _te_global_amax_1d(x_te._amax_rowwise, device)
    amax_col = _te_global_amax_1d(
        x_te._amax_columnwise if x_te._amax_columnwise is not None else x_te._amax_rowwise,
        device,
    )

    fp4_gemm.quantize_transpose_nvfp4(
        x,
        amax_row,
        amax_col,
        out,
        out_s,
        out_t,
        out_st,
        with_2d_quantization,
        return_transpose,
    )
    torch.cuda.synchronize()

    if not bit_exact:
        assert torch.isfinite(out_s.to(torch.float32)).all()
        if return_transpose:
            assert qt_te is not None and st_te is not None
            assert torch.isfinite(out_st.to(torch.float32)).all()
        return

    torch.testing.assert_close(out, q_te, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        out_s.view(torch.uint8), s_te.view(torch.uint8), atol=0.0, rtol=0.0
    )
    if return_transpose:
        assert qt_te is not None and st_te is not None
        torch.testing.assert_close(out_t, qt_te, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            out_st.view(torch.uint8), st_te.view(torch.uint8), atol=0.0, rtol=0.0
        )


@pytest.mark.skipif(NVFP4Quantizer is None, reason="transformer_engine not installed")
def test_te_vs_reference_bf16():
    check_quantization_nvfp4_versus_reference(
        x_dtype=torch.bfloat16,
        M=1024,
        N=4096,
        return_transpose=True,
        with_2d_quantization=True,
    )


@pytest.mark.skipif(_SKIP_CUDA_TE is not None, reason=_SKIP_CUDA_TE or "")
@pytest.mark.parametrize("return_transpose", [False, True])
@pytest.mark.parametrize("with_2d_quantization", [False, True])
def test_cuda_quantize_transpose_vs_te(return_transpose: bool, with_2d_quantization: bool):
    check_quantization_cuda_versus_te(
        M=1024,
        N=4096,
        return_transpose=return_transpose,
        with_2d_quantization=with_2d_quantization,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
