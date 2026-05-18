"""
Binding tests for ``fp4_gemm.quantize_transpose_nvfp4`` (csrc/ops.h).

The CUDA implementation targets Blackwell (sm_100+) TMA paths; on older GPUs
the kernel is not usable. Tests skip when the extension is missing or the
device capability is below 10.0.
"""

import pytest
import torch
from miniTransformer.ops.nvfp4_quantization import quantize_nvfp4_cuda

try:
    import fp4_gemm
except ImportError:
    fp4_gemm = None

_SKIP_REASON = None
if fp4_gemm is None:
    _SKIP_REASON = "fp4_gemm extension not built"
elif not torch.cuda.is_available():
    _SKIP_REASON = "CUDA is not available"
else:
    _maj, _min = torch.cuda.get_device_capability()
    if _maj < 10:
        _SKIP_REASON = (
            f"quantize_transpose_nvfp4 requires sm_100+ (got capability {_maj}.{_min})"
        )


def _alloc_nvfp4_buffers(rows: int, cols: int, return_transpose: bool):
    """
    Allocate output buffers consistent with quantize_transpose_nvfp4.cu:

    - input: bf16 [rows, cols]
    - output: uint8 [rows, cols // 2] (packed e2m1 x2 per byte)
    - output_scale: float8_e4m3fn [rows, cols // 16] (one scale per 16 columns)
    - if return_transpose: uint8 [cols, rows // 2] and float8_e4m3fn [cols, rows // 16]
    """
    out = torch.empty((rows, cols // 2), dtype=torch.uint8, device="cuda")
    out_s = torch.empty((rows, cols // 16), dtype=torch.float8_e4m3fn, device="cuda")
    if return_transpose:
        out_t = torch.empty((cols, rows // 2), dtype=torch.uint8, device="cuda")
        out_st = torch.empty((cols, rows // 16), dtype=torch.float8_e4m3fn, device="cuda")
    else:
        # Host still passes valid tensors; the RETURN_TRANSPOSE=false kernel does not use them.
        out_t = torch.empty((1, 1), dtype=torch.uint8, device="cuda")
        out_st = torch.empty((1, 1), dtype=torch.float8_e4m3fn, device="cuda")
    return out, out_s, out_t, out_st


@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON)
@pytest.mark.parametrize("use_2d_quantization", [False, True])
@pytest.mark.parametrize("return_transpose", [False, True])
def test_quantize_transpose_nvfp4_runs(use_2d_quantization, return_transpose):
    rows, cols = 128, 128
    inp = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)

    out, out_s, out_t, out_st = _alloc_nvfp4_buffers(rows, cols, return_transpose)

    # Single-element tensors; 2D kernel reads *amax_* when use_2d_quantization is True.
    amax_row = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    amax_col = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    fp4_gemm.quantize_transpose_nvfp4(
        inp,
        amax_row,
        amax_col,
        out,
        out_s,
        out_t,
        out_st,
        use_2d_quantization,
        return_transpose,
    )
    torch.cuda.synchronize()

    assert out.shape == (rows, cols // 2)
    assert out_s.shape == (rows, cols // 16)
    if return_transpose:
        assert out_t.shape == (cols, rows // 2)
        assert out_st.shape == (cols, rows // 16)


@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON)
def test_quantize_transpose_nvfp4_zeros_deterministic():
    """Smoke test: all-zero input should complete without NaN scales."""
    rows, cols = 256, 256
    inp = torch.zeros(rows, cols, device="cuda", dtype=torch.bfloat16)
    out, out_s, out_t, out_st = _alloc_nvfp4_buffers(rows, cols, return_transpose=True)
    amax_row = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    amax_col = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    fp4_gemm.quantize_transpose_nvfp4(
        inp,
        amax_row,
        amax_col,
        out,
        out_s,
        out_t,
        out_st,
        use_2d_quantization=False,
        return_transpose=True,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(out_s.to(torch.float32)).all()
    assert torch.isfinite(out_st.to(torch.float32)).all()


@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON)
@pytest.mark.parametrize("return_transpose", [False, True])
def test_quantize_nvfp4_cuda_wrapper_shapes(return_transpose: bool):
    rows, cols = 128, 128
    inp = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    result = quantize_nvfp4_cuda(
        inp,
        with_2d_quantization=True,
        return_transpose=return_transpose,
    )

    assert result.data.shape == (rows, cols // 2)
    assert result.scale.shape == (rows, cols // 16)
    assert result.tensor_scale is not None and result.tensor_scale.shape == (1,)

    if return_transpose:
        assert result.data_t is not None and result.data_t.shape == (cols, rows // 2)
        assert result.scale_t is not None and result.scale_t.shape == (cols, rows // 16)
        assert result.tensor_scale_t is not None and result.tensor_scale_t.shape == (1,)
    else:
        assert result.data_t is None
        assert result.scale_t is None
