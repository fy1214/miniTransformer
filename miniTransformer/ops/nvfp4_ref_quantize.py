"""
NVFP4 quantize-dequantize reference — torch only (no miniTransformer / fp4_gemm).

Matches ``Nvfp4TiledQuantizeRefOp`` in ``nvfp4_quantize_op_ref.py`` for the path
used by ``tests/python/quantization/test_multi_quant_method.quant_dequant_nvfp4_REF``:

- ``quant_tile_shape=(1, 16)``
- ``pow_2_scales=False``
- ``with_2d_quantization=False``
- ``with_rht=False`` / ``return_transpose=False``
- ``per_token_amax=False`` → tensor-wide global amax (same as ref without ``per_token_amax`` on QParams)

Optional kwargs expose the rest of the ref op when needed.

CUDA Graph / SGLang
-------------------
Use ``quant_dequant_nvfp4_ref_for_cudagraph`` for replay inside a captured graph:

- Call from a single CUDA stream; keep ``x.shape`` / dtypes / ``per_token_amax`` /
  ``quant_tile_shape`` identical across warmup capture and replay.
- Pass a preallocated ``out`` to avoid a fresh allocation in the graph body.
- No ``.item()`` / host sync on GPU tensors in the hot path.
- If you use CUDAGraph trees / cudagraph-splitting, call
  ``maybe_cudagraph_mark_step_begin()`` once before the region that owns this op
  (see PyTorch ``torch.compiler.cudagraph_mark_step_begin``).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0


def maybe_cudagraph_mark_step_begin() -> None:
    """Call before a subgraph that uses this module if you rely on CUDA graph splitting."""
    fn = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if fn is not None:
        fn()


def cast_to_e8(decode_scale: torch.Tensor) -> torch.Tensor:
    max_exponent = 127
    exponent = torch.ceil(torch.log2(decode_scale))
    exponent = torch.clamp(exponent, min=-max_exponent, max=max_exponent)
    return torch.exp2(exponent)


def cast_to_fp4x2(x: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(x, dtype=torch.uint8)
    result[(x >= 0.0) & (x <= 0.25)] = 0
    result[(x > 0.25) & (x < 0.75)] = 1
    result[(x >= 0.75) & (x <= 1.25)] = 2
    result[(x > 1.25) & (x < 1.75)] = 3
    result[(x >= 1.75) & (x <= 2.5)] = 4
    result[(x > 2.5) & (x < 3.5)] = 5
    result[(x >= 3.5) & (x <= 5.0)] = 6
    result[x > 5.0] = 7

    result[(x >= -0.25) & (x < -0.0)] = 8
    result[(x < -0.25) & (x > -0.75)] = 9
    result[(x <= -0.75) & (x >= -1.25)] = 10
    result[(x < -1.25) & (x > -1.75)] = 11
    result[(x <= -1.75) & (x >= -2.5)] = 12
    result[(x < -2.5) & (x > -3.5)] = 13
    result[(x <= -3.5) & (x >= -5.0)] = 14
    result[x < -5.0] = 15

    return result[:, ::2] + result[:, 1::2] * 16


def _fp4_nibble_to_float(v: torch.Tensor, dq_dtype: torch.dtype) -> torch.Tensor:
    sign = (v >> 3) & 1
    exp = (v >> 1) & 0x3
    mant = v & 1
    normal_val = (1.0 + mant * 0.5) * (2.0 ** (exp.float() - 1))
    subnormal_val = mant * 0.5
    val = torch.where(exp == 0, subnormal_val.to(dq_dtype), normal_val.to(dq_dtype))
    val = torch.where(sign == 1, -val, val)
    return val


def cast_from_fp4x2(x: torch.Tensor, dq_dtype: torch.dtype) -> torch.Tensor:
    second_bit = torch.div(x, 16, rounding_mode="floor").to(torch.long)
    first_bit = (x - second_bit * 16).to(torch.long)
    first_bit_values = _fp4_nibble_to_float(first_bit, dq_dtype)
    second_bit_values = _fp4_nibble_to_float(second_bit, dq_dtype)
    result = torch.zeros(
        (x.shape[0], x.shape[1] * 2),
        device=x.device,
        dtype=dq_dtype,
    )
    result[:, ::2] = first_bit_values
    result[:, 1::2] = second_bit_values
    return result


def _pad_tensor(
    tensor: torch.Tensor,
    row_divisor: Optional[int],
    col_divisor: Optional[int],
) -> torch.Tensor:
    """Pad to row/col multiples using a single ``F.pad`` path (CUDA Graph friendly)."""
    assert tensor.dim() == 2, "only supports 2D tensors"
    m, n = int(tensor.shape[0]), int(tensor.shape[1])
    rd = row_divisor if row_divisor is not None else 1
    cd = col_divisor if col_divisor is not None else 1
    assert rd > 0 and cd > 0
    ph = (rd - (m % rd)) % rd
    pw = (cd - (n % cd)) % cd
    return torch.nn.functional.pad(
        tensor, (0, pw, 0, ph), mode="constant", value=0.0
    ).contiguous()


def _rm_pad_tensor(tensor: torch.Tensor, original_size: Tuple[int, int]) -> torch.Tensor:
    assert tensor.dim() == 2, "only supports 2D tensors"
    m, n = original_size
    return tensor[:m, :n].contiguous()


def _quantize_blockwise_reference(
    x: torch.Tensor,
    global_amax: torch.Tensor,
    tile_len_x: int,
    tile_len_y: int,
    *,
    pow_2_scales: bool,
    with_2d_quantization: bool,
    eps: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del eps  # unused; kept for signature parity with ref
    assert x.ndim == 2
    m, n = x.shape
    if with_2d_quantization:
        x_blocks = (
            x.unfold(0, tile_len_y, tile_len_y)
            .unfold(1, tile_len_x, tile_len_x)
            .to(torch.float32)
        )
        block_amax = torch.amax(torch.abs(x_blocks), dim=(-1, -2))
        vec_max = block_amax.repeat_interleave(tile_len_y, dim=0).unsqueeze(-1)
    else:
        x_reshaped = x.view(m, n // tile_len_x, tile_len_x)
        vec_max = torch.amax(torch.abs(x_reshaped), dim=-1, keepdim=True).to(torch.float32)
    x = x.view(m, n // tile_len_x, tile_len_x)
    decode_scale = torch.div(vec_max, FLOAT4_E2M1_MAX)

    if pow_2_scales:
        decode_scale = cast_to_e8(decode_scale)
        encode_scale = torch.reciprocal(decode_scale.to(torch.float32))
    else:
        # [1], [1,1], [M,1] -> broadcastable [*, 1, 1] without a Python rank branch
        ga2 = global_amax.reshape(-1, 1)
        global_amax_bc = ga2.unsqueeze(1)
        global_encode_scale = torch.div(
            FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX, global_amax_bc
        )
        global_encode_scale = torch.clamp(
            global_encode_scale,
            max=torch.finfo(torch.float32).max,
        )
        global_encode_scale = torch.where(
            global_encode_scale == 0.0,
            torch.ones_like(global_encode_scale),
            global_encode_scale,
        )
        global_decode_scale = torch.div(1.0, global_encode_scale)

        decode_scale = decode_scale * global_encode_scale
        decode_scale = torch.clamp(
            decode_scale,
            max=torch.finfo(torch.float32).max,
        )
        decode_scale = torch.clamp(
            decode_scale, min=-FLOAT8_E4M3_MAX, max=FLOAT8_E4M3_MAX
        )
        decode_scale = decode_scale.to(torch.float8_e4m3fn)

        encode_scale = torch.clamp(
            torch.div(1.0, decode_scale.to(torch.float32) * global_decode_scale),
            max=torch.finfo(torch.float32).max,
        )

    scaled_x = x.to(torch.float32) * encode_scale
    clipped_x = torch.clamp(scaled_x, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).reshape(m, n)
    return cast_to_fp4x2(clipped_x), decode_scale.squeeze(-1)


def _dequantize_blockwise_reference(
    qx: torch.Tensor,
    decode_scale: torch.Tensor,
    original_shape: Tuple[int, int],
    tile_len_x: int,
    tile_len_y: int,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    del tile_len_y
    m_orig, n_orig = original_shape
    x_fp4 = cast_from_fp4x2(qx, torch.float32)
    m, n = x_fp4.shape
    x_tiles = x_fp4.view(m, n // tile_len_x, tile_len_x)
    scale = decode_scale.to(torch.float32)
    x_dequant = x_tiles * scale.unsqueeze(-1)
    x_dequant = x_dequant.view(m, n)
    x_dequant = x_dequant[:m_orig, :n_orig].contiguous()
    y = x_dequant.to(out_dtype)
    if out is not None:
        out.copy_(y)
        return out
    return y


def quant_dequant_nvfp4_ref_standalone(
    x: torch.Tensor,
    *,
    quant_tile_shape: Tuple[int, int] = (1, 16),
    pow_2_scales: bool = False,
    per_token_amax: bool = False,
    with_2d_quantization: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Quantize ``x`` with the ref NVFP4 scheme and return the dequantized tensor.

    Parameters mirror ``nvfp4_quantize_op_ref.Nvfp4TiledQuantizeRefOp`` defaults
    used by ``quant_dequant_nvfp4_REF`` in the multi-quant test.
    """
    assert x.ndim == 2, "NVFP4 ref expects a 2D tensor [M, K]"
    assert quant_tile_shape in ((1, 16), (16, 16)), "NVFP4 ref supports (1,16) or (16,16) tiles"

    row_input = x
    if per_token_amax:
        global_amax_row = torch.amax(torch.abs(row_input), dim=-1, keepdim=True).to(
            torch.float32
        )
    else:
        global_amax_row = torch.max(torch.abs(row_input)).to(torch.float32).view(1)

    m, n = x.shape
    x_padded = _pad_tensor(
        row_input,
        row_divisor=quant_tile_shape[0],
        col_divisor=quant_tile_shape[1],
    )
    qx, sx = _quantize_blockwise_reference(
        x_padded,
        global_amax_row,
        quant_tile_shape[1],
        quant_tile_shape[0],
        pow_2_scales=pow_2_scales,
        with_2d_quantization=with_2d_quantization,
        eps=0.0,
    )
    qx = _rm_pad_tensor(qx, (m, n // 2))

    mm = qx.shape[0]
    nn = qx.shape[1] * 2
    if pow_2_scales:
        actual_scale = sx
    else:
        global_decode_scale = global_amax_row.to(torch.float32) / (
            FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX
        )
        actual_scale = sx.to(torch.float32) * global_decode_scale

    return _dequantize_blockwise_reference(
        qx,
        actual_scale,
        original_shape=(mm, nn),
        tile_len_x=quant_tile_shape[1],
        tile_len_y=quant_tile_shape[0],
        out_dtype=out_dtype,
        out=None,
    )

def quantize_nvfp4_ref_pertoken_amax(
    x: torch.Tensor,
    *,
    quant_tile_shape: Tuple[int, int] = (1, 16),
    out_qx: Optional[torch.Tensor] = None,
    out_sx: Optional[torch.Tensor] = None,
    out_global_amax_row: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    NVFP4 per-token amax quantize (reference path).

    Returns
    -------
    qx : torch.uint8, shape [M, K//2]
    sx : torch.float8_e4m3fn, shape [M, K//tile]
    global_amax_row : torch.float32, shape [M, 1]
    """
    assert x.ndim == 2, "NVFP4 ref expects a 2D tensor [M, K]"
    assert quant_tile_shape in ((1, 16), (16, 16)), "NVFP4 ref supports (1,16) or (16,16) tiles"

    m, n = int(x.shape[0]), int(x.shape[1])
    global_amax_row = torch.amax(torch.abs(x), dim=-1, keepdim=True).to(torch.float32)

    x_padded = _pad_tensor(
        x,
        row_divisor=quant_tile_shape[0],
        col_divisor=quant_tile_shape[1],
    )
    qx, sx = _quantize_blockwise_reference(
        x_padded,
        global_amax_row,
        quant_tile_shape[1],
        quant_tile_shape[0],
        pow_2_scales=False,
        with_2d_quantization=False,
        eps=0.0,
    )
    qx = _rm_pad_tensor(qx, (m, n // 2))
    sx = sx[:m, :].contiguous()

    if out_qx is not None:
        out_qx.copy_(qx)
        qx = out_qx
    if out_sx is not None:
        out_sx.copy_(sx)
        sx = out_sx
    if out_global_amax_row is not None:
        out_global_amax_row.copy_(global_amax_row)
        global_amax_row = out_global_amax_row
    return qx, sx, global_amax_row


def dequantize_nvfp4_ref_pertoken_amax(
    qx: torch.Tensor,
    sx: torch.Tensor,
    global_amax_row: torch.Tensor,
    *,
    quant_tile_shape: Tuple[int, int] = (1, 16),
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize NVFP4 tensors produced by ``quantize_nvfp4_ref_pertoken_amax``."""
    assert qx.ndim == 2 and sx.ndim == 2, "qx/sx must be 2D"
    assert global_amax_row.ndim == 2 and global_amax_row.shape[1] == 1, "global_amax_row must be [M, 1]"
    assert quant_tile_shape in ((1, 16), (16, 16)), "NVFP4 ref supports (1,16) or (16,16) tiles"

    m = int(qx.shape[0])
    n = int(qx.shape[1]) * 2
    global_decode_scale = global_amax_row.to(torch.float32) / (
        FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX
    )
    actual_scale = sx.to(torch.float32) * global_decode_scale
    return _dequantize_blockwise_reference(
        qx,
        actual_scale,
        original_shape=(m, n),
        tile_len_x=quant_tile_shape[1],
        tile_len_y=quant_tile_shape[0],
        out_dtype=out_dtype,
        out=out,
    )


def quant_dequant_nvfp4_ref_pertoken_amax(
    x: torch.Tensor,
    *,
    quant_tile_shape: Tuple[int, int] = (1, 16),
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convenience wrapper: per-token amax quantize + dequantize."""
    qx, sx, global_amax_row = quantize_nvfp4_ref_pertoken_amax(
        x,
        quant_tile_shape=quant_tile_shape,
    )
    return dequantize_nvfp4_ref_pertoken_amax(
        qx,
        sx,
        global_amax_row,
        quant_tile_shape=quant_tile_shape,
        out_dtype=out_dtype,
        out=out,
    )


def quant_dequant_nvfp4_ref_for_cudagraph(
    x: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    per_token_amax: bool = False,
    quant_tile_shape: Tuple[int, int] = (1, 16),
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    NVFP4 ref Q/DQ tuned for ``torch.cuda.graph`` / SGLang-style capture.

    Compared to :func:`quant_dequant_nvfp4_ref_standalone`, this entry point
    fixes ``pow_2_scales=False`` and ``with_2d_quantization=False`` inside the
    body so the recorded op sequence does not depend on those Python flags.

    Parameters
    ----------
    x
        2D tensor on CUDA; keep shape/dtype fixed across capture and replay.
    out
        Optional preallocated buffer with shape ``x.shape`` and dtype ``out_dtype``
        (same storage must be used for warmup, capture, and replay).
    per_token_amax
        If ``True``, same semantics as ``per_token_amax=True`` in the class ref
        (must stay constant across capture vs replay).
    quant_tile_shape
        ``(1, 16)`` or ``(16, 16)``; must stay constant across capture vs replay.
    """
    assert x.ndim == 2, "NVFP4 ref expects a 2D tensor [M, K]"
    assert quant_tile_shape in ((1, 16), (16, 16)), "NVFP4 ref supports (1,16) or (16,16) tiles"

    row_input = x
    if per_token_amax:
        global_amax_row = torch.amax(torch.abs(row_input), dim=-1, keepdim=True).to(
            torch.float32
        )
    else:
        global_amax_row = torch.max(torch.abs(row_input)).to(torch.float32).reshape(1, 1)

    m, n = x.shape[0], x.shape[1]
    x_padded = _pad_tensor(
        row_input,
        row_divisor=quant_tile_shape[0],
        col_divisor=quant_tile_shape[1],
    )
    qx, sx = _quantize_blockwise_reference(
        x_padded,
        global_amax_row,
        quant_tile_shape[1],
        quant_tile_shape[0],
        pow_2_scales=False,
        with_2d_quantization=False,
        eps=0.0,
    )
    qx = _rm_pad_tensor(qx, (m, n // 2))

    mm = qx.shape[0]
    nn = qx.shape[1] * 2
    global_decode_scale = global_amax_row.to(torch.float32) / (
        FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX
    )
    actual_scale = sx.to(torch.float32) * global_decode_scale

    return _dequantize_blockwise_reference(
        qx,
        actual_scale,
        original_shape=(mm, nn),
        tile_len_x=quant_tile_shape[1],
        tile_len_y=quant_tile_shape[0],
        out_dtype=out_dtype,
        out=out,
    )


def quantize_nvfp4_ref_pertoken_amax_for_cudagraph(
    x: torch.Tensor,
    out_qx: Optional[torch.Tensor] = None,
    out_sx: Optional[torch.Tensor] = None,
    out_global_amax_row: Optional[torch.Tensor] = None,
    *,
    quant_tile_shape: Tuple[int, int] = (1, 16),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CUDA-graph friendly per-token amax quantize.

    This is the split quant half of ``quant_dequant_nvfp4_ref_for_cudagraph`` with
    per-token global amax enabled.
    """
    return quantize_nvfp4_ref_pertoken_amax(
        x,
        quant_tile_shape=quant_tile_shape,
        out_qx=out_qx,
        out_sx=out_sx,
        out_global_amax_row=out_global_amax_row,
    )


def dequantize_nvfp4_ref_pertoken_amax_for_cudagraph(
    qx: torch.Tensor,
    sx: torch.Tensor,
    global_amax_row: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    quant_tile_shape: Tuple[int, int] = (1, 16),
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """CUDA-graph friendly per-token amax dequantize."""
    return dequantize_nvfp4_ref_pertoken_amax(
        qx,
        sx,
        global_amax_row,
        quant_tile_shape=quant_tile_shape,
        out_dtype=out_dtype,
        out=out,
    )
