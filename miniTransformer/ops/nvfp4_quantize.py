"""NVFP4 quantize/dequant wrappers over fp4_gemm CUDA kernels."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import List, Optional, Sequence

import fp4_gemm
import torch

from miniTransformer.quantiztion import quantization
from miniTransformer.utils import utils

nvfp4_output_t = torch.uint8
nvfp4_scale_t = torch.float8_e4m3fn

_MMA_TILE_ROWS = 128
_NVFP4_GLOBAL_SCALE_DENOM = 448.0 * 6.0
_FUSED_PERTOKEN_COL_LIMIT = 8192
_SCALE_VIEW_PERM = (3, 4, 1, 5, 2, 0)

# --- Quant APIs (production) -------------------------------------------------
# group_quantize_nvfp4_pertoken_swizzled   MoE activation X  → grouped GEMM (QAT 6–12)
# group_quantize_nvfp4_pertoken_split_amax MoE activation X  → grouped GEMM (QAT 15, optional row_amax)
# group_quantize_nvfp4_tensor_global_packed  MoE weight W      → grouped GEMM (QAT 6–14)
# quantize_weight_nvfp4_swizzled           single weight [N,K] → grouped_gemm_nvfp4_api
# quantize_nvfp4_cuda                      single weight (low-level; via grouped_gemm_nvfp4_impl)
#
# --- Quant APIs (tests / reference only) ------------------------------------
# quantize_nvfp4_pertoken                    single activation [M,K], row-major scales
# quantize_nvfp4_pertoken_swizzled           single activation [M,K], swizzled scales
# dequantize_*                             round-trip checks, eval scripts, log tools
# -----------------------------------------------------------------------------


# =============================================================================
# Production — grouped MoE activation (GroupedLinear X)
# =============================================================================


@dataclass
class GroupQuantizeSwizzledResult:
    """Packed NVFP4 activation for grouped GEMM (one contiguous buffer + offsets).

    Produced by ``group_quantize_nvfp4_pertoken_swizzled`` or
    ``group_quantize_nvfp4_pertoken_split_amax``. Consumed by cursor2 / cutlass
    quantize ops via ``extra_info`` (data, scale_flat, global_scale, offsets).
    """

    data: torch.Tensor
    global_scale: torch.Tensor
    scale_flat: torch.Tensor
    row_offsets_gpu: torch.Tensor
    scale_offsets_gpu: torch.Tensor
    unpadded_row_offsets_gpu: Optional[torch.Tensor] = None


# MoE activation quant — GroupedLinear forward, QAT 6–12 (cursor2 / cutlass ops).
# What:  BF16 [total_m, K] + m_splits → FP4 + per-row global scale + swizzled block scales.
# Where: quantiztion/nvfp4_cursor2_op*.group_quantize(X).
# Note:  fuse_pad=True pads each expert to 128 rows inside the kernel.
def group_quantize_nvfp4_pertoken_swizzled(
    x: torch.Tensor,
    m_splits: List[int],
    *,
    four_six: bool = False,
    scale_override: float = 1.0,
    fuse_pad: bool = False,
) -> GroupQuantizeSwizzledResult:
    assert x.is_cuda and x.dtype == torch.bfloat16, "x must be CUDA bfloat16."
    assert x.dim() == 2 and x.is_contiguous(), "x must be a contiguous 2-D tensor."
    assert sum(m_splits) == x.shape[0], "sum(m_splits) must equal x.shape[0]"

    total_m, k = x.shape
    device = x.device
    sf_k = k // 16

    if total_m == 0:
        return _empty_swizzled_group_result(len(m_splits), k, device, fuse_pad=fuse_pad)

    m_padded = [_pad_rows(m) for m in m_splits]
    scale_flat = torch.zeros(
        sum(mp * sf_k for mp in m_padded), dtype=nvfp4_scale_t, device=device,
    )
    scale_offsets_gpu = _cumsum_offsets([mp * sf_k for mp in m_padded], torch.int64, device)

    if fuse_pad:
        unpadded_row_offsets_gpu = _cumsum_offsets(m_splits, torch.int32, device)
        padded_row_offsets_gpu = _cumsum_offsets(m_padded, torch.int32, device)
        padded_total = sum(m_padded)
        data = torch.empty((padded_total, k // 2), dtype=nvfp4_output_t, device=device)
        global_scale = torch.empty(padded_total, dtype=torch.float32, device=device)

        fn = (
            fp4_gemm.four_six_fp4_group_pertoken_pad_fuse
            if four_six
            else fp4_gemm.rtn_fp4_group_pertoken_pad_fuse
        )
        fn(
            data, scale_flat, global_scale, x,
            padded_row_offsets_gpu, unpadded_row_offsets_gpu,
            scale_offsets_gpu, scale_override,
        )
        return GroupQuantizeSwizzledResult(
            data=data,
            global_scale=global_scale,
            scale_flat=scale_flat,
            row_offsets_gpu=padded_row_offsets_gpu,
            scale_offsets_gpu=scale_offsets_gpu,
            unpadded_row_offsets_gpu=unpadded_row_offsets_gpu,
        )

    assert all(m % _MMA_TILE_ROWS == 0 for m in m_splits), (
        f"fuse_pad=False requires each m_splits entry to be a multiple of "
        f"{_MMA_TILE_ROWS}; got {m_splits}"
    )
    data = torch.empty((total_m, k // 2), dtype=nvfp4_output_t, device=device)
    global_scale = torch.empty(total_m, dtype=torch.float32, device=device)
    row_offsets_gpu = _cumsum_offsets(m_splits, torch.int32, device)

    fn = fp4_gemm.four_six_fp4_group_pertoken if four_six else fp4_gemm.rtn_fp4_group_pertoken
    fn(data, scale_flat, global_scale, x, row_offsets_gpu, scale_offsets_gpu, scale_override)

    return GroupQuantizeSwizzledResult(
        data=data,
        global_scale=global_scale,
        scale_flat=scale_flat,
        row_offsets_gpu=row_offsets_gpu,
        scale_offsets_gpu=scale_offsets_gpu,
    )


# MoE activation quant — GroupedLinear forward, QAT 15 (pertoken + amax-fuse path).
# What:  same output as group_quantize_nvfp4_pertoken_swizzled; two-kernel amax + quant.
# Where: quantiztion/nvfp4_cursor2_op_cutlass_v2.group_quantize(X, row_amax=...).
# Note:  pass row_amax to skip the per-row amax kernel when upstream already fused it.
def group_quantize_nvfp4_pertoken_split_amax(
    x: torch.Tensor,
    m_splits: List[int],
    *,
    four_six: bool = False,
    scale_override: float = 1.0,
    row_amax: Optional[torch.Tensor] = None,
) -> GroupQuantizeSwizzledResult:
    assert x.is_cuda and x.dtype == torch.bfloat16
    assert x.dim() == 2 and x.is_contiguous()
    assert sum(m_splits) == x.shape[0]

    total_m, k = x.shape
    device = x.device
    sf_k = k // 16

    if total_m == 0:
        return _empty_swizzled_group_result(len(m_splits), k, device)

    scale_flat = torch.zeros(sum(m * sf_k for m in m_splits), dtype=nvfp4_scale_t, device=device)
    scale_offsets_gpu = _cumsum_offsets([m * sf_k for m in m_splits], torch.int64, device)
    row_offsets_gpu = _cumsum_offsets(m_splits, torch.int32, device)

    data = torch.empty((total_m, k // 2), dtype=nvfp4_output_t, device=device)
    global_scale = torch.empty(total_m, dtype=torch.float32, device=device)

    if row_amax is not None:
        assert row_amax.shape == (total_m,), (
            f"row_amax shape {tuple(row_amax.shape)} != ({total_m},)"
        )
        assert row_amax.dtype == torch.float32 and row_amax.is_cuda
        amax_per_row = row_amax
    else:
        amax_per_row = torch.empty(total_m, dtype=torch.float32, device=device)
        fp4_gemm.fp4_perrow_amax_te(amax_per_row, x)

    fn = (
        fp4_gemm.four_six_fp4_group_pertoken_split_amax
        if four_six
        else fp4_gemm.rtn_fp4_group_pertoken_split_amax
    )
    fn(
        data, scale_flat, global_scale, x, amax_per_row,
        row_offsets_gpu, scale_offsets_gpu, scale_override,
    )

    return GroupQuantizeSwizzledResult(
        data=data,
        global_scale=global_scale,
        scale_flat=scale_flat,
        row_offsets_gpu=row_offsets_gpu,
        scale_offsets_gpu=scale_offsets_gpu,
    )


# =============================================================================
# Production — grouped MoE weight (GroupedLinear W)
# =============================================================================


@dataclass
class GroupWeightQuantizeResult:
    """Packed NVFP4 weights for all experts in one buffer.

    Produced by ``group_quantize_nvfp4_tensor_global_packed``. One global scale
    per expert (amax over that expert's weight rows). Consumed by cursor2 /
    cutlass grouped GEMM ops.
    """

    data: torch.Tensor
    cute_data: torch.Tensor
    global_scale: torch.Tensor
    scale_flat: torch.Tensor
    row_offsets_gpu: torch.Tensor
    scale_offsets_gpu: torch.Tensor


# MoE weight quant — GroupedLinear forward, QAT 6–14 (cursor2 / cutlass ops).
# What:  BF16 [E, N, K] → one packed FP4 buffer; one global scale per expert.
# Where: quantiztion/nvfp4_cursor2_op*.group_quantize(WEIGHT).
def group_quantize_nvfp4_tensor_global_packed(
    x: torch.Tensor,
    n_per_expert: int,
    *,
    four_six: bool = False,
    scale_override: float = 1.0,
    skip_cute_data: bool = False,
) -> GroupWeightQuantizeResult:
    assert x.is_cuda, "x must be a CUDA tensor."
    assert x.dtype == torch.bfloat16, "x must be bfloat16."
    assert x.ndim == 3, "x must be 3-D [num_experts, n_per_expert, k]."
    assert x.is_contiguous(), "x must be contiguous."
    num_experts, n, k = x.shape
    assert n == n_per_expert, f"x.shape[1]={n} != n_per_expert={n_per_expert}"
    assert k % 128 == 0, "k must be divisible by 128."
    assert k // 8 <= 512, "k must be <= 4096."
    device = x.device

    total_n = num_experts * n_per_expert
    x_2d = x.view(total_n, k)

    n_padded = _pad_rows(n_per_expert)
    sf_stride = n_padded * k // 16

    data = torch.empty((total_n, k // 2), dtype=nvfp4_output_t, device=device)
    global_scale = torch.empty(num_experts, dtype=torch.float32, device=device)
    scale_flat = torch.zeros(num_experts * sf_stride, dtype=nvfp4_scale_t, device=device)

    row_offsets_gpu = torch.arange(num_experts + 1, dtype=torch.int32, device=device) * n_per_expert
    scale_offsets_gpu = torch.arange(num_experts + 1, dtype=torch.int64, device=device) * sf_stride

    fn = fp4_gemm.four_six_fp4_group_tensor_global if four_six else fp4_gemm.rtn_fp4_group_tensor_global
    amax = (
        x_2d.abs().amax(dim=1)
        .reshape(num_experts, n_per_expert)
        .amax(dim=1)
        .to(torch.float32)
        .contiguous()
    )
    fn(
        data, scale_flat, global_scale, x_2d,
        row_offsets_gpu, scale_offsets_gpu, amax, scale_override, swizzle=True,
    )

    if skip_cute_data:
        cute_data = None
    else:
        cute_data = torch.zeros(total_n * k, dtype=torch.int8, device=device)
        rows_per_expert = total_n // num_experts
        for g in range(num_experts):
            src = data[g * rows_per_expert:(g + 1) * rows_per_expert].reshape(-1).view(torch.int8)
            dst_start = g * rows_per_expert * k
            cute_data[dst_start:dst_start + rows_per_expert * k // 2].copy_(src)

    return GroupWeightQuantizeResult(
        data=data,
        cute_data=cute_data,
        global_scale=global_scale,
        scale_flat=scale_flat,
        row_offsets_gpu=row_offsets_gpu,
        scale_offsets_gpu=scale_offsets_gpu,
    )


# =============================================================================
# Production — single-matrix weight
# =============================================================================


# Single weight [N, K] — one global amax over all rows, swizzled scales + CUTE layout.
# Where: ops/cuteDSL/nvfp4/grouped_gemm_nvfp4_api (exported). MoE W uses group_quantize_* above.
def quantize_weight_nvfp4_swizzled(
    x: torch.Tensor,
    *,
    four_six: bool = False,
    scale_override: float = 1.0,
) -> "TensorGlobalNvfp4Tensor":
    from miniTransformer.ops.cuteDSL.nvfp4.grouped_gemm_nvfp4_impl import (
        TensorGlobalNvfp4Tensor,
        _make_fp4_tensor_for_cutedsl,
    )

    assert x.is_cuda, "quantize_weight_nvfp4_swizzled requires a CUDA tensor."
    assert x.dtype in utils.HIGH_PRECISION_FLOAT_DTYPES, "Unsupported input dtype."
    assert x.ndim == 2, "x must be 2-D [N, K]."
    n, k = x.shape
    assert k % 128 == 0, "K must be divisible by 128."
    assert k // 8 <= 512, "K must be <= 4096."
    device = x.device

    amax = x.abs().max().to(torch.float32).view(1).contiguous()
    data = torch.empty((n, k // 2), dtype=nvfp4_output_t, device=device)
    n_padded = _pad_rows(n)
    scale_flat = torch.zeros(n_padded * (k // 16), dtype=nvfp4_scale_t, device=device)
    gs = torch.empty(1, dtype=torch.float32, device=device)

    fn = fp4_gemm.four_six_fp4 if four_six else fp4_gemm.rtn_fp4
    fn(data, scale_flat, gs, x, amax, scale_override)

    nMT = n_padded // _MMA_TILE_ROWS
    nKT = k // 64
    cute_sfb = scale_flat.view(1, nMT, nKT, 32, 4, 4).permute(*_SCALE_VIEW_PERM)
    cute_fp4 = _make_fp4_tensor_for_cutedsl(data, n, k, device)

    t = TensorGlobalNvfp4Tensor(
        data_uint8=data,
        block_scale_fp8=scale_flat,
        amax=amax,
    )
    t._alpha_b_cached = float(amax.item()) / _NVFP4_GLOBAL_SCALE_DENOM
    t._cute_fp4 = cute_fp4
    t._cute_sfb = cute_sfb
    return t


# Single weight [N, K] — low-level path via fp4_gemm.quantize_transpose_nvfp4.
# Where: grouped_gemm_nvfp4_impl.quantize_weight_nvfp4_tensor_global (not GroupedLinear direct).
# Note:  optional 2D quant and transpose; prefer quantize_weight_nvfp4_swizzled for grouped GEMM API.
def quantize_nvfp4_cuda(
    x: torch.Tensor,
    *,
    with_2d_quantization: bool,
    return_transpose: bool = False,
    amax_rowwise: torch.Tensor | None = None,
    amax_colwise: torch.Tensor | None = None,
) -> quantization.QuantizeResult:
    assert x.is_cuda, "quantize_nvfp4_cuda expects a CUDA tensor."
    assert x.dtype in utils.HIGH_PRECISION_FLOAT_DTYPES, "Unsupported input dtype."
    assert x.dim() == 2, "quantize_nvfp4_cuda expects a 2D tensor."

    rows, cols = x.shape
    assert rows % 32 == 0 and cols % 32 == 0, "Input shape must be multiples of 32."

    amax_rowwise = _as_unit_amax(amax_rowwise, x)
    amax_colwise = _as_unit_amax(amax_colwise, x, fallback=amax_rowwise)

    out = torch.empty((rows, cols // 2), dtype=nvfp4_output_t, device=x.device)
    out_s = torch.empty((rows, cols // 16), dtype=nvfp4_scale_t, device=x.device)
    out_t = torch.empty((cols, rows // 2), dtype=nvfp4_output_t, device=x.device)
    out_st = torch.empty((cols, rows // 16), dtype=nvfp4_scale_t, device=x.device)

    fp4_gemm.quantize_transpose_nvfp4(
        x,
        amax_rowwise,
        amax_colwise,
        out,
        out_s,
        out_t,
        out_st,
        with_2d_quantization,
        True,
    )

    if return_transpose:
        return quantization.QuantizeResult(
            data=out,
            scale=out_s,
            data_t=out_t,
            scale_t=out_st,
            tensor_scale=amax_rowwise,
            tensor_scale_t=amax_colwise,
        )
    return quantization.QuantizeResult(
        data=out,
        scale=out_s,
        tensor_scale=amax_rowwise,
    )


# =============================================================================
# Test / reference — single-matrix activation (not GroupedLinear forward)
# =============================================================================


# Single activation [M, K] — pertoken quant, row-major block scales (debug / golden layout).
# Where: unit tests, nvfp4_cursor2_op_ref STE, log tools. Not GroupedLinear production forward.
def quantize_nvfp4_pertoken(
    x: torch.Tensor,
    *,
    four_six: bool = False,
    scale_override: float = 1.0,
) -> quantization.QuantizeResult:
    assert x.is_cuda and x.dtype == torch.bfloat16, "Input must be CUDA bfloat16."
    assert x.dim() == 2, "Input must be 2-D."
    rows, cols = x.shape

    x_c = x.contiguous()
    out = torch.empty((rows, cols // 2), dtype=nvfp4_output_t, device=x.device)
    scale = torch.empty((rows, cols // 16), dtype=nvfp4_scale_t, device=x.device)
    global_scale = torch.empty(rows, dtype=torch.float32, device=x.device)

    _launch_pertoken_quant(
        x_c, out, scale, global_scale,
        four_six=four_six, scale_override=scale_override, swizzled=False,
    )
    return quantization.QuantizeResult(data=out, scale=scale, tensor_scale=global_scale)


# Single activation [M, K] — same as above but MMA swizzled block scales (matches grouped X kernel).
# Where: unit tests only; validates kernels against group_quantize_nvfp4_pertoken_swizzled.
def quantize_nvfp4_pertoken_swizzled(
    x: torch.Tensor,
    *,
    four_six: bool = False,
    scale_override: float = 1.0,
) -> quantization.QuantizeResult:
    assert x.is_cuda and x.dtype == torch.bfloat16, "Input must be CUDA bfloat16."
    assert x.dim() == 2, "Input must be 2-D."
    rows, cols = x.shape

    x_c = x.contiguous()
    out = torch.empty((rows, cols // 2), dtype=nvfp4_output_t, device=x.device)
    scale = torch.empty(rows * cols // 16, dtype=nvfp4_scale_t, device=x.device)
    global_scale = torch.empty(rows, dtype=torch.float32, device=x.device)

    _launch_pertoken_quant(
        x_c, out, scale, global_scale,
        four_six=four_six, scale_override=scale_override, swizzled=True,
    )
    return quantization.QuantizeResult(data=out, scale=scale, tensor_scale=global_scale)


# =============================================================================
# Dequantize (tests / eval — not training forward)
# =============================================================================


# Inverse of quantize_nvfp4_pertoken.
def dequantize_nvfp4_pertoken(result: quantization.QuantizeResult) -> torch.Tensor:
    return _dequant_fp4_block(result.data, result.scale, result.tensor_scale)


# Inverse of quantize_nvfp4_pertoken_swizzled.
def dequantize_nvfp4_pertoken_swizzled(result: quantization.QuantizeResult) -> torch.Tensor:
    m, nhalf = result.data.shape
    n = nhalf * 2
    scale_rm = _unswizzle_scale(result.scale.to(torch.float32), m, n)
    return _dequant_fp4_block(result.data, scale_rm, result.tensor_scale)


# Inverse of quantize_weight_nvfp4_swizzled.
def dequantize_weight_nvfp4_swizzled(t: "TensorGlobalNvfp4Tensor") -> torch.Tensor:
    data = t.data_uint8
    n, nhalf = data.shape
    k = nhalf * 2
    n_padded = _pad_rows(n)
    scale_rm = _unswizzle_scale(t.block_scale_fp8.to(torch.float32), n_padded, k)[:n]
    alpha = t.amax.to(torch.float32) / _NVFP4_GLOBAL_SCALE_DENOM
    return _dequant_fp4_block(data, scale_rm, alpha)


# Inverse of group_quantize_nvfp4_pertoken_swizzled / split_amax.
def dequantize_group_nvfp4_pertoken_swizzled(result: GroupQuantizeSwizzledResult) -> torch.Tensor:
    k = result.data.shape[1] * 2
    return _dequant_group_swizzled_slices(
        result.data,
        result.scale_flat,
        result.global_scale,
        result.row_offsets_gpu.tolist(),
        result.scale_offsets_gpu.tolist(),
        k,
        global_scale_per_group=False,
    )


# Inverse of group_quantize_nvfp4_tensor_global_packed.
def dequantize_group_nvfp4_tensor_global_packed(result: GroupWeightQuantizeResult) -> torch.Tensor:
    k = result.data.shape[1] * 2
    return _dequant_group_swizzled_slices(
        result.data,
        result.scale_flat,
        result.global_scale,
        result.row_offsets_gpu.tolist(),
        result.scale_offsets_gpu.tolist(),
        k,
        global_scale_per_group=True,
    )


# =============================================================================
# Internal helpers
# =============================================================================

_FP4_GRID = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


@functools.lru_cache(maxsize=None)
def _fp4_grid(device_str: str) -> torch.Tensor:
    return _FP4_GRID.to(torch.device(device_str))


def _get_fp4_grid(device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        key = f"cuda:{idx}"
    else:
        key = str(device)
    return _fp4_grid(key)


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    p = packed.to(torch.int32)
    return torch.stack([p & 0x0F, (p >> 4) & 0x0F], dim=-1).flatten(start_dim=-2)


def _unswizzle_scale(scale_flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    n_row_tiles = rows // 128
    n_col_tiles = (cols // 16) // 4
    s = scale_flat.reshape(n_row_tiles * n_col_tiles, 32, 4, 4)
    s = s.permute(0, 2, 1, 3).reshape(n_row_tiles, n_col_tiles, 128, 4)
    return s.permute(0, 2, 1, 3).reshape(rows, n_col_tiles * 4)


def _pad_rows(rows: int) -> int:
    return ((rows + _MMA_TILE_ROWS - 1) // _MMA_TILE_ROWS) * _MMA_TILE_ROWS


def _as_unit_amax(
    amax: torch.Tensor | None,
    x: torch.Tensor,
    *,
    fallback: torch.Tensor | None = None,
) -> torch.Tensor:
    if amax is None:
        src = fallback if fallback is not None else torch.max(torch.abs(x))
        return src.to(device=x.device, dtype=torch.float32).reshape(1).contiguous()
    return amax.to(device=x.device, dtype=torch.float32).reshape(1).contiguous()


def _cumsum_offsets(
    sizes: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    return torch.tensor(offsets, dtype=dtype).pin_memory().to(device, non_blocking=True)


def _dequant_fp4_block(
    data: torch.Tensor,
    scale_rm: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    m, nhalf = data.shape
    n = nhalf * 2
    grid = _get_fp4_grid(data.device)
    fp4_vals = grid[_unpack_fp4(data)].to(torch.float32)
    gs = global_scale.to(torch.float32)
    if gs.dim() == 0 or gs.numel() == 1:
        actual_scale = scale_rm.to(torch.float32) * gs
    else:
        actual_scale = scale_rm.to(torch.float32) * gs.unsqueeze(1)
    return (
        fp4_vals.view(m, n // 16, 16) * actual_scale.unsqueeze(-1)
    ).view(m, n).to(torch.bfloat16)


def _launch_pertoken_quant(
    x: torch.Tensor,
    out: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    four_six: bool,
    scale_override: float,
    swizzled: bool,
) -> None:
    cols = x.shape[1]
    if swizzled:
        fused = fp4_gemm.four_six_fp4_pertoken_fused if four_six else fp4_gemm.rtn_fp4_pertoken_fused
        linear = fp4_gemm.four_six_fp4_pertoken if four_six else fp4_gemm.rtn_fp4_pertoken
    else:
        fused = (
            fp4_gemm.four_six_fp4_pertoken_linear_fused
            if four_six
            else fp4_gemm.rtn_fp4_pertoken_linear_fused
        )
        linear = (
            fp4_gemm.four_six_fp4_pertoken_linear
            if four_six
            else fp4_gemm.rtn_fp4_pertoken_linear
        )

    if cols <= _FUSED_PERTOKEN_COL_LIMIT:
        fused(out, scale, global_scale, x, scale_override)
        return

    amax_per_row = x.abs().amax(dim=1).to(torch.float32)
    linear(out, scale, global_scale, x, amax_per_row, scale_override)


def _empty_swizzled_group_result(
    num_groups: int,
    k: int,
    device: torch.device,
    *,
    fuse_pad: bool = False,
) -> GroupQuantizeSwizzledResult:
    empty_off = torch.zeros(num_groups + 1, dtype=torch.int32, device=device)
    return GroupQuantizeSwizzledResult(
        data=torch.empty((0, k // 2), dtype=nvfp4_output_t, device=device),
        global_scale=torch.empty(0, dtype=torch.float32, device=device),
        scale_flat=torch.empty(0, dtype=nvfp4_scale_t, device=device),
        row_offsets_gpu=empty_off,
        scale_offsets_gpu=torch.zeros(num_groups + 1, dtype=torch.int64, device=device),
        unpadded_row_offsets_gpu=empty_off.clone() if fuse_pad else None,
    )


def _dequant_group_swizzled_slices(
    data: torch.Tensor,
    scale_flat: torch.Tensor,
    global_scale: torch.Tensor,
    row_offsets: List[int],
    scale_offsets: List[int],
    k: int,
    *,
    global_scale_per_group: bool,
) -> torch.Tensor:
    device = data.device
    if data.shape[0] == 0:
        return torch.empty((0, k), dtype=torch.bfloat16, device=device)

    parts: List[torch.Tensor] = []
    for g in range(len(row_offsets) - 1):
        r0, r1 = row_offsets[g], row_offsets[g + 1]
        m_i = r1 - r0
        if m_i == 0:
            parts.append(torch.empty((0, k), dtype=torch.bfloat16, device=device))
            continue

        m_pad = _pad_rows(m_i)
        s0, s1 = scale_offsets[g], scale_offsets[g + 1]
        scale_rm = _unswizzle_scale(scale_flat[s0:s1].to(torch.float32), m_pad, k)[:m_i]
        gs = global_scale[g] if global_scale_per_group else global_scale[r0:r1]
        parts.append(_dequant_fp4_block(data[r0:r1], scale_rm, gs))
    return torch.cat(parts, dim=0)
