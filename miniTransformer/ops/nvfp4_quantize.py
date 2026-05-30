from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import List

import torch
import fp4_gemm

from miniTransformer.quantiztion import quantization
from miniTransformer.utils import utils

nvfp4_output_t = torch.uint8
nvfp4_scale_t = torch.float8_e4m3fn

_MMA_TILE_ROWS = 128   # tcgen05 MMA tile height; scale buffers must be padded to this

# FP4 E2M1 value lookup table: index (nibble 0–15) → float (CPU master copy)
_FP4_GRID = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


@functools.lru_cache(maxsize=None)
def _fp4_grid(device_str: str) -> torch.Tensor:
    # Cached per-device GPU copy.  lru_cache ensures the H2D transfer happens
    # only once (during warmup), so CUDA graph capture sees no host→device copy.
    return _FP4_GRID.to(torch.device(device_str))


def _get_fp4_grid(device: torch.device) -> torch.Tensor:
    # Normalize to "cuda:N" so lru_cache always hits after first call.
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        key = f"cuda:{idx}"
    else:
        key = str(device)
    return _fp4_grid(key)


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 [m, n//2] → int32 [m, n] nibble indices 0–15."""
    p = packed.to(torch.int32)
    return torch.stack([p & 0x0F, (p >> 4) & 0x0F], dim=-1).flatten(start_dim=-2)


def _unswizzle_scale(scale_flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Convert tcgen05 swizzled flat scale [m*n//16] → row-major [m, n//16]."""
    n_row_tiles = rows // 128
    n_col_tiles = (cols // 16) // 4
    s = scale_flat.reshape(n_row_tiles * n_col_tiles, 32, 4, 4)
    s = s.permute(0, 2, 1, 3).reshape(n_row_tiles, n_col_tiles, 128, 4)
    return s.permute(0, 2, 1, 3).reshape(rows, n_col_tiles * 4)


def quantize_nvfp4_cuda(
    x: torch.Tensor,
    *,
    with_2d_quantization: bool,
    return_transpose: bool = False,
    amax_rowwise: torch.Tensor | None = None,
    amax_colwise: torch.Tensor | None = None,
) -> quantization.QuantizeResult:
    """
    Standalone NVFP4 CUDA quantization wrapper around fp4_gemm binding.

    Returns QuantizeResult:
      - always: data (x_q), scale (sx), tensor_scale (global amax rowwise)
      - when return_transpose=True: data_t (x_q^T), scale_t (sx_t),
        tensor_scale_t (global amax colwise)
    """
    assert x.is_cuda, "quantize_nvfp4_cuda expects a CUDA tensor."
    assert x.dtype in utils.HIGH_PRECISION_FLOAT_DTYPES, "Unsupported input dtype."
    assert x.dim() == 2, "quantize_nvfp4_cuda expects a 2D tensor."

    rows, cols = x.shape
    assert rows % 32 == 0 and cols % 32 == 0, "Input shape must be multiples of 32."

    # Default global amax from input (1-element float32 tensor).
    if amax_rowwise is None:
        amax_rowwise = torch.max(torch.abs(x)).to(torch.float32).view(1).contiguous()
    else:
        amax_rowwise = amax_rowwise.to(device=x.device, dtype=torch.float32).reshape(1).contiguous()

    if amax_colwise is None:
        amax_colwise = amax_rowwise
    else:
        amax_colwise = amax_colwise.to(device=x.device, dtype=torch.float32).reshape(1).contiguous()

    out = torch.empty((rows, cols // 2), dtype=nvfp4_output_t, device=x.device)
    out_s = torch.empty((rows, cols // 16), dtype=nvfp4_scale_t, device=x.device)

    # Keep CUDA launch stable by always passing valid transpose buffers.
    out_t_full = torch.empty((cols, rows // 2), dtype=nvfp4_output_t, device=x.device)
    out_st_full = torch.empty((cols, rows // 16), dtype=nvfp4_scale_t, device=x.device)
    cuda_return_transpose = True

    fp4_gemm.quantize_transpose_nvfp4(
        x,
        amax_rowwise,
        amax_colwise,
        out,
        out_s,
        out_t_full,
        out_st_full,
        with_2d_quantization,
        cuda_return_transpose,
    )

    if return_transpose:
        return quantization.QuantizeResult(
            data=out,
            scale=out_s,
            data_t=out_t_full,
            scale_t=out_st_full,
            tensor_scale=amax_rowwise,
            tensor_scale_t=amax_colwise,
        )
    return quantization.QuantizeResult(
        data=out,
        scale=out_s,
        tensor_scale=amax_rowwise,
    )


def quantize_nvfp4_pertoken(
    x: torch.Tensor,
    *,
    four_six: bool = False,
    scale_override: float = 1.0,
) -> quantization.QuantizeResult:
    """Per-token NV FP4 quantization with row-major scale-factor layout.

    Computes one global scale per row on-the-fly (no pre-computed amax needed).
    The per-group FP8 scale tensor is written in plain row-major order,
    suitable for inspection, dequantisation in Python, or custom kernels.

    Args:
        x:              2-D BF16 input tensor of shape [m, n].
                        n must be divisible by 128 and <= 4096.
        four_six:       If True, use the Round-4-or-6 algorithm (tries FP4 max=6
                        and max=4, keeps the candidate with lower L2 error).
                        If False (default), use plain RTN (FP4 max=6 only).
        scale_override: Scalar multiplier applied to the scale computation.
                        Defaults to 1.0.

    Returns:
        QuantizeResult:
          data          – packed FP4 values, shape [m, n//2], dtype uint8
          scale         – per-group FP8 scale, shape [m, n//16], dtype float8_e4m3fn,
                          in row-major layout
          tensor_scale  – per-token global scale, shape [m], dtype float32
    """
    assert x.is_cuda and x.dtype == torch.bfloat16, "Input must be a CUDA bfloat16 tensor."
    assert x.dim() == 2, "Input must be 2-D."
    rows, cols = x.shape

    x_c = x.contiguous()
    out = torch.empty((rows, cols // 2), dtype=nvfp4_output_t, device=x.device)
    scale = torch.empty((rows, cols // 16), dtype=nvfp4_scale_t, device=x.device)
    global_scale = torch.empty(rows, dtype=torch.float32, device=x.device)

    if cols <= 8192:
        fn = fp4_gemm.four_six_fp4_pertoken_linear_fused if four_six else fp4_gemm.rtn_fp4_pertoken_linear_fused
        fn(out, scale, global_scale, x_c, scale_override)
    else:
        amax_per_row = x_c.abs().amax(dim=1).to(torch.float32)
        fn = fp4_gemm.four_six_fp4_pertoken_linear if four_six else fp4_gemm.rtn_fp4_pertoken_linear
        fn(out, scale, global_scale, x_c, amax_per_row, scale_override)

    return quantization.QuantizeResult(
        data=out,
        scale=scale,
        tensor_scale=global_scale,
    )


def quantize_nvfp4_pertoken_swizzled(
    x: torch.Tensor,
    *,
    four_six: bool = False,
    scale_override: float = 1.0,
) -> quantization.QuantizeResult:
    """Per-token NV FP4 quantization with tcgen05 swizzled scale-factor layout.

    Same as :func:`quantize_nvfp4_pertoken` but the per-group FP8 scale tensor
    is written in the tcgen05 MMA swizzled layout, ready for direct use with
    Blackwell tensor-core GEMM.

    Args:
        x:              2-D BF16 input tensor of shape [m, n].
                        n must be divisible by 128 and <= 4096.
        four_six:       See :func:`quantize_nvfp4_pertoken`.
        scale_override: See :func:`quantize_nvfp4_pertoken`.

    Returns:
        QuantizeResult:
          data          – packed FP4 values, shape [m, n//2], dtype uint8
          scale         – per-group FP8 scale, flat shape [m*n//16], dtype float8_e4m3fn,
                          in tcgen05 swizzled layout
          tensor_scale  – per-token global scale, shape [m], dtype float32
    """
    assert x.is_cuda and x.dtype == torch.bfloat16, "Input must be a CUDA bfloat16 tensor."
    assert x.dim() == 2, "Input must be 2-D."
    rows, cols = x.shape

    x_c = x.contiguous()
    out = torch.empty((rows, cols // 2), dtype=nvfp4_output_t, device=x.device)
    scale = torch.empty(rows * cols // 16, dtype=nvfp4_scale_t, device=x.device)
    global_scale = torch.empty(rows, dtype=torch.float32, device=x.device)

    if cols <= 8192:
        fn = fp4_gemm.four_six_fp4_pertoken_fused if four_six else fp4_gemm.rtn_fp4_pertoken_fused
        fn(out, scale, global_scale, x_c, scale_override)
    else:
        amax_per_row = x_c.abs().amax(dim=1).to(torch.float32)
        fn = fp4_gemm.four_six_fp4_pertoken if four_six else fp4_gemm.rtn_fp4_pertoken
        fn(out, scale, global_scale, x_c, amax_per_row, scale_override)

    return quantization.QuantizeResult(
        data=out,
        scale=scale,
        tensor_scale=global_scale,
    )


def dequantize_nvfp4_pertoken(result: quantization.QuantizeResult) -> torch.Tensor:
    """Dequantize the output of :func:`quantize_nvfp4_pertoken` back to BF16.

    Args:
        result: QuantizeResult from quantize_nvfp4_pertoken.
                data          – uint8 [m, n//2]
                scale         – float8_e4m3fn [m, n//16], row-major
                tensor_scale  – float32 [m]

    Returns:
        Reconstructed BF16 tensor of shape [m, n].
    """
    data = result.data
    scale = result.scale
    global_scale = result.tensor_scale

    grid = _get_fp4_grid(data.device)
    fp4_vals = grid[_unpack_fp4(data)].to(torch.float32)      # [m, n]
    gs = global_scale.to(torch.float32).unsqueeze(1)           # [m, 1]
    # merge scale at [m, n//16] granularity before expanding, matching golden formula:
    # actual_scale = fp8_scale * global_decode_scale, then broadcast over 16 elements
    actual_scale = scale.to(torch.float32) * gs                # [m, n//16]
    m = data.shape[0]
    n = data.shape[1] * 2
    return (fp4_vals.view(m, n // 16, 16) * actual_scale.unsqueeze(-1)).view(m, n).to(torch.bfloat16)


def dequantize_nvfp4_pertoken_swizzled(result: quantization.QuantizeResult) -> torch.Tensor:
    """Dequantize the output of :func:`quantize_nvfp4_pertoken_swizzled` back to BF16.

    Args:
        result: QuantizeResult from quantize_nvfp4_pertoken_swizzled.
                data          – uint8 [m, n//2]
                scale         – float8_e4m3fn [m*n//16], tcgen05 swizzled flat
                tensor_scale  – float32 [m]

    Returns:
        Reconstructed BF16 tensor of shape [m, n].
    """
    data = result.data
    global_scale = result.tensor_scale

    m, nhalf = data.shape
    n = nhalf * 2

    scale_rm = _unswizzle_scale(result.scale.to(torch.float32), m, n)  # [m, n//16]
    grid = _get_fp4_grid(data.device)
    fp4_vals = grid[_unpack_fp4(data)].to(torch.float32)               # [m, n]
    gs = global_scale.to(torch.float32).unsqueeze(1)                   # [m, 1]
    actual_scale = scale_rm * gs                                        # [m, n//16]
    return (fp4_vals.view(m, n // 16, 16) * actual_scale.unsqueeze(-1)).view(m, n).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Grouped per-token quantization with per-group tcgen05 swizzled scale layout
# ---------------------------------------------------------------------------

@dataclass
class GroupQuantizeSwizzledResult:
    """Output of group_quantize_nvfp4_pertoken_swizzled.

    data          – [total_m, k//2] uint8, packed FP4x2, row-major.
    global_scale  – [total_m] float32, per-row global scale.
    scale_flat    – flat FP8 buffer; group g lives at
                    scale_flat[scale_offsets[g]:scale_offsets[g+1]].
    scale_views   – list[Tensor | None], one per group, shaped (32,4,nMT,4,nKT,1)
                    matching the tensor returned by _make_cute_sf, ready for the
                    Sm100 GEMM kernel.  None for empty groups (m_i == 0).
    scale_offsets – list[int], len = num_groups+1, cumulative FP8 element counts.
    row_offsets   – list[int], len = num_groups+1, cumulative row counts.
    """
    data:          torch.Tensor
    global_scale:  torch.Tensor
    scale_flat:    torch.Tensor
    scale_views:   List[torch.Tensor]
    scale_offsets: List[int]
    row_offsets:   List[int]


_SCALE_VIEW_PERM = (3, 4, 1, 5, 2, 0)   # matches mma_permute_order in nvfp4_qgemm.py


def quantize_weight_nvfp4_swizzled(
    x: torch.Tensor,
    *,
    four_six: bool = False,
    scale_override: float = 1.0,
) -> "TensorGlobalNvfp4Tensor":
    """Quantize weight [N, K] to NVFP4 and build CUTE-layout tensors fully on GPU.

    Replaces the ``quantize_weight_nvfp4_tensor_global`` + ``_make_cute_sf`` path:
    ``rtn_fp4`` / ``four_six_fp4`` already writes tcgen05 swizzled scale in the
    kernel, so no D2H → CPU CuTeDSL → H2D is needed.

    The returned ``TensorGlobalNvfp4Tensor`` has ``_cute_fp4`` and ``_cute_sfb``
    pre-populated so ``grouped_gemm_cutedsl_pertoken`` never calls ``_make_cute_sf``.
    """
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

    data       = torch.empty((n, k // 2), dtype=nvfp4_output_t, device=device)
    n_padded   = ((n + _MMA_TILE_ROWS - 1) // _MMA_TILE_ROWS) * _MMA_TILE_ROWS
    scale_flat = torch.zeros(n_padded * (k // 16), dtype=nvfp4_scale_t, device=device)
    gs         = torch.empty(1, dtype=torch.float32, device=device)

    fn = fp4_gemm.four_six_fp4 if four_six else fp4_gemm.rtn_fp4
    fn(data, scale_flat, gs, x, amax, scale_override)

    nMT = n_padded // _MMA_TILE_ROWS
    nKT = k // 64
    cute_sfb = scale_flat.view(1, nMT, nKT, 32, 4, 4).permute(*_SCALE_VIEW_PERM)
    # Non-contiguous intentionally: strides must match create_cute_scale_factor_tensor output
    # (physical (1,nMT,nKT,32,4,4) C-order) so the CuTe kernel stub sees identical strides.
    cute_fp4 = _make_fp4_tensor_for_cutedsl(data, n, k, device)

    t = TensorGlobalNvfp4Tensor(
        data_uint8=data,
        block_scale_fp8=scale_flat,
        amax=amax,
    )
    t._alpha_b_cached = float(amax.item()) / (448.0 * 6.0)
    t._cute_fp4 = cute_fp4
    t._cute_sfb = cute_sfb
    return t


def dequantize_weight_nvfp4_swizzled(t: "TensorGlobalNvfp4Tensor") -> torch.Tensor:
    """Dequantize the output of :func:`quantize_weight_nvfp4_swizzled` back to BF16.

    Args:
        t: TensorGlobalNvfp4Tensor with data_uint8 [N, K//2], block_scale_fp8
           [N_padded*K//16] swizzled, and amax [1] (raw global amax).

    Returns:
        Reconstructed BF16 tensor of shape [N, K].
    """
    data = t.data_uint8
    n, nhalf = data.shape
    k = nhalf * 2
    n_padded = ((n + _MMA_TILE_ROWS - 1) // _MMA_TILE_ROWS) * _MMA_TILE_ROWS
    scale_rm = _unswizzle_scale(t.block_scale_fp8.to(torch.float32), n_padded, k)[:n]
    grid = _get_fp4_grid(data.device)
    fp4_vals = grid[_unpack_fp4(data)].to(torch.float32)
    global_decode_scale = t.amax.to(torch.float32) / (448.0 * 6.0)  # scalar
    actual_scale = scale_rm * global_decode_scale                    # [n, k//16]
    return (fp4_vals.view(n, k // 16, 16) * actual_scale.unsqueeze(-1)).view(n, k).to(torch.bfloat16)


def group_quantize_nvfp4_pertoken_swizzled(
    x: torch.Tensor,
    m_splits: List[int],
    *,
    four_six: bool = False,
    scale_override: float = 1.0,
    skip_scale_views: bool = False,
) -> GroupQuantizeSwizzledResult:
    """Grouped per-token NVFP4 quantization with per-group tcgen05 swizzled scales.

    Replaces the D2H → CPU CuTeDSL layout-conversion → H2D path in
    grouped_gemm_nvfp4_impl._make_cute_sf with a single GPU kernel launch.

    Args:
        x:              2-D BF16 tensor [total_m, k], contiguous, on CUDA.
                        k must be divisible by 128 and <= 4096.
        m_splits:       Row counts per group; sum must equal x.shape[0].
        four_six:       Round-4-or-6 if True; RTN otherwise (default).
        scale_override: Scalar multiplier for the per-row global scale.

    Returns:
        GroupQuantizeSwizzledResult.  scale_views[g] is shaped (32,4,nMT,4,nKT,1)
        and has the same memory layout as what _make_cute_sf returns, so it can
        be passed directly to Sm100GroupedBlockScaledGemmKernel without changes.

    Scale buffer sizing per group:
        Each group allocates ceil(m_i/128)*128 * k//16 elements (zero-padded).
        Padding to the next 128-row tile boundary is mandatory: the tcgen05
        swizzle formula addresses the full tile even when m_i < 128.
    """
    assert x.is_cuda and x.dtype == torch.bfloat16, "x must be a CUDA bfloat16 tensor"
    assert x.dim() == 2 and x.is_contiguous(), "x must be a contiguous 2-D tensor"
    assert sum(m_splits) == x.shape[0], "sum(m_splits) must equal x.shape[0]"

    total_m, k = x.shape
    device = x.device

    if total_m == 0:
        nKT = k // 64
        return GroupQuantizeSwizzledResult(
            data=torch.empty((0, k // 2), dtype=nvfp4_output_t, device=device),
            global_scale=torch.empty(0, dtype=torch.float32, device=device),
            scale_flat=torch.empty(0, dtype=nvfp4_scale_t, device=device),
            scale_views=[None] * len(m_splits),
            scale_offsets=[0] * (len(m_splits) + 1),
            row_offsets=[0] * (len(m_splits) + 1),
        )

    row_offsets: List[int] = [0]
    scale_offsets: List[int] = [0]
    for m_i in m_splits:
        row_offsets.append(row_offsets[-1] + m_i)
        m_padded = ((m_i + _MMA_TILE_ROWS - 1) // _MMA_TILE_ROWS) * _MMA_TILE_ROWS
        scale_offsets.append(scale_offsets[-1] + m_padded * k // 16)

    data         = torch.empty((total_m, k // 2), dtype=nvfp4_output_t, device=device)
    global_scale = torch.empty(total_m, dtype=torch.float32, device=device)
    scale_flat   = torch.zeros(scale_offsets[-1], dtype=nvfp4_scale_t, device=device)

    row_offsets_gpu   = torch.tensor(row_offsets,   dtype=torch.int32, device=device)
    scale_offsets_gpu = torch.tensor(scale_offsets, dtype=torch.int64, device=device)

    fn = fp4_gemm.four_six_fp4_group_pertoken if four_six else fp4_gemm.rtn_fp4_group_pertoken
    fn(data, scale_flat, global_scale, x, row_offsets_gpu, scale_offsets_gpu, scale_override)

    nKT = k // 64
    if skip_scale_views:
        scale_views = []
    else:
        scale_views: List[torch.Tensor] = []
        for g, m_i in enumerate(m_splits):
            if m_i == 0:
                scale_views.append(None)
                continue
            m_padded = ((m_i + _MMA_TILE_ROWS - 1) // _MMA_TILE_ROWS) * _MMA_TILE_ROWS
            nMT  = m_padded // _MMA_TILE_ROWS
            sl   = scale_flat[scale_offsets[g]: scale_offsets[g + 1]]
            view = sl.view(1, nMT, nKT, 32, 4, 4).permute(*_SCALE_VIEW_PERM)
            scale_views.append(view)

    return GroupQuantizeSwizzledResult(
        data=data,
        global_scale=global_scale,
        scale_flat=scale_flat,
        scale_views=scale_views,
        scale_offsets=scale_offsets,
        row_offsets=row_offsets,
    )


def dequantize_group_nvfp4_pertoken_swizzled(result: GroupQuantizeSwizzledResult) -> torch.Tensor:
    """Dequantize the output of :func:`group_quantize_nvfp4_pertoken_swizzled` back to BF16.

    Each group has its own swizzled scale block padded to the next 128-row boundary
    independently; this function unswizzles and dequantizes per group then concatenates.

    Args:
        result: GroupQuantizeSwizzledResult from group_quantize_nvfp4_pertoken_swizzled.

    Returns:
        Reconstructed BF16 tensor of shape [total_m, k].
    """
    total_m, nhalf = result.data.shape
    k = nhalf * 2
    device = result.data.device

    if total_m == 0:
        return torch.empty((0, k), dtype=torch.bfloat16, device=device)

    grid = _get_fp4_grid(device)
    parts = []
    for g in range(len(result.row_offsets) - 1):
        r0, r1 = result.row_offsets[g], result.row_offsets[g + 1]
        m_i = r1 - r0
        if m_i == 0:
            parts.append(torch.empty((0, k), dtype=torch.bfloat16, device=device))
            continue
        m_padded = ((m_i + _MMA_TILE_ROWS - 1) // _MMA_TILE_ROWS) * _MMA_TILE_ROWS
        s0, s1 = result.scale_offsets[g], result.scale_offsets[g + 1]
        scale_rm = _unswizzle_scale(result.scale_flat[s0:s1].to(torch.float32), m_padded, k)[:m_i]
        data_g = result.data[r0:r1]
        gs = result.global_scale[r0:r1].to(torch.float32).unsqueeze(1)
        fp4_vals = grid[_unpack_fp4(data_g)].to(torch.float32)
        actual_scale = scale_rm * gs
        parts.append(
            (fp4_vals.view(m_i, k // 16, 16) * actual_scale.unsqueeze(-1))
            .view(m_i, k).to(torch.bfloat16)
        )
    return torch.cat(parts, dim=0)


@dataclass
class GroupWeightQuantizeResult:
    """Output of group_quantize_nvfp4_tensor_global_packed.

    data          – [total_n, k//2] uint8, packed FP4x2, all experts contiguous.
    cute_data     – [total_n * k] int8, CUTE-layout (first half data, second half zero padding).
                    cute_fp4_for(g) slices this zero-copy.
    global_scale  – [num_experts] float32, per-expert alpha = amax / (448*6).
                    Same value as TensorGlobalNvfp4Tensor._alpha_b_cached, ready to
                    use directly as alpha_b_vec in grouped_gemm_cutedsl_pertoken.
    scale_flat    – flat FP8 buffer; expert g lives at scale_flat[scale_offsets[g]:scale_offsets[g+1]].
    scale_views   – list[Tensor], one per expert, shaped (32,4,nMT,4,nKT,1), CUTE-ready.
    scale_offsets – list[int], len = num_experts+1, cumulative FP8 element counts.
    row_offsets   – list[int], len = num_experts+1, cumulative row counts (fixed stride = n_per_expert).
    """
    data:          torch.Tensor
    cute_data:     torch.Tensor
    global_scale:  torch.Tensor
    scale_flat:    torch.Tensor
    scale_views:   List[torch.Tensor]
    scale_offsets: List[int]
    row_offsets:   List[int]


_NVFP4_GLOBAL_SCALE_DENOM = 448.0 * 6.0


def group_quantize_nvfp4_tensor_global_packed(
    x: torch.Tensor,
    n_per_expert: int,
    *,
    four_six: bool = False,
    scale_override: float = 1.0,
    skip_cute_data: bool = False,
) -> GroupWeightQuantizeResult:
    """Quantize all MoE expert weights in one kernel launch.

    Args:
        x:             3-D BF16 tensor [num_experts, n_per_expert, k], contiguous, on CUDA.
                       k must be divisible by 128 and <= 4096.
        n_per_expert:  Number of rows per expert (must equal x.shape[1]).
        four_six:      Round-4-or-6 if True; RTN otherwise (default).
        scale_override: Scalar multiplier for the per-expert global scale.

    Returns:
        GroupWeightQuantizeResult with all experts packed into contiguous buffers.
        scale_views[g] is shaped (32,4,nMT,4,nKT,1), directly usable by the
        Sm100 grouped GEMM kernel.
    """
    assert x.is_cuda, "x must be a CUDA tensor."
    assert x.dtype == torch.bfloat16, "x must be bfloat16."
    assert x.ndim == 3, "x must be 3-D [num_experts, n_per_expert, k]."
    assert x.is_contiguous(), "x must be contiguous."
    num_experts, n, k = x.shape
    assert n == n_per_expert, f"x.shape[1]={n} != n_per_expert={n_per_expert}"
    assert k % 128 == 0, "k must be divisible by 128."
    assert k // 8 <= 512, "k must be <= 4096."
    device = x.device

    # Flatten to [total_n, k] for the kernel (experts are already contiguous).
    total_n = num_experts * n_per_expert
    x_2d = x.view(total_n, k)

    # Build offsets (fixed stride per expert).
    n_padded = ((n_per_expert + _MMA_TILE_ROWS - 1) // _MMA_TILE_ROWS) * _MMA_TILE_ROWS
    sf_stride = n_padded * k // 16

    row_offsets: List[int] = [i * n_per_expert for i in range(num_experts + 1)]
    scale_offsets: List[int] = [i * sf_stride for i in range(num_experts + 1)]

    data         = torch.empty((total_n, k // 2), dtype=nvfp4_output_t, device=device)
    global_scale = torch.empty(num_experts, dtype=torch.float32, device=device)
    scale_flat   = torch.zeros(scale_offsets[-1], dtype=nvfp4_scale_t, device=device)

    row_offsets_gpu   = torch.tensor(row_offsets,   dtype=torch.int32, device=device)
    scale_offsets_gpu = torch.tensor(scale_offsets, dtype=torch.int64, device=device)

    # Per-expert amax: one reduction per expert, fully on GPU.
    amax = x.abs().amax(dim=(1, 2)).to(torch.float32).contiguous()

    fn = fp4_gemm.four_six_fp4_group_tensor_global if four_six else fp4_gemm.rtn_fp4_group_tensor_global
    fn(data, scale_flat, global_scale, x_2d, row_offsets_gpu, scale_offsets_gpu, amax, scale_override)

    # Build one CUTE int8 buffer for all experts: [total_n * k] int8.
    # Each expert occupies n_per_expert*k bytes: first half is FP4 data, second half zeros.
    # cute_fp4_for(g) = cute_data[g*n*k : (g+1)*n*k], so each expert must be interleaved.
    if skip_cute_data:
        cute_data = None
    else:
        cute_data = torch.zeros(total_n * k, dtype=torch.int8, device=device)
        n_per_expert = total_n // num_experts
        for g in range(num_experts):
            src = data[g * n_per_expert: (g + 1) * n_per_expert].reshape(-1).view(torch.int8)
            dst_start = g * n_per_expert * k
            cute_data[dst_start: dst_start + n_per_expert * k // 2].copy_(src)

    nMT = n_padded // _MMA_TILE_ROWS
    nKT = k // 64
    scale_views: List[torch.Tensor] = []
    for g in range(num_experts):
        sl   = scale_flat[scale_offsets[g]: scale_offsets[g + 1]]
        view = sl.view(1, nMT, nKT, 32, 4, 4).permute(*_SCALE_VIEW_PERM)
        scale_views.append(view)

    return GroupWeightQuantizeResult(
        data=data,
        cute_data=cute_data,
        global_scale=global_scale,
        scale_flat=scale_flat,
        scale_views=scale_views,
        scale_offsets=scale_offsets,
        row_offsets=row_offsets,
    )


def dequantize_group_nvfp4_tensor_global_packed(result: GroupWeightQuantizeResult) -> torch.Tensor:
    """Dequantize the output of :func:`group_quantize_nvfp4_tensor_global_packed` back to BF16.

    Each expert has its own tensor-global amax stored in global_scale[g] as
    alpha = amax / (448*6).  Scale layout per expert is tcgen05 swizzled,
    padded to the next 128-row boundary independently.

    Args:
        result: GroupWeightQuantizeResult from group_quantize_nvfp4_tensor_global_packed.

    Returns:
        Reconstructed BF16 tensor of shape [total_n, k].
    """
    total_n, nhalf = result.data.shape
    k = nhalf * 2
    device = result.data.device
    num_experts = len(result.row_offsets) - 1

    if total_n == 0:
        return torch.empty((0, k), dtype=torch.bfloat16, device=device)

    grid = _get_fp4_grid(device)
    parts = []
    for g in range(num_experts):
        r0, r1 = result.row_offsets[g], result.row_offsets[g + 1]
        n_i = r1 - r0
        if n_i == 0:
            parts.append(torch.empty((0, k), dtype=torch.bfloat16, device=device))
            continue
        n_padded = ((n_i + _MMA_TILE_ROWS - 1) // _MMA_TILE_ROWS) * _MMA_TILE_ROWS
        s0, s1 = result.scale_offsets[g], result.scale_offsets[g + 1]
        scale_rm = _unswizzle_scale(result.scale_flat[s0:s1].to(torch.float32), n_padded, k)[:n_i]
        data_g = result.data[r0:r1]
        gs = result.global_scale[g].to(torch.float32)          # scalar alpha = amax/(448*6)
        fp4_vals = grid[_unpack_fp4(data_g)].to(torch.float32)
        actual_scale = scale_rm * gs                            # [n_i, k//16]
        parts.append(
            (fp4_vals.view(n_i, k // 16, 16) * actual_scale.unsqueeze(-1))
            .view(n_i, k).to(torch.bfloat16)
        )
    return torch.cat(parts, dim=0)
