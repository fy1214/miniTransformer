"""
NVFP4 grouped GEMM helpers (quantization + host alpha preparation).

Kernel execution lives in ``blockwise_group_gemm_cute_alpha_epilogue``.

Quantization sources:
  - Per-token (rowwise) amax: ``miniTransformer.ops.triton.nvfp4_quantization``.
  - Tensor-global amax (e.g. weights): ``miniTransformer.ops.nvfp4_quantization``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

import torch

from miniTransformer.ops.nvfp4_qgemm import create_cute_scale_factor_tensor
from miniTransformer.ops.nvfp4_ref_quantize import cast_from_fp4x2, cast_to_fp4x2

_NVFP4_SF_VEC_SIZE = 16
_NVFP4_FP4_MAX = 6.0
_NVFP4_FP8_MAX = 448.0
_NVFP4_GLOBAL_SCALE_DENOM = _NVFP4_FP8_MAX * _NVFP4_FP4_MAX
_NVFP4_ALPHA_DENOM = _NVFP4_GLOBAL_SCALE_DENOM * _NVFP4_GLOBAL_SCALE_DENOM


@dataclass
class PertokenNvfp4Tensor:
    """Per-token NVFP4 activation/weight: packed FP4 + block/global scales."""

    data_uint8: torch.Tensor
    block_scale: torch.Tensor
    global_scale: torch.Tensor

    def __post_init__(self) -> None:
        if self.data_uint8.dtype != torch.uint8:
            raise TypeError("data_uint8 must be torch.uint8 (packed FP4x2).")
        if self.block_scale.ndim != 2:
            raise ValueError("block_scale must be 2D [M, K//16].")
        if self.global_scale.ndim != 1:
            raise ValueError("global_scale must be 1D [M].")


@dataclass
class TensorGlobalNvfp4Tensor:
    """Tensor-global NVFP4 weight from ``quantize_nvfp4_cuda`` (fp4_gemm)."""

    data_uint8: torch.Tensor
    block_scale_fp8: torch.Tensor
    amax: torch.Tensor


def is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


def _make_fp4_tensor_for_cutedsl(
    data_uint8: torch.Tensor, mode0: int, K: int, device: torch.device
) -> torch.Tensor:
    buf = torch.empty(mode0 * K, dtype=torch.int8, device=device)
    buf[: mode0 * K // 2].copy_(data_uint8.reshape(-1).view(torch.int8))
    return buf.view(mode0, K, 1)


def _make_cute_sf(sf_2d: torch.Tensor, mn: int, sf_k: int, l: int = 1) -> torch.Tensor:
    if sf_2d.dtype == torch.uint8:
        sf_fp8 = sf_2d[:mn, :sf_k].contiguous().view(torch.float8_e4m3fn)
    else:
        sf_fp8 = sf_2d[:mn, :sf_k]
    ref = sf_fp8.cpu().unsqueeze(-1)
    return create_cute_scale_factor_tensor(l, mn, sf_k, ref)


def block_scale_to_sf_fp8(
    block_scale: torch.Tensor,
    mn: int,
    k: int,
    block_size: int = _NVFP4_SF_VEC_SIZE,
) -> torch.Tensor:
    sf_k = k // block_size
    return (
        block_scale[:mn, :sf_k]
        .reshape(mn, sf_k)
        .float()
        .clamp(min=1e-38, max=_NVFP4_FP8_MAX)
        .to(torch.float8_e4m3fn)
    )


def pertoken_scales_to_sfa_2d(
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    m: int,
    k: int,
    block_size: int = _NVFP4_SF_VEC_SIZE,
) -> torch.Tensor:
    """Merge per-token global scale into block SFA, matching block-FP8 then multiply semantics."""
    sf_k = k // block_size
    block_fp8 = block_scale_to_sf_fp8(block_scale, m, k)
    bs_dec = block_fp8[:m, :sf_k].reshape(m, sf_k).float()
    gs = global_scale[:m].reshape(m, 1).float()
    merged = (bs_dec * gs).clamp(min=1e-38, max=_NVFP4_FP8_MAX)
    return merged.to(torch.float8_e4m3fn)


def pack_fp4_grid_to_uint8(x_q: torch.Tensor) -> torch.Tensor:
    # Align discrete levels with Triton pertoken kernel / golden (mx E2M1 table).
    from miniTransformer.ops.triton.nvfp4_quantization import mx_e2m1_quantize

    return cast_to_fp4x2(mx_e2m1_quantize(x_q.float()))


def quantize_activation_nvfp4_pertoken(
    x: torch.Tensor,
    block_size: int = _NVFP4_SF_VEC_SIZE,
) -> PertokenNvfp4Tensor:
    # Token-wise amax: Triton path (aligned with quant_nvfp4_pertoken_golden).
    from miniTransformer.ops.triton.nvfp4_quantization import quant_nvfp4_pertoken_triton

    assert x.ndim == 2
    m, k = x.shape
    assert k % block_size == 0
    x_q, global_scale, block_scale = quant_nvfp4_pertoken_triton(x, block_size=block_size)
    bs = block_scale.reshape(m, k // block_size) if block_scale.ndim == 1 else block_scale
    return PertokenNvfp4Tensor(
        data_uint8=pack_fp4_grid_to_uint8(x_q),
        block_scale=bs,
        global_scale=global_scale.reshape(m),
    )


def _pad_2d_to_multiples_of_32(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """``quantize_nvfp4_cuda`` requires both dimensions divisible by 32."""
    n, k = x.shape
    n_pad = ((n + 31) // 32) * 32
    k_pad = ((k + 31) // 32) * 32
    if n_pad == n and k_pad == k:
        return x, (n, k)
    out = torch.zeros(n_pad, k_pad, dtype=x.dtype, device=x.device)
    out[:n, :k] = x
    return out, (n, k)


def quantize_weight_nvfp4_tensor_global(x: torch.Tensor) -> TensorGlobalNvfp4Tensor:
    # Tensor-global amax: CUDA fp4_gemm path.
    from miniTransformer.ops.nvfp4_quantization import quantize_nvfp4_cuda

    assert x.ndim == 2
    assert x.is_cuda, "quantize_weight_nvfp4_tensor_global requires a CUDA tensor."
    n, k = x.shape
    assert k % _NVFP4_SF_VEC_SIZE == 0

    x_pad, (n0, k0) = _pad_2d_to_multiples_of_32(x)
    qr = quantize_nvfp4_cuda(x_pad, with_2d_quantization=False, return_transpose=False)
    data_uint8 = qr.data[:n0, : k0 // 2].contiguous()
    block_scale_fp8 = qr.scale[:n0, : k0 // _NVFP4_SF_VEC_SIZE].contiguous()
    assert qr.tensor_scale is not None
    amax = qr.tensor_scale.reshape(())
    return TensorGlobalNvfp4Tensor(
        data_uint8=data_uint8,
        block_scale_fp8=block_scale_fp8,
        amax=amax,
    )


def build_group_alpha_pertoken(
    act: PertokenNvfp4Tensor,
    b: Union[TensorGlobalNvfp4Tensor, PertokenNvfp4Tensor, object],
    alpha_b: float,
    n: int,
) -> Union[float, torch.Tensor]:
    """
    Per-group alpha applied in kernel epilogue (float32), not on FP8 block scales.

    - A pertoken + B tensor-global: ``global_a[M] * alpha_b`` (1D broadcast)
    - A pertoken + B pertoken: ``global_a[:, None] * global_b[None, :]`` (2D)
    """
    ga = act.global_scale.float()
    if isinstance(b, PertokenNvfp4Tensor):
        gb = b.global_scale[:n].float()
        return ga.unsqueeze(1) * gb.unsqueeze(0)
    return ga * alpha_b


def _resolve_pertoken_activation(a: Union[PertokenNvfp4Tensor, torch.Tensor]) -> PertokenNvfp4Tensor:
    if isinstance(a, PertokenNvfp4Tensor):
        return a
    if isinstance(a, torch.Tensor):
        return quantize_activation_nvfp4_pertoken(a)
    raise TypeError(f"Unsupported activation type: {type(a)}")


def _resolve_group_weight(
    b: Union[TensorGlobalNvfp4Tensor, PertokenNvfp4Tensor, object],
    n_i: int,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, float, object]:
    if isinstance(b, PertokenNvfp4Tensor):
        return (
            b.data_uint8,
            block_scale_to_sf_fp8(b.block_scale, n_i, k),
            1.0,
            b,
        )
    if isinstance(b, TensorGlobalNvfp4Tensor):
        amax_b = float(b.amax.item())
        alpha_b = amax_b / _NVFP4_GLOBAL_SCALE_DENOM
        return (
            b.data_uint8,
            b.block_scale_fp8[:n_i, : k // _NVFP4_SF_VEC_SIZE],
            alpha_b,
            b,
        )
    if hasattr(b, "_rowwise_data") and hasattr(b, "_rowwise_scale_inv"):
        amax_b = float(b._amax_rowwise.float().item())
        alpha_b = amax_b / _NVFP4_GLOBAL_SCALE_DENOM
        return (
            b._rowwise_data,
            b._rowwise_scale_inv[:n_i, : k // _NVFP4_SF_VEC_SIZE],
            alpha_b,
            b,
        )
    raise TypeError(f"Unsupported weight type: {type(b)}")


def grouped_gemm_cutedsl_te(
    A: list,
    B: list,
    m_splits: list,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Original TE NVFP4 grouped GEMM (tensor-global amax).

    Expects ``A[i]``/``B[i]`` to be Transformer Engine NVFP4 quantized tensors
    (``_rowwise_data``, ``_rowwise_scale_inv``, ``_amax_rowwise``).
    """
    from miniTransformer.ops.cuteDSL.nvfp4.blockwise_group_gemm_cute_alpha_epilogue import (
        GroupedBlockScaledProfile,
        Sm100GroupedBlockScaledGemmKernel,
    )

    device = A[0]._rowwise_data.device
    k = A[0]._rowwise_data.shape[1] * 2
    sf_k = k // _NVFP4_SF_VEC_SIZE

    abc_torch, sfasfb_torch, problem_sizes, alphas = [], [], [], []
    for i, m_i in enumerate(m_splits):
        n_i = B[i]._rowwise_data.shape[0]
        a_fp4 = _make_fp4_tensor_for_cutedsl(A[i]._rowwise_data, m_i, k, device)
        b_fp4 = _make_fp4_tensor_for_cutedsl(B[i]._rowwise_data, n_i, k, device)
        c_out = torch.empty((m_i, n_i), dtype=out_dtype, device=device)
        sfa = _make_cute_sf(A[i]._rowwise_scale_inv, m_i, sf_k)
        sfb = _make_cute_sf(B[i]._rowwise_scale_inv, n_i, sf_k)
        abc_torch.append([a_fp4, b_fp4, c_out])
        sfasfb_torch.append([sfa, sfb])
        problem_sizes.append((m_i, n_i, k, 1))
        amax_a = A[i]._amax_rowwise.float().item()
        amax_b = B[i]._amax_rowwise.float().item()
        alphas.append(amax_a * amax_b / _NVFP4_ALPHA_DENOM)

    kernel = Sm100GroupedBlockScaledGemmKernel(profile=GroupedBlockScaledProfile.NVFP4)
    return torch.cat(
        kernel.run_from_grouped_inputs(
            abc_tensors={"torch": abc_torch},
            sfasfb_tensors={"torch": sfasfb_torch},
            problem_sizes=problem_sizes,
            out_dtype=out_dtype,
            accumulate=False,
            alphas=alphas,
        ),
        dim=0,
    )


def grouped_gemm_cutedsl_pertoken(
    A: List[Union[PertokenNvfp4Tensor, torch.Tensor]],
    B: List[Union[TensorGlobalNvfp4Tensor, PertokenNvfp4Tensor, object]],
    m_splits: List[int],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Pertoken activation quant + block scales in SFA/SFB; global scales via ``alphas``.

    Scalar alphas are fused in the kernel epilogue. Non-scalar alphas (1D ``[M]``, 2D
    ``[M, N]``, etc.) are applied once on the host inside
    ``Sm100GroupedBlockScaledGemmKernel.run_from_grouped_inputs`` — do not multiply again.
    """
    from miniTransformer.ops.cuteDSL.nvfp4.blockwise_group_gemm_cute_alpha_epilogue import (
        GroupedBlockScaledProfile,
        Sm100GroupedBlockScaledGemmKernel,
    )

    if len(A) != len(B) or len(A) != len(m_splits):
        raise ValueError("A, B, and m_splits must have the same length.")

    act0 = _resolve_pertoken_activation(A[0])
    device = act0.data_uint8.device
    k = act0.data_uint8.shape[1] * 2
    sf_k = k // _NVFP4_SF_VEC_SIZE

    abc_torch, sfasfb_torch, problem_sizes, alphas = [], [], [], []
    for i, m_i in enumerate(m_splits):
        act = _resolve_pertoken_activation(A[i])
        if isinstance(B[i], (PertokenNvfp4Tensor, TensorGlobalNvfp4Tensor)):
            n_i = B[i].data_uint8.shape[0]
        else:
            n_i = B[i]._rowwise_data.shape[0]

        b_data, sfb_2d, alpha_b, b_for_alpha = _resolve_group_weight(B[i], n_i, k)
        a_fp4 = _make_fp4_tensor_for_cutedsl(act.data_uint8, m_i, k, device)
        b_fp4 = _make_fp4_tensor_for_cutedsl(b_data, n_i, k, device)
        c_out = torch.empty((m_i, n_i), dtype=out_dtype, device=device)
        sfa = _make_cute_sf(block_scale_to_sf_fp8(act.block_scale, m_i, k), m_i, sf_k)
        sfb = _make_cute_sf(sfb_2d, n_i, sf_k)
        abc_torch.append([a_fp4, b_fp4, c_out])
        sfasfb_torch.append([sfa, sfb])
        problem_sizes.append((m_i, n_i, k, 1))
        alphas.append(build_group_alpha_pertoken(act, b_for_alpha, alpha_b, n_i))

    kernel = Sm100GroupedBlockScaledGemmKernel(profile=GroupedBlockScaledProfile.NVFP4)
    outputs = kernel.run_from_grouped_inputs(
        abc_tensors={"torch": abc_torch},
        sfasfb_tensors={"torch": sfasfb_torch},
        problem_sizes=problem_sizes,
        out_dtype=out_dtype,
        accumulate=False,
        alphas=alphas,
    )

    return torch.cat(outputs, dim=0)


def dequant_pertoken_nvfp4(
    tensor: PertokenNvfp4Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:

    m, k_half = tensor.data_uint8.shape
    k = k_half * 2
    x = cast_from_fp4x2(tensor.data_uint8, torch.float32)
    # Match kernel path: activation block scale is consumed as FP8 E4M3 in SFA.
    bs_fp8 = block_scale_to_sf_fp8(tensor.block_scale, m, k)
    bs = bs_fp8[:, : k // _NVFP4_SF_VEC_SIZE].float().repeat_interleave(_NVFP4_SF_VEC_SIZE, dim=1)
    gs = tensor.global_scale.reshape(m, 1).float()
    return (x * bs * gs).to(out_dtype)


def dequant_tensor_global_nvfp4(
    tensor: TensorGlobalNvfp4Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:

    n, k_half = tensor.data_uint8.shape
    k = k_half * 2
    x = cast_from_fp4x2(tensor.data_uint8, torch.float32)
    sx = tensor.block_scale_fp8[:, : k // _NVFP4_SF_VEC_SIZE].float().repeat_interleave(
        _NVFP4_SF_VEC_SIZE, dim=1
    )
    gs = tensor.amax.float() / _NVFP4_GLOBAL_SCALE_DENOM
    return (x * sx * gs).to(out_dtype)


def reference_grouped_gemm_pertoken(
    A: List[Union[PertokenNvfp4Tensor, torch.Tensor]],
    B: List[Union[TensorGlobalNvfp4Tensor, PertokenNvfp4Tensor, object]],
    m_splits: List[int],
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    outputs = []
    for i, m_i in enumerate(m_splits):
        act = _resolve_pertoken_activation(A[i])
        b_item = B[i]
        if isinstance(b_item, PertokenNvfp4Tensor):
            b_dq = dequant_pertoken_nvfp4(b_item, torch.float32)
        elif isinstance(b_item, TensorGlobalNvfp4Tensor):
            b_dq = dequant_tensor_global_nvfp4(b_item, torch.float32)
        elif hasattr(b_item, "_rowwise_data"):
            n_i = b_item._rowwise_data.shape[0]
            k = act.data_uint8.shape[1] * 2
            wrap = TensorGlobalNvfp4Tensor(
                data_uint8=b_item._rowwise_data.view(torch.uint8),
                block_scale_fp8=b_item._rowwise_scale_inv[:n_i, : k // 16],
                amax=b_item._amax_rowwise.reshape(()),
            )
            b_dq = dequant_tensor_global_nvfp4(wrap, torch.float32)
        else:
            raise TypeError(f"Unsupported B type: {type(b_item)}")
        outputs.append(dequant_pertoken_nvfp4(act, torch.float32) @ b_dq.T)
    return torch.cat(outputs, dim=0).to(out_dtype)
