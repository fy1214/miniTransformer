"""Wrapper-level tests for ``group_quantize_nvfp4_pertoken_swizzled(fuse_pad=...)``.

We already have a kernel-level test (``test_nvfp4_group_pertoken_pad_fuse.py``)
that covers the C++ binding directly. These tests target the Python wrapper:

* ``fuse_pad=True``   – output must be bit-identical to invoking
  ``fp4_gemm.rtn_fp4_group_pertoken_pad_fuse`` with the same offsets that the
  wrapper builds internally. The wrapper also exposes
  ``unpadded_row_offsets_gpu`` for the caller to slice real rows back.
* ``fuse_pad=False``  – legacy path: bit-identical to
  ``fp4_gemm.rtn_fp4_group_pertoken`` (this is what the wrapper called before
  the change), and ``unpadded_row_offsets_gpu`` is ``None``.
"""
from __future__ import annotations

from typing import List

import pytest
import torch

# Skip the whole module on hardware that lacks NVFP4 PTX support.
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
_MAJOR, _MINOR = torch.cuda.get_device_capability()
if _MAJOR < 10:
    pytest.skip(f"NVFP4 requires sm_100+, found sm_{_MAJOR}{_MINOR}", allow_module_level=True)

import fp4_gemm  # noqa: E402

from miniTransformer.ops.nvfp4_quantize import (  # noqa: E402
    GroupQuantizeSwizzledResult,
    group_quantize_nvfp4_pertoken_swizzled,
    nvfp4_output_t,
    nvfp4_scale_t,
)


_MMA = 128


def _padded_total(m_splits: List[int]) -> int:
    return sum(((m + _MMA - 1) // _MMA) * _MMA for m in m_splits)


def _build_unpadded_offsets(m_splits: List[int], device) -> torch.Tensor:
    return torch.tensor(
        [0] + list(torch.tensor(m_splits, dtype=torch.int32).cumsum(0).tolist()),
        dtype=torch.int32, device=device,
    )


def _build_padded_offsets(m_splits: List[int], device) -> torch.Tensor:
    padded = [((m + _MMA - 1) // _MMA) * _MMA for m in m_splits]
    return torch.tensor(
        [0] + list(torch.tensor(padded, dtype=torch.int32).cumsum(0).tolist()),
        dtype=torch.int32, device=device,
    )


def _build_scale_offsets(m_splits: List[int], k: int, device) -> torch.Tensor:
    padded = [((m + _MMA - 1) // _MMA) * _MMA for m in m_splits]
    return torch.tensor(
        [0] + list(torch.tensor(padded, dtype=torch.int64).cumsum(0).tolist()),
        dtype=torch.int64, device=device,
    ) * (k // 16)


def _make_input(m_splits: List[int], k: int, *, dtype=torch.bfloat16, seed: int = 0xC0FFEE) -> torch.Tensor:
    """Unpadded activation [sum(m_splits), k] with heterogeneous per-row magnitude."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    total = sum(m_splits)
    if total == 0:
        return torch.empty((0, k), dtype=dtype, device="cuda")
    x = torch.randn(total, k, generator=g, dtype=torch.float32, device="cuda") * 4.0
    row_scale = torch.empty(total, 1, dtype=torch.float32, device="cuda").uniform_(
        0.25, 4.0, generator=g
    )
    return (x * row_scale).to(dtype).contiguous()


def _direct_pad_fuse(
    x_unpadded: torch.Tensor,
    m_splits: List[int],
    *,
    four_six: bool,
    scale_override: float,
):
    """Reference impl that calls the C++ binding directly using offsets we build
    here (i.e. no Python wrapper involvement)."""
    assert x_unpadded.is_cuda
    total_m, k = x_unpadded.shape
    device = x_unpadded.device
    padded_total = _padded_total(m_splits)
    data = torch.empty((padded_total, k // 2), dtype=nvfp4_output_t, device=device)
    gs = torch.empty(padded_total, dtype=torch.float32, device=device)
    sf = torch.zeros(padded_total * (k // 16), dtype=nvfp4_scale_t, device=device)
    pad_off = _build_padded_offsets(m_splits, device)
    unp_off = _build_unpadded_offsets(m_splits, device)
    sf_off = _build_scale_offsets(m_splits, k, device)
    fn = (fp4_gemm.four_six_fp4_group_pertoken_pad_fuse if four_six
          else fp4_gemm.rtn_fp4_group_pertoken_pad_fuse)
    fn(data, sf, gs, x_unpadded, pad_off, unp_off, sf_off, scale_override)
    return data, sf, gs, pad_off, unp_off, sf_off


def _direct_legacy(
    x_padded: torch.Tensor,
    m_splits: List[int],
    *,
    four_six: bool,
    scale_override: float,
):
    """Legacy reference: x is already padded row-for-row to m_splits."""
    assert x_padded.is_cuda
    total_m, k = x_padded.shape
    device = x_padded.device
    data = torch.empty((total_m, k // 2), dtype=nvfp4_output_t, device=device)
    gs = torch.empty(total_m, dtype=torch.float32, device=device)
    padded_total = _padded_total(m_splits)
    sf = torch.zeros(padded_total * (k // 16), dtype=nvfp4_scale_t, device=device)
    row_off = _build_unpadded_offsets(m_splits, device)
    sf_off = _build_scale_offsets(m_splits, k, device)
    fn = fp4_gemm.four_six_fp4_group_pertoken if four_six else fp4_gemm.rtn_fp4_group_pertoken
    fn(data, sf, gs, x_padded, row_off, sf_off, scale_override)
    return data, sf, gs, row_off, sf_off


# ---------------------------------------------------------------------------
# fuse_pad=True
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "m_splits,k",
    [
        ([100, 50, 200, 64], 2048),
        ([0, 64, 0, 128, 256, 0], 2048),
        ([1, 2, 3, 4, 5, 6, 7, 8], 768),
        ([512, 384, 257, 129, 0, 1023], 2048),
    ],
    ids=["mix4", "empty_groups", "tiny", "bigger"],
)
@pytest.mark.parametrize("four_six", [False, True], ids=["rtn", "four_six"])
def test_fuse_pad_true_matches_direct_kernel(m_splits, k, four_six):
    x = _make_input(m_splits, k)
    scale_override = 1.0

    res = group_quantize_nvfp4_pertoken_swizzled(
        x, m_splits, four_six=four_six, scale_override=scale_override, fuse_pad=True
    )
    data_ref, sf_ref, gs_ref, pad_off, unp_off, sf_off = _direct_pad_fuse(
        x, m_splits, four_six=four_six, scale_override=scale_override
    )

    assert torch.equal(res.data, data_ref), "data tensor differs from direct kernel call"
    assert torch.equal(res.scale_flat, sf_ref), "scale_flat differs"
    assert torch.equal(res.global_scale, gs_ref), "global_scale differs"

    assert torch.equal(res.row_offsets_gpu, pad_off)
    assert res.unpadded_row_offsets_gpu is not None
    assert torch.equal(res.unpadded_row_offsets_gpu, unp_off)
    assert torch.equal(res.scale_offsets_gpu, sf_off)


def test_fuse_pad_true_empty_input():
    """sum(m_splits)=0 still returns a well-formed result with zero-length buffers
    and unpadded_row_offsets_gpu populated to the expected zeros tensor."""
    m_splits = [0, 0, 0, 0]
    k = 2048
    x = torch.empty((0, k), dtype=torch.bfloat16, device="cuda")
    res = group_quantize_nvfp4_pertoken_swizzled(x, m_splits, fuse_pad=True)
    assert res.data.shape == (0, k // 2)
    assert res.global_scale.numel() == 0
    assert res.scale_flat.numel() == 0
    assert torch.equal(res.row_offsets_gpu, torch.zeros(5, dtype=torch.int32, device="cuda"))
    assert res.unpadded_row_offsets_gpu is not None
    assert torch.equal(
        res.unpadded_row_offsets_gpu, torch.zeros(5, dtype=torch.int32, device="cuda")
    )


# ---------------------------------------------------------------------------
# fuse_pad=False (legacy path)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "m_splits,k",
    [
        ([128, 128, 256, 128], 2048),
        ([512, 384, 256, 128], 768),
    ],
    ids=["aligned_small", "aligned_bigger"],
)
@pytest.mark.parametrize("four_six", [False, True], ids=["rtn", "four_six"])
def test_legacy_path_matches_direct_kernel(m_splits, k, four_six):
    x = _make_input(m_splits, k)
    scale_override = 1.0

    res = group_quantize_nvfp4_pertoken_swizzled(
        x, m_splits, four_six=four_six, scale_override=scale_override, fuse_pad=False
    )
    data_ref, sf_ref, gs_ref, row_off, sf_off = _direct_legacy(
        x, m_splits, four_six=four_six, scale_override=scale_override
    )

    assert torch.equal(res.data, data_ref), "legacy data differs from direct kernel"
    assert torch.equal(res.scale_flat, sf_ref), "legacy scale_flat differs"
    assert torch.equal(res.global_scale, gs_ref), "legacy global_scale differs"
    assert torch.equal(res.row_offsets_gpu, row_off)
    assert torch.equal(res.scale_offsets_gpu, sf_off)
    assert res.unpadded_row_offsets_gpu is None, "legacy path should not populate unpadded offsets"


def test_legacy_default_is_unchanged():
    """fuse_pad omitted → must default to False, yielding legacy behavior."""
    m_splits = [128, 256, 128]
    k = 2048
    x = _make_input(m_splits, k)
    res_default = group_quantize_nvfp4_pertoken_swizzled(x, m_splits)
    res_explicit = group_quantize_nvfp4_pertoken_swizzled(x, m_splits, fuse_pad=False)
    assert torch.equal(res_default.data, res_explicit.data)
    assert torch.equal(res_default.scale_flat, res_explicit.scale_flat)
    assert torch.equal(res_default.global_scale, res_explicit.global_scale)
    assert res_default.unpadded_row_offsets_gpu is None
    assert res_explicit.unpadded_row_offsets_gpu is None


# ---------------------------------------------------------------------------
# Equivalence between fuse_pad=True (unpadded x) and the legacy path fed an
# Fp8Padding-style pre-padded x. This is the chain that pad-fuse replaces in
# Megatron, so an end-to-end bit-exact match is the load-bearing invariant.
# ---------------------------------------------------------------------------

def _padded_input_via_zero_copy(x_unp: torch.Tensor, m_splits: List[int]) -> torch.Tensor:
    """Mimic Megatron's Fp8Padding D2D copy: real rows in their group's slot, pad rows zero."""
    device = x_unp.device
    k = x_unp.shape[1]
    padded_total = _padded_total(m_splits)
    out = torch.zeros(padded_total, k, dtype=x_unp.dtype, device=device)
    src_off = 0
    dst_off = 0
    for m in m_splits:
        if m > 0:
            out[dst_off:dst_off + m].copy_(x_unp[src_off:src_off + m])
        src_off += m
        dst_off += ((m + _MMA - 1) // _MMA) * _MMA
    return out


@pytest.mark.parametrize(
    "m_splits,k",
    [
        ([100, 50, 200, 64], 2048),
        ([1, 2, 3, 4, 5, 6, 7, 8], 768),
    ],
    ids=["mix", "tiny"],
)
def test_fuse_pad_true_equals_legacy_on_pre_padded_input(m_splits, k):
    """Wrapper(fuse_pad=True, x=unpadded) ≡ wrapper(fuse_pad=False, x=Fp8Padding(unpadded))
    on every byte that lives in the padded layout (data, scale_flat, gs)."""
    x_unp = _make_input(m_splits, k)
    x_padded = _padded_input_via_zero_copy(x_unp, m_splits)
    padded_splits = [((m + _MMA - 1) // _MMA) * _MMA for m in m_splits]

    fuse = group_quantize_nvfp4_pertoken_swizzled(x_unp, m_splits, fuse_pad=True)
    legacy = group_quantize_nvfp4_pertoken_swizzled(x_padded, padded_splits, fuse_pad=False)

    assert torch.equal(fuse.data, legacy.data), "fuse_pad data ≠ legacy data on pre-padded input"
    assert torch.equal(fuse.scale_flat, legacy.scale_flat), "fuse_pad scale_flat ≠ legacy scale_flat"
    assert torch.equal(fuse.global_scale, legacy.global_scale), "fuse_pad gs ≠ legacy gs"
