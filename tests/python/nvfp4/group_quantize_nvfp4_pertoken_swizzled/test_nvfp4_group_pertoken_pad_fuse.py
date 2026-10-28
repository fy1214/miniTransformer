"""Pad-fuse CUDA kernels for MoE activation quant (optional ``fuse_pad`` path).

What: ``fp4_gemm.rtn_fp4_group_pertoken_pad_fuse`` / ``four_six_…`` (unpadded X →
      padded FP4 buffers; pad rows zeroed for GEMM).
Where: ``group_quantize_nvfp4_pertoken_swizzled(fuse_pad=True)``; not yet wired
       in production cursor2 ops (default ``fuse_pad=False``).
Compare: zero-pad then ``rtn_fp4_group_pertoken``; pad rows must be exact zero.

Markers: ``production`` (API/kernel shipped; keep until fuse_pad is removed).
Requires: fp4_gemm with pad_fuse binding, CUDA sm_100+.
"""

from __future__ import annotations

from typing import List, Tuple

import pytest
import torch

try:
    import fp4_gemm
except ImportError:
    fp4_gemm = None


# ---------------------------------------------------------------------------
# Skip guard (same shape as the sibling per-row test)
# ---------------------------------------------------------------------------

_SKIP_REASON: str | None = None
if fp4_gemm is None:
    _SKIP_REASON = "fp4_gemm extension not built"
elif not torch.cuda.is_available():
    _SKIP_REASON = "CUDA is not available"
elif not hasattr(fp4_gemm, "rtn_fp4_group_pertoken_pad_fuse"):
    _SKIP_REASON = "rtn_fp4_group_pertoken_pad_fuse binding missing — rebuild fp4_gemm"
else:
    _maj, _min = torch.cuda.get_device_capability()
    if _maj < 10:
        _SKIP_REASON = (
            f"NVFP4 pertoken pad-fuse kernel requires sm_100+ (got {_maj}.{_min})"
        )

pytestmark = [
    pytest.mark.production,
    pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or ""),
]


# ---------------------------------------------------------------------------
# Constants & dequant helpers (re-used from the per-row test, copied to keep
# this file self-contained — the sibling file is a sibling, not an importable
# module)
# ---------------------------------------------------------------------------

NVFP4_DATA  = torch.uint8
NVFP4_SCALE = torch.float8_e4m3fn
PAD         = 128         # tcgen05 MMA tile height
_MMA_TILE_ROWS = PAD

_FP4_GRID = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _fp4_grid(device) -> torch.Tensor:
    return _FP4_GRID.to(device)


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    p = packed.to(torch.int32)
    return torch.stack([p & 0x0F, (p >> 4) & 0x0F], dim=-1).flatten(start_dim=-2)


def _unswizzle_scale(scale_flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Undo tcgen05 swizzle: flat [nMT*nKT*512] → row-major [rows, cols//16]."""
    nMT = rows // _MMA_TILE_ROWS
    nKT = cols // 64
    s = scale_flat.reshape(nMT * nKT, 32, 4, 4)
    s = s.permute(0, 2, 1, 3).reshape(nMT, nKT, _MMA_TILE_ROWS, 4)
    return s.permute(0, 2, 1, 3).reshape(rows, nKT * 4)


def _dequant_real_rows(
    data: torch.Tensor,         # [m_pad, K//2]  (we slice [:m_unp])
    scale_flat: torch.Tensor,   # [m_pad * K//16]  (we use all m_pad rows)
    global_scale: torch.Tensor, # [m_pad]
    m_unp: int,
    m_pad: int,
    K: int,
) -> torch.Tensor:
    """Dequant the first m_unp rows of one group from the fuse-kernel output."""
    scale_rm = _unswizzle_scale(scale_flat.to(torch.float32), m_pad, K)[:m_unp]
    grid     = _fp4_grid(data.device)
    fp4_vals = grid[_unpack_fp4(data[:m_unp])].to(torch.float32)   # [m_unp, K]
    scale_e  = scale_rm.repeat_interleave(16, dim=1)               # [m_unp, K]
    gs       = global_scale[:m_unp].to(torch.float32).unsqueeze(1) # [m_unp, 1]
    return (fp4_vals * scale_e * gs).to(torch.bfloat16)


def _eff_bitwidth(x: torch.Tensor, x_dq: torch.Tensor) -> float:
    x_f = x.to(torch.float32)
    d_f = x_dq.to(torch.float32)
    row_err = (x_f - d_f).pow(2).sum(dim=-1) / x_f.pow(2).sum(dim=-1).clamp(min=1e-9)
    return (-torch.log2(row_err.mean()) / 2).item()


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------


def _build_offsets(
    m_unpadded_splits: List[int],
    m_padded_splits:   List[int],
    K: int,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unpad = [0]
    pad   = [0]
    sf    = [0]
    for mu, mp in zip(m_unpadded_splits, m_padded_splits):
        unpad.append(unpad[-1] + mu)
        pad  .append(pad[-1]   + mp)
        sf   .append(sf[-1]    + mp * (K // 16))
    return (
        torch.tensor(unpad, dtype=torch.int32, device=device),
        torch.tensor(pad,   dtype=torch.int32, device=device),
        torch.tensor(sf,    dtype=torch.int64, device=device),
    )


def _build_scatter_indices(
    m_unpadded_splits: List[int],
    m_padded_splits:   List[int],
    device: str = "cuda",
) -> torch.Tensor:
    """dst[i] = padded position for unpadded row i (Megatron Fp8Padding semantics)."""
    dst = []
    p_off = 0
    for mu, mp in zip(m_unpadded_splits, m_padded_splits):
        for i in range(mu):
            dst.append(p_off + i)
        p_off += mp
    return torch.tensor(dst, dtype=torch.long, device=device)


def _baseline_pad_then_quant(
    x_unp:             torch.Tensor,
    m_unpadded_splits: List[int],
    m_padded_splits:   List[int],
    K: int,
    four_six: bool = False,
    scale_override: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mock Megatron Fp8Padding + per-row NVFP4 quant.  Returns (data, sf, gs)."""
    device = x_unp.device
    total_pad = sum(m_padded_splits)
    pad_off, sf_off = (
        torch.tensor([0] + list(torch.tensor(m_padded_splits).cumsum(0).tolist()),
                     dtype=torch.int32, device=device),
        torch.tensor(
            [0] + list((torch.tensor(m_padded_splits) * (K // 16)).cumsum(0).tolist()),
            dtype=torch.int64, device=device,
        ),
    )

    # zero-pad then index_copy_ — exactly what Fp8Padding does.
    dst = _build_scatter_indices(m_unpadded_splits, m_padded_splits, str(device))
    x_pad = torch.zeros(total_pad, K, dtype=torch.bfloat16, device=device)
    x_pad.index_copy_(0, dst, x_unp)

    sf_bytes = total_pad * (K // 16)
    data = torch.empty((total_pad, K // 2), dtype=NVFP4_DATA, device=device)
    sf   = torch.zeros(sf_bytes,             dtype=NVFP4_SCALE, device=device)
    gs   = torch.empty(total_pad,            dtype=torch.float32, device=device)
    fn   = (fp4_gemm.four_six_fp4_group_pertoken if four_six
            else fp4_gemm.rtn_fp4_group_pertoken)
    fn(data, sf, gs, x_pad, pad_off, sf_off, scale_override)
    return data, sf, gs


def _run_pad_fuse(
    x_unp:             torch.Tensor,
    m_unpadded_splits: List[int],
    m_padded_splits:   List[int],
    K: int,
    four_six: bool = False,
    scale_override: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = x_unp.device
    unpad_off, pad_off, sf_off = _build_offsets(
        m_unpadded_splits, m_padded_splits, K, str(device)
    )
    total_pad = sum(m_padded_splits)
    sf_bytes  = total_pad * (K // 16)
    data = torch.empty((total_pad, K // 2), dtype=NVFP4_DATA, device=device)
    sf   = torch.zeros(sf_bytes,             dtype=NVFP4_SCALE, device=device)
    gs   = torch.empty(total_pad,            dtype=torch.float32, device=device)
    fn   = (fp4_gemm.four_six_fp4_group_pertoken_pad_fuse if four_six
            else fp4_gemm.rtn_fp4_group_pertoken_pad_fuse)
    fn(data, sf, gs, x_unp, pad_off, unpad_off, sf_off, scale_override)
    return data, sf, gs


# ---------------------------------------------------------------------------
# Shape catalog
# ---------------------------------------------------------------------------

# (G, m_unp_per_group, K) — m_unp ≠ multiple of 128 forces pad waste
TYPICAL_PAD_FUSE_SHAPES = [
    # G   m_unp  K       comment
    (8,   100,   2048),  # 28-row pad per group   (heavy waste)
    (8,   128,   2048),  # zero pad
    (16,  50,    768),   # 78-row pad per group   (severe waste)
    (32,  200,   2048),  # 56-row pad per group   (mid waste)
    (64,  128,   2048),  # zero pad, large G
    (128, 64,    2048),  # Qwen3 1k-token gate_up, 50% waste
    (128, 64,    768),   # Qwen3 1k-token down,    50% waste
]

# Irregular RL-style splits including empty groups
_RL_UNP_SPLITS = [
    100, 0, 96, 192, 64, 100, 0, 48, 128, 160,
    77,  80, 0, 128, 96, 50, 100, 0, 128, 112,
    96,  100, 128, 0, 64, 100, 80, 128, 0, 100,
    128, 96, 50, 100, 128, 0, 80, 128, 64, 100,
    0,   100, 112, 96, 128, 0, 100, 50, 128, 96,
    100, 80, 100, 0, 96, 100, 64, 128, 128, 0,
    100, 112,
]  # 62 groups, mixed empty / non-empty, mostly non-128-aligned


def _pad_splits(m_unp_splits: List[int]) -> List[int]:
    return [((m + PAD - 1) // PAD) * PAD for m in m_unp_splits]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("G,m_unp,K", TYPICAL_PAD_FUSE_SHAPES)
def test_pad_fuse_bitexact_vs_baseline(G: int, m_unp: int, K: int):
    """FP4 data, SF, global_scale must be bit-identical to (Fp8Padding + per-row)."""
    torch.manual_seed(42)
    m_unp_splits = [m_unp] * G
    m_pad_splits = _pad_splits(m_unp_splits)
    total_unp = sum(m_unp_splits)
    x_unp = torch.randn(total_unp, K, dtype=torch.bfloat16, device="cuda")

    data_a, sf_a, gs_a = _baseline_pad_then_quant(x_unp, m_unp_splits, m_pad_splits, K)
    data_b, sf_b, gs_b = _run_pad_fuse        (x_unp, m_unp_splits, m_pad_splits, K)

    assert torch.equal(data_a, data_b), "FP4 data bytes diverge"
    assert torch.equal(sf_a.view(torch.uint8), sf_b.view(torch.uint8)), "SF bytes diverge"
    assert torch.equal(gs_a, gs_b), "global_scale diverge"


@pytest.mark.parametrize("G,m_unp,K", TYPICAL_PAD_FUSE_SHAPES)
def test_pad_fuse_pad_rows_are_zero(G: int, m_unp: int, K: int):
    """Pad rows must carry zero FP4 + zero SF so the downstream GEMM yields zero.

    Pre-fill the output buffers with non-zero junk so the check actually verifies
    the kernel writes zeros (rather than merely preserving a pre-zeroed buffer).
    """
    if m_unp % PAD == 0:
        pytest.skip("no pad rows in this shape")
    torch.manual_seed(7)
    m_unp_splits = [m_unp] * G
    m_pad_splits = _pad_splits(m_unp_splits)
    total_unp = sum(m_unp_splits)
    total_pad = sum(m_pad_splits)
    x_unp = torch.randn(total_unp, K, dtype=torch.bfloat16, device="cuda")

    unpad_off, pad_off_t, sf_off_t = _build_offsets(m_unp_splits, m_pad_splits, K)
    sf_bytes = total_pad * (K // 16)

    # Non-zero junk in both output buffers — kernel MUST overwrite pad rows.
    data = torch.full((total_pad, K // 2), 0xAB, dtype=NVFP4_DATA, device="cuda")
    sf   = torch.full((sf_bytes,),         0xCD, dtype=torch.uint8, device="cuda") \
              .view(NVFP4_SCALE)
    gs   = torch.empty(total_pad, dtype=torch.float32, device="cuda")
    fp4_gemm.rtn_fp4_group_pertoken_pad_fuse(
        data, sf, gs, x_unp, pad_off_t, unpad_off, sf_off_t, 1.0
    )

    pad_acc = 0
    sf_acc  = 0
    for mu, mp in zip(m_unp_splits, m_pad_splits):
        if mu < mp:
            # FP4 data pad rows: must be byte-zero
            pad_rows = data[pad_acc + mu:pad_acc + mp]
            assert torch.all(pad_rows == 0), (
                f"FP4 pad rows [{pad_acc+mu}:{pad_acc+mp}] are not all zero"
            )
            # SF pad rows: un-swizzle the group's SF and check rows mu..mp
            sf_g  = sf[sf_acc:sf_acc + mp * (K // 16)]
            sf_rm = _unswizzle_scale(
                sf_g.view(torch.uint8).to(torch.int32), mp, K
            )  # [mp, K//16] int32 byte values
            assert torch.all(sf_rm[mu:mp] == 0), (
                f"SF pad rows for group (m_unp={mu}, m_pad={mp}) are not all zero"
            )
        pad_acc += mp
        sf_acc  += mp * (K // 16)


@pytest.mark.parametrize("G,m_unp,K", TYPICAL_PAD_FUSE_SHAPES)
def test_pad_fuse_dequant_precision(G: int, m_unp: int, K: int):
    """Dequantize the real (unpadded) rows; eff-bits > 3.3 on Gaussian input."""
    torch.manual_seed(0)
    m_unp_splits = [m_unp] * G
    m_pad_splits = _pad_splits(m_unp_splits)
    total_unp = sum(m_unp_splits)
    x_unp = torch.randn(total_unp, K, dtype=torch.bfloat16, device="cuda")

    data, sf, gs = _run_pad_fuse(x_unp, m_unp_splits, m_pad_splits, K)

    bits = []
    unp_off = 0
    pad_off = 0
    sf_off  = 0
    for mu, mp in zip(m_unp_splits, m_pad_splits):
        if mu == 0:
            pad_off += mp
            sf_off  += mp * (K // 16)
            continue
        x_g  = x_unp[unp_off:unp_off + mu]
        d_g  = data[pad_off:pad_off + mp]
        sf_g = sf[sf_off:sf_off + mp * (K // 16)]
        gs_g = gs[pad_off:pad_off + mp]
        dq   = _dequant_real_rows(d_g, sf_g, gs_g, mu, mp, K)
        bits.append(_eff_bitwidth(x_g, dq))
        unp_off += mu
        pad_off += mp
        sf_off  += mp * (K // 16)

    mean_bits = sum(bits) / len(bits)
    assert mean_bits > 3.3, (
        f"mean eff bits {mean_bits:.3f} < 3.3  (G={G} m_unp={m_unp} K={K})"
    )


def test_pad_fuse_rl_splits_bitexact():
    """Irregular RL m_unp_splits (includes m_unp=0 groups) → bit-exact match."""
    K = 2048
    torch.manual_seed(13)
    m_unp_splits = _RL_UNP_SPLITS
    m_pad_splits = _pad_splits(m_unp_splits)
    total_unp    = sum(m_unp_splits)
    x_unp = torch.randn(total_unp, K, dtype=torch.bfloat16, device="cuda")

    data_a, sf_a, gs_a = _baseline_pad_then_quant(x_unp, m_unp_splits, m_pad_splits, K)
    data_b, sf_b, gs_b = _run_pad_fuse        (x_unp, m_unp_splits, m_pad_splits, K)

    assert torch.equal(data_a, data_b)
    assert torch.equal(sf_a.view(torch.uint8), sf_b.view(torch.uint8))
    assert torch.equal(gs_a, gs_b)


def test_pad_fuse_rl_splits_dequant():
    """RL splits dequant precision (real rows only)."""
    K = 2048
    torch.manual_seed(33)
    m_unp_splits = _RL_UNP_SPLITS
    m_pad_splits = _pad_splits(m_unp_splits)
    total_unp    = sum(m_unp_splits)
    x_unp = torch.randn(total_unp, K, dtype=torch.bfloat16, device="cuda")

    data, sf, gs = _run_pad_fuse(x_unp, m_unp_splits, m_pad_splits, K)

    bits = []
    unp_off = 0
    pad_off = 0
    sf_off  = 0
    for mu, mp in zip(m_unp_splits, m_pad_splits):
        if mu == 0:
            pad_off += mp
            sf_off  += mp * (K // 16)
            continue
        x_g  = x_unp[unp_off:unp_off + mu]
        d_g  = data[pad_off:pad_off + mp]
        sf_g = sf[sf_off:sf_off + mp * (K // 16)]
        gs_g = gs[pad_off:pad_off + mp]
        dq   = _dequant_real_rows(d_g, sf_g, gs_g, mu, mp, K)
        bits.append(_eff_bitwidth(x_g, dq))
        unp_off += mu
        pad_off += mp
        sf_off  += mp * (K // 16)

    mean_bits = sum(bits) / len(bits)
    assert mean_bits > 3.3, f"RL splits: mean eff bits {mean_bits:.3f} < 3.3"


def test_pad_fuse_empty_groups_handled():
    """m_unp=0 groups must produce well-formed output (no crashes, no NaN)."""
    K = 512
    m_unp_splits = [128, 0, 64, 0, 100]
    m_pad_splits = _pad_splits(m_unp_splits)   # [128, 0, 128, 0, 128]
    total_unp = sum(m_unp_splits)
    x_unp = torch.randn(total_unp, K, dtype=torch.bfloat16, device="cuda")

    data_a, sf_a, gs_a = _baseline_pad_then_quant(x_unp, m_unp_splits, m_pad_splits, K)
    data_b, sf_b, gs_b = _run_pad_fuse        (x_unp, m_unp_splits, m_pad_splits, K)

    assert torch.equal(data_a, data_b)
    assert torch.equal(sf_a.view(torch.uint8), sf_b.view(torch.uint8))
    assert torch.equal(gs_a, gs_b)


def test_pad_fuse_all_zeros_stable():
    """All-zero unpadded input → finite global_scale, all-zero FP4 + SF."""
    K = 256
    m_unp_splits = [100, 50, 128, 32]    # all < 128 pad
    m_pad_splits = _pad_splits(m_unp_splits)
    total_unp = sum(m_unp_splits)
    x_unp = torch.zeros(total_unp, K, dtype=torch.bfloat16, device="cuda")

    data, sf, gs = _run_pad_fuse(x_unp, m_unp_splits, m_pad_splits, K)
    assert torch.isfinite(gs).all(), "NaN/Inf in global_scale on all-zero input"
    assert torch.all(data == 0), "FP4 data must be all zero for zero input"
    assert torch.all(sf.view(torch.uint8) == 0), "SF must be all zero for zero input"


@pytest.mark.parametrize("G,m_unp,K", [(8, 100, 2048), (32, 200, 2048)])
def test_pad_fuse_four_six_variant(G: int, m_unp: int, K: int):
    """four_six variant: bit-exact vs (Fp8Padding + per-row four_six)."""
    if not hasattr(fp4_gemm, "four_six_fp4_group_pertoken_pad_fuse"):
        pytest.skip("four_six_fp4_group_pertoken_pad_fuse binding missing")
    torch.manual_seed(99)
    m_unp_splits = [m_unp] * G
    m_pad_splits = _pad_splits(m_unp_splits)
    total_unp    = sum(m_unp_splits)
    x_unp = torch.randn(total_unp, K, dtype=torch.bfloat16, device="cuda")

    data_a, sf_a, gs_a = _baseline_pad_then_quant(
        x_unp, m_unp_splits, m_pad_splits, K, four_six=True
    )
    data_b, sf_b, gs_b = _run_pad_fuse(
        x_unp, m_unp_splits, m_pad_splits, K, four_six=True
    )

    assert torch.equal(data_a, data_b), "four_six FP4 data diverges"
    assert torch.equal(sf_a.view(torch.uint8), sf_b.view(torch.uint8)), \
        "four_six SF diverges"
    assert torch.equal(gs_a, gs_b), "four_six global_scale diverges"


def test_pad_fuse_overallocated_pad():
    """m_pad > ceil(m_unp/128)*128 (caller over-allocates) → still bit-exact."""
    G  = 4
    K  = 1024
    m_unp_splits = [50, 100, 128, 64]
    # Over-allocate one pad block beyond the strict ceil(m_unp/128)*128.
    m_pad_splits = [
        ((m + PAD - 1) // PAD + 1) * PAD if m > 0 else 0
        for m in m_unp_splits
    ]   # +128 extra in every non-empty group
    total_unp = sum(m_unp_splits)
    torch.manual_seed(123)
    x_unp = torch.randn(total_unp, K, dtype=torch.bfloat16, device="cuda")

    data_a, sf_a, gs_a = _baseline_pad_then_quant(x_unp, m_unp_splits, m_pad_splits, K)
    data_b, sf_b, gs_b = _run_pad_fuse        (x_unp, m_unp_splits, m_pad_splits, K)

    assert torch.equal(data_a, data_b)
    assert torch.equal(sf_a.view(torch.uint8), sf_b.view(torch.uint8))
    assert torch.equal(gs_a, gs_b)


# ---------------------------------------------------------------------------
# Standalone runner (mirrors the sibling per-row test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if _SKIP_REASON:
        print(f"Skipped: {_SKIP_REASON}")
        raise SystemExit(0)

    print("=" * 72)
    print("NVFP4 pertoken pad-fuse — precision check")
    print("=" * 72)
    for G, m_unp, K in TYPICAL_PAD_FUSE_SHAPES:
        torch.manual_seed(42)
        m_unp_splits = [m_unp] * G
        m_pad_splits = _pad_splits(m_unp_splits)
        total_unp    = sum(m_unp_splits)
        x_unp = torch.randn(total_unp, K, dtype=torch.bfloat16, device="cuda")

        data_a, sf_a, gs_a = _baseline_pad_then_quant(x_unp, m_unp_splits, m_pad_splits, K)
        data_b, sf_b, gs_b = _run_pad_fuse        (x_unp, m_unp_splits, m_pad_splits, K)

        ok_d  = torch.equal(data_a, data_b)
        ok_s  = torch.equal(sf_a.view(torch.uint8), sf_b.view(torch.uint8))
        ok_gs = torch.equal(gs_a, gs_b)

        # Dequant only the real rows
        bits = []
        unp_off = 0; pad_off = 0; sf_off = 0
        for mu, mp in zip(m_unp_splits, m_pad_splits):
            if mu == 0:
                pad_off += mp; sf_off += mp * (K // 16); continue
            x_g  = x_unp[unp_off:unp_off + mu]
            d_g  = data_b[pad_off:pad_off + mp]
            sf_g = sf_b[sf_off:sf_off + mp * (K // 16)]
            gs_g = gs_b[pad_off:pad_off + mp]
            dq   = _dequant_real_rows(d_g, sf_g, gs_g, mu, mp, K)
            bits.append(_eff_bitwidth(x_g, dq))
            unp_off += mu; pad_off += mp; sf_off += mp * (K // 16)

        mean_bits = sum(bits) / len(bits) if bits else float("nan")
        status = "OK" if (ok_d and ok_s and ok_gs) else "MISMATCH"
        print(
            f"  G={G:>3} m_unp={m_unp:>4} K={K:>4}:  "
            f"data={'✓' if ok_d else '✗'}  "
            f"sf={'✓' if ok_s else '✗'}  "
            f"gs={'✓' if ok_gs else '✗'}  "
            f"eff-bits={mean_bits:.3f}  [{status}]"
        )
