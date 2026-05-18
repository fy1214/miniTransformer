"""Shared utilities for quantization precision experiments."""
import torch

# Lazy device cache for lookup tables — CUDA Graph compatible.
# H2D copies happen during warmup before graph capture, not during capture itself.
_POS_VALS_CACHE: dict = {}


def _get_cached_pos_vals(name: str, vals_cpu: torch.Tensor, device) -> torch.Tensor:
    key = f"{name}:{device}"
    if key not in _POS_VALS_CACHE:
        _POS_VALS_CACHE[key] = vals_cpu.to(device)
    return _POS_VALS_CACHE[key]


# Physical max of torch.float8_e4m3fn (OCP FP8 E4M3 standard)
_FP8_E4M3_MAX = 448.0


# ── Lookup tables (all positive representable values for each format) ─────────

_POS_VALS_E5M1 = torch.tensor(
    [0, 3.05176e-05, 6.10352e-05, 9.15527e-05, 1.220703e-04, 1.831055e-04,
     2.441406e-04, 3.662109e-04, 4.882812e-04, 7.324219e-04, 9.765625e-04,
     1.4648438e-03, 1.953125e-03, 2.9296875e-03, 3.90625e-03, 5.859375e-03,
     7.8125e-03, 1.171875e-02, 1.5625e-02, 2.34375e-02, 3.125e-02, 4.6875e-02,
     6.25e-02, 9.375e-02, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1, 1.5, 2, 3,
     4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
     1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152,
     65536, 98304],
    dtype=torch.float32,
)
_POS_VALS_E4M1 = torch.tensor(
    [0, 7.8125e-03, 1.5625e-02, 2.34375e-02, 3.125e-02, 4.6875e-02, 6.25e-02,
     9.375e-02, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8,
     12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384],
    dtype=torch.float32,
)
_POS_VALS_E3M1 = torch.tensor(
    [0, 0.125, 0.25, 0.375, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24],
    dtype=torch.float32,
)
_POS_VALS_E2M4 = torch.tensor(
    [0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625,
     0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1, 1.0625, 1.125, 1.1875,
     1.25, 1.3125, 1.375, 1.4375, 1.5, 1.5625, 1.625, 1.6875, 1.75, 1.8125,
     1.875, 1.9375, 2, 2.125, 2.25, 2.375, 2.5, 2.625, 2.75, 2.875, 3, 3.125,
     3.25, 3.375, 3.5, 3.625, 3.75, 3.875, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5,
     5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75],
    dtype=torch.float32,
)
_POS_VALS_E2M3 = torch.tensor(
    [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1.125, 1.25, 1.375,
     1.5, 1.625, 1.75, 1.875, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.5,
     5, 5.5, 6, 6.5, 7, 7.5],
    dtype=torch.float32,
)
_POS_VALS_E2M2 = torch.tensor(
    [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 7],
    dtype=torch.float32,
)
_POS_VALS_E2M1 = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6],
    dtype=torch.float32,
)


# ── Lookup quantizer ──────────────────────────────────────────────────────────

def _lookup_quantize(x: torch.Tensor, pos_vals: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest quantization via exhaustive distance lookup."""
    x32 = x.float()
    sign = torch.sign(x32)
    sign = torch.where(x32 == 0, torch.ones_like(sign), sign)
    idx = (x32.abs().unsqueeze(-1) - pos_vals.unsqueeze(0)).abs().argmin(dim=-1)
    return sign * pos_vals[idx]


def _quantize_e5m1(x: torch.Tensor) -> torch.Tensor:
    return _lookup_quantize(x, _get_cached_pos_vals("E5M1", _POS_VALS_E5M1, x.device))

def _quantize_e4m1(x: torch.Tensor) -> torch.Tensor:
    return _lookup_quantize(x, _get_cached_pos_vals("E4M1", _POS_VALS_E4M1, x.device))

def _quantize_e3m1(x: torch.Tensor) -> torch.Tensor:
    return _lookup_quantize(x, _get_cached_pos_vals("E3M1", _POS_VALS_E3M1, x.device))

def _quantize_e2m4(x: torch.Tensor) -> torch.Tensor:
    return _lookup_quantize(x, _get_cached_pos_vals("E2M4", _POS_VALS_E2M4, x.device))

def _quantize_e2m3(x: torch.Tensor) -> torch.Tensor:
    return _lookup_quantize(x, _get_cached_pos_vals("E2M3", _POS_VALS_E2M3, x.device))

def _quantize_e2m2(x: torch.Tensor) -> torch.Tensor:
    return _lookup_quantize(x, _get_cached_pos_vals("E2M2", _POS_VALS_E2M2, x.device))

def _quantize_e2m1(x: torch.Tensor) -> torch.Tensor:
    return _lookup_quantize(x, _get_cached_pos_vals("E2M1", _POS_VALS_E2M1, x.device))


# ── Scale utilities ───────────────────────────────────────────────────────────

def fp_format_max(e_bits: int, m_bits: int) -> float:
    """Max representable value for formats where all exponent patterns are valid (no NaN/Inf)."""
    bias = (1 << (e_bits - 1)) - 1
    max_exp = (1 << e_bits) - 1
    m_max = 1 << m_bits
    return (2.0 ** (max_exp - bias)) * (1.0 + (m_max - 1) / m_max)


def cast_to_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Round scale to the nearest power of 2 (ceiling in log2 space)."""
    exponent = torch.ceil(torch.log2(scale.clamp(min=1e-38))).clamp(-127, 127)
    return torch.pow(2.0, exponent)


def cast_to_e4m3(scale: torch.Tensor) -> torch.Tensor:
    """Round scale to the nearest FP8 E4M3 representable value (round-to-nearest)."""
    return scale.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn).to(torch.float32)


# ── Per-block quantization framework ─────────────────────────────────────────

def quant_dequant_block(
    x: torch.Tensor,
    block_size: int,
    fp_max: float,
    quantize_fn,
    scale_type: str,  # "e8m0" | "e4m3"
) -> torch.Tensor:
    """Per-block quant/dequant (nvfp4-style: encode_scale derived from quantized decode_scale).

    encode_scale = 1 / round(decode_scale) ensures strict encode/decode invertibility.
    clamp after scaling guards against overflow when round(decode_scale) < decode_scale.
    """
    orig_shape = x.shape
    x32 = x.float().reshape(-1)
    assert x32.numel() % block_size == 0, \
        f"numel()={x32.numel()} must be divisible by block_size={block_size}"
    x_blocks = x32.reshape(-1, block_size)

    block_max = x_blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-38)
    decode_scale_raw = block_max / fp_max

    if scale_type == "e8m0":
        decode_scale_q = cast_to_e8m0(decode_scale_raw)
    elif scale_type == "e4m3":
        decode_scale_q = cast_to_e4m3(decode_scale_raw)
    else:
        raise ValueError(f"unknown scale_type: {scale_type!r}")

    encode_scale = torch.reciprocal(decode_scale_q.clamp(min=1e-38))
    x_scaled = torch.clamp(x_blocks * encode_scale, -fp_max, fp_max)
    x_q = quantize_fn(x_scaled.reshape(-1)).reshape(-1, block_size)
    return (x_q * decode_scale_q).reshape(orig_shape).bfloat16()


# ── Per-token two-level quantization framework ────────────────────────────────

def quant_dequant_pertoken(
    x: torch.Tensor,
    block_size: int,
    fp_max: float,
    quantize_fn,
    scale_type: str,   # "e8m0" | "e4m3"
    norm_max: float = 448.0,
) -> torch.Tensor:
    """Two-level quant: per-token global scale (fp32) + per-block decode scale (e4m3/e8m0).

    Level-1  global_scale[i] = amax(|x[i,:]|) / (norm_max * fp_max)  — not quantized
    Level-2  decode_scale    = amax(|x_norm_block|) / fp_max          — round-to-nearest
    Rebuild  x_dq = x_q * decode_scale_q * global_scale
    """
    assert x.dim() == 2, f"expected 2D [M, K], got {x.shape}"
    M, K = x.shape
    assert K % block_size == 0, f"K={K} must be divisible by block_size={block_size}"
    B = K // block_size
    x32 = x.float()

    token_amax = x32.abs().amax(dim=1, keepdim=True).clamp(min=1e-38)
    global_scale = token_amax / (norm_max * fp_max)
    x_norm = x32 / global_scale

    x_blocks = x_norm.reshape(M * B, block_size)
    block_amax = x_blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-38)
    decode_scale_raw = block_amax / fp_max

    if scale_type == "e4m3":
        decode_scale_q = cast_to_e4m3(decode_scale_raw)
    elif scale_type == "e8m0":
        decode_scale_q = cast_to_e8m0(decode_scale_raw)
    else:
        raise ValueError(f"unknown scale_type: {scale_type!r}")

    encode_scale = torch.reciprocal(decode_scale_q.clamp(min=1e-38))
    x_scaled = torch.clamp(x_blocks * encode_scale, -fp_max, fp_max)
    x_q = quantize_fn(x_scaled.reshape(-1)).reshape(M * B, block_size)

    global_scale_exp = global_scale.unsqueeze(1).expand(M, B, 1).reshape(M * B, 1)
    return (x_q * decode_scale_q * global_scale_exp).reshape(M, K).to(x.dtype)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(x_orig: torch.Tensor, x_dq: torch.Tensor) -> dict:
    x32, dq32 = x_orig.float(), x_dq.float()
    err = (x32 - dq32).abs()
    snr = 10 * torch.log10(x32.pow(2).mean() / (err.pow(2).mean() + 1e-20))
    rel_l2 = torch.norm(x32 - dq32) / (torch.norm(x32) + 1e-20)
    return dict(
        mae=err.mean().item(),
        max_abs=err.max().item(),
        mean_rel=(err / (x32.abs() + 1e-8)).mean().item(),
        snr=snr.item(),
        rel_l2=rel_l2.item(),
    )


def print_metrics(name: str, x_orig: torch.Tensor, x_dq: torch.Tensor) -> dict:
    x32, dq32 = x_orig.float(), x_dq.float()
    m = compute_metrics(x_orig, x_dq)
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Range     : [{x32.min():.4f}, {x32.max():.4f}]"
          f" → [{dq32.min():.4f}, {dq32.max():.4f}]")
    print(f"  MAE       : {m['mae']:.6f}")
    print(f"  Max Err   : {m['max_abs']:.6f}")
    print(f"  Rel Err   : {m['mean_rel']:.4%}")
    print(f"  SNR       : {m['snr']:.2f} dB")
    print(f"  Rel L2    : {m['rel_l2']:.6f}")
    return m


def print_summary_table(results: dict) -> None:
    width = max(len(n) for n in results) + 2
    print(f"\n{'=' * (width + 34)}")
    print(f"  {'Format + Scale':<{width}} {'MAE':>9} {'SNR(dB)':>9} {'Rel_L2':>10}")
    print(f"  {'-' * (width + 30)}")
    for name, m in results.items():
        print(f"  {name:<{width}} {m['mae']:>9.6f} {m['snr']:>9.2f} {m['rel_l2']:>10.6f}")
    print()


def make_test_tensor(m: int, n: int, device: str = "cpu") -> torch.Tensor:
    """Mixed-distribution 2D tensor: 50% N(0,0.5), 25% U(-2,2), 25% N(0,5)."""
    torch.manual_seed(42)
    N = m * n
    return torch.cat([
        torch.randn(N // 2) * 0.5,
        torch.rand(N // 4) * 4 - 2,
        torch.randn(N // 4) * 5,
    ]).reshape(m, n).bfloat16().to(device)
