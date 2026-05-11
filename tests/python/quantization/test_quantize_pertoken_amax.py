"""
quantize_exp_1_pertoken.py

在 quantize_exp_1.py 的基础上，参考 quant_nvfp4_pertoken 的实现思路，
为所有 E2Mx 格式加入 per-token global amax 两级量化：

  Level-1 (global): global_scale[i] = amax(|x[i,:]|) / (E4M3_MAX * fp_max)
                    per-token，每行一个，fp32 不量化
  Level-2 (block) : decode_scale    = amax(|x_norm_block|) / fp_max
                    per-block，round-to-nearest (e4m3 or e8m0)

重建：x_dq = x_q * block_decode_scale * global_scale

与 quantize_exp_1.py 的区别：
  - 输入必须是 2D [M, K]
  - 多了 global_scale 一级，block_scale 在 normalized 空间计算
  - encode_scale 从量化后的 block_decode_scale 反算（nvfp4 风格）
  - 量化后 clamp 兜底
"""

import torch

# CUDA Graph compatible: lazy pos_vals cache per device
_POS_VALS_CACHE: dict = {}

def _get_cached_pos_vals(name: str, vals_cpu: torch.Tensor, device) -> torch.Tensor:
    """First call moves tensor to device and caches; subsequent calls return cache.
    Warmup before CUDA Graph capture triggers the H2D copy; capture sees no H2D ops."""
    key = f"{name}:{device}"
    if key not in _POS_VALS_CACHE:
        _POS_VALS_CACHE[key] = vals_cpu.to(device)
    return _POS_VALS_CACHE[key]

FLOAT8_E4M3_MAX = 240.0


# ============================================================
# 核心：浮点分解 + 重建
# ============================================================

def get_fp_components(x: torch.Tensor, e_bits: int, m_bits: int):
    bias    = (1 << (e_bits - 1)) - 1
    max_exp = (1 << e_bits) - 1
    m_max   = (1 << m_bits)

    fp_max = (2.0 ** (max_exp - bias)) * (1.0 + (m_max - 1) / m_max)

    x32   = x.float()
    sign  = torch.sign(x32)
    sign  = torch.where(x32 == 0, torch.ones_like(sign), sign)
    abs_x = x32.abs().clamp(max=fp_max)

    is_zero = (abs_x == 0)

    safe_abs    = abs_x.clamp(min=2.0 ** (1 - bias - m_bits))
    e_real_f    = torch.floor(torch.log2(safe_abs))
    e_stored_f  = (e_real_f + bias).clamp(1, max_exp)
    e_real_used = e_stored_f - bias

    scale   = torch.pow(2.0, e_real_used)
    frac    = (abs_x / scale - 1.0).clamp(0.0, 1.0)
    m_int_f = torch.round(frac * m_max)

    carry      = (m_int_f >= m_max)
    e_stored_f = torch.where(carry, (e_stored_f + 1).clamp(1, max_exp), e_stored_f)
    m_int_f    = torch.where(carry, torch.zeros_like(m_int_f), m_int_f)

    denorm_threshold = 2.0 ** (1 - bias)
    is_denorm        = (abs_x < denorm_threshold) & (~is_zero)
    m_int_denorm     = torch.round(abs_x / denorm_threshold * m_max).clamp(0, m_max - 1).long()

    e_final = torch.where(is_denorm | is_zero,
                          torch.zeros_like(e_stored_f.long()),
                          e_stored_f.long())
    m_final = torch.where(is_denorm, m_int_denorm,
              torch.where(is_zero,   torch.zeros_like(m_int_f.long()),
                                     m_int_f.long()))
    return sign, e_final, m_final, bias, max_exp, m_max


def reconstruct_fp(sign, e_stored, m_int, bias, m_max):
    e_f = e_stored.float()
    m_f = m_int.float()

    is_zero   = (e_stored == 0) & (m_int == 0)
    is_denorm = (e_stored == 0) & (m_int != 0)

    val_norm   = torch.pow(2.0, e_f - bias) * (1.0 + m_f / m_max)
    val_denorm = (2.0 ** (1.0 - bias)) * (m_f / m_max)

    val = torch.where(~is_zero & ~is_denorm, val_norm,
          torch.where(is_denorm,             val_denorm,
                                             torch.zeros_like(val_norm)))
    return sign * val


# ============================================================
# scale 量化工具
# ============================================================

def cast_to_e8m0(decode_scale: torch.Tensor) -> torch.Tensor:
    """E8M0: 2^ceil(log2(x))，纯 2 的幂次，round 语义（ceil = 向上）"""
    exponent = torch.ceil(torch.log2(decode_scale.clamp(min=1e-38)))
    exponent = torch.clamp(exponent, -127, 127)
    return torch.pow(2.0, exponent)


def cast_to_e4m3(decode_scale: torch.Tensor) -> torch.Tensor:
    """E4M3 round-to-nearest，直接用 PyTorch 原生转换，max=448"""
    decode_scale = decode_scale.clamp(-FLOAT8_E4M3_MAX, FLOAT8_E4M3_MAX)
    return decode_scale.to(torch.float8_e4m3fn).to(torch.float32)


# ============================================================
# 各格式量化函数
# ============================================================

_POS_VALS_E2M4 = torch.tensor(
    [0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1, 1.0625, 1.125, 1.1875, 1.25, 1.3125, 1.375, 1.4375, 1.5, 1.5625, 1.625, 1.6875, 1.75, 1.8125, 1.875, 1.9375, 2, 2.125, 2.25, 2.375, 2.5, 2.625, 2.75, 2.875, 3, 3.125, 3.25, 3.375, 3.5, 3.625, 3.75, 3.875, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75],
    dtype=torch.float32
)

def _quantize_e2m4(x: torch.Tensor) -> torch.Tensor:
    pos_vals = _get_cached_pos_vals("E2M4", _POS_VALS_E2M4, x.device)
    x32  = x.float()
    sign = torch.sign(x32)
    sign = torch.where(x32 == 0, torch.ones_like(sign), sign)
    diff = (x32.abs().unsqueeze(-1) - pos_vals.unsqueeze(0)).abs()
    idx  = diff.argmin(dim=-1)
    return sign * pos_vals[idx]

_POS_VALS_E2M3 = torch.tensor(
    [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5],
    dtype=torch.float32
)

def _quantize_e2m3(x: torch.Tensor) -> torch.Tensor:
    pos_vals = _get_cached_pos_vals("E2M3", _POS_VALS_E2M3, x.device)
    x32  = x.float()
    sign = torch.sign(x32)
    sign = torch.where(x32 == 0, torch.ones_like(sign), sign)
    diff = (x32.abs().unsqueeze(-1) - pos_vals.unsqueeze(0)).abs()
    idx  = diff.argmin(dim=-1)
    return sign * pos_vals[idx]

_POS_VALS_E2M2 = torch.tensor(
    [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 7],
    dtype=torch.float32
)

def _quantize_e2m2(x: torch.Tensor) -> torch.Tensor:
    pos_vals = _get_cached_pos_vals("E2M2", _POS_VALS_E2M2, x.device)
    x32  = x.float()
    sign = torch.sign(x32)
    sign = torch.where(x32 == 0, torch.ones_like(sign), sign)
    diff = (x32.abs().unsqueeze(-1) - pos_vals.unsqueeze(0)).abs()
    idx  = diff.argmin(dim=-1)
    return sign * pos_vals[idx]

_POS_VALS_E2M1 = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6],
    dtype=torch.float32
)

def _quantize_e2m1(x: torch.Tensor) -> torch.Tensor:
    pos_vals = _get_cached_pos_vals("E2M1", _POS_VALS_E2M1, x.device)
    x32  = x.float()
    sign = torch.sign(x32)
    sign = torch.where(x32 == 0, torch.ones_like(sign), sign)
    diff = (x32.abs().unsqueeze(-1) - pos_vals.unsqueeze(0)).abs()
    idx  = diff.argmin(dim=-1)
    return sign * pos_vals[idx]


def fp_format_max(e_bits: int, m_bits: int) -> float:
    bias    = (1 << (e_bits - 1)) - 1
    max_exp = (1 << e_bits) - 1
    m_max   = (1 << m_bits)
    return (2.0 ** (max_exp - bias)) * (1.0 + (m_max - 1) / m_max)


# ============================================================
# 核心：带 per-token global amax 的两级量化框架
# ============================================================

def quant_dequant_pertoken(
    x          : torch.Tensor,   # [M, K], bfloat16 or float32
    block_size : int,
    fp_max     : float,           # 数据格式的最大值（E2M1=6, E2M3=7.5 ...）
    quantize_fn,
    scale_type : str,             # "e4m3" | "e8m0"
    debug_name : str = "",
    debug      : bool = False,
) -> torch.Tensor:
    """
    两级量化框架（nvfp4 pertoken 风格）：

    Level-1  per-token global scale（fp32，不量化）:
        global_scale[i] = amax(|x[i,:]|) / (FLOAT8_E4M3_MAX * fp_max)
        normalize: x_norm = x / global_scale        → 每行 amax ≈ E4M3_MAX * fp_max

    Level-2  per-block decode scale（e4m3 or e8m0，round-to-nearest）:
        decode_scale = amax(|x_norm_block|) / fp_max → 值域在 [0, E4M3_MAX]
        decode_scale_q = round(decode_scale)          → 存储为 e4m3/e8m0
        encode_scale   = 1 / decode_scale_q           → 反算，保证严格互逆
        x_scaled = x_norm * encode_scale              → 值域 ≈ [-fp_max, fp_max]
        x_scaled = clamp(x_scaled, -fp_max, fp_max)   → clamp 兜底
        x_q      = quantize_fn(x_scaled)

    重建：
        x_dq = x_q * decode_scale_q * global_scale
    """
    assert x.dim() == 2, f"输入必须是 2D [M, K]，got {x.shape}"
    M, K   = x.shape
    x32    = x.float()

    assert K % block_size == 0, (
        f"K={K} 必须是 block_size={block_size} 的整数倍，请在调用前 pad"
    )
    B = K // block_size  # 每行的 block 数

    # ── Level-1: per-token global scale（fp32，不量化）───────
    token_amax   = x32.abs().amax(dim=1, keepdim=True).clamp(min=1e-38)  # [M, 1]
    global_scale = token_amax / (FLOAT8_E4M3_MAX * fp_max)               # [M, 1]



    # ── normalize ─────────────────────────────────────────────
    x_norm = x32 / global_scale                           # [M, K]
    # 每行 amax 现在约等于 FLOAT8_E4M3_MAX * fp_max



    # ── Level-2: per-block decode scale ───────────────────────
    x_blocks   = x_norm.reshape(M * B, block_size)        # [M*B, block_size]
    block_amax = x_blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-38)  # [M*B, 1]
    decode_scale_raw = block_amax / fp_max                 # [M*B, 1]，值域 [0, E4M3_MAX]

    # round-to-nearest 量化 decode_scale
    if scale_type == "e4m3":
        decode_scale_q = cast_to_e4m3(decode_scale_raw)   # [M*B, 1]
    elif scale_type == "e8m0":
        decode_scale_q = cast_to_e8m0(decode_scale_raw)   # [M*B, 1]
    else:
        raise ValueError(f"未知 scale_type: {scale_type}")

    # encode_scale 从量化后的 decode_scale 反算（nvfp4 风格）
    encode_scale = torch.reciprocal(decode_scale_q.clamp(min=1e-38))  # [M*B, 1]



    # ── 量化 ──────────────────────────────────────────────────
    x_scaled = x_blocks * encode_scale                     # [M*B, block_size]
    x_scaled = torch.clamp(x_scaled, -fp_max, fp_max)      # clamp 兜底



    x_q = quantize_fn(x_scaled.reshape(-1)).reshape(M * B, block_size)

    # ── 重建 ──────────────────────────────────────────────────
    # decode_scale_q : [M*B, 1]
    # global_scale   : [M, 1]  → expand 到 [M*B, 1]
    global_scale_expanded = global_scale.unsqueeze(1).expand(M, B, 1).reshape(M * B, 1)

    x_dq = (x_q * decode_scale_q * global_scale_expanded).reshape(M, K)

    return x_dq.to(x.dtype)


# ============================================================
# 对外接口：E2Mx × {e4m3, e8m0} 共 8 种
# ============================================================

def quant_dequant_e2m4_e4m3scale(x, block_size=16, debug=False):
    return quant_dequant_pertoken(x, block_size, fp_format_max(2, 4),
        _quantize_e2m4, "e4m3", "E2M4+global | E4M3 scale", debug)

def quant_dequant_e2m3_e4m3scale(x, block_size=16, debug=False):
    return quant_dequant_pertoken(x, block_size, fp_format_max(2, 3),
        _quantize_e2m3, "e4m3", "E2M3+global | E4M3 scale", debug)

def quant_dequant_e2m2_e4m3scale(x, block_size=16, debug=False):
    return quant_dequant_pertoken(x, block_size, fp_format_max(2, 2),
        _quantize_e2m2, "e4m3", "E2M2+global | E4M3 scale", debug)

def quant_dequant_e2m1_e4m3scale(x, block_size=16, debug=False):
    return quant_dequant_pertoken(x, block_size, fp_format_max(2, 1),
        _quantize_e2m1, "e4m3", "E2M1+global | E4M3 scale", debug)

def quant_dequant_e2m4_e8m0scale(x, block_size=16, debug=False):
    return quant_dequant_pertoken(x, block_size, fp_format_max(2, 4),
        _quantize_e2m4, "e8m0", "E2M4+global | E8M0 scale", debug)

def quant_dequant_e2m3_e8m0scale(x, block_size=16, debug=False):
    return quant_dequant_pertoken(x, block_size, fp_format_max(2, 3),
        _quantize_e2m3, "e8m0", "E2M3+global | E8M0 scale", debug)

def quant_dequant_e2m2_e8m0scale(x, block_size=16, debug=False):
    return quant_dequant_pertoken(x, block_size, fp_format_max(2, 2),
        _quantize_e2m2, "e8m0", "E2M2+global | E8M0 scale", debug)

def quant_dequant_e2m1_e8m0scale(x, block_size=16, debug=False):
    return quant_dequant_pertoken(x, block_size, fp_format_max(2, 1),
        _quantize_e2m1, "e8m0", "E2M1+global | E8M0 scale", debug)


# ============================================================
# 误差统计 & 主测试
# ============================================================

def compute_metrics(x_orig, x_dq, name):
    x32  = x_orig.float()
    dq32 = x_dq.float()
    ae   = (x32 - dq32).abs()
    re   = ae / (x32.abs() + 1e-8)
    snr  = 10 * torch.log10(x32.pow(2).mean() / (ae.pow(2).mean() + 1e-20))

    print(f"\n{'=' * 62}")
    print(f"  {name}")
    print(f"{'=' * 62}")
    print(f"  原始范围     : [{x32.min():.4f}, {x32.max():.4f}]")
    print(f"  量化后范围   : [{dq32.min():.4f}, {dq32.max():.4f}]")
    print(f"  MAE          : {ae.mean():.6f}")
    print(f"  Max Abs Err  : {ae.max():.6f}")
    print(f"  Mean Rel Err : {re.mean():.4%}")
    print(f"  SNR          : {snr:.2f} dB")
    return snr.item(), ae.mean().item()


def run_all_tests():
    torch.manual_seed(42)
    M, K = 32, 128   # 2D 输入，M 个 token，每个 token K 维
    x = torch.cat([
        torch.randn(M * K // 2) * 0.5,
        torch.rand(M * K // 4)  * 4 - 2,
        torch.randn(M * K // 4) * 5,
    ]).reshape(M, K).bfloat16()

    print(f"输入 Tensor: shape={x.shape}, dtype={x.dtype}")
    print(f"值域: [{x.float().min():.4f}, {x.float().max():.4f}]")
    print(f"均值: {x.float().mean():.4f}, 标准差: {x.float().std():.4f}")

    configs = [
        ("E2M4+global | E4M3 scale", lambda: quant_dequant_e2m4_e4m3scale(x, debug=True)),
        ("E2M3+global | E4M3 scale", lambda: quant_dequant_e2m3_e4m3scale(x, debug=True)),
        ("E2M2+global | E4M3 scale", lambda: quant_dequant_e2m2_e4m3scale(x, debug=True)),
        ("E2M1+global | E4M3 scale", lambda: quant_dequant_e2m1_e4m3scale(x, debug=True)),
        ("E2M4+global | E8M0 scale", lambda: quant_dequant_e2m4_e8m0scale(x, debug=True)),
        ("E2M3+global | E8M0 scale", lambda: quant_dequant_e2m3_e8m0scale(x, debug=True)),
        ("E2M2+global | E8M0 scale", lambda: quant_dequant_e2m2_e8m0scale(x, debug=True)),
        ("E2M1+global | E8M0 scale", lambda: quant_dequant_e2m1_e8m0scale(x, debug=True)),
    ]

    results = {}
    for name, fn in configs:
        xq = fn()
        snr, mae = compute_metrics(x, xq, name)
        results[name] = (snr, mae)

    print(f"\n{'=' * 68}")
    print(f"  {'格式 + Scale':<42} {'MAE':>9} {'SNR(dB)':>9}")
    print(f"  {'-' * 62}")
    for name, (snr, mae) in results.items():
        print(f"  {name:<42} {mae:>9.6f} {snr:>9.2f}")
    print()


if __name__ == "__main__":
    run_all_tests()
