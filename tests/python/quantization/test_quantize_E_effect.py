"""
quantize_exp.py
===============
作者: mingfa
日期: 2026-04-29

功能:
    探索固定尾数位数(M=1)、变化指数位数(E=2/3/4/5)的低精度浮点量化格式，
    结合两种 block scale 存储方式（E8M0 / E4M3）的量化精度对比实验。

量化方案（共 8 种）:
    数据格式: E5M1 / E4M1 / E3M1 / E2M1
    Scale  : E8M0（2的幂次，ceil取整） / E4M3（round-to-nearest，clamp兜底）

核心设计（参考 nvfp4_quantize_op_ref._quantize_blockwise_reference）:
    1. decode_scale = round-to-nearest(block_amax / fp_max)，不做 ceil
    2. encode_scale = 1 / decode_scale，从量化后值反算，保证 encode/decode 互逆
    3. x_scaled = x * encode_scale，配合 clamp(x_scaled, -fp_max, fp_max) 兜底

评估指标: MAE / Mean Rel Err / SNR / Rel_L2
"""
import torch

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

    # 进位处理
    carry      = (m_int_f >= m_max)
    e_stored_f = torch.where(carry, (e_stored_f + 1).clamp(1, max_exp), e_stored_f)
    m_int_f    = torch.where(carry, torch.zeros_like(m_int_f), m_int_f)

    # 非规格化
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
    val_denorm = torch.pow(2.0, torch.tensor(1.0 - bias)) * (m_f / m_max)

    val = torch.where(~is_zero & ~is_denorm, val_norm,
          torch.where(is_denorm,             val_denorm,
                                             torch.zeros_like(val_norm)))
    return sign * val


# ============================================================
# scale 量化工具
# ============================================================

def cast_to_e8m0(decode_scale: torch.Tensor) -> torch.Tensor:
    """Cast to E8M0: 2^ceil(log2(x))，纯 2 的幂次"""
    exponent = torch.ceil(torch.log2(decode_scale.clamp(min=1e-38)))
    exponent = torch.clamp(exponent, -127, 127)
    return torch.pow(2.0, exponent)


def cast_to_e4m3(decode_scale: torch.Tensor) -> torch.Tensor:
    """round-to-nearest 到 E4M3，直接用 PyTorch 原生转换"""
    E4M3_MAX = 448.0
    decode_scale = decode_scale.clamp(-E4M3_MAX, E4M3_MAX)
    return decode_scale.to(torch.float8_e4m3fn).to(torch.float32)


# ============================================================
# 通用 per-block 量化框架（nvfp4 风格：round + clamp，encode_scale 反算）
# ============================================================

def quant_dequant_block(
    x          : torch.Tensor,
    block_size : int,
    fp_max     : float,
    quantize_fn,            # 将 x_scaled (float32) 量化并返回 float32
    scale_type : str,       # "e8m0" | "e4m3"
    debug_name : str = "",
    debug      : bool = False,
) -> torch.Tensor:
    """
    通用 per-block 量化/反量化框架，与 nvfp4_quantize_op_ref 实现思路一致：
    - decode_scale = round-to-nearest（e8m0 或 e4m3），不做 ceil
    - encode_scale 从量化后的 decode_scale 反算，保证 encode/decode 严格互逆
    - 量化后用 clamp 兜底，防止因 round 偏小导致的溢出
    """
    orig_shape = x.shape
    x32 = x.float().reshape(-1)
    N   = x32.numel()

    # 补齐
    pad = (block_size - N % block_size) % block_size
    if pad > 0:
        x32 = torch.cat([x32, torch.zeros(pad)])
    x_blocks = x32.reshape(-1, block_size)           # [B, block_size]

    # ── step1: 计算 raw decode_scale ─────────────────────────
    block_max  = x_blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-38)
    decode_scale_raw = block_max / fp_max            # = vec_max / fp_max

    # ── step2: round-to-nearest 量化 decode_scale ────────────
    if scale_type == "e8m0":
        decode_scale_q = cast_to_e8m0(decode_scale_raw)
    elif scale_type == "e4m3":
        decode_scale_q = cast_to_e4m3(decode_scale_raw)
    else:
        raise ValueError(f"未知 scale_type: {scale_type}")

    # ── step3: 从量化后的 decode_scale 反算 encode_scale ─────
    # encode_scale = 1 / decode_scale_q
    encode_scale = torch.reciprocal(decode_scale_q.clamp(min=1e-38))

    # ── step4: 缩放 + clamp 兜底 ──────────────────────────────
    x_scaled = x_blocks * encode_scale              # 缩放到 [-fp_max, fp_max] 附近
    x_scaled = torch.clamp(x_scaled, -fp_max, fp_max)  # clamp 兜底（round 偏小时保护）

    if debug and debug_name:
        print(f"  [{debug_name}] fp_max={fp_max:.2f}, scale_type={scale_type}")
        print(f"    x_scaled 范围: [{x_scaled.min():.4f}, {x_scaled.max():.4f}]")
        print(f"    decode_scale_q 范围: [{decode_scale_q.min():.6f}, {decode_scale_q.max():.6f}]")

    # ── step5: 量化数据 ───────────────────────────────────────
    x_q = quantize_fn(x_scaled.reshape(-1)).reshape(-1, block_size)

    # ── step6: 反量化（乘以 decode_scale_q）─────────────────
    x_dq = (x_q * decode_scale_q).reshape(-1)[:N].reshape(orig_shape)
    return x_dq.bfloat16()


# ============================================================
# 各格式量化函数
# ============================================================

def _quantize_e5m1(x_scaled: torch.Tensor) -> torch.Tensor:
    sign, e, m, bias, _, mm = get_fp_components(x_scaled, e_bits=5, m_bits=1)
    return reconstruct_fp(sign, e, m, bias, mm)

def _quantize_e4m1(x_scaled: torch.Tensor) -> torch.Tensor:
    sign, e, m, bias, _, mm = get_fp_components(x_scaled, e_bits=4, m_bits=1)
    return reconstruct_fp(sign, e, m, bias, mm)

def _quantize_e3m1(x_scaled: torch.Tensor) -> torch.Tensor:
    sign, e, m, bias, _, mm = get_fp_components(x_scaled, e_bits=3, m_bits=1)
    return reconstruct_fp(sign, e, m, bias, mm)

def _quantize_e2m1(x_scaled: torch.Tensor) -> torch.Tensor:
    sign, e, m, bias, _, mm = get_fp_components(x_scaled, e_bits=2, m_bits=1)
    return reconstruct_fp(sign, e, m, bias, mm)


def fp_format_max(e_bits: int, m_bits: int) -> float:
    bias    = (1 << (e_bits - 1)) - 1
    max_exp = (1 << e_bits) - 1
    m_max   = (1 << m_bits)
    return (2.0 ** (max_exp - bias)) * (1.0 + (m_max - 1) / m_max)


# ============================================================
# 对外接口
# ============================================================

def quant_dequant_e5m1_e4m3scale(x, block_size=16, debug=False):
    return quant_dequant_block(x, block_size, fp_max=fp_format_max(5, 1),
        quantize_fn=_quantize_e5m1, scale_type="e4m3",
        debug_name="E5M1 + E4M3 scale", debug=debug)

def quant_dequant_e4m1_e4m3scale(x, block_size=16, debug=False):
    return quant_dequant_block(x, block_size, fp_max=fp_format_max(4, 1),
        quantize_fn=_quantize_e4m1, scale_type="e4m3",
        debug_name="E4M1 + E4M3 scale", debug=debug)

def quant_dequant_e3m1_e4m3scale(x, block_size=16, debug=False):
    return quant_dequant_block(x, block_size, fp_max=fp_format_max(3, 1),
        quantize_fn=_quantize_e3m1, scale_type="e4m3",
        debug_name="E3M1 + E4M3 scale", debug=debug)

def quant_dequant_e2m1_e4m3scale(x, block_size=16, debug=False):
    return quant_dequant_block(x, block_size, fp_max=fp_format_max(2, 1),
        quantize_fn=_quantize_e2m1, scale_type="e4m3",
        debug_name="E2M1 + E4M3 scale", debug=debug)

def quant_dequant_e5m1_e8m0scale(x, block_size=16, debug=False):
    return quant_dequant_block(x, block_size, fp_max=fp_format_max(5, 1),
        quantize_fn=_quantize_e5m1, scale_type="e8m0",
        debug_name="E5M1 + E8M0 scale", debug=debug)

def quant_dequant_e4m1_e8m0scale(x, block_size=16, debug=False):
    return quant_dequant_block(x, block_size, fp_max=fp_format_max(4, 1),
        quantize_fn=_quantize_e4m1, scale_type="e8m0",
        debug_name="E4M1 + E8M0 scale", debug=debug)

def quant_dequant_e3m1_e8m0scale(x, block_size=16, debug=False):
    return quant_dequant_block(x, block_size, fp_max=fp_format_max(3, 1),
        quantize_fn=_quantize_e3m1, scale_type="e8m0",
        debug_name="E3M1 + E8M0 scale", debug=debug)

def quant_dequant_e2m1_e8m0scale(x, block_size=16, debug=False):
    return quant_dequant_block(x, block_size, fp_max=fp_format_max(2, 1),
        quantize_fn=_quantize_e2m1, scale_type="e8m0",
        debug_name="E2M1 + E8M0 scale", debug=debug)


# ============================================================
# 误差统计 & 主测试
# ============================================================

def compute_metrics(x_orig, x_dq, name):
    x32    = x_orig.float()
    dq32   = x_dq.float()
    ae     = (x32 - dq32).abs()
    re     = ae / (x32.abs() + 1e-8)
    snr    = 10 * torch.log10(x32.pow(2).mean() / (ae.pow(2).mean() + 1e-20))
    rel_l2 = torch.norm(x32 - dq32) / (torch.norm(x32) + 1e-20)

    print(f"\n{'=' * 58}")
    print(f"  {name}")
    print(f"{'=' * 58}")
    print(f"  原始范围     : [{x32.min():.4f}, {x32.max():.4f}]")
    print(f"  量化后范围   : [{dq32.min():.4f}, {dq32.max():.4f}]")
    print(f"  MAE          : {ae.mean():.6f}")
    print(f"  Max Abs Err  : {ae.max():.6f}")
    print(f"  Mean Rel Err : {re.mean():.4%}")
    print(f"  SNR          : {snr:.2f} dB")
    print(f"  Rel_L2       : {rel_l2:.6f}")
    return snr.item(), ae.mean().item(), rel_l2.item()


def run_all_tests():
    torch.manual_seed(42)
    m, n = 32, 32
    N = m * n
    x = torch.cat([
        torch.randn(N // 2) * 0.5,
        torch.rand(N // 4)  * 4 - 2,
        torch.randn(N // 4) * 5,
    ]).reshape(m, n).bfloat16()

    print(f"输入 Tensor: shape={x.shape}, dtype={x.dtype}")
    print(f"值域: [{x.float().min():.4f}, {x.float().max():.4f}]")
    print(f"均值: {x.float().mean():.4f}, 标准差: {x.float().std():.4f}")

    configs = [
        ("E5M1         | E4M3 scale", lambda: quant_dequant_e5m1_e4m3scale(x, debug=True)),
        ("E4M1         | E4M3 scale", lambda: quant_dequant_e4m1_e4m3scale(x, debug=True)),
        ("E3M1         | E4M3 scale", lambda: quant_dequant_e3m1_e4m3scale(x, debug=True)),
        ("E2M1         | E4M3 scale", lambda: quant_dequant_e2m1_e4m3scale(x, debug=True)),
        ("E5M1         | E8M0 scale", lambda: quant_dequant_e5m1_e8m0scale(x, debug=True)),
        ("E4M1         | E8M0 scale", lambda: quant_dequant_e4m1_e8m0scale(x, debug=True)),
        ("E3M1         | E8M0 scale", lambda: quant_dequant_e3m1_e8m0scale(x, debug=True)),
        ("E2M1         | E8M0 scale", lambda: quant_dequant_e2m1_e8m0scale(x, debug=True)),
    ]

    results = {}
    for name, fn in configs:
        xq = fn()
        snr, mae, rel_l2 = compute_metrics(x, xq, name)
        results[name] = (snr, mae, rel_l2)

    print(f"\n{'=' * 65}")
    print(f"  {'格式 + Scale':<38} {'MAE':>9} {'SNR(dB)':>9} {'Rel_L2':>10}")
    print(f"  {'-' * 70}")
    for name, (snr, mae, rel_l2) in results.items():
        print(f"  {name:<38} {mae:>9.6f} {snr:>9.2f} {rel_l2:>10.6f}")
    print()


if __name__ == "__main__":
    run_all_tests()
