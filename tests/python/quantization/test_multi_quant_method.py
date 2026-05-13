"""
mxfp8_e4m1_e2m3_e2m1_nvfp4.py
==============================
作者: mingfa
日期: 2026-04-29

功能:
    对比 MXFP8 / NVFP4 及若干自定义低精度格式的量化精度，
    包含 global amax 和 per-token amax 两种 scale 策略。
    支持用真实推理 activation 批量评测。

量化方案:
    MXFP8  E4M3 | E8M0 scale      — TransformerEngine 官方实现（基准）
    E4M1        | E4M3 scale      — 4bit指数+1bit尾数，E4M3 block scale
    E2M3        | E4M3 scale      — 2bit指数+3bit尾数，E4M3 block scale
    MXFP4  E2M1 | E4M3 scale      — MX规格 FP4，E4M3 block scale（无global）
    NVFP4  E2M1 | global scale    — NVIDIA规格：global amax + E4M3 block scale
    NVFP4  E2M1 | pertoken scale  — per-token global amax + E4M3 block scale
    NVFP4  E2M1 | TE              — TransformerEngine NVFP4 官方实现

block scale 设计（参考 nvfp4_quantize_op_ref._quantize_blockwise_reference）:
    1. decode_scale = round-to-nearest(.to(float8_e4m3fn))，不做 ceil
    2. encode_scale = 1 / decode_scale，从量化后值反算，保证 encode/decode 互逆
    3. x_scaled = x_norm * encode_scale，配合 clamp 兜底防溢出

注意:
    MXFP8 TE / NVFP4 TE 要求输入行数能被 32/16 整除，
    实际 activation shape=(146, 2560) 时会 skip（146 不整除）
"""
import torch

# CUDA Graph compatible: lazy pos_vals cache per device
_POS_VALS_CACHE: dict = {}

def _get_cached_pos_vals(name: str, vals_cpu: torch.Tensor, device) -> torch.Tensor:
    key = f"{name}:{device}"
    if key not in _POS_VALS_CACHE:
        _POS_VALS_CACHE[key] = vals_cpu.to(device)
    return _POS_VALS_CACHE[key]
from transformer_engine.pytorch import NVFP4Quantizer, MXFP8Quantizer


# ============================================================
# 核心：浮点分解 + 重建
# ============================================================

def get_fp_components(x: torch.Tensor, e_bits: int, m_bits: int):
    bias = (1 << (e_bits - 1)) - 1
    max_exp = (1 << e_bits) - 1
    m_max = (1 << m_bits)

    fp_max = (2.0 ** (max_exp - 1 - bias)) * (1.0 + (m_max - 1) / m_max)

    x32 = x.float()
    sign = torch.sign(x32)
    sign = torch.where(x32 == 0, torch.ones_like(sign), sign)
    abs_x = x32.abs().clamp(max=fp_max)

    is_zero = (abs_x == 0)

    safe_abs = abs_x.clamp(min=2.0 ** (1 - bias - m_bits))
    e_real_f = torch.floor(torch.log2(safe_abs))
    e_stored_f = (e_real_f + bias).clamp(1, max_exp - 1)
    e_real_used = e_stored_f - bias

    scale = torch.pow(2.0, e_real_used)
    frac = (abs_x / scale - 1.0).clamp(0.0, 1.0)
    m_int_f = torch.round(frac * m_max)

    # 进位处理
    carry = (m_int_f >= m_max)
    e_stored_f = torch.where(carry, (e_stored_f + 1).clamp(1, max_exp - 1), e_stored_f)
    m_int_f = torch.where(carry, torch.zeros_like(m_int_f), m_int_f)

    # 非规格化
    denorm_threshold = 2.0 ** (1 - bias)
    is_denorm = (abs_x < denorm_threshold) & (~is_zero)
    m_int_denorm = torch.round(abs_x / denorm_threshold * m_max).clamp(0, m_max - 1).long()

    e_final = torch.where(is_denorm | is_zero,
                          torch.zeros_like(e_stored_f.long()),
                          e_stored_f.long())
    m_final = torch.where(is_denorm, m_int_denorm,
                          torch.where(is_zero, torch.zeros_like(m_int_f.long()),
                                      m_int_f.long()))
    return sign, e_final, m_final, bias, max_exp, m_max


def reconstruct_fp(sign, e_stored, m_int, bias, m_max):
    e_f = e_stored.float()
    m_f = m_int.float()

    is_zero = (e_stored == 0) & (m_int == 0)
    is_denorm = (e_stored == 0) & (m_int != 0)

    val_norm = torch.pow(2.0, e_f - bias) * (1.0 + m_f / m_max)
    val_denorm = (2.0 ** (1.0 - bias)) * (m_f / m_max)

    val = torch.where(~is_zero & ~is_denorm, val_norm,
                      torch.where(is_denorm, val_denorm,
                                  torch.zeros_like(val_norm)))
    return sign * val


# ============================================================
# E4M3 scale 量化工具（正数 only，用于存储 block scale）
# ============================================================

def quantize_scale_to_e4m3(scale: torch.Tensor) -> torch.Tensor:
    """
    将正数 scale 量化到 E4M3 精度后返回 float32 值
    E4M3: bias=7, max=240.0
    scale 全为正数，直接用 sign=+1
    """
    s_flat = scale.reshape(-1).abs().clamp(min=1e-38)
    sign, e_stored, m_int, bias, max_exp, m_max = get_fp_components(
        s_flat, e_bits=4, m_bits=3
    )
    # 强制正数
    q = reconstruct_fp(torch.ones_like(sign), e_stored, m_int, bias, m_max)
    return q.reshape(scale.shape)


# ============================================================
# MX E2M1 查表量化（用于 FP4 格式）
# ============================================================

_MX_E2M1_POS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)

def mx_e2m1_quantize(x: torch.Tensor) -> torch.Tensor:
    pos_vals = _get_cached_pos_vals("MX_E2M1", _MX_E2M1_POS, x.device)
    x32  = x.float()
    sign = torch.sign(x32)
    sign = torch.where(x32 == 0, torch.ones_like(sign), sign)
    diff = (x32.abs().unsqueeze(-1) - pos_vals.unsqueeze(0)).abs()
    idx  = diff.argmin(dim=-1)
    return sign * pos_vals[idx]


# ============================================================
# 通用 per-block 量化框架
# ============================================================

def quant_dequant_block(
    x: torch.Tensor,
    block_size: int,
    fp_max: float,
    quantize_fn,  # 将 x_scaled (float32) 量化并返回 float32
    scale_type: str,  # "e8m0" | "e4m3"
    debug_name: str = "",
    debug: bool = False,
) -> torch.Tensor:
    """
    通用 per-block 量化/反量化框架

    参数
    ----
    scale_type : "e8m0"  → scale = 2^ceil(log2(block_max/fp_max))
                 "e4m3"  → scale = quantize_e4m3(block_max / fp_max)
    """
    orig_shape = x.shape
    x32 = x.float().reshape(-1)
    N   = x32.numel()
    assert N % block_size == 0, (
        f"numel()={N} 必须是 block_size={block_size} 的整数倍，请在调用前 pad"
    )
    x_blocks = x32.reshape(-1, block_size)  # [B, block_size]

    # ── 计算 block scale ──────────────────────────────────────
    block_max = x_blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-38)
    raw_scale = block_max / fp_max  # 未量化的 scale

    if scale_type == "e8m0":
        log2_s  = torch.ceil(torch.log2(raw_scale)).clamp(-127, 127)
        scale_q = torch.pow(2.0, log2_s)
    elif scale_type == "e4m3":
        scale_q = raw_scale.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32)
    else:
        raise ValueError(f"未知 scale_type: {scale_type}")

    # encode_scale 从量化后 scale_q 反算，保证 encode/decode 互逆
    encode_scale = torch.reciprocal(scale_q.clamp(min=1e-38))
    x_scaled = x_blocks * encode_scale



    x_q = quantize_fn(x_scaled.reshape(-1)).reshape(-1, block_size)

    # ── 反量化 ────────────────────────────────────────────────
    x_dq = (x_q * scale_q).reshape(-1).reshape(orig_shape)
    return x_dq.bfloat16()


def _e4m3_ceil(raw_scale: torch.Tensor) -> torch.Tensor:
    """
    将正数 scale 向上取整到最近的 E4M3 可表示值。
    实现：先量化，若量化后偏小则用 "下一个" E4M3 值替代。
    下一个 E4M3 值 = 当前值 * (1 + 1/8) 再量化（或直接 +1ulp）。

    更简洁的办法：对 raw_scale 轻微放大（乘以 1+eps 使 round→ceil）
    E4M3 的最小相对步长 = 1/8 = 0.125
    只需放大 1 + 0.5/8 = 1.0625 后再做 nearest round 即可实现 ceil 效果
    """
    # 放大到 "介于当前值和下一个 E4M3 值之间的中点以上"
    # E4M3 相对精度 = 1/(2*m_max) = 1/16，放大 1+1/16 足够
    slightly_larger = raw_scale * (1.0 + 1.0 / 16.0)
    return quantize_scale_to_e4m3(slightly_larger).clamp(min=1e-38)


# ============================================================
# 各格式量化函数
# ============================================================

_POS_VALS_E4M3 = torch.tensor(
    [0, 0.001953125, 0.00390625, 0.005859375, 0.0078125, 0.01171875, 0.015625, 0.0234375, 0.03125, 0.046875, 0.0625, 0.09375, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0, 384.0, 448.0],
    dtype=torch.float32
)

def _quantize_e4m3(x: torch.Tensor) -> torch.Tensor:
    pos_vals = _get_cached_pos_vals("E4M3", _POS_VALS_E4M3, x.device)
    x32  = x.float()
    sign = torch.sign(x32)
    sign = torch.where(x32 == 0, torch.ones_like(sign), sign)
    diff = (x32.abs().unsqueeze(-1) - pos_vals.unsqueeze(0)).abs()
    idx  = diff.argmin(dim=-1)
    return sign * pos_vals[idx]


_POS_VALS_E4M1 = torch.tensor(
    [0, 0.0078125, 0.015625, 0.0234375, 0.03125, 0.046875, 0.0625, 0.09375, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384],
    dtype=torch.float32
)

def _quantize_e4m1(x: torch.Tensor) -> torch.Tensor:
    pos_vals = _get_cached_pos_vals("E4M1", _POS_VALS_E4M1, x.device)
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


def fp_format_max(e_bits: int, m_bits: int) -> float:
    bias = (1 << (e_bits - 1)) - 1
    max_exp = (1 << e_bits) - 1
    m_max = (1 << m_bits)
    return (2.0 ** (max_exp - 1 - bias)) * (1.0 + (m_max - 1) / m_max)


# ============================================================
# 对外接口
# ============================================================

def quant_dequant_mxfp8(x, block_size=32, debug=False):
    """MXFP8 E4M3，E8M0 block scale（基准）"""
    '''
    return quant_dequant_block(
        x, block_size,
        fp_max      = fp_format_max(4, 3),   # 240.0
        quantize_fn = _quantize_e4m3,
        scale_type  = "e8m0",
        debug_name  = "MXFP8 E4M3",
        debug       = debug,
    )
    '''
    import transformer_engine.pytorch.cpp_extensions as tex
    mxfp8_quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )
    x_q = mxfp8_quantizer(x)
    return x_q.dequantize()


def quant_dequant_e4m1_e4m3scale(x, block_size=16, debug=False):
    """
    E4M1 数据格式 + E4M3 block scale
    fp_max of E4M1 = 2^(14-7) * 1.5 = 192.0
    """
    return quant_dequant_block(
        x, block_size,
        fp_max=fp_format_max(4, 1),  # 192.0
        quantize_fn=_quantize_e4m1,
        scale_type="e4m3",
        debug_name="E4M1 + E4M3 scale",
        debug=debug,
    )


def quant_dequant_e2m3_e4m3scale(x, block_size=16, debug=False):
    """
    E2M3 数据格式 + E4M3 block scale
    fp_max of E2M3 = 2^(2-1) * 1.875 = 3.75
    E2M3 本身范围窄，但 E4M3 scale 覆盖了动态范围
    """
    return quant_dequant_block(
        x, block_size,
        fp_max=fp_format_max(2, 3),  # 3.75
        quantize_fn=_quantize_e2m3,
        scale_type="e4m3",
        debug_name="E2M3 + E4M3 scale",
        debug=debug,
    )


def quant_dequant_mxfp4(x, block_size=16, debug=False):
    """MXFP4 E2M1，E8M0 block scale（对照）"""
    return quant_dequant_block(
        x, block_size,
        fp_max=6.0,
        quantize_fn=mx_e2m1_quantize,
        scale_type="e4m3",
        debug_name="MXFP4 E2M1",
        debug=debug,
    )


def cast_bscale_to_e4m3(raw_scale: torch.Tensor) -> torch.Tensor:
    """
    将 block scale round-to-nearest 到 FP8 E4M3，直接用 PyTorch 原生转换。
    参考 nvfp4_quantize_op_ref._quantize_blockwise_reference 的实现：
    使用 round-to-nearest，encode_scale 从量化后的 decode_scale 反算，
    配合 clamp 兜底保证不溢出，无需 ceil。
    """
    return raw_scale.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX) \
        .to(torch.float8_e4m3fn).to(torch.float32)


_FP8_E4M3_MAX = 448.0  # E4M3 标准最大正值（OCP FP8 E4M3）
_FP4_E2M1_POS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32
)
_FP4_E2M1_MAX = 6.0


def quant_dequant_nvfp4(x, block_size=16, debug=False):
    """
    严格按照 NVIDIA 规格实现，每步打印中间量用于对比

    NVIDIA 真实流程：
      1. 计算 global_scale（fp32，不量化）
         global_scale = amax(|W|) / (fp8_e4m3_max * fp4_max)
                      = amax(|W|) / (448.0 * 6.0)
      2. W_norm = W / global_scale
      3. 对每个 block：
         block_scale = amax(|W_norm_block|) / fp4_max   (fp32)
         block_scale_q = UE4M3_ceil(block_scale)         (fp8)
      4. W_scaled = W_norm_block / block_scale_q
      5. W_q = E2M1_quantize(W_scaled)
      6. 重建：W_dq = W_q * block_scale_q * global_scale
    """
    orig_shape = x.shape
    x32 = x.float().reshape(-1)
    N   = x32.numel()
    if N == 0:
        return x
    assert N % block_size == 0, f"numel()={N} 必须是 block_size={block_size} 的整数倍"
    x_blk = x32.reshape(-1, block_size)  # [B, 16]
    # ── Step 1: global scale（fp32，不量化）───────────────
    amax = x32.abs().amax().clamp(min=1e-38)
    global_scale = amax / (_FP8_E4M3_MAX * _FP4_E2M1_MAX)
    # global_scale ∈ fp32，原始 NVIDIA 实现不对它量化

    # ── Step 2: normalize ─────────────────────────────────
    x_norm = x_blk / global_scale  # 值域应在 ±448*6 内
    # 理想情况：x_norm 的 amax ≈ 448 * 6 = 2688

    # ── Step 3: per-block decode_scale，round-to-nearest（ref 风格）──
    block_amax = x_norm.abs().amax(dim=1, keepdim=True).clamp(min=1e-38)
    raw_bscale = block_amax / _FP4_E2M1_MAX  # 应在 [0, 448] 范围内
    bscale_q = cast_bscale_to_e4m3(raw_bscale)  # round-to-nearest，不 ceil
    # encode_scale 从量化后的 bscale_q 反算，保证 encode/decode 严格互逆
    encode_scale = torch.reciprocal(bscale_q.clamp(min=1e-38))

    # ── Step 4 & 5: scale + clamp 兜底 + quantize ─────────
    x_scaled = x_norm * encode_scale  # 应在 ±6.0 附近
    x_scaled = torch.clamp(x_scaled, -_FP4_E2M1_MAX, _FP4_E2M1_MAX)  # clamp 兜底

    x_q = mx_e2m1_quantize(x_scaled.reshape(-1)).reshape(-1, block_size)
    # ── Step 6: 重建 ──────────────────────────────────────
    x_dq = (x_q * bscale_q * global_scale).reshape(-1).reshape(orig_shape)
    return x_dq.bfloat16()


def quant_nvfp4_pertoken(x, block_size=16, debug=False):
    """
    NVFP4 Per-Token Global Scale 版本

    原始：global_scale = amax(|整个W|) / (448 * 6)        → 1个标量
    改造：global_scale[i] = amax(|W[i,:]|) / (448 * 6)   → 每行1个

    shape 约定：x: [M, K]
      - global_scale : [M, 1]      每 token（行）一个
      - block_scale  : [M*B, 1]    每 block 一个，B = K // block_size
    """
    orig_shape = x.shape
    M, K = x.shape
    if x.numel() == 0:
        return x
    if x.dim() != 2:
        raise ValueError(f"期望 2D 输入 [M, K]，得到 {x.shape}")
    x32 = x.float()  # [M, K]

    assert K % block_size == 0, (
        f"K={K} 必须是 block_size={block_size} 的整数倍，请在调用前 pad"
    )
    B = K // block_size  # 每行的 block 数

    # ── Step 1: Per-Token Global Scale ────────────────────────
    # 原始：amax(全局) / (448 * 6)
    # 现在：每行 amax / (448 * 6)
    token_amax = x32.abs().amax(dim=1, keepdim=True).clamp(min=1e-38)  # [M,1]
    global_scale = token_amax / (_FP8_E4M3_MAX * _FP4_E2M1_MAX)  # [M,1]
    # global_scale 保持 fp32，不量化（与原始一致）



    # ── Step 2: Per-Token Normalize ───────────────────────────
    x_norm = x32 / global_scale  # [M, K]
    # 每行的 amax 现在都是 448 * 6 = 2688



    # ── Step 3: Per-Block decode_scale，round-to-nearest（ref 风格）──
    x_blocks = x_norm.reshape(M * B, block_size)  # [M*B, 16]
    block_amax = x_blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-38)
    raw_bscale = block_amax / _FP4_E2M1_MAX  # [M*B, 1]
    bscale_q = cast_bscale_to_e4m3(raw_bscale)  # round-to-nearest
    # encode_scale 从量化后的 bscale_q 反算，保证 encode/decode 严格互逆
    encode_scale = torch.reciprocal(bscale_q.clamp(min=1e-38))  # [M*B, 1]



    # ── Step 4: scale + clamp 兜底 + quantize ─────────────────
    x_scaled = x_blocks * encode_scale  # [M*B, 16]
    x_scaled = torch.clamp(x_scaled, -_FP4_E2M1_MAX, _FP4_E2M1_MAX)  # clamp 兜底

    x_q = mx_e2m1_quantize(x_scaled.reshape(-1)).reshape(M * B, block_size)

    # ── Step 5: Dequantize ────────────────────────────────────
    # bscale_q:    [M*B, 1]
    # global_scale:[M,   1]  → 需要扩展到 [M*B, 1]
    global_scale_expanded = global_scale.unsqueeze(1).expand(M, B, 1).reshape(M * B, 1)

    x_dq = (x_q * bscale_q * global_scale_expanded).reshape(M, K)

    return x_dq.to(x.dtype)


def quant_dequant_nvfp4_TE(x, block_size=16, debug=False):
    nvfp4_quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False
    )
    nvfp4_tensor = nvfp4_quantizer(x)
    w_1 = nvfp4_tensor.dequantize()

    return w_1


def quant_dequant_nvfp4_REF(x, block_size=16, debug=False):
    from miniTransformer.quantiztion.nvfp4_quantize_op_ref import Nvfp4TiledQuantizeRefOp
    from miniTransformer.quantiztion import quantization

    op = Nvfp4TiledQuantizeRefOp()
    x_params = quantization.QParams(
        scaling_dtype=torch.float8_e4m3fn,
        output_dtype=torch.uint8,
        eps=0.0,
        pow_2_scales=False,
        quant_tile_shape=(1, 16),
        with_2d_quantization=False,
        with_rht=False,
    )
    qx, sx, _, _, global_amax_row, _ = op._quantize(x, x_params, False, False)

    return op.dequantize(qx, sx, global_amax_row)


# ============================================================
# 误差统计 & 主测试
# ============================================================

def compute_metrics(x_orig, x_dq, name):
    x32 = x_orig.float()
    dq32 = x_dq.float()
    ae = (x32 - dq32).abs()
    re = ae / (x32.abs() + 1e-8)
    snr = 10 * torch.log10(x32.pow(2).mean() / (ae.pow(2).mean() + 1e-20))

    print(f"\n{'=' * 58}")
    print(f"  {name}")
    print(f"{'=' * 58}")
    print(f"  原始范围     : [{x32.min():.4f}, {x32.max():.4f}]")
    print(f"  量化后范围   : [{dq32.min():.4f}, {dq32.max():.4f}]")
    print(f"  MAE          : {ae.mean():.6f}")
    print(f"  Max Abs Err  : {ae.max():.6f}")
    print(f"  Mean Rel Err : {re.mean():.4%}")
    print(f"  SNR          : {snr:.2f} dB")
    return snr.item(), ae.mean().item()


def run_all_tests():
    torch.manual_seed(42)
    m, n = 32, 32  # 2D 矩阵维度，总元素数 = m * n
    N = m * n
    x = torch.cat([
        torch.randn(N // 2) * 0.5,
        torch.rand(N // 4) * 4 - 2,
        torch.randn(N // 4) * 5,
    ]).reshape(m, n).bfloat16().cuda()  # ← reshape 成 2D

    print(f"输入 Tensor: shape={x.shape}, dtype={x.dtype}")
    print(f"值域: [{x.float().min():.4f}, {x.float().max():.4f}]")
    print(f"均值: {x.float().mean():.4f}, 标准差: {x.float().std():.4f}")

    configs = [
        ("MXFP8  E4M3  | E8M0 scale (基准)",
         lambda: quant_dequant_mxfp8(x, debug=True)),
        # ── 修改后的两个格式 ──────────────────────────────────
        ("E4M1         | E4M3 scale ★",
         lambda: quant_dequant_e4m1_e4m3scale(x, debug=True)),
        ("E2M3         | E4M3 scale ★",
         lambda: quant_dequant_e2m3_e4m3scale(x, debug=True)),
        # ── 对照 ──────────────────────────────────────────────
        ("MXFP4  E2M1                | E4M3 scale",
         lambda: quant_dequant_mxfp4(x, debug=True)),
        ("NVFP4  E2M1                | UE4M3+global scale",
         lambda: quant_dequant_nvfp4(x, debug=True)),
        ("PERTOKEN AMAX NVFP4  E2M1  | UE4M3+global scale",
         lambda: quant_nvfp4_pertoken(x, debug=True)),
        ("TE NVFP4  E2M1             | UE4M3+global scale",
         lambda: quant_dequant_nvfp4_TE(x, debug=True)),
        ("REF NVFP4  E2M1            | UE4M3+global scale",
         lambda: quant_dequant_nvfp4_REF(x, debug=True)),
    ]

    results = {}
    for name, fn in configs:
        xq = fn()
        snr, mae = compute_metrics(x, xq, name)
        results[name] = (snr, mae)

    # 汇总
    print(f"\n{'=' * 65}")
    print(f"  {'格式 + Scale':<38} {'MAE':>9} {'SNR(dB)':>9}")
    print(f"  {'-' * 58}")
    for name, (snr, mae) in results.items():
        star = " ←★" if "★" in name else ""
        print(f"  {name:<38} {mae:>9.6f} {snr:>9.2f}{star}")
    print()


if __name__ == "__main__":
    run_all_tests()
