import torch
import triton
import triton.language as tl
import numpy as np

# ============================================================================
# Golden实现
# ============================================================================

_FP8_E4M3_MAX = 448.0
_FP4_E2M1_MAX = 6.0
_MX_E2M1_POS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def mx_e2m1_quantize(x: torch.Tensor) -> torch.Tensor:
    pos_vals = _MX_E2M1_POS.to(x.device)
    x32 = x.float()
    sign = torch.sign(x32)
    sign = torch.where(x32 == 0, torch.ones_like(sign), sign)
    diff = (x32.abs().unsqueeze(-1) - pos_vals.unsqueeze(0)).abs()
    idx = diff.argmin(dim=-1)
    return sign * pos_vals[idx]


def quant_nvfp4_pertoken_golden(x, block_size=16):
    M, K = x.shape
    if x.numel() == 0:
        return x, None, None

    x32 = x.float()
    B = K // block_size

    token_amax = x32.abs().amax(dim=1, keepdim=True).clamp(min=1e-38)
    global_scale = token_amax / (_FP8_E4M3_MAX * _FP4_E2M1_MAX)

    x_norm = x32 / global_scale
    x_blocks = x_norm.reshape(M * B, block_size)
    block_amax = x_blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-38)
    raw_bscale = block_amax / _FP4_E2M1_MAX
    bscale_q = torch.clamp(raw_bscale, 1e-38, _FP8_E4M3_MAX)
    encode_scale = torch.reciprocal(bscale_q.clamp(min=1e-38))

    x_scaled = x_blocks * encode_scale
    x_scaled = torch.clamp(x_scaled, -_FP4_E2M1_MAX, _FP4_E2M1_MAX)
    x_q = mx_e2m1_quantize(x_scaled.reshape(-1)).reshape(M * B, block_size)

    return x_q.reshape(M, K), global_scale.squeeze(-1), bscale_q.squeeze(-1)


# ============================================================================
# Triton实现
# ============================================================================

# constexpr 常量
FP8_E4M3_MAX = tl.constexpr(448.0)
FP4_E2M1_MAX_VAL = tl.constexpr(6.0)  # 改名避免与后续变量冲突
EPS = tl.constexpr(1e-38)


@triton.jit
def _rtn_e2m1_triton(x):
    """
    E2M1量化，对齐golden的mx_e2m1_quantize
    量化级别:[0, 0.5, 1, 1.5, 2, 3, 4, 6]
    """
    x_abs = tl.abs(x)
    x_sign = tl.where(x >= 0, 1.0, -1.0)
    x_sign = tl.where(x == 0.0, 1.0, x_sign)

    x_fp4_abs = tl.where(
        x_abs >= 5.0, 6.0,
        tl.where(
            x_abs >= 3.5, 4.0,
            tl.where(
                x_abs >= 2.5, 3.0,
                tl.where(
                    x_abs >= 1.75, 2.0,
                    tl.where(
                        x_abs >= 1.25, 1.5,
                        tl.where(
                            x_abs >= 0.75, 1.0,
                            tl.where(
                                x_abs >= 0.25, 0.5,
                                0.0
                            )
                        )
                    )
                )
            )
        )
    )
    return x_fp4_abs * x_sign


@triton.jit
def _compute_token_amax(
        x_ptr, stride_x_m, M, K, token_offs, token_mask,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    """分块扫描K维度计算每个token的amax"""
    token_amax = tl.zeros([BLOCK_M], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        x_block = tl.load(
            x_ptr + token_offs[:, None] * stride_x_m + k_offs[None, :],
            mask=token_mask[:, None] & k_mask[None, :],
            other=0.0
        )
        block_amax = tl.max(tl.abs(x_block), axis=1)
        token_amax = tl.maximum(token_amax, block_amax)

    return token_amax


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 8, 'BLOCK_BLOCKS': 4, 'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_BLOCKS': 4, 'BLOCK_K': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_BLOCKS': 8, 'BLOCK_K': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_BLOCKS': 8, 'BLOCK_K': 512}, num_warps=16),
    ],
    key=['M', 'K'],
)
@triton.jit
def _quant_nvfp4_pertoken_kernel_golden_aligned(
        x_ptr,
        x_q_ptr,
        global_scale_ptr,
        block_scale_ptr,
        M, K,
        stride_x_m,
        stride_xq_m,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_BLOCKS: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    token_start = pid * BLOCK_M
    token_offs = token_start + tl.arange(0, BLOCK_M)
    token_mask = token_offs < M

    B = K // BLOCK_SIZE

    # Step 1: Per-Token Global Scale
    token_amax = _compute_token_amax(
        x_ptr, stride_x_m, M, K, token_offs, token_mask,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )

    token_amax = tl.maximum(token_amax, EPS)
    global_scale = token_amax / (FP8_E4M3_MAX * FP4_E2M1_MAX_VAL)
    tl.store(global_scale_ptr + token_offs, global_scale, mask=token_mask)

    # Step 2-4: 逐block处理
    for b_start in range(0, B, BLOCK_BLOCKS):
        b_end = tl.minimum(b_start + BLOCK_BLOCKS, B)

        for b in range(b_start, b_end):
            k_block_start = b * BLOCK_SIZE
            k_offs = k_block_start + tl.arange(0, BLOCK_SIZE)

            x_block = tl.load(
                x_ptr + token_offs[:, None] * stride_x_m + k_offs[None, :],
                mask=token_mask[:, None],
                other=0.0
            )

            x_norm = x_block / global_scale[:, None]

            block_amax = tl.max(tl.abs(x_norm), axis=1)
            block_amax = tl.maximum(block_amax, EPS)

            raw_bscale = block_amax / FP4_E2M1_MAX_VAL

            # 用 tl.minimum + tl.maximum 替代 tl.clamp
            bscale_q = tl.maximum(raw_bscale, EPS)
            bscale_q = tl.minimum(bscale_q, FP8_E4M3_MAX)
            bscale_q = tl.where(bscale_q == 0, 1.0, bscale_q)

            tl.store(block_scale_ptr + token_offs * B + b, bscale_q, mask=token_mask)

            # Scale
            encode_scale = 1.0 / tl.maximum(bscale_q, EPS)
            x_scaled = x_norm * encode_scale[:, None]

            # Clamp: 用 minimum + maximum 替代
            x_scaled = tl.maximum(x_scaled, -FP4_E2M1_MAX_VAL)
            x_scaled = tl.minimum(x_scaled, FP4_E2M1_MAX_VAL)

            # E2M1量化
            x_q = _rtn_e2m1_triton(x_scaled)

            tl.store(
                x_q_ptr + token_offs[:, None] * stride_xq_m + k_offs[None, :],
                x_q,
                mask=token_mask[:, None]
            )


def quant_nvfp4_pertoken_triton(x, block_size=16):
    """Triton实现"""
    M, K = x.shape

    if x.numel() == 0:
        B = K // block_size
        return (
            torch.empty_like(x),
            torch.empty(M, dtype=torch.float32, device=x.device),
            torch.empty(M * B, dtype=torch.float32, device=x.device),
        )

    assert K % block_size == 0, f"K={K} 必须整除 block_size={block_size}"

    B = K // block_size

    x_q = torch.empty_like(x)
    global_scale = torch.empty(M, dtype=torch.float32, device=x.device)
    block_scale = torch.empty(M * B, dtype=torch.float32, device=x.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)

    _quant_nvfp4_pertoken_kernel_golden_aligned[grid](
        x, x_q, global_scale, block_scale,
        M, K,
        x.stride(0), x_q.stride(0),
        BLOCK_SIZE=block_size,
    )

    return x_q, global_scale, block_scale


# ============================================================================
# dequant实现
# ============================================================================

@triton.jit
def _dequant_nvfp4_pertoken_kernel(
    x_q_ptr,  # [M, K] 量化后的值（E2M1浮点值）
    global_scale_ptr,  # [M] per-token global scale
    block_scale_ptr,  # [M * B] per-block scale
    output_ptr,  # [M, K] 反量化输出
    M, K,
    block_size,
    stride_xq_m,
    stride_out_m,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_BLOCKS: tl.constexpr,
):
    """
    反量化 kernel

    反量化公式: output = x_q * block_scale * global_scale

    x_q:        [M, K]     量化值: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
    block_scale: [M * B]   每个block的缩放因子
    global_scale: [M]      每行的全局缩放因子

    输出: output [M, K] 反量化回原始精度
    """
    pid = tl.program_id(0)

    token_start = pid * BLOCK_M
    token_offs = token_start + tl.arange(0, BLOCK_M)
    token_mask = token_offs < M

    B = K // BLOCK_SIZE

    # 加载 global_scale [BLOCK_M]
    gs = tl.load(global_scale_ptr + token_offs, mask=token_mask, other=1.0)

    # 逐block反量化
    for b_start in range(0, B, BLOCK_BLOCKS):
        b_end = tl.minimum(b_start + BLOCK_BLOCKS, B)

        for b in range(b_start, b_end):
            k_block_start = b * BLOCK_SIZE
            k_offs = k_block_start + tl.arange(0, BLOCK_SIZE)

            # 加载量化值 [BLOCK_M, BLOCK_SIZE]
            x_q = tl.load(
                x_q_ptr + token_offs[:, None] * stride_xq_m + k_offs[None, :],
                mask=token_mask[:, None],
                other=0.0
            )

            # 加载 block_scale [BLOCK_M]
            bs = tl.load(
                block_scale_ptr + token_offs * B + b,
                mask=token_mask,
                other=1.0
            )

            # 反量化: output = x_q * block_scale * global_scale
            x_dq = x_q * bs[:, None] * gs[:, None]

            # 存储结果
            tl.store(
                output_ptr + token_offs[:, None] * stride_out_m + k_offs[None, :],
                x_dq,
                mask=token_mask[:, None]
            )


def dequant_nvfp4_pertoken(x_q, global_scale, block_scale, block_size=16,
                           BLOCK_M=None, BLOCK_BLOCKS=None, num_warps=None):
    """
    反量化函数

    Args:
        x_q: [M, K] 量化后的值（E2M1浮点值）
        global_scale: [M] per-token global scale
        block_scale: [M * B] per-block scale
        block_size: block大小

    Returns:
        output: [M, K] 反量化结果，dtype与x_q相同
    """
    M, K = x_q.shape

    if x_q.numel() == 0:
        return x_q

    B = K // block_size

    # 自动选择参数
    if BLOCK_M is None:
        if M <= 32:
            BLOCK_M = 8
            BLOCK_BLOCKS = 4
            num_warps = 4
        elif M <= 256:
            BLOCK_M = 16
            BLOCK_BLOCKS = 4
            num_warps = 8
        else:
            BLOCK_M = 32
            BLOCK_BLOCKS = 8
            num_warps = 16

    output = torch.empty_like(x_q)

    grid = (triton.cdiv(M, BLOCK_M),)

    _dequant_nvfp4_pertoken_kernel[grid](
        x_q, global_scale, block_scale, output,
        M, K, block_size,
        x_q.stride(0), output.stride(0),
        BLOCK_SIZE=block_size,
        BLOCK_M=BLOCK_M,
        BLOCK_BLOCKS=BLOCK_BLOCKS,
        num_warps=num_warps,
    )

    return output
