"""
E4M1 FP6 quantization operator (simulation).

E4M1 format: 1 (sign) + 4 (exponent) + 1 (mantissa) = 6 bits
  - Exponent bias = 2^(4-1) - 1 = 7
  - Normal range: 2^(1-7) x 1.0 ~ 2^(14-7) x 1.5 = 0.015625 ~ 192.0
  - Subnormal (e=0000): (-1)^s x 2^(1-7) x (m/2) = +/-0.0 or +/-0.0078125
  - inf/nan (e=1111): not used

This file simulates quantize/dequantize precision without actual bit packing.
Quantized results are stored as float32 tensors (E4M1 representable values), not packed uint8.

Reference: miniTransformer/quantiztion/nvfp4_quantize_op_ref.py for blockwise quantization structure.
"""

import torch
from typing import Optional, Tuple

from miniTransformer.quantiztion import quantization
from miniTransformer.quantiztion.quantization import QuantizeResult, GroupQuantizeResult


E4M1_EXPONENT_BITS = 4
E4M1_MANTISSA_BITS = 1
E4M1_BIAS          = (1 << (E4M1_EXPONENT_BITS - 1)) - 1   # = 7
E4M1_MAX_EXP       = (1 << E4M1_EXPONENT_BITS) - 2         # = 14 (1111 reserved for inf/nan)
E4M1_MIN_NORMAL_EXP = 1
E4M1_MAX_VALUE      = (2 ** (E4M1_MAX_EXP - E4M1_BIAS)) * (1.0 + 0.5)   # = 192.0
E4M1_MIN_NORMAL     = 2 ** (E4M1_MIN_NORMAL_EXP - E4M1_BIAS)             # = 2^(-6) ~ 0.015625
E4M1_MIN_SUBNORMAL  = E4M1_MIN_NORMAL * 0.5                               # = 2^(-7) ~ 0.0078125

DEFAULT_TILE_LEN_X = 16
DEFAULT_TILE_LEN_Y = 1


def _build_e4m1_lookup() -> torch.Tensor:
    """
    Build a lookup table of all E4M1 representable values (positive and negative, excluding inf/nan).

    E4M1 has 64 codes (6 bits):
      - 0b000000 ~ 0b011111: non-negative values
      - 0b100000 ~ 0b111111: negative values
      - e=1111: inf/nan; mapped to +/-E4M1_MAX_VALUE

    Returns a float32 tensor of shape [64] indexed by 6-bit code.
    """
    values = []
    for code in range(64):
        sign = -1.0 if (code >> 5) & 1 else 1.0
        exp  = (code >> 1) & 0xF   # 4-bit exponent
        mant = code & 0x1           # 1-bit mantissa

        if exp == 0b1111:
            # inf/nan -> clamp to max representable value
            values.append(sign * E4M1_MAX_VALUE)
        elif exp == 0:
            # subnormal: (-1)^s x 2^(1-bias) x (m/2)
            values.append(sign * (2 ** (1 - E4M1_BIAS)) * (mant * 0.5))
        else:
            # normal: (-1)^s x 2^(e-bias) x (1 + m/2)
            values.append(sign * (2 ** (exp - E4M1_BIAS)) * (1.0 + mant * 0.5))

    return torch.tensor(values, dtype=torch.float32)


_E4M1_LOOKUP: Optional[torch.Tensor] = None


def _get_e4m1_lookup(device: torch.device) -> torch.Tensor:
    global _E4M1_LOOKUP
    if _E4M1_LOOKUP is None:
        _E4M1_LOOKUP = _build_e4m1_lookup()
    return _E4M1_LOOKUP.to(device)


def quantize_to_e4m1(x: torch.Tensor) -> torch.Tensor:
    """
    Quantize each element to the nearest E4M1 representable value using a lookup table.

    Args:
        x: Float32 tensor of arbitrary shape.

    Returns:
        Float32 tensor of the same shape; each value is a valid E4M1 value.
    """
    lookup = _get_e4m1_lookup(x.device)              # [64]
    x_flat = x.reshape(-1, 1)                         # [N, 1]
    diff   = (x_flat - lookup.unsqueeze(0)).abs()     # [N, 64]
    idx    = diff.argmin(dim=-1)                      # [N]
    return lookup[idx].reshape(x.shape)


def dequantize_from_e4m1(qx: torch.Tensor) -> torch.Tensor:
    """
    E4M1 quantized values are already in float32 representation, so dequantize is identity.
    Provided for symmetry with nvfp4's packed uint8 dequantize.
    """
    return qx


class E4M1QuantizeOp(quantization.QuantizeOpBase):
    """
    E4M1 FP6 quantization operator (simulation).

    Designed to match Nvfp4TiledQuantizeRefOp:
      - Blockwise scale: one scale per tile_len_x elements
      - Supports per-tensor and per-token global_amax strategies
      - No bit packing; quantized results stored as float32

    Quantization steps:
      1. Compute global_amax (per-tensor or per-token)
      2. Compute per-tile amax (vec_max)
      3. encode_scale = 1 / (vec_max / E4M1_MAX * global_encode_scale)
      4. Map x * encode_scale to the nearest E4M1 value
      5. decode_scale = vec_max / E4M1_MAX * global_encode_scale (stored as float32)

    Dequantization: qx * decode_scale (restores original magnitude per tile)
    """

    def __init__(self, per_token_amax: bool = False):
        """
        Args:
            per_token_amax: True  -> per-token global_amax (one per row, shape [M, 1])
                            False -> per-tensor global_amax (one global value, shape [1])
        """
        super().__init__()
        self.per_token_amax = per_token_amax

    @property
    def supports_allgather(self) -> bool:
        return False

    def _compute_global_amax(self, x: torch.Tensor) -> torch.Tensor:
        """Return global amax in per-tensor ([1]) or per-token ([M, 1]) mode."""
        if self.per_token_amax:
            return x.float().abs().amax(dim=-1, keepdim=True)    # [M, 1]
        else:
            return x.float().abs().max().to(torch.float32).view(1)  # [1]

    def _quantize_blockwise(
        self,
        x: torch.Tensor,
        global_amax: torch.Tensor,
        tile_len_x: int = DEFAULT_TILE_LEN_X,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Blockwise E4M1 quantization.

        Args:
            x          : [M, N], float32
            global_amax: [1] or [M, 1], float32
            tile_len_x : number of columns per tile

        Returns:
            qx         : [M, N], float32 (E4M1 representable values)
            sx         : [M, N//tile_len_x], float32 (decode scale, one per tile)
            global_amax: returned as-is for use in dequantize
        """
        assert x.ndim == 2, "only 2D tensors supported"
        x = x.float()
        M, N = x.shape

        pad_cols = (tile_len_x - N % tile_len_x) % tile_len_x
        if pad_cols > 0:
            x_padded = torch.nn.functional.pad(x, (0, pad_cols))
        else:
            x_padded = x
        _, N_pad = x_padded.shape

        x_tiled = x_padded.view(M, N_pad // tile_len_x, tile_len_x)
        vec_max  = x_tiled.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)  # [M, T, 1]

        # Broadcast global_amax to [M, 1, 1] or [1, 1, 1] for correct tile-wise scaling
        if global_amax.dim() == 2:
            global_amax_bc = global_amax.unsqueeze(1)    # [M, 1, 1]
        else:
            global_amax_bc = global_amax.view(1, 1, 1)  # [1, 1, 1]

        global_encode_scale = (E4M1_MAX_VALUE / global_amax_bc).clamp(
            min=0.0, max=torch.finfo(torch.float32).max
        )

        # decode_scale stores the per-tile magnitude; encode_scale inverts it to map to [-E4M1_MAX, E4M1_MAX]
        decode_scale = (vec_max / E4M1_MAX_VALUE) * global_encode_scale   # [M, T, 1]
        decode_scale = decode_scale.clamp(max=torch.finfo(torch.float32).max)

        encode_scale = (1.0 / decode_scale.clamp(min=1e-12))             # [M, T, 1]

        scaled_x  = (x_tiled * encode_scale).clamp(-E4M1_MAX_VALUE, E4M1_MAX_VALUE)
        scaled_flat = scaled_x.reshape(-1)
        qx_flat     = quantize_to_e4m1(scaled_flat)
        qx_padded   = qx_flat.reshape(M, N_pad)

        qx = qx_padded[:, :N].contiguous()
        sx = decode_scale.squeeze(-1)[:, :N // tile_len_x].contiguous()   # [M, T]

        return qx, sx, global_amax

    def _dequantize_blockwise(
        self,
        qx: torch.Tensor,
        sx: torch.Tensor,
        global_amax: torch.Tensor,
        tile_len_x: int = DEFAULT_TILE_LEN_X,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Blockwise E4M1 dequantization (qx * decode_scale).

        Args:
            qx         : [M, N], float32 (E4M1 representable values)
            sx         : [M, N//tile_len_x], float32 (decode scale)
            global_amax: [1] or [M, 1], float32 (unused in current implementation; reserved)
            tile_len_x : number of columns per tile
            out_dtype  : output dtype

        Returns:
            [M, N], out_dtype
        """
        M, N = qx.shape
        qx_f  = qx.float()

        x_tiled   = qx_f.view(M, N // tile_len_x, tile_len_x)        # [M, T, tile]
        sx_expand = sx.float().unsqueeze(-1)                          # [M, T, 1]
        x_dequant = (x_tiled * sx_expand).view(M, N)                 # [M, N]

        return x_dequant.to(out_dtype)

    def quantize(
        self,
        x: torch.Tensor,
        qparams: Optional[quantization.QParams] = None,
        return_transpose: bool = False,
        reduce_amax: bool = False,
        tp_group=None,
        input_meta=None,
    ) -> QuantizeResult:
        """
        Apply blockwise E4M1 quantization to x.

        Args:
            x      : [M, K], bf16/fp16/fp32
            qparams: unused; retained for interface compatibility

        Returns:
            QuantizeResult:
              data         = qx  [M, K]  float32 (E4M1 values)
              scale        = sx  [M, K//tile] float32 (decode scale)
              tensor_scale = global_amax [1] or [M, 1]
        """
        x_2d = x.float().view(-1, x.shape[-1])
        tile  = DEFAULT_TILE_LEN_X

        global_amax = self._compute_global_amax(x_2d)
        qx, sx, ga  = self._quantize_blockwise(x_2d, global_amax, tile)

        return QuantizeResult(
            data=qx,
            scale=sx,
            data_t=None,
            scale_t=None,
            tensor_scale=ga,
            tensor_scale_t=None,
        )

    def dequantize(
        self,
        qx: torch.Tensor,
        sx: torch.Tensor,
        global_amax: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
        qparams=None,
    ) -> torch.Tensor:
        """
        Dequantize E4M1 quantized results back to high precision.

        Args:
            qx         : [M, K], float32 (E4M1 representable values)
            sx         : [M, K//tile] float32 (decode scale)
            global_amax: reserved; not used in current implementation
            out_dtype  : output dtype

        Returns:
            [M, K], out_dtype
        """
        return self._dequantize_blockwise(
            qx, sx, global_amax,
            tile_len_x=DEFAULT_TILE_LEN_X,
            out_dtype=out_dtype,
        )

    def quant_dequant(
        self,
        x: torch.Tensor,
        out_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Convenience method: quantize then immediately dequantize to simulate E4M1 precision loss.

        Args:
            x       : arbitrary shape, bf16/fp16/fp32
            out_dtype: output dtype; defaults to input dtype

        Returns:
            Tensor of the same shape as x, after E4M1 round-trip.
        """
        if out_dtype is None:
            out_dtype = x.dtype
        orig_shape = x.shape
        x_2d = x.float().reshape(-1, x.shape[-1])

        result = self.quantize(x_2d)
        x_dq = self.dequantize(
            result.data, result.scale, result.tensor_scale, out_dtype=out_dtype
        )
        return x_dq.reshape(orig_shape)

    def qgemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        m_params,
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        grad: bool = False,
        accumulate: bool = False,
        layout: str = "TN",
        global_amax_qx: Optional[torch.Tensor] = None,
        global_amax_qw: Optional[torch.Tensor] = None,
        qparams_x=None,
        qparams_w=None,
        **kwargs,
    ) -> torch.Tensor:
        """E4M1 GEMM: dequantize x and w, then run a standard bf16 matmul."""
        x_dq = self._dequantize_blockwise(qx, sx, global_amax_qx, out_dtype=torch.float32)
        w_dq = self._dequantize_blockwise(qw, sw, global_amax_qw, out_dtype=torch.float32)

        y = x_dq @ w_dq.T
        if bias is not None:
            y = y + bias.float()

        return y.to(out_dtype)

    def group_quantize(self, *args, **kwargs):
        raise NotImplementedError("E4M1QuantizeOp does not support group_quantize")

    def grouped_gemm(self, *args, **kwargs):
        raise NotImplementedError("E4M1QuantizeOp does not support grouped_gemm")
