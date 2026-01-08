import math
import torch
import fp4_gemm
from typing import Optional, Tuple
from transformer.utils import utils
from transformer.ops import quantization

nvfp4_output_t = torch.uint8
nvfp4_scale_t = torch.float8_e4m3fn


def cast_to_e8(decode_scale):
    """Cast to a value that is representable in FP8 E8M0.

    The result is in FP32, not FP8 E8M0.
    """
    max_exponent = torch.tensor(127, device=decode_scale.device, dtype=torch.float32)
    exponent = torch.ceil(torch.log2(decode_scale))
    exponent = torch.clamp(exponent, min=-max_exponent, max=max_exponent)

    return torch.tensor(2.0, device=decode_scale.device, dtype=torch.float32) ** exponent


def cast_to_e4m3(decode_scale, global_amax):
    """Scale and cast to FP8 E4M3.

    decode_scale is actually the encoding scaling factor. global_amax
    can be any data tensor and not just the amax.

    TODO(etsykunov): Make less unintuitive.
    """
    decode_scale = decode_scale * global_amax
    FLOAT8_E4M3_MAX = torch.tensor(448.0, device=decode_scale.device, dtype=torch.float32)
    decode_scale = torch.clamp(decode_scale, min=-FLOAT8_E4M3_MAX, max=FLOAT8_E4M3_MAX)
    return decode_scale.to(torch.float8_e4m3fn)


def cast_to_fp4x2(x):
    """Quantize a tensor to FP4 E2M1 and store in a byte tensor"""

    result = torch.zeros_like(x, dtype=torch.uint8)
    result[(x >= 0.0) & (x <= 0.25)] = 0
    result[(x > 0.25) & (x < 0.75)] = 1
    result[(x >= 0.75) & (x <= 1.25)] = 2
    result[(x > 1.25) & (x < 1.75)] = 3
    result[(x >= 1.75) & (x <= 2.5)] = 4
    result[(x > 2.5) & (x < 3.5)] = 5
    result[(x >= 3.5) & (x <= 5.0)] = 6
    result[x > 5.0] = 7

    result[(x >= -0.25) & (x < -0.0)] = 8
    result[(x < -0.25) & (x > -0.75)] = 9
    result[(x <= -0.75) & (x >= -1.25)] = 10
    result[(x < -1.25) & (x > -1.75)] = 11
    result[(x <= -1.75) & (x >= -2.5)] = 12
    result[(x < -2.5) & (x > -3.5)] = 13
    result[(x <= -3.5) & (x >= -5.0)] = 14
    result[x < -5.0] = 15

    return result[:, ::2] + result[:, 1::2] * 16


def cast_from_fp4x2(x, dq_dtype):
    """Dequantize FP4 E2M1 tensor that has been represented in a byte tensor"""
    fp4_values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device=x.device,
        dtype=dq_dtype,
    )

    # Convert to long integers for indexing
    second_bit = torch.div(x, 16, rounding_mode="floor").to(torch.long)
    first_bit = (x - second_bit * 16).to(torch.long)

    # Use the long integers to index fp4_values
    first_bit_values = fp4_values[first_bit]
    second_bit_values = fp4_values[second_bit]

    result = torch.zeros(
        (first_bit_values.shape[0], first_bit_values.shape[1] * 2),
        device=x.device,
        dtype=dq_dtype,
    )
    result[:, ::2] = first_bit_values
    result[:, 1::2] = second_bit_values

    return result


def high_precision_gemm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
    accumulate: bool = False,
    is_a_transposed: bool = False,
    is_b_transposed: bool = False,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scale_alpha: float = 1.0,
) -> torch.Tensor:
    """GEMM implementation with unquantized data"""
    # Handle transpositions
    mat1, mat2 = a, b
    if is_a_transposed:
        mat1 = a.T
    if is_b_transposed:
        mat2 = b.T

    # Ensure dtype compatibility for torch.addmm
    mat1 = mat1.to(out_dtype)
    mat2 = mat2.to(out_dtype)

    # Determine output shape
    y_shape = (mat1.size(0), mat2.size(1))

    if bias is not None:
        assert not accumulate, "Bias is not supported with accumulation"
        bias = bias.to(out_dtype)
        # With bias case
        if out_dtype == torch.float32:
            y_ref = torch.addmm(bias.repeat(mat1.size(0), 1), mat1, mat2, beta=1, alpha=1)
        else:
            y_ref = torch.addmm(bias, mat1, mat2, beta=1, alpha=scale_alpha)
    else:
        # Without bias case
        if accumulate and out is not None:
            y_ref = out.clone().to(out_dtype)
        else:
            y_ref = torch.zeros(y_shape, dtype=out_dtype, device=a.device)
        torch.addmm(y_ref, mat1, mat2, beta=1, alpha=scale_alpha, out=y_ref)

    return y_ref


def get_wgrad_sign_vector() -> torch.Tensor:
    """Hard-coded signs for Hadamard transform"""
    return torch.tensor(
        [1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0],
        dtype=torch.float32,
    )


class Nvfp4TiledQuantizeRefOp(quantization.QuantizeOpBase):
    def __init__(self):
        super().__init__()

    @property
    def supports_allgather(self) -> bool:
        return False
    
    @staticmethod
    def _build_hadamard_matrix(
        size: int, device: torch.device, dtype: torch.dtype, with_random_sign_mask: bool = True
    ) -> torch.Tensor:
        """Construct a Hadamard matrix of given power-of-two size with entries +-1.

        Uses Sylvester construction to avoid SciPy dependency.
        """
        assert (size & (size - 1)) == 0, "Hadamard size must be a power of two"
        h = torch.ones((1, 1), device=device, dtype=torch.float32)
        while h.shape[0] < size:
            h = torch.cat(
                [
                    torch.cat([h, h], dim=1),
                    torch.cat([h, -h], dim=1),
                ],
                dim=0,
            )
        if with_random_sign_mask:
            sign_mat = get_wgrad_sign_vector().to(device) * torch.eye(
                size, device=device, dtype=torch.float32
            )
            h = sign_mat @ h
        return h.to(dtype)
    
    def _apply_rht(self, x: torch.Tensor, qparams: quantization.QParams) -> torch.Tensor:
        """Apply randomized Hadamard transform without random signs (reference path).

        This matches the reference used in tests: x_reshaped @ (H * (1/sqrt(g))).
        """
        # Only apply when enabled
        if not qparams.with_rht:
            return x

        # RHT dimension equals the quantization tile length (NVFP4 uses 16)
        rht_dim = qparams.quant_tile_shape[1]
        assert (
            x.shape[-1] % rht_dim == 0
        ), f"Inner dimension {x.shape[-1]} must be divisible by hadamard dimension {rht_dim}"

        # Build H and scale
        H = self._build_hadamard_matrix(rht_dim, x.device, x.dtype, True)
        scale = 1.0 / float(rht_dim) ** 0.5

        # Perform blockwise transform along the last dimension
        original_shape = x.shape
        x_mat = x.contiguous().view(-1, rht_dim)
        # Random sign matrix is identity in this reference (no sign flipping)
        transform = H * scale
        out = x_mat @ transform
        return out.view(original_shape)
    
    @staticmethod
    def _pad_tensor(
        tensor: torch.Tensor, row_divisor: Optional[int], col_divisor: Optional[int]
    ) -> torch.Tensor:

        assert tensor.dim() == 2, "only supports 2D tensors"
        M, N = tensor.shape
        padding_needed_rows = 0
        padding_needed_cols = 0

        if row_divisor is not None and M % row_divisor != 0:
            padding_needed_rows = row_divisor - (M % row_divisor)
        # Check and calculate column padding if col_divisor is provided
        if col_divisor is not None and N % col_divisor != 0:
            padding_needed_cols = col_divisor - (N % col_divisor)

        # Return original tensor if no padding is needed
        if padding_needed_rows == 0 and padding_needed_cols == 0:
            return tensor

        # pad the tensor
        out = torch.nn.functional.pad(
            tensor,
            (0, padding_needed_cols, 0, padding_needed_rows),
            mode="constant",
            value=0.0,
        ).contiguous()

        return out

    @staticmethod
    def _rm_pad_tensor(tensor: torch.Tensor, original_size: tuple[int, ...]) -> torch.Tensor:

        assert tensor.dim() == 2, "only supports 2D tensors"
        M, N = original_size
        out = tensor[:M, :N].contiguous()
        return out
    
    @classmethod
    def _quantize_blockwise_reference(
        cls,
        x: torch.Tensor,
        global_amax: torch.Tensor,
        tile_len_x: int,
        tile_len_y: int,
        *,
        pow_2_scales: bool,
        with_2d_quantization: bool,
        eps: float,  # pylint: disable=unused-argument
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert x.ndim == 2
        m, n = x.shape
        # Compute vec_max based on the original x (before reshape)
        # For 1D quantization: amax over each row chunk of 16
        # For 2D quantization: amax over each 16x16 block, but output shape is still (128, 8, 1), filled with block amax
        if with_2d_quantization:
            # x shape: (128, 128)
            x_blocks = (
                x.unfold(0, tile_len_y, tile_len_y)
                .unfold(1, tile_len_x, tile_len_x)
                .to(torch.float32)
            )  # (8, 8, 16, 16)
            block_amax = torch.amax(torch.abs(x_blocks), dim=(-1, -2))  # (8, 8)
            # Now, expand to (128, 8, 1) by repeating each block_amax for 16 rows
            vec_max = block_amax.repeat_interleave(tile_len_y, dim=0).unsqueeze(-1)  # (128, 8, 1)
        else:
            # x shape: (128, 128)
            x_reshaped = x.view(m, n // tile_len_x, tile_len_x)  # (128, 8, 16)
            vec_max = torch.amax(torch.abs(x_reshaped), dim=-1, keepdim=True).to(
                torch.float32
            )  # (128, 8, 1)
        x = x.view(m, n // tile_len_x, tile_len_x)
        FLOAT4_E2M1_MAX = torch.tensor(6.0, device=x.device, dtype=torch.float32)
        FLOAT8_E4M3_MAX = torch.tensor(448.0, device=x.device, dtype=torch.float32)
        decode_scale = torch.div(vec_max, FLOAT4_E2M1_MAX)

        if pow_2_scales:
            decode_scale = cast_to_e8(decode_scale)
            encode_scale = torch.div(
                torch.tensor(1.0, device=x.device, dtype=torch.float32),
                decode_scale.to(torch.float32),
            )
        else:
            global_encode_scale = torch.div(FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX, global_amax)
            global_encode_scale = torch.min(
                global_encode_scale,
                torch.tensor(
                    torch.finfo(torch.float32).max,
                    device=global_encode_scale.device,
                    dtype=torch.float32,
                ),
            )
            if global_encode_scale == torch.tensor(0.0, device=x.device, dtype=torch.float32):
                global_encode_scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
            global_decode_scale = torch.div(1.0, global_encode_scale)

            decode_scale = decode_scale * global_encode_scale
            decode_scale = torch.min(
                decode_scale,
                torch.tensor(
                    torch.finfo(torch.float32).max,
                    device=decode_scale.device,
                    dtype=torch.float32,
                ),
            )
            decode_scale = torch.clamp(decode_scale, min=-FLOAT8_E4M3_MAX, max=FLOAT8_E4M3_MAX)
            decode_scale = decode_scale.to(torch.float8_e4m3fn)

            encode_scale = torch.min(
                torch.div(1.0, decode_scale.to(torch.float32) * global_decode_scale),
                torch.tensor(
                    torch.finfo(torch.float32).max,
                    device=decode_scale.device,
                    dtype=torch.float32,
                ),
            )

        scaled_x = x.to(torch.float32) * encode_scale

        clipped_x = torch.clamp(scaled_x, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).reshape(m, n)

        return cast_to_fp4x2(clipped_x), decode_scale.squeeze(-1)

    def _quantize(
        self,
        x: torch.Tensor,
        qparams: quantization.QParams,
        return_transpose: bool,
        with_2d_quantization: bool
    ):
        assert qparams.quant_tile_shape in (
            (1, 16),
            (16, 16),
        ), "NVFP4 only supports 1x16 or 16x16 tile shape."
        # Prepare inputs once so we can reuse for both amax and quantization
        # Row-input will always be the original input.
        row_input = x
        col_input = (
            self._apply_rht(x.t().contiguous(), qparams)
            if qparams.with_rht
            else x.t().contiguous()
        )
        # Compute amax for rowwise and columnwise paths separately
        global_amax_row = torch.max(torch.abs(row_input)).to(torch.float32).view(1)
        global_amax_col = (
            torch.max(torch.abs(col_input)).to(torch.float32).view(1)
            if return_transpose
            else global_amax_row
        )

        transpose_scales = False

        M, N = x.shape
        x_input = row_input
        x_padded = self._pad_tensor(
            x_input, row_divisor=qparams.quant_tile_shape[0], col_divisor=qparams.quant_tile_shape[1]
        )

        qx, sx = self._quantize_blockwise_reference(
            x_padded,
            global_amax_row,
            qparams.quant_tile_shape[1],
            qparams.quant_tile_shape[0],
            pow_2_scales=qparams.pow_2_scales,
            eps=qparams.eps,
            with_2d_quantization=with_2d_quantization
        )
        if transpose_scales:
            sx = sx.T

        qx = self._rm_pad_tensor(qx, (M, N // 2))

        if return_transpose:
            x_t = col_input
            x_t_padded = self._pad_tensor(
                x_t, row_divisor=qparams.quant_tile_shape[0], col_divisor=qparams.quant_tile_shape[1]
            )

            qx_t, sx_t = self._quantize_blockwise_reference(
                x_t_padded,
                global_amax_col,
                qparams.quant_tile_shape[1],
                qparams.quant_tile_shape[0],
                pow_2_scales=qparams.pow_2_scales,
                eps=qparams.eps,
                with_2d_quantization=with_2d_quantization
            )

            qx_t = self._rm_pad_tensor(qx_t, (N, M // 2))

            if transpose_scales:
                sx_t = sx_t.T
        else:
            qx_t = None
            sx_t = None

        return qx, sx, qx_t, sx_t, global_amax_row, global_amax_col

    def quantize(
        self,
        x: torch.Tensor,
        qparams: quantization.QParams,
        return_transpose: bool = False,
        reduce_amax: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        input_meta: Optional[quantization.QuantizeInputMetaBase] = None,
    ) -> quantization.QuantizeResult:
        
        assert x.dtype in utils.HIGH_PRECISION_FLOAT_DTYPES, "Unsupported input dtype."

        assert qparams.quant_tile_shape is not None
        assert len(qparams.quant_tile_shape) == 2
        
        (
            qx, 
            sx, 
            qx_t, 
            sx_t, 
            global_amax_row, 
            global_amax_col
         ) = self._quantize(
            x,
            qparams,
            return_transpose=return_transpose,
            with_2d_quantization=qparams.with_2d_quantization,
        )
        return quantization.QuantizeResult(qx, sx, qx_t, sx_t, global_amax_row, global_amax_col)

    def _ref_qgemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        m_params: quantization.MMParams,  # pylint: disable=unused-argument
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        accumulate: bool = False,
        pow_2_scales: bool = False,
        global_amax_qx: torch.Tensor | None = None,
        global_amax_qw: torch.Tensor | None = None,
        qparams_x: Optional[quantization.QParams] = None,
        qparams_w: Optional[quantization.QParams] = None,
    ) -> torch.Tensor:
        """Python implementation of microblock FP4 GEMM."""
        assert bias is None, "Bias is implemented for FP4 GEMM."

        high_precision_x = cast_from_fp4x2(qx, out_dtype)
        high_precision_w = cast_from_fp4x2(qw, out_dtype)

        if pow_2_scales:

            if sx.dtype == torch.uint8:
                # if scaling factor is stored in uint8 container
                sx = torch.tensor(2.0, device=sx.device, dtype=torch.float32) ** (
                    (
                        sx.to(torch.float32)
                        - torch.tensor(127, device=sx.device, dtype=torch.float32)
                    )
                )
                sw = torch.tensor(2.0, device=sw.device, dtype=torch.float32) ** (
                    (
                        sw.to(torch.float32)
                        - torch.tensor(127, device=sw.device, dtype=torch.float32)
                    )
                )
            else:
                # if scaling factor is torch.float8_e8m0fnu
                sx = sx.to(torch.float32)
                sw = sw.to(torch.float32)

            alpha = torch.tensor(1.0, device=high_precision_x.device, dtype=torch.float32)

        else:

            assert global_amax_qx is not None
            assert global_amax_qw is not None

            sx = sx.to(torch.float32)
            sw = sw.to(torch.float32)

            factor = 6.0 * 6.0 * 448.0 * 448.0

            partial_alpha = global_amax_qx * global_amax_qw
            alpha = torch.div(partial_alpha, factor).squeeze(-1)

        M, K = high_precision_x.shape
        N, K_w = high_precision_w.shape
        assert K == K_w, "K dimension mismatch between qx and qw"

        assert K % 32 == 0 if pow_2_scales else K % 16 == 0, "K dimension must be divisible by 16 or 32"
        assert N % 8 == 0, "N dimension must be divisible by 8"

        block_length = 32 if pow_2_scales else 16

        grid_k = K // block_length

        assert sx.shape == (
            M,
            K // block_length,
        ), f"sx shape mismatch: expected ({M}, {K//block_length}), got {sx.shape}"
        assert sw.shape == (
            N,
            K // block_length,
        ), f"sw shape mismatch: expected ({N}, {K//block_length}), got {sw.shape}"

        y = torch.zeros(M, N, dtype=torch.float32, device=qx.device)

        # below implementation is to match the FP4 tensor core implementation
        # Each output element (i, j) is fp32 accumulation of (K // block_length) inner products
        # Each inner product is sx * sw * (1, block_length) x (block_length, 1) with precision in fp32
        # Then batch the computation in M, N dimension
        for k in range(grid_k):
            k_start = k * block_length
            k_end = k_start + block_length

            qx_block = high_precision_x[:, k_start:k_end].clone().contiguous()
            qw_block = high_precision_w[:, k_start:k_end].clone().contiguous()

            # Extract scaling factors for the current blocks
            sx_block = sx[:, k]
            sw_block = sw[:, k]

            y += torch.outer(sx_block, sw_block) * high_precision_gemm_ref(
                qx_block, qw_block, torch.float32, is_b_transposed=True
            )

        if not pow_2_scales and K > 0:
            # only apply global scale for NVFP4 and non-empty cases
            y = alpha * y

        # accumulation happens at epilogue in float32
        if accumulate:
            assert out is not None, "Output tensor must be provided for accumulation."
            out = (out.to(torch.float32) + y).to(out_dtype)
            return out
        else:
            y = y.to(out_dtype)
            return y

    def qgemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        m_params: quantization.MMParams,
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        accumulate: bool = False,
        global_amax_qx: torch.Tensor | None = None,
        global_amax_qw: torch.Tensor | None = None,
        qparams_x: Optional[quantization.QParams] = None,
        qparams_w: Optional[quantization.QParams] = None,
    ) -> torch.Tensor:
        if out is not None:
            assert out_dtype == out.dtype, "Output dtype mismatch."

        return self._ref_qgemm(
            qx,
            qw,
            m_params,
            out_dtype,
            sx,
            sw,
            bias=bias,
            out=out,
            accumulate=accumulate,
            pow_2_scales=qparams_x.pow_2_scales and qparams_w.pow_2_scales,
            global_amax_qx=global_amax_qx,
            global_amax_qw=global_amax_qw,
            qparams_x=qparams_x,
            qparams_w=qparams_w,
        )