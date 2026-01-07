import math
import torch
import fp4_gemm
from typing import Optional
from transformer_engine.pytorch.custom_recipes.quantization_nvfp4 import NVFP4QuantizerRef
from transformer.ops import quantization

nvfp4_output_t = torch.uint8
nvfp4_scale_t = torch.float8_e4m3fn

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


class Nvfp4TiledQuantizeOp(quantization.QuantizeOpBase):
    def __init__(self):
        super().__init__()

    @property
    def supports_allgather(self) -> bool:
        return False   

    def _quantize(
        self,
        x: torch.Tensor,
        qparams: quantization.QParams,
        return_transpose: bool,
        with_2d_quantization: bool
    ):
        if x.ndim > 2:
            x = x.view(-1, x.shape[-1])

        assert x.dim() == 2
        assert x.is_cuda
        assert x.is_contiguous(), "Input tensor must be contiguous"
        M, N = x.shape
        dtype = x.dtype
        device = x.device

        tile_m, tile_n = qparams.quant_tile_shape[0], qparams.quant_tile_shape[1]

        qx = torch.empty((M, math.ceil(N // 2)), dtype=dtype, device=device).to(nvfp4_output_t)
        sx = torch.empty((M, math.ceil(N // tile_n)), dtype=dtype, device=device).to(nvfp4_scale_t)
        if return_transpose:
            qx_t = torch.empty((N, math.ceil(M // 2)), dtype=dtype, device=device).to(nvfp4_output_t)
            sx_t = torch.empty((N, math.ceil(M // tile_n)), dtype=dtype, device=device).to(nvfp4_scale_t)
        else:
            qx_t = None
            sx_t = None

        fp4_gemm.quantize_transpose_nvfp4(x, qx, sx, qx_t, sx_t, with_2d_quantization, return_transpose)

        return qx, sx, qx_t, sx_t, None, None

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
        

        qx, sx, qx_t, sx_t, global_amax_row, global_amax_col = self._quantize(
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
        gemm_type: quantization.GEMMType = quantization.GEMMType.FPROP,
        qresult_x: QuantizedTensorStorage | None = None,
        qresult_w: QuantizedTensorStorage | None = None,
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

            assert qresult_x is not None
            assert qresult_w is not None

            assert qresult_x.global_amax_row is not None
            assert qresult_w.global_amax_col is not None

            sx = sx.to(torch.float32)
            sw = sw.to(torch.float32)

            factor = 6.0 * 6.0 * 448.0 * 448.0

            if gemm_type == quantization.GEMMType.WGRAD:
                partial_alpha = qresult_x.global_amax_col * qresult_w.global_amax_col
            else:
                partial_alpha = qresult_x.global_amax_row * qresult_w.global_amax_row
            alpha = torch.div(partial_alpha, factor).squeeze(-1)

        M, K = high_precision_x.shape
        N, K_w = high_precision_w.shape
        assert K == K_w, "K dimension mismatch between qx and qw"

        assert K % 32 == 0, "K dimension must be divisible by 32"
        assert N % 8 == 0, "N dimension must be divisible by 8"

        block_length = 32 if self.pow_2_scales else 16

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

        if not self.pow_2_scales and K > 0:
            # only apply global scale for NVFP4 and non-empty cases
            y = alpha * y

        # accumulation happens at epilogue in float32
        if accumulate:
            assert out is not None, "Output tensor must be provided for accumulation."
            y += out.to(torch.float32)
        else:
            assert out is None, "Output tensor should be None when accumulate is False."

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
        return self.ref_gemm.qgemm(
            qx,
            qw,
            m_params,
            out_dtype,
            sx,
            sw,
            bias=bias,
            out=out,
            accumulate=accumulate,
            qparams_x=qparams_x,
            qparams_w=qparams_w,
        )