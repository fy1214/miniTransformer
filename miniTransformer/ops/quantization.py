import abc
import dataclasses
import enum
import math
import torch
from typing import Optional, Protocol, Tuple

from ..utils import utils


@dataclasses.dataclass(frozen=True)
class QParams:
    """Quantization parameters.

    scaling_type: Scaling granularity.
    quant_dtype: Quantized type. If None then quantization is not applied.
    eps: Optional minimum value of amax.
    pow_2_scales: Rounds the scale derived from argmax to a power of 2. Applying
        power of 2 constraint unlocks possibilities within hardware, however
        it also can lessen or increase the magnitude of sampling error.
    quant_tile_shape: For ScalingTypes that define a fixed shape for a
        quantization tiling granularity for scale factor calculation, this defines
        the tile shape. For example 1x128 or 128x128.
    """

    scaling_dtype: torch.dtype | None = None
    output_dtype: torch.dtype | None = None
    eps: float = 0.0
    pow_2_scales: bool = False
    quant_tile_shape: Tuple[int, int] | None = None
    with_2d_quantization:  bool = False
    with_rht: bool = False


@dataclasses.dataclass(frozen=True)
class MMParams:
    """Matrix multiplication parameters."""

    out_dtype: torch.dtype | None = None
    # Use split accumulator for more accurate FP8 GEMM
    use_split_accumulator: bool = True


@dataclasses.dataclass
class QuantizeResult:
    """A container to hold quantization result, including quantized tensor, optional
    transposed quantized tensor, and corresponding decoding scales.

    data: the quantized tensor.
    scale: the decoding scale for the quantized tensor. Shape depends on the scaling granularity.
        - if scaling type is PER_TENSOR, it should be a 1D scalar tensor.
    data_t: the transposed quantized tensor.
    scale_t: the decoding scale for the transposed quantized tensor.
    """

    data: torch.Tensor
    scale: torch.Tensor
    data_t: torch.Tensor | None = None
    scale_t: torch.Tensor | None = None
    global_amax_row: torch.Tensor | None = None
    global_amax_col: torch.Tensor | None = None


@dataclasses.dataclass
class QuantizeResultCache:
    """
    Cache of quantized tensors
    """

    data: torch.Tensor | None = None
    scale: torch.Tensor | None = None
    data_t: torch.Tensor | None = None
    scale_t: torch.Tensor | None = None
    global_amax_row: torch.Tensor | None = None
    global_amax_col: torch.Tensor | None = None

    def to_qresult(self) -> QuantizeResult:
        assert self.data is not None
        assert self.scale is not None
        return QuantizeResult(
            data=self.data, 
            scale=self.scale, 
            data_t=self.data_t, 
            scale_t=self.scale_t,
            global_amax_row=self.global_amax_row,
            global_amax_col=self.global_amax_col
        )


@dataclasses.dataclass
class QuantizeInputMetaBase:
    """Metadata for quantization input, such as holding amax from the previous normalization layer.
    This should be subclassed for specific quantize operations.
    """



class QuantizeOpBase():

    def quantize(
        self,
        x: torch.Tensor,
        qparams: QParams,
        return_transpose: bool = False,
        reduce_amax: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        input_meta: Optional[QuantizeInputMetaBase] = None,
    ) -> QuantizeResult:
        """Quantizes input tensor.

        Args:
            x: Input tensor.
            qparams: Quantizaton parameters.
            return_transpose: Whether to return transposed quantized tensor.
            reduce_amax: Whether to reduce parallel tensor for amax.
            tp_group: Parallel group to reduce over for reduce_amax.
            input_meta: Metadata for quantization input.
        """
        NotImplementedError()
    
    def qgemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        m_params: MMParams,
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        accumulate: bool = False,
        global_amax_qx: torch.Tensor | None = None,
        global_amax_qw: torch.Tensor | None = None,
        qparams_x: Optional[QParams] = None,
        qparams_w: Optional[QParams] = None,
    ) -> torch.Tensor:
        """Quantized GEMM interface.

        It computes the quantized GEMM operation `y = qx * qw.T + bias` with
        the given input tensors. In row-major notation, the shapes are
        `qx` (M, K), `qw` (N, K), and `y` (M, N). Bias should have shape (N) or None.

        When accumulate is True, the GEMM result is accumulated into the `out` tensor in-place.

        Args:
            qx: The quantized input tensor (left side), shape (M, K).
            qw: The quantized weight tensor (right side), shape (N, K).
            m_params: Gemm parameters.
            out_dtype: Output dtype for the GEMM result.
            sx: Decode scale for qx, shape depends on the scaling granularity.
            sw: Decode scale for qw, shape depends on the scaling granularity.
            bias: Bias tensor, shape should be (N) or None.
            out: Output tensor, must be provided if accumulate is True, shape (M, N).
            accumulate: Whether to accumulate the GEMM result into the `out` tensor in-place.
                        Final result in the `out` tensor will be returned.

        Returns:
            The quantized GEMM result tensor `y` with shape (M, N).
        """
        NotImplementedError()

    @property
    def supports_allgather(self) -> bool:
        """
        Whether the TensorQuantizer supports a quantize -> allgather
        order of operations when tensor parallelism applies.

        If True, then quantization should expect to be called with a
        parallel sharded input when tensor parallelism applies. The
        quantized tensor can be all-gathered and transpose_qresult
        can calculate the transpose.

        If False, then the allgather is the responsibility of the
        caller on the high precision input before calling quantize,
        and quantize has visibility of the entire gathered tensor
        and can transpose in one shot.

        It is a design choice that communication be explicitly in the
        structure of the linear layer rather than hidden in
        the TensorQuantizers.
        """
        NotImplementedError()


def _scale_from_amax_tensor(
    x_dtype: torch.dtype,
    amax: torch.Tensor,
    quant_dtype: torch.dtype,
    *,
    eps: float,
    pow_2_scales: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derives quantization and dequantization from amax and options.

    Reference implementation for scale calculation.

    Returns:
    - scale: quantization scales
    - scale_inv: dequantization scales
    - amax: Amax tensor with updates made for extrema values.
    """
    assert amax.dtype == torch.float, "amax must be a float tensor."
    fp8_max = torch.finfo(quant_dtype).max
    # Clamping amax to avoid division by small numbers
    amax = torch.max(amax, torch.tensor(eps))

    # Compute scale factor
    scale = torch.div(fp8_max, amax)
    # Note frexp doesn't give back inf for exponent with an inf input
    # We take care of inf before pow_2_scales
    scale = torch.where(scale == torch.inf, torch.finfo(x_dtype).max, scale)
    if pow_2_scales:
        # Calculate rounded down exponent
        _, exp = torch.frexp(scale)
        # Positive numbers are always returned as mant, exp with
        # a mantissa in [0.5, 1.0). Because a normal float has a mantissa with
        # hidden bit in [1.0, 2.0), the exponent will be off by exactly one because
        # of the shift. Subnormal and zero cases need not be considered because
        # the smallest possible result of fp8_max / amax is still normal.
        exp = exp - 1
        # No subnormals and zero.
        assert (exp > -127).all()
        # TODO: If/when adding a URM option an option is to cap to 126
        # rather than allowing the full range of FP32 (2 - 2^23) x 2^127
        # addresses cases where adding a mantissa overflows into inf scales.
        # Not necessary currently without additional scale smudging options.
        unity = torch.tensor([1.0], device=exp.device)
        torch.ldexp(unity, exp, out=scale)
        # Case where amax is inf. The frexp, ldexp logic changes 0.0 scales
        # Return 0.0 for 0.0 scale for consistency with non-pow2 scale
        # calculation.
        scale = torch.where(amax == float("inf"), 0.0, scale)

    # Handle overflow cases for amax zero causing NaN
    scale = torch.where(amax == 0, 1.0, scale)

    # Compute scale_inv
    scale_inv = torch.reciprocal(scale)

    return scale, scale_inv, amax


class QuantizeOpNonQuantize(QuantizeOpBase):
    """
    A 'quantization' implementation that does not quantize.

    Two supported modes are
    1. Identity quantization
    2. Simplistic cast between high precision types such as fp32 -> bf16

    The GEMM executes in high precision.
    """

    supported_scaling_types = (None,)

    @property
    def supports_allgather_fp8(self) -> bool:
        return False

    def quantize(
        self,
        x: torch.Tensor,
        qparams: QParams,
        return_transpose: bool = False,
        reduce_amax: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        input_meta: Optional[QuantizeInputMetaBase] = None,
    ):
        assert (
            x.dtype in utils.HIGH_PRECISION_FLOAT_DTYPES
        ), f"Unsupported input dtype, {x.dtype}"
        assert not reduce_amax
        if qparams.quant_dtype is None:
            # If qparams is None, use the input tensor as is
            qx = x
        else:
            # Otherwise, cast the input tensor to quant_dtype
            assert (
                qparams.scaling_type in self.supported_scaling_types
            ), f"Unsupported scaling type"
            assert (
                qparams.quant_dtype in utils.HIGH_PRECISION_FLOAT_DTYPES
            ), "Unsupported quant dtype."
            qx = x.to(qparams.quant_dtype)

        sx = torch.empty(0)
        if return_transpose:
            qx_t = qx.t()
            sx_t = torch.empty(0)
        else:
            qx_t = None
            sx_t = None

        return QuantizeResult(data=qx, scale=sx, data_t=qx_t, scale_t=sx_t)

    def qgemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        m_params: MMParams,
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        accumulate: bool = False,
        qparams_x: Optional[QParams] = None,
        qparams_w: Optional[QParams] = None,
    ) -> torch.Tensor:

        assert (
            qx.dtype in utils.HIGH_PRECISION_FLOAT_DTYPES
        ), f"Unsupported input dtype, {qx.dtype}"
        assert (
            qw.dtype in utils.HIGH_PRECISION_FLOAT_DTYPES
        ), f"Unsupported weight dtype, {qw.dtype}"

        if bias is not None and bias.numel():
            # fused matmul + bias
            y = torch.nn.functional.linear(qx, qw, bias=bias)
        else:
            y = torch.matmul(qx, qw.t())

        if accumulate:
            assert out is not None, "Output tensor must be provided for accumulation."
            out.add_(y)
            y = out
        else:
            assert out is None, "Output tensor should be None when accumulate is False."

        return y
