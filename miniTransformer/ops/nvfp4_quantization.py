import torch
import fp4_gemm

from miniTransformer.quantiztion import quantization
from miniTransformer.utils import utils

nvfp4_output_t = torch.uint8
nvfp4_scale_t = torch.float8_e4m3fn


def quantize_nvfp4_cuda(
    x: torch.Tensor,
    *,
    with_2d_quantization: bool,
    return_transpose: bool = False,
    amax_rowwise: torch.Tensor | None = None,
    amax_colwise: torch.Tensor | None = None,
) -> quantization.QuantizeResult:
    """
    Standalone NVFP4 CUDA quantization wrapper around fp4_gemm binding.

    Returns QuantizeResult:
      - always: data (x_q), scale (sx), tensor_scale (global amax rowwise)
      - when return_transpose=True: data_t (x_q^T), scale_t (sx_t),
        tensor_scale_t (global amax colwise)
    """
    assert x.is_cuda, "quantize_nvfp4_cuda expects a CUDA tensor."
    assert x.dtype in utils.HIGH_PRECISION_FLOAT_DTYPES, "Unsupported input dtype."
    assert x.dim() == 2, "quantize_nvfp4_cuda expects a 2D tensor."

    rows, cols = x.shape
    assert rows % 32 == 0 and cols % 32 == 0, "Input shape must be multiples of 32."

    # Default global amax from input (1-element float32 tensor).
    if amax_rowwise is None:
        amax_rowwise = torch.max(torch.abs(x)).to(torch.float32).view(1).contiguous()
    else:
        amax_rowwise = amax_rowwise.to(device=x.device, dtype=torch.float32).reshape(1).contiguous()

    if amax_colwise is None:
        amax_colwise = amax_rowwise
    else:
        amax_colwise = amax_colwise.to(device=x.device, dtype=torch.float32).reshape(1).contiguous()

    out = torch.empty((rows, cols // 2), dtype=nvfp4_output_t, device=x.device)
    out_s = torch.empty((rows, cols // 16), dtype=nvfp4_scale_t, device=x.device)

    # Keep CUDA launch stable by always passing valid transpose buffers.
    out_t_full = torch.empty((cols, rows // 2), dtype=nvfp4_output_t, device=x.device)
    out_st_full = torch.empty((cols, rows // 16), dtype=nvfp4_scale_t, device=x.device)
    cuda_return_transpose = True

    fp4_gemm.quantize_transpose_nvfp4(
        x,
        amax_rowwise,
        amax_colwise,
        out,
        out_s,
        out_t_full,
        out_st_full,
        with_2d_quantization,
        cuda_return_transpose,
    )

    if return_transpose:
        return quantization.QuantizeResult(
            data=out,
            scale=out_s,
            data_t=out_t_full,
            scale_t=out_st_full,
            tensor_scale=amax_rowwise,
            tensor_scale_t=amax_colwise,
        )
    return quantization.QuantizeResult(
        data=out,
        scale=out_s,
        tensor_scale=amax_rowwise,
    )