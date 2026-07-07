import inspect
from transformer_engine.pytorch.cpp_extensions.gemm import (
    general_gemm,
    general_grouped_gemm as _te_general_grouped_gemm,
)

# Detect once at import time whether this TE version needs quantization_params.
# Older TE: general_grouped_gemm(A, B, out, out_dtype, ...)
# Newer TE: general_grouped_gemm(A, B, out, quantization_params, out_dtype, ...)
_HAS_QUANT_PARAMS = (
    'quantization_params' in inspect.signature(_te_general_grouped_gemm).parameters
)


def general_grouped_gemm(A, B, out, out_dtype, quantization_params=None, **kwargs):
    if _HAS_QUANT_PARAMS:
        # TE expects a list[Optional[Quantizer]], one entry per GEMM group.
        if quantization_params is None:
            quantization_params = [None] * len(A)
        return _te_general_grouped_gemm(A, B, out, quantization_params, out_dtype, **kwargs)
    else:
        return _te_general_grouped_gemm(A, B, out, out_dtype, **kwargs)
