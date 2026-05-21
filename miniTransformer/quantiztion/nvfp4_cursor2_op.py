import math
import os
import torch
import fp4_gemm
from typing import Optional, List, Tuple, Union

from miniTransformer.quantiztion import quantization
from miniTransformer.quantiztion.quantization import GroupQuantizeResult
from miniTransformer.utils import utils

import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch import NVFP4Quantizer, MXFP8Quantizer
from transformer_engine.pytorch.cpp_extensions import split_quantize
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm, general_grouped_gemm
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor, QuantizedTensorStorage


nvfp4_output_t = torch.uint8
nvfp4_scale_t = torch.float8_e4m3fn

use_cpp_allocator = False

# Print a timing breakdown every N grouped-GEMM calls.
# Set to 0 to disable entirely (zero overhead on non-printing iterations).
_GEMM_TIMING_INTERVAL: int = 50
_GEMM_CALL_COUNT: int = 0

# Capture one torch.profiler trace at this call index (0 = first call).
# Set to -1 to disable. Output written to NVFP4_PROFILE_DIR (default /tmp).
_GEMM_PROFILE_AT: int = int(os.environ.get("NVFP4_PROFILE_AT", "50"))
_GEMM_PROFILED: bool = False


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


def _is_cutedsl_pertoken_path(A: list, B: list) -> bool:
    from miniTransformer.ops.cuteDSL.nvfp4.grouped_gemm_nvfp4_api import (
        PertokenNvfp4Tensor,
        TensorGlobalNvfp4Tensor,
    )

    if not A or not B:
        return False
    return isinstance(A[0], PertokenNvfp4Tensor) and isinstance(
        B[0], TensorGlobalNvfp4Tensor
    )


def _grouped_gemm_cutedsl_pertoken(
    A: list,
    B: list,
    m_splits: list,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    from miniTransformer.ops.cuteDSL.nvfp4.grouped_gemm_nvfp4_api import (
        grouped_gemm_cutedsl_pertoken,
    )
    import miniTransformer.ops.cuteDSL.nvfp4.grouped_gemm_nvfp4_impl as _impl
    import time
    global _GEMM_CALL_COUNT, _GEMM_PROFILED

    do_timing = _GEMM_TIMING_INTERVAL > 0 and _GEMM_CALL_COUNT % _GEMM_TIMING_INTERVAL == 0
    do_profile = (
        not _GEMM_PROFILED
        and _GEMM_PROFILE_AT >= 0
        and _GEMM_CALL_COUNT == _GEMM_PROFILE_AT
    )

    if do_profile:
        profile_dir = os.environ.get("NVFP4_PROFILE_DIR", "/tmp")
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        trace_path = os.path.join(profile_dir, f"nvfp4_gemm_rank{rank}_call{_GEMM_CALL_COUNT}")
        print(f"[GEMM #{_GEMM_CALL_COUNT:5d}] capturing torch.profiler trace → {trace_path}")
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        )
        prof.__enter__()

    if do_timing:
        _impl._timing_enabled = True
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    res = grouped_gemm_cutedsl_pertoken(A, B, m_splits, out_dtype)

    if do_timing:
        torch.cuda.synchronize()
        t_us = (time.perf_counter() - t0) * 1e6
        _impl._timing_enabled = False
        t = _impl._last_timing_us
        print(
            f"[GEMM #{_GEMM_CALL_COUNT:5d}] total={t_us:8.1f} µs\n"
            f"  loop={t.get('loop_us', float('nan')):8.1f} µs"
            f"  | make_fp4={t.get('make_fp4_us', float('nan')):7.1f} µs"
            f"  | block_scale_fp8={t.get('block_scale_fp8_us', float('nan')):7.1f} µs\n"
            f"  sfa_d2h={t.get('make_cute_sf_d2h_us', float('nan')):7.1f} µs"
            f"  | sfa_cvt={t.get('make_cute_sf_cvt_us', float('nan')):7.1f} µs"
            f"  (alloc={t.get('create_sf_alloc_us', float('nan')):6.1f} µs"
            f"  cvt_jit={t.get('create_sf_cvt_jit_us', float('nan')):6.1f} µs"
            f"  h2d={t.get('create_sf_h2d_us', float('nan')):6.1f} µs)\n"
            f"  alpha_build={t.get('alpha_build_us', float('nan')):7.1f} µs"
            f"  | kernel_dispatch={t.get('kernel_dispatch_us', float('nan')):7.1f} µs"
        )

    if do_profile:
        prof.__exit__(None, None, None)
        prof.export_chrome_trace(trace_path + ".json")
        print(f"[GEMM #{_GEMM_CALL_COUNT:5d}] trace saved to {trace_path}.json")
        _GEMM_PROFILED = True

    _GEMM_CALL_COUNT += 1
    return res


class Nvfp4Cursor2QuantizeOp(quantization.QuantizeOpBase):
    """
  NVFP4 quantize op for GroupLinear (cursor2 path):

  - Activation: per-token Triton quant → ``PertokenNvfp4Tensor``
  - Weight: tensor-global ``quantize_nvfp4_cuda`` → ``TensorGlobalNvfp4Tensor``
  - Forward grouped GEMM: CuTeDSL ``grouped_gemm_cutedsl_pertoken`` (SM100+)

  Use with ``disable_backward_quant=True`` in GroupLinear when activation has no
  columnwise TE storage. Weight backward with columnwise still uses TE when
  ``return_transpose=True``.
    """

    def __init__(self):
        super().__init__()
        self.should_update_quantize_weights = True

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
        M, N = x.shape
        device = x.device
        x_dtype = x.dtype

        nvfp4_quantizer = NVFP4Quantizer(  # NVFP4Quantizer
            rowwise=True,
            columnwise=return_transpose,
            with_amax_reduction=False,
            amax_reduction_group=None,
            with_rht=qparams.with_rht,
            with_post_rht_amax=qparams.with_rht,
            with_2d_quantization=with_2d_quantization,
            stochastic_rounding=qparams.stochastic_rounding
        )
        if use_cpp_allocator:
            nvfp4_tensor = nvfp4_quantizer(x)
        else:
            nvfp4_tensor = nvfp4_quantizer.make_empty(
                (M, N), dtype=x_dtype, device=device, requires_grad=False
            )
            nvfp4_tensor = nvfp4_quantizer.update_quantized(x, nvfp4_tensor)

        return nvfp4_tensor

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
        

        nvfp4_tensor = self._quantize(
            x,
            qparams,
            return_transpose=return_transpose,
            with_2d_quantization=qparams.with_2d_quantization,
        )
        return quantization.QuantizeResult(data=nvfp4_tensor, scale=nvfp4_tensor, data_t=nvfp4_tensor, scale_t=nvfp4_tensor)


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
        grad: bool = False,
        accumulate: bool = False,
        layout: str = "TN",
        global_amax_qx: torch.Tensor | None = None,
        global_amax_qw: torch.Tensor | None = None,
        qparams_x: Optional[quantization.QParams] = None,
        qparams_w: Optional[quantization.QParams] = None,
        custom_gemm_backend: str = "cutlass",  # cutlass, triton, cuteDSL
    ) -> torch.Tensor:

        gemm_out = general_gemm(
            A=qw,
            B=qx,
            out_dtype=out_dtype,
            quantization_params=None,
            gelu=False,
            gelu_in=None,
            alpha=1.0,
            beta=None,
            accumulate=accumulate,
            layout=layout,
            out=out,
            bias=bias,
            use_split_accumulator=False,
            grad=grad,
            ub=None,
            ub_type=None,
            extra_output=None,
            bulk_overlap=False
        )[0]
        return gemm_out


    def group_quantize(
        self,
        x: torch.Tensor,
        qparams: quantization.QParams,
        m_splits: List[int] = None,
        return_transpose: bool = False,
        reduce_amax: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        input_meta: Optional[quantization.QuantizeInputMetaBase] = None,
    ) -> GroupQuantizeResult:
        """
        Grouped quantize for ``GroupLinear`` forward (CuTeDSL pertoken GEMM).

        - Activation (X): per-token Triton NVFP4 → list of ``PertokenNvfp4Tensor``
        - Weight: tensor-global NVFP4 → list of ``TensorGlobalNvfp4Tensor``
        - Weight + ``return_transpose``: falls back to TE NVFP4 (columnwise for dgrad)
        """
        from miniTransformer.ops.cuteDSL.nvfp4.grouped_gemm_nvfp4_api import (
            PertokenNvfp4Tensor,
            quantize_activation_nvfp4_pertoken
        )

        num_gemms = len(m_splits)

        if qparams.param_type == qparams.param_type.WEIGHT:
            quantize_tensor_list = []
            quantize_tensor_list_transpose = []
            for i in range(num_gemms):
                w_quantizer_fp4 = NVFP4Quantizer(
                    rowwise=True,
                    columnwise=True,
                    with_amax_reduction=False,
                    amax_reduction_group=None,
                    with_rht=qparams.with_rht,
                    with_post_rht_amax=qparams.with_rht,
                    with_2d_quantization=qparams.with_2d_quantization,
                    stochastic_rounding=qparams.stochastic_rounding,
                )
                quantize_tensor_list.append(w_quantizer_fp4.quantize(x[i]))
                if return_transpose:
                    w_quantizer_fp8 = MXFP8Quantizer(rowwise=False, columnwise=True, fp8_dtype=tex.DType.kFloat8E4M3)
                    quantize_tensor_list_transpose.append(w_quantizer_fp8.quantize(x[i]))

            return GroupQuantizeResult(
                data=quantize_tensor_list,
                data_t=quantize_tensor_list_transpose,
                scale=quantize_tensor_list,
                scale_t=quantize_tensor_list_transpose,
                tensor_scale=quantize_tensor_list,
                tensor_scale_t=quantize_tensor_list_transpose,
            )

        if qparams.param_type == qparams.param_type.X:
            # Per-token quant is row-independent: quantize the full tensor once,
            # then split the result by m_splits.  This avoids num_gemms kernel
            # launches and sidesteps zero-chunk issues from empty experts.
            qresult_full = quantize_activation_nvfp4_pertoken(x)
            act_list: List[PertokenNvfp4Tensor] = [
                PertokenNvfp4Tensor(
                    data_uint8=d,
                    block_scale=bs,
                    global_scale_full=qresult_full.global_scale,
                )
                for d, bs in zip(
                    torch.split(qresult_full.data_uint8, m_splits, dim=0),
                    torch.split(qresult_full.block_scale, m_splits, dim=0),
                )
            ]

            # Activation: MXFP8 quant for backward
            quantizers = []
            for i in range(num_gemms):
                x_quantizer = MXFP8Quantizer(
                    rowwise=False,
                    columnwise=True,
                    fp8_dtype=tex.DType.kFloat8E4M3
                )
                quantizers.append(x_quantizer)
            quantize_tensor_list = tex.split_quantize(
                x,
                m_splits,
                quantizers,
            )
            return GroupQuantizeResult(
                data=act_list,
                scale=[a.block_scale for a in act_list],
                data_t=quantize_tensor_list,
                scale_t=quantize_tensor_list,
                tensor_scale=[a.global_scale for a in act_list],
                tensor_scale_t=quantize_tensor_list,
            )

        # Dy: MXFP8 quant for backward
        quantizers = []
        for i in range(num_gemms):
            x_quantizer = MXFP8Quantizer(
                rowwise=True,
                columnwise=True,
                fp8_dtype=tex.DType.kFloat8E4M3
            )
            quantizers.append(x_quantizer)
        quantize_tensor_list = tex.split_quantize(
            x,
            m_splits,
            quantizers,
        )
        return GroupQuantizeResult(
            data=quantize_tensor_list,
            scale=quantize_tensor_list,
            data_t=quantize_tensor_list,
            scale_t=quantize_tensor_list,
            tensor_scale=quantize_tensor_list,
            tensor_scale_t=quantize_tensor_list,
        )


    def grouped_gemm(
        self,
        A: List[torch.Tensor],
        B: List[torch.Tensor],
        sA: List[torch.Tensor],
        sB: List[torch.Tensor],
        global_amax_A: List[torch.Tensor],
        global_amax_B: List[torch.Tensor],
        out: List[torch.Tensor],
        out_dtype: torch.dtype,
        layout: str = "TN",
        m_splits: List[int] = None,
        gelu: bool = False,
        grad=False,
        accumulate: bool = False,
        bias: Optional[List[torch.Tensor]] = None,
        use_bias: bool = False,
        qparams_x: Optional[quantization.QParams] = None,
        qparams_w: Optional[quantization.QParams] = None,
    ) -> Tuple[List[torch.Tensor], ...]:
        activation_dtype = out_dtype

        a0 = A[0]
        is_te_quantized = isinstance(a0, (QuantizedTensor, QuantizedTensorStorage))

        if (
            _is_sm100()
            and not grad
            and not gelu
            and not accumulate
            and layout == "TN"
        ):
            try:
                return _grouped_gemm_cutedsl_pertoken(A, B, m_splits, activation_dtype)
            except (TypeError, NotImplementedError) as exc:
                raise RuntimeError(
                    "CuTeDSL pertoken grouped GEMM failed on SM100; "
                    "set disable_backward_quant and use Nvfp4Cursor2QuantizeOp."
                ) from exc

        if grad:
            if layout == "NN":
                if is_te_quantized:
                    n_cols = B[0]._columnwise_data.size(0)
                else:
                    n_cols = B[0]._columnwise_data.size(0)
                out = torch.empty(
                    [sum(m_splits), n_cols],
                    dtype=activation_dtype,
                    device=B[0].device,
                )
                out = [out]
        else:
            if is_te_quantized:
                n_cols = B[0]._rowwise_data.size(0)
            else:
                n_cols = B[0].size(0)
            out = torch.empty(
                [sum(m_splits), n_cols],
                dtype=activation_dtype,
                device=B[0].device,
            )
            out = [out]

        general_grouped_gemm(
            A=B,
            B=A,
            out=out,
            out_dtype=activation_dtype,
            layout=layout,
            m_splits=m_splits,
            single_output=False if (grad and layout == "NT") else True,
            grad=grad,
            bias=bias,
            use_bias=use_bias,
            use_split_accumulator=False,
            accumulate=accumulate
        )

        if grad and layout == "NT":
            return out
        else:
            return out[0]
