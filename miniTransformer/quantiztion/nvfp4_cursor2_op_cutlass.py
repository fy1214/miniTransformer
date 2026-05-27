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

# Independent debug counter for quant/GEMM prints (not gated by _GEMM_CALL_COUNT).
_DEBUG_CALL_COUNT: int = 0
# Separate counter for backward GEMM prints (reset independently).
_DEBUG_BGRAD_COUNT: int = 0
_DEBUG_BGRAD_LIMIT: int = 5   # show first 5 backward calls (catches both zero and NaN)

# Capture one torch.profiler trace at this call index (0 = first call).
# Set to -1 to disable. Output written to NVFP4_PROFILE_DIR (default /tmp).
_GEMM_PROFILE_AT: int = int(os.environ.get("NVFP4_PROFILE_AT", "50"))
_GEMM_PROFILED: bool = False

# Forward NaN detection: print on first occurrence, -1 = not yet detected.
_XQUANT_NAN_CALL: int = -1   # X-quant first NaN (gs or sfb)
_WQUANT_NAN_CALL: int = -1   # W-quant first NaN (sfb or alpha_b)
_GEMM_NAN_CALL:   int = -1   # forward GEMM output first NaN


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
    global _GEMM_CALL_COUNT, _GEMM_PROFILED, _DEBUG_CALL_COUNT
    global _GEMM_NAN_CALL

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

    if _DEBUG_CALL_COUNT < 3:
        for i, (a, b) in enumerate(zip(A, B)):
            a_data = getattr(a, 'data_uint8', a)
            b_data = getattr(b, 'data_uint8', b)
            a_gs   = getattr(a, 'global_scale', None)
            b_alpha = getattr(b, '_alpha_b_cached', None)
            if a_gs is not None and a_gs.numel() > 0:
                gs_str     = f"{a_gs.abs().max().item():.4g}"
                gs_nan_str = f"{a_gs.isnan().any().item()}"
                gs_large   = (a_gs > 1.0).sum().item()
                gs_large_str = f"{gs_large}/{a_gs.numel()}"
            else:
                gs_str = gs_nan_str = gs_large_str = "N/A"
            alpha_str = f"{b_alpha:.4g}" if b_alpha is not None else "N/A"
            print(f"[GEMM PRE g={i}] A.data={a_data.shape} B.data={b_data.shape} "
                  f"A.gs_nan={gs_nan_str} A.gs_max={gs_str} A.gs>1={gs_large_str} B.alpha_b={alpha_str}", flush=True)

    res = grouped_gemm_cutedsl_pertoken(A, B, m_splits, out_dtype)

    # Always check for first NaN in forward GEMM output.
    if _GEMM_NAN_CALL < 0:
        res_t = res if isinstance(res, torch.Tensor) else (res[0] if res else None)
        if res_t is not None and res_t.numel() > 0 and res_t.isnan().any():
            _GEMM_NAN_CALL = _GEMM_CALL_COUNT
            nan_cnt = res_t.isnan().sum().item()
            print(f"[GEMM NAN! call={_GEMM_CALL_COUNT}] out={res_t.shape} "
                  f"nan_cnt={nan_cnt}/{res_t.numel()}", flush=True)
            for i, (a, b) in enumerate(zip(A, B)):
                a_gs    = getattr(a, 'global_scale', None)
                b_alpha = getattr(b, '_alpha_b_cached', None)
                a_data  = getattr(a, 'data_uint8', a)
                if a_gs is not None and a_gs.numel() > 0:
                    gs_nan   = a_gs.isnan().any().item()
                    gs_inf   = a_gs.isinf().any().item()
                    gs_max   = a_gs.abs().max().item()
                    gs_large = (a_gs > 1.0).sum().item()
                else:
                    gs_nan = gs_inf = gs_max = gs_large = "N/A"
                alpha_s = f"{b_alpha:.6g}" if b_alpha is not None else "N/A"
                print(f"  g={i}: rows={a_data.shape[0] if hasattr(a_data,'shape') else '?'} "
                      f"gs_nan={gs_nan} gs_inf={gs_inf} gs_max={gs_max} gs>1={gs_large} "
                      f"alpha_b={alpha_s}", flush=True)

    if _DEBUG_CALL_COUNT < 3:
        res_t = res if isinstance(res, torch.Tensor) else res[0] if res else None
        if res_t is not None:
            print(f"[GEMM POST   ] out={res_t.shape} nan={res_t.isnan().any().item()} "
                  f"inf={res_t.isinf().any().item()} max={res_t.abs().max().item():.4g}", flush=True)

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
        # Cached quantizers — rebuilt only when num_gemms or qparams change.
        self._cached_num_gemms: int = 0
        self._w_quantizers_fp8: List = []   # rowwise=False, columnwise=True
        self._x_quantizers: List = []       # rowwise=False, columnwise=True
        self._dy_quantizers: List = []      # rowwise=True,  columnwise=True
        self._w_quantizer_fp4: Optional[NVFP4Quantizer] = None
        self._cached_fp4_key: tuple = ()    # (with_rht, with_2d_quantization, stochastic_rounding)

    def _refresh_quantizer_cache(self, num_gemms: int) -> None:
        """Rebuild MXFP8 quantizer lists when num_gemms changes."""
        if self._cached_num_gemms == num_gemms:
            return
        self._cached_num_gemms = num_gemms
        fp8_col = dict(rowwise=False, columnwise=True, fp8_dtype=tex.DType.kFloat8E4M3)
        fp8_both = dict(rowwise=True, columnwise=True, fp8_dtype=tex.DType.kFloat8E4M3)
        self._w_quantizers_fp8 = [MXFP8Quantizer(**fp8_col) for _ in range(num_gemms)]
        self._x_quantizers = [MXFP8Quantizer(**fp8_col) for _ in range(num_gemms)]
        self._dy_quantizers = [MXFP8Quantizer(**fp8_both) for _ in range(num_gemms)]

    def _get_w_quantizer_fp4(self, qparams) -> NVFP4Quantizer:
        """Return a cached NVFP4Quantizer, rebuilding only when qparams change."""
        key = (qparams.with_rht, qparams.with_2d_quantization, qparams.stochastic_rounding)
        if self._w_quantizer_fp4 is None or self._cached_fp4_key != key:
            self._cached_fp4_key = key
            self._w_quantizer_fp4 = NVFP4Quantizer(
                rowwise=True,
                columnwise=True,
                with_amax_reduction=False,
                amax_reduction_group=None,
                with_rht=qparams.with_rht,
                with_post_rht_amax=qparams.with_rht,
                with_2d_quantization=qparams.with_2d_quantization,
                stochastic_rounding=qparams.stochastic_rounding,
            )
        return self._w_quantizer_fp4

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
        )

        global _DEBUG_CALL_COUNT
        num_gemms = len(m_splits)
        self._refresh_quantizer_cache(num_gemms)

        if qparams.param_type == qparams.param_type.WEIGHT:
            from miniTransformer.ops.nvfp4_quantize import group_quantize_nvfp4_tensor_global_packed
            from miniTransformer.ops.cuteDSL.nvfp4.grouped_gemm_nvfp4_impl import PackedWeightNvfp4Tensor

            if _DEBUG_CALL_COUNT < 3:
                print(f"[W-QUANT PRE ] shape={x.shape} dtype={x.dtype} "
                      f"nan={x.isnan().any().item()} inf={x.isinf().any().item()} "
                      f"max={x.abs().max().item():.4g}", flush=True)

            gw = group_quantize_nvfp4_tensor_global_packed(x, x.shape[1])
            packed_b = PackedWeightNvfp4Tensor(
                cute_data=gw.cute_data,
                scale_flat=gw.scale_flat,
                global_scale=gw.global_scale,
                scale_views=gw.scale_views,
                scale_offsets=gw.scale_offsets,
                row_offsets=gw.row_offsets,
                n_per_expert=x.shape[1],
                k=x.shape[2],
            )
            # Return a list of the same packed object — grouped_gemm_cutedsl_pertoken
            # will index into it by expert index.
            quantize_tensor_list = [packed_b] * num_gemms

            if _DEBUG_CALL_COUNT < 3:
                print(f"[W-QUANT POST] data={gw.data.shape} "
                      f"global_scale={gw.global_scale[:4]} "
                      f"scale_flat={gw.scale_flat.shape}", flush=True)

            if return_transpose:
                w_out_splits = [x[i].shape[0] for i in range(num_gemms)]
                quantize_tensor_list_transpose = tex.split_quantize(
                    x.reshape(-1, x.shape[-1]),
                    w_out_splits,
                    self._w_quantizers_fp8,
                )
            else:
                quantize_tensor_list_transpose = []

            return GroupQuantizeResult(
                data=quantize_tensor_list,
                data_t=quantize_tensor_list_transpose,
                scale=quantize_tensor_list,
                scale_t=quantize_tensor_list_transpose,
                tensor_scale=quantize_tensor_list,
                tensor_scale_t=quantize_tensor_list_transpose,
            )

        if qparams.param_type == qparams.param_type.X:
            from miniTransformer.ops.nvfp4_quantize import group_quantize_nvfp4_pertoken_swizzled

            if _DEBUG_CALL_COUNT < 3:
                print(f"[X-QUANT PRE ] shape={x.shape} dtype={x.dtype} "
                      f"nan={x.isnan().any().item()} inf={x.isinf().any().item()} "
                      f"max={x.abs().max().item():.4g} m_splits={m_splits}", flush=True)

            gq = group_quantize_nvfp4_pertoken_swizzled(x, m_splits)

            if _DEBUG_CALL_COUNT < 3:
                sfb = gq.scale_flat.float()
                gs = gq.global_scale
                gs_large = (gs > 1.0).sum().item()
                print(f"[X-QUANT POST] data={gq.data.shape} "
                      f"gs_nan={gs.isnan().any().item()} "
                      f"gs_inf={gs.isinf().any().item()} "
                      f"gs_max={gs.abs().max().item():.4g} "
                      f"gs_mean={gs.mean().item():.4g} "
                      f"gs>1={gs_large}/{gs.numel()} "
                      f"sfb_nan={sfb.isnan().any().item()} "
                      f"sfb_inf={sfb.isinf().any().item()} "
                      f"sfb_max={sfb.abs().max().item():.4g}", flush=True)

            act_list: List[PertokenNvfp4Tensor] = [
                PertokenNvfp4Tensor(
                    data_uint8=gq.data[gq.row_offsets[g]: gq.row_offsets[g + 1]],
                    block_scale=None,
                    global_scale=gq.global_scale[gq.row_offsets[g]: gq.row_offsets[g + 1]],
                    global_scale_full=gq.global_scale,
                    sfa_swizzled=gq.scale_views[g],
                )
                for g in range(num_gemms)
            ]

            quantize_tensor_list = tex.split_quantize(
                x,
                m_splits,
                self._x_quantizers,
            )
            _DEBUG_CALL_COUNT += 1
            return GroupQuantizeResult(
                data=act_list,
                scale=[a.block_scale for a in act_list],
                data_t=quantize_tensor_list,
                scale_t=quantize_tensor_list,
                tensor_scale=[a.global_scale for a in act_list],
                tensor_scale_t=quantize_tensor_list,
            )

        # Dy: MXFP8 quant for backward
        if _DEBUG_BGRAD_COUNT < _DEBUG_BGRAD_LIMIT:
            dy_nan = x.isnan().any().item()
            dy_inf = x.isinf().any().item()
            dy_max = x.abs().max().item() if x.numel() > 0 else 0.0
            dy_nan_cnt = x.isnan().sum().item() if dy_nan else 0
            print(f"[DY-QUANT PRE] dy.shape={x.shape} dtype={x.dtype} "
                  f"nan={dy_nan}(cnt={dy_nan_cnt}) inf={dy_inf} "
                  f"max={dy_max:.4g} m_splits={m_splits}", flush=True)
        quantize_tensor_list = tex.split_quantize(
            x,
            m_splits,
            self._dy_quantizers,
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
        global _DEBUG_CALL_COUNT, _DEBUG_BGRAD_COUNT

        # Unconditional entry log — confirms whether grad=True path is ever reached.
        if grad and _DEBUG_BGRAD_COUNT < 5:
            print(f"[GROUPED_GEMM GRAD=TRUE ENTRY] layout={layout} m_splits_sum={sum(m_splits)} "
                  f"_DEBUG_BGRAD_COUNT={_DEBUG_BGRAD_COUNT}", flush=True)

        if _DEBUG_CALL_COUNT <= 3:
            from miniTransformer.ops.cuteDSL.nvfp4.grouped_gemm_nvfp4_impl import (
                PertokenNvfp4Tensor, TensorGlobalNvfp4Tensor,
            )
            is_pertoken_path = isinstance(a0, PertokenNvfp4Tensor) and isinstance(B[0], TensorGlobalNvfp4Tensor)
            print(f"[GROUPED_GEMM ENTER] grad={grad} layout={layout} m_splits={m_splits} "
                  f"is_pertoken={is_pertoken_path} sm100={_is_sm100()} "
                  f"A[0] type={type(a0).__name__} B[0] type={type(B[0]).__name__}", flush=True)

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

        if _DEBUG_BGRAD_COUNT < _DEBUG_BGRAD_LIMIT and grad:
            for i, (a, b) in enumerate(zip(A, B)):
                def _tstat(t, name):
                    try:
                        # TE QuantizedTensor: has _rowwise_data
                        if hasattr(t, '_rowwise_data') and t._rowwise_data is not None:
                            rw = t._rowwise_data
                            amax = getattr(t, '_amax_rowwise', None)
                            amax_s = f"amax={amax.float().item():.4g}" if (amax is not None and amax.numel() > 0) else ""
                            # FP8 E4M3 NaN = byte 0x7F(+) or 0xFF(-): (byte & 0x7F)==0x7F
                            if rw.dtype == torch.uint8:
                                fp8_nan = ((rw & 0x7F) == 0x7F).any().item()
                                fp8_cnt = ((rw & 0x7F) == 0x7F).sum().item()
                                nan_s = f"fp8_nan={fp8_nan}(cnt={fp8_cnt})"
                            else:
                                f = rw.float()
                                nan_s = f"nan={f.isnan().any().item()} inf={f.isinf().any().item()}"
                            return f"{name}.rw={rw.shape} {nan_s} {amax_s}"
                        # TE QuantizedTensor: columnwise only
                        if hasattr(t, '_columnwise_data') and t._columnwise_data is not None:
                            cw = t._columnwise_data
                            if cw.dtype == torch.uint8:
                                fp8_nan = ((cw & 0x7F) == 0x7F).any().item()
                                fp8_cnt = ((cw & 0x7F) == 0x7F).sum().item()
                                nan_s = f"fp8_nan={fp8_nan}(cnt={fp8_cnt})"
                            else:
                                f = cw.float()
                                nan_s = f"nan={f.isnan().any().item()} inf={f.isinf().any().item()}"
                            return f"{name}.cw={cw.shape} {nan_s}"
                        # Plain or FP4 tensor
                        if isinstance(t, torch.Tensor):
                            if t.numel() == 0:
                                return f"{name}={t.shape}(empty)"
                            f = t.float() if t.dtype not in (torch.float32, torch.bfloat16, torch.float16) else t
                            return (f"{name}.shape={t.shape} nan={f.isnan().any().item()} "
                                    f"inf={f.isinf().any().item()} max={f.abs().max().item():.4g}")
                        return f"{name}=type:{type(t).__name__}"
                    except Exception as e:
                        return f"{name}=stat-err:{e}"
                print(f"[BGRAD PRE g={i}] layout={layout} "
                      f"{_tstat(a, 'A')} | {_tstat(b, 'B')}", flush=True)

        if grad:
            if layout == "NN":
                # dgrad = dy @ W → in_features = last dim of _columnwise_data.
                # size(0) is out_features; size(-1) is in_features.
                n_cols = B[0]._columnwise_data.size(-1)
                out = torch.empty(
                    [sum(m_splits), n_cols],
                    dtype=activation_dtype,
                    device=B[0]._columnwise_data.device,
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

        if _DEBUG_BGRAD_COUNT < _DEBUG_BGRAD_LIMIT and grad:
            out_tensors = out if isinstance(out, list) else [out]
            for i, ot in enumerate(out_tensors):
                if ot is None:
                    continue
                try:
                    if ot.numel() == 0:
                        print(f"[BGRAD POST i={i}] layout={layout} shape={ot.shape} (empty)", flush=True)
                        continue
                    print(f"[BGRAD POST i={i}] layout={layout} shape={ot.shape} "
                          f"nan={ot.isnan().any().item()} inf={ot.isinf().any().item()} "
                          f"max={ot.abs().max().item():.4g}", flush=True)
                except Exception as e:
                    print(f"[BGRAD POST i={i}] stat-err={e}", flush=True)
            _DEBUG_BGRAD_COUNT += 1

        if grad and layout == "NT":
            return out
        else:
            return out[0]
