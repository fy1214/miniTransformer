import torch
import fp4_gemm
from typing import Optional, List, Tuple, Union
from torch.profiler import record_function

from miniTransformer.quantiztion import quantization
from miniTransformer.quantiztion.quantization import GroupQuantizeResult
from miniTransformer.utils import utils
from miniTransformer.ops.nvfp4_grouped_cutlass_gemm import (
    V2GemmInputs,
    grouped_cutlass_gemm_v2,
)
from miniTransformer.ops.mxfp8_quantize import (
    mxfp8_grouped_quantize_chunked_hw,
    mxfp8_grouped_transpose_quantize,
)
from miniTransformer.ops.mxfp8_grouped_cutlass_gemm import (
    MxFP8V2GemmInputs,
    mxfp8_grouped_gemm_v2,
)

import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch import NVFP4Quantizer, MXFP8Quantizer
from transformer_engine.pytorch.cpp_extensions import split_quantize
from miniTransformer.ops._te_compat import general_gemm, general_grouped_gemm
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor, QuantizedTensorStorage


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


class Nvfp4Cursor2CutlassBackwardQuantizeOp(quantization.QuantizeOpBase):
    """
  NVFP4 quantize op for GroupLinear (cursor2 path):

  - Activation: per-token NVFP4 (group_quantize_nvfp4_pertoken_swizzled)
  - Weight: tensor-global NVFP4 (group_quantize_nvfp4_tensor_global_packed)
  - Forward grouped GEMM: CUTLASS v2 grouped FP4 GEMM (SM100+)

  Use with ``disable_backward_quant=True`` in GroupLinear when activation has no
  columnwise TE storage. Weight backward with columnwise still uses TE when
  ``return_transpose=True``.
    """

    def __init__(self):
        super().__init__()
        self.should_update_quantize_weights = False
        # Cached quantizers — rebuilt only when num_gemms changes.
        self._cached_num_gemms: int = 0
        self._w_quantizers_fp8: List = []   # rowwise=False, columnwise=True
        self._x_quantizers: List = []       # rowwise=False, columnwise=True
        self._dy_quantizers: List = []      # rowwise=True,  columnwise=True

    def _refresh_quantizer_cache(self, num_gemms: int) -> None:
        """Rebuild MXFP8 quantizer lists when num_gemms changes."""
        if self._cached_num_gemms == num_gemms:
            return
        self._cached_num_gemms = num_gemms
        fp8_col = dict(rowwise=False, columnwise=True, fp8_dtype=tex.DType.kFloat8E4M3)
        self._w_quantizers_fp8 = [MXFP8Quantizer(**fp8_col) for _ in range(num_gemms)]
        self._x_quantizers = [MXFP8Quantizer(**fp8_col) for _ in range(num_gemms)]
        self._dy_quantizers = [MXFP8Quantizer(**fp8_col) for _ in range(num_gemms)]

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
        Grouped quantize for ``GroupLinear`` forward (v2 CUTLASS GEMM).

        - Activation (X): per-token NVFP4 → ``extra_info`` with swizzled data/scales
        - Weight: tensor-global NVFP4 → ``extra_info`` with packed data/scales + n/k
        - Weight + ``return_transpose``: also produces TE FP8 columnwise tensors for dgrad
        """
        num_gemms = len(m_splits)
        self._refresh_quantizer_cache(num_gemms)

        if qparams.param_type == qparams.param_type.WEIGHT:
            from miniTransformer.ops.nvfp4_quantize import group_quantize_nvfp4_tensor_global_packed

            with record_function("sq_w_nvfp4_pack"):
                gw = group_quantize_nvfp4_tensor_global_packed(x, x.shape[1], skip_cute_data=True)
            n, k = x.shape[1], x.shape[2]

            if return_transpose:
                with record_function("sq_w_mxfp8_transpose_quantize"):
                    wt_data, wt_sf, wt_sf_off = mxfp8_grouped_transpose_quantize(
                        x.reshape(-1, x.shape[-1]), n, num_gemms
                    )
                extra_info_t = {
                    "data": wt_data,
                    "sf": wt_sf,
                    "sf_offsets": wt_sf_off,
                    "n": n,
                    "k": k,
                }
            else:
                extra_info_t = None

            wt_sentinel = [gw] if return_transpose else []
            return GroupQuantizeResult(
                data=[gw],
                data_t=wt_sentinel,
                scale=[gw],
                scale_t=wt_sentinel,
                tensor_scale=[gw],
                tensor_scale_t=wt_sentinel,
                extra_info={
                    "data":          gw.data,
                    "scale_flat":    gw.scale_flat,
                    "global_scale":  gw.global_scale,
                    "row_offsets":   gw.row_offsets_gpu,
                    "scale_offsets": gw.scale_offsets_gpu,
                    "n": n,
                    "k": k,
                    "extra_info_t": extra_info_t,
                },
            )

        if qparams.param_type == qparams.param_type.X:
            from miniTransformer.ops.nvfp4_quantize import group_quantize_nvfp4_pertoken_swizzled

            with record_function("sq_x_nvfp4_pertoken"):
                gq = group_quantize_nvfp4_pertoken_swizzled(x, m_splits)

            if return_transpose:
                with record_function("sq_x_split_quantize_fp8"):
                    quantize_tensor_list = tex.split_quantize(x, m_splits, self._x_quantizers)
            else:
                quantize_tensor_list = []
            return GroupQuantizeResult(
                data=[gq],
                scale=[gq],
                data_t=quantize_tensor_list,
                scale_t=quantize_tensor_list,
                tensor_scale=[gq],
                tensor_scale_t=quantize_tensor_list,
                extra_info={
                    "data":          gq.data,
                    "scale_flat":    gq.scale_flat,
                    "global_scale":  gq.global_scale,
                    "row_offsets":   gq.row_offsets_gpu,
                    "scale_offsets": gq.scale_offsets_gpu,
                },
            )

        # Dy: rowwise → mxfp8_grouped_quantize_chunked_hw (for dgrad MXFP8 CUTLASS)
        #     columnwise → TE split_quantize (for wgrad general_grouped_gemm NT)
        with record_function("sq_dy_mxfp8_chunked_hw"):
            dy_data, dy_sf, dy_rows, dy_sf_off = mxfp8_grouped_quantize_chunked_hw(
                x, m_splits
            )
        with record_function("sq_dy_te_columnwise"):
            quantize_tensor_list_col = tex.split_quantize(
                x, m_splits, self._dy_quantizers
            )
        extra_info_dy = {
            "rowwise": {
                "data": dy_data,
                "sf": dy_sf,
                "row_offsets": dy_rows,
                "sf_offsets": dy_sf_off,
            },
        }
        return GroupQuantizeResult(
            data=quantize_tensor_list_col,
            scale=quantize_tensor_list_col,
            data_t=quantize_tensor_list_col,
            scale_t=quantize_tensor_list_col,
            tensor_scale=quantize_tensor_list_col,
            tensor_scale_t=quantize_tensor_list_col,
            extra_info=extra_info_dy,
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
        extra_info_A: Optional[dict] = None,
        extra_info_B: Optional[dict] = None,
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
            and extra_info_A is not None
            and extra_info_B is not None
        ):
            ng = len(m_splits)
            n, k = extra_info_B["n"], extra_info_B["k"]
            prob_host = torch.tensor(
                [[m, n, k] for m in m_splits], dtype=torch.int32
            ).reshape(-1)
            inputs = V2GemmInputs(
                act_data=extra_info_A["data"],
                act_sf=extra_info_A["scale_flat"],
                act_ts=extra_info_A["global_scale"],
                a_rows=extra_info_A["row_offsets"],
                a_sf_offsets=extra_info_A["scale_offsets"],
                wgt_data=extra_info_B["data"],
                wgt_sf=extra_info_B["scale_flat"],
                wgt_ts=extra_info_B["global_scale"],
                b_rows=extra_info_B["row_offsets"],
                b_sf_offsets=extra_info_B["scale_offsets"],
                prob_host=prob_host,
                m_splits=m_splits,
            )
            return grouped_cutlass_gemm_v2(inputs, activation_dtype)

        if grad:
            if (
                layout == "NN"
                and _is_sm100()
                and extra_info_A is not None
                and extra_info_B is not None
                and "rowwise" in extra_info_A
                and "extra_info_t" in extra_info_B
                and extra_info_B["extra_info_t"] is not None
            ):
                # MXFP8 CUTLASS dgrad: dx = dy @ W via TN GEMM
                # TN: D[M, N_gemm] = A[M, K_gemm] @ B[N_gemm, K_gemm]^T
                # dgrad: dx[M, K_in] = dy[M, N_out] @ W^T[K_in, N_out]^T
                #   → A=dy (rowwise [total_M, N_out]), K_gemm=N_out
                #   → B=W^T (per-group [K_in, N_out] stacked), N_gemm=K_in
                dy_info = extra_info_A["rowwise"]
                wt_info = extra_info_B["extra_info_t"]
                ng = len(m_splits)
                n_out = wt_info["n"]   # N_out (K_gemm = reduction dim)
                k_in = wt_info["k"]    # K_in  (N_gemm = output cols)

                # prob_host: per group [M_g, N_gemm=K_in, K_gemm=N_out]
                prob_host = torch.tensor(
                    sum(([m, k_in, n_out] for m in m_splits), []),
                    dtype=torch.int32,
                )

                inputs = MxFP8V2GemmInputs(
                    act_data=dy_info["data"],
                    act_sf=dy_info["sf"],
                    a_rows=dy_info["row_offsets"],
                    a_sf_offsets=dy_info["sf_offsets"],
                    wgt_data=wt_info["data"],
                    wgt_sf=wt_info["sf"],
                    b_rows=torch.arange(0, (ng + 1) * k_in, k_in, dtype=torch.int32, device=dy_info["data"].device),
                    b_sf_offsets=wt_info["sf_offsets"],
                    prob_host=prob_host,
                )
                return mxfp8_grouped_gemm_v2(inputs, activation_dtype)

            if layout == "NN":
                # Fallback: TE general_grouped_gemm
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

        with record_function(f"gg_te_grouped_gemm_{layout}_grad{grad}"):
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