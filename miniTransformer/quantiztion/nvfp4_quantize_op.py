import math
import torch
import fp4_gemm
from typing import Optional, List, Tuple

from miniTransformer.quantiztion import quantization
from miniTransformer.quantiztion.quantization import GroupQuantizeResult
from miniTransformer.utils import utils

import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch import NVFP4Quantizer, MXFP8Quantizer
from transformer_engine.pytorch.cpp_extensions import split_quantize
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm, general_grouped_gemm
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor, QuantizedTensorStorage

from miniTransformer.ops.cuteDSL.nvfp4.blockwise_group_gemm_cute import (
    Sm100GroupedBlockScaledGemmKernel,
    GroupedBlockScaledProfile,
)
from miniTransformer.ops.nvfp4_qgemm import create_cute_scale_factor_tensor

nvfp4_output_t = torch.uint8
nvfp4_scale_t = torch.float8_e4m3fn

use_cpp_allocator = False

_NVFP4_SF_VEC_SIZE = 16


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)


def _make_fp4_tensor_for_cutedsl(
    data_uint8: torch.Tensor, mode0: int, K: int, device: torch.device
) -> torch.Tensor:
    """Pack (mode0, K//2) uint8 FP4 data into an (mode0, K, 1) int8 tensor.

    _infer_strides_from_abc_tensors uses shape[1] to compute the CUTLASS stride in
    Float4E2M1FN element units. We need shape[1]=K so it produces stride (K, 1).
    The kernel reads only the first mode0*K//2 bytes via (K,1) Float4E2M1FN strides.
    """
    buf = torch.empty(mode0 * K, dtype=torch.int8, device=device)
    buf[: mode0 * K // 2].copy_(data_uint8.reshape(-1).view(torch.int8))
    return buf.view(mode0, K, 1)


def _make_cute_sf(
    sf_2d: torch.Tensor, mn: int, sf_k: int, l: int = 1
) -> torch.Tensor:
    """Convert _rowwise_scale_inv to CuTe MMA layout on GPU.

    Production NVFP4Tensor stores _rowwise_scale_inv as uint8 bytes encoding
    float8_e4m3fn decode_scale = block_max * FP8_MAX / global_amax.
    Mock tensors (float8_e4m3fn dtype) store true block scales (block_max / FP4_MAX).
    The kernel applies these as per-block multipliers; callers using production
    tensors must apply global alpha = amax_a * amax_b / (FP8_MAX^2 * FP4_MAX^2).
    """
    if sf_2d.dtype == torch.uint8:
        sf_fp8 = sf_2d[:mn, :sf_k].contiguous().view(torch.float8_e4m3fn)
    else:
        sf_fp8 = sf_2d[:mn, :sf_k]
    ref = sf_fp8.cpu().unsqueeze(-1)  # (mn, sf_k, l) on CPU
    return create_cute_scale_factor_tensor(l, mn, sf_k, ref)


_NVFP4_FP4_MAX = 6.0
_NVFP4_FP8_MAX = 448.0


def _grouped_gemm_cutedsl(
    A: list,
    B: list,
    m_splits: list,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Dispatch grouped NVFP4 GEMM to Sm100GroupedBlockScaledGemmKernel.

    _rowwise_scale_inv stores decode_scale = block_max * FP8_MAX / global_amax.
    The kernel computes output = sum(fp4_a * decode_a * fp4_b * decode_b), which
    equals true_output * (FP8_MAX^2 * FP4_MAX^2) / (amax_a * amax_b).
    We correct with per-expert alpha = amax_a * amax_b / (FP8_MAX^2 * FP4_MAX^2).
    """
    device = A[0]._rowwise_data.device
    K = A[0]._rowwise_data.shape[1] * 2  # packed K//2 uint8 → K FP4
    sf_k = K // _NVFP4_SF_VEC_SIZE
    alpha_denom = _NVFP4_FP8_MAX * _NVFP4_FP8_MAX * _NVFP4_FP4_MAX * _NVFP4_FP4_MAX

    abc_torch, sfasfb_torch, problem_sizes, alphas = [], [], [], []
    for i, m_i in enumerate(m_splits):
        N_i = B[i]._rowwise_data.shape[0]

        a_fp4 = _make_fp4_tensor_for_cutedsl(A[i]._rowwise_data, m_i, K, device)
        b_fp4 = _make_fp4_tensor_for_cutedsl(B[i]._rowwise_data, N_i, K, device)
        c_out = torch.empty((m_i, N_i), dtype=out_dtype, device=device)

        sfa = _make_cute_sf(A[i]._rowwise_scale_inv, m_i, sf_k)
        sfb = _make_cute_sf(B[i]._rowwise_scale_inv, N_i, sf_k)

        abc_torch.append([a_fp4, b_fp4, c_out])
        sfasfb_torch.append([sfa, sfb])
        problem_sizes.append((m_i, N_i, K, 1))

        amax_a = A[i]._amax_rowwise.float().item()
        amax_b = B[i]._amax_rowwise.float().item()
        alphas.append(amax_a * amax_b / alpha_denom)

    kernel = Sm100GroupedBlockScaledGemmKernel(profile=GroupedBlockScaledProfile.NVFP4)
    results = kernel.run_from_grouped_inputs(
        abc_tensors={"torch": abc_torch},
        sfasfb_tensors={"torch": sfasfb_torch},
        problem_sizes=problem_sizes,
        out_dtype=out_dtype,
        accumulate=False,
    )
    corrected = [r.float() * a for r, a in zip(results, alphas)]
    return torch.cat(corrected, dim=0).to(out_dtype)


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
        M, N = x.shape
        device = x.device
        x_dtype = x.dtype

        nvfp4_quantizer = NVFP4Quantizer(
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
        If param type is weight, the shape is [num_gemms, n, k].
        If param type is X, the shape is [m_sum, k], m_sum should split into m_splits.
        If param type is Y, the shape is [m_sum, n].
        """
        num_gemms = len(m_splits)
        quantizers = []
        for i in range(num_gemms):
            x_quantizer = NVFP4Quantizer(
                rowwise=True,
                columnwise=return_transpose,
                with_amax_reduction=False,
                amax_reduction_group=None,
                with_rht=qparams.with_rht,
                with_post_rht_amax=qparams.with_rht,
                with_2d_quantization=qparams.with_2d_quantization,
                stochastic_rounding=qparams.stochastic_rounding
            )
            quantizers.append(x_quantizer)

        if qparams.param_type == qparams.param_type.WEIGHT:
            quantize_tensor_list = []
            for i in range(num_gemms):
                quantize_tensor_list.append(quantizers[i](x[i]))
        else:
            quantize_tensor_list = tex.split_quantize(
                x,
                m_splits,
                quantizers,
            )

        return GroupQuantizeResult(
            data=quantize_tensor_list,
            data_t=quantize_tensor_list,
            scale=quantize_tensor_list,
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

        # 判断输入是量化 tensor 还是 BF16 plain tensor
        a0 = A[0]
        is_quantized = isinstance(a0, (QuantizedTensor, QuantizedTensorStorage))

        # Dispatch to cuteDSL Sm100 kernel for NVFP4 TN fprop on Blackwell GPUs
        if (is_quantized and not grad and not gelu and not accumulate
                and layout == "TN" and _is_sm100()):
            try:
                return _grouped_gemm_cutedsl(A, B, m_splits, activation_dtype)
            except (TypeError, NotImplementedError):
                pass  # fall through to TE path

        if grad:
            if layout == "NN":
                if is_quantized:
                    n_cols = B[0]._rowwise_data.size(1) * 2
                else:
                    # BF16: B[i] shape [n, k]，dgrad shape [m_sum, k]
                    n_cols = B[0].size(1)
                out = torch.empty(
                    [sum(m_splits), n_cols],
                    dtype=activation_dtype,
                    device=B[0].device,
                )
                out = [out]
        else:
            if is_quantized:
                n_cols = B[0]._rowwise_data.size(0)
            else:
                # BF16 fprop: B[i] shape [n, k]，output shape [m_sum, n]
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
