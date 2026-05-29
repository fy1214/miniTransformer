import os
import torch
import fp4_gemm
from typing import Optional, List, Tuple
from scipy.linalg import hadamard
from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor, QuantizedTensorStorage

from miniTransformer.ops.nvfp4_gemm_backend import dispatch_gemm_backend
from miniTransformer.ops.nvfp4_quartet_quantize import quant_fp4, rht128_quant_eden, rht128_requant, NVFP4QuantMode
from miniTransformer.quantiztion import quantization
from miniTransformer.utils import utils


def get_hadamard_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    return torch.tensor(
        hadamard(group_size) * group_size**-0.5,
        dtype=dtype,
        device=device,
        requires_grad=False,
    )


def rerotate_hadamard(hadamard_matrix):
    signs = torch.randint(
            0, 2, (hadamard_matrix.size(0),),
            device=hadamard_matrix.device,
            dtype=hadamard_matrix.dtype
        ) * 2 - 1
    return hadamard_matrix * signs[None, :] # NOTE: rerotate along last dim, inner dim for TN GEMM


@torch.compile(dynamic=False)
def abs_max(x):
    return x.abs().max().to(torch.float32)


def unblock(blocked_scales, rows, cols):
    n_row_blocks, n_col_blocks = rows // 128, (cols // 16) // 4
    rearranged = blocked_scales.reshape(-1, 32, 4, 4)
    rearranged = rearranged.permute(0, 2, 1, 3).reshape(-1, n_col_blocks, 128, 4)
    rearranged = rearranged.permute(0, 2, 1, 3)
    # Reverse: view(n_row_blocks, 128, n_col_blocks, 4)
    return rearranged.reshape(n_row_blocks * 128, n_col_blocks * 4)


_NVFP4_SF_VEC_SIZE = 16
_NVFP4_FP4_MAX = 6.0
_NVFP4_FP8_MAX = 448.0
_GROUPED_GEMM_CUTEDSL_ENV = "MINITRANSFORMER_ENABLE_GROUPED_GEMM_CUTEDSL"


def is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)


def is_grouped_gemm_cutedsl_enabled() -> bool:
    value = os.environ.get(_GROUPED_GEMM_CUTEDSL_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _make_fp4_tensor_for_cutedsl(
    data_uint8: torch.Tensor, mode0: int, K: int, device: torch.device
) -> torch.Tensor:
    buf = torch.empty(mode0 * K, dtype=torch.int8, device=device)
    buf[: mode0 * K // 2].copy_(data_uint8.reshape(-1).view(torch.int8))
    return buf.view(mode0, K, 1)


def _make_cute_sf(
    sf_2d: torch.Tensor, mn: int, sf_k: int, l: int = 1
) -> torch.Tensor:
    from miniTransformer.ops.nvfp4_qgemm import create_cute_scale_factor_tensor

    if sf_2d.dtype == torch.uint8:
        sf_fp8 = sf_2d[:mn, :sf_k].contiguous().view(torch.float8_e4m3fn)
    else:
        sf_fp8 = sf_2d[:mn, :sf_k]
    ref = sf_fp8.cpu().unsqueeze(-1)
    return create_cute_scale_factor_tensor(l, mn, sf_k, ref)


def grouped_gemm_cutedsl(
    A: list,
    B: list,
    m_splits: list,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    from miniTransformer.ops.cuteDSL.nvfp4.group_gemm_blockscaled_nvfp4_mxfp_cute import (
        Sm100GroupedBlockScaledGemmKernel,
        GroupedBlockScaledProfile,
    )

    device = A[0]._rowwise_data.device
    K = A[0]._rowwise_data.shape[1] * 2
    sf_k = K // _NVFP4_SF_VEC_SIZE
    alpha_denom = _NVFP4_FP8_MAX * _NVFP4_FP8_MAX * _NVFP4_FP4_MAX * _NVFP4_FP4_MAX

    abc_torch, sfasfb_torch, problem_sizes, alphas = [], [], [], []
    for i, m_i in enumerate(m_splits):
        n_i = B[i]._rowwise_data.shape[0]

        a_fp4 = _make_fp4_tensor_for_cutedsl(A[i]._rowwise_data, m_i, K, device)
        b_fp4 = _make_fp4_tensor_for_cutedsl(B[i]._rowwise_data, n_i, K, device)
        c_out = torch.empty((m_i, n_i), dtype=out_dtype, device=device)

        sfa = _make_cute_sf(A[i]._rowwise_scale_inv, m_i, sf_k)
        sfb = _make_cute_sf(B[i]._rowwise_scale_inv, n_i, sf_k)

        abc_torch.append([a_fp4, b_fp4, c_out])
        sfasfb_torch.append([sfa, sfb])
        problem_sizes.append((m_i, n_i, K, 1))

        amax_a = A[i]._amax_rowwise.float().item()
        amax_b = B[i]._amax_rowwise.float().item()
        alphas.append(amax_a * amax_b / alpha_denom)

    try:
        kernel = Sm100GroupedBlockScaledGemmKernel(profile=GroupedBlockScaledProfile.NVFP4)
    except AttributeError as exc:
        raise NotImplementedError(
            "grouped_gemm_cutedsl requires newer cuteDSL/cutlass with required arch helpers"
        ) from exc

    results = kernel.run_from_grouped_inputs(
        abc_tensors={"torch": abc_torch},
        sfasfb_tensors={"torch": sfasfb_torch},
        problem_sizes=problem_sizes,
        out_dtype=out_dtype,
        accumulate=False,
    )
    corrected = [r.float() * a for r, a in zip(results, alphas)]
    return torch.cat(corrected, dim=0).to(out_dtype)


class Nvfp4QuartetQuantizeOp(quantization.QuantizeOpBase):

    def __init__(self, for_over_six=True):
        super().__init__()
        self.should_update_quantize_weights = True
        self.forward_scale_override = 1.0
        self.backward_scale_override = (17 / 16) * 0.93
        self.mode = NVFP4QuantMode.FOUR_SIX if for_over_six else NVFP4QuantMode.RNE
        self.scratch_amax = torch.empty((), dtype=torch.int32, device='cuda')
        self.had = get_hadamard_matrix(128, torch.bfloat16, 'cuda')
        # Re-randomize the rotation
        self.had_t = rerotate_hadamard(self.had)

    @property
    def supports_allgather(self) -> bool:
        return False

    def update_had_t(self):
        self.had_t = rerotate_hadamard(self.had)

    def _quantize(
        self,
        x: torch.Tensor,
        qparams: quantization.QParams,
        return_transpose: bool,
        with_2d_quantization: bool
    ):
        global_amax = abs_max(x)

        with utils.nvtx_annotate("Quant", color="yellow"):
            if qparams.param_type == quantization.ParamType.Y:
                x_ht_fp4, x_ht_ms, x_ht_ts = rht128_quant_eden(
                    x=x,
                    h=self.had_t[:16, :],
                    scale_override=self.backward_scale_override,
                    scratch_amax=self.scratch_amax
                )
            else:
                quant_weight = quant_fp4(x, amax=global_amax, scale_override=self.forward_scale_override, mode=self.mode)

            if return_transpose:
                if qparams.param_type == quantization.ParamType.Y:
                    xt_ht_fp4, xt_ht_ms, xt_ht_ts = rht128_quant_eden(
                        x=x,
                        h=self.had_t[:16, :],
                        scale_override=self.backward_scale_override,
                        transpose=True,
                        scratch_amax=self.scratch_amax
                    )
                else:
                    xt_ht_fp4, xt_ht_ms, xt_ht_ts = rht128_requant(
                        x=quant_weight.fp4,
                        x_group_scales=quant_weight.micro_scales,
                        x_tensor_scale=quant_weight.tensor_scale,
                        h=self.had_t[:16, :],
                        scale_override=self.backward_scale_override,
                        scratch_amax=self.scratch_amax
                    )

        nvfp4_tensor = quantization.QuantizeResult(
            data=x_ht_fp4 if qparams.param_type == quantization.ParamType.Y else quant_weight.fp4,
            scale=x_ht_ms if qparams.param_type == quantization.ParamType.Y else quant_weight.micro_scales,
            data_t=xt_ht_fp4,
            scale_t=xt_ht_ms,
            tensor_scale=x_ht_ts if qparams.param_type == quantization.ParamType.Y else quant_weight.tensor_scale,
            tensor_scale_t=xt_ht_ts
        )

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

        # Y is the last one quantize, so we update had_t here
        if qparams.param_type == quantization.ParamType.Y:
            self.update_had_t()

        return nvfp4_tensor

    def dequantize(self, x_e2m1: torch.Tensor, x_e4m3: torch.Tensor, alpha: float):
        device = x_e2m1.device

        x_e2m1_i32 = x_e2m1.view(dtype=torch.uint8).to(dtype=torch.int32)
        x_e2m1_unpacked = torch.stack(
            [x_e2m1_i32 & 0xF, (x_e2m1_i32 >> 4) & 0xF], dim=-1
        ).flatten(start_dim=-2)

        grid_dq = torch.tensor(
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
            dtype=torch.float32,
            device=device,
        )
        x_fp4_dq = grid_dq[x_e2m1_unpacked]

        scales_dq = x_e4m3.to(torch.float32)
        scales_dq = unblock(scales_dq, x_e2m1.shape[0], x_e2m1.shape[1] * 2)
        x_dq = (x_fp4_dq.unflatten(dim=-1, sizes=(-1, 16)) * scales_dq[..., None]).flatten(
            start_dim=-2
        ) * alpha
        return x_dq.to(torch.bfloat16)

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
        custom_gemm_backend: str = "cublas",
    ) -> torch.Tensor:

        alpha = 1.0 * global_amax_qw
        backend = dispatch_gemm_backend(custom_gemm_backend, "nvfp4")
        out = backend.gemm(
            qx, qw, out_dtype, sx, sw,
            bias, out, grad, accumulate,
            layout, alpha, None, None, True
        )

        return out

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
        grad: bool = False,
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
        is_quantized = isinstance(a0, (QuantizedTensor, QuantizedTensorStorage))

        if (
            is_grouped_gemm_cutedsl_enabled()
            and is_quantized
            and not grad
            and not gelu
            and not accumulate
            and layout == "TN"
            and is_sm100()
        ):
            try:
                # Keep compatibility with existing tests/tools that patch the
                # legacy nvfp4_quantize_op entrypoint.
                from miniTransformer.quantiztion.nvfp4_quantize_op import _grouped_gemm_cutedsl

                return _grouped_gemm_cutedsl(A, B, m_splits, activation_dtype)
            except (TypeError, NotImplementedError):
                pass

        if grad:
            if layout == "NN":
                if is_quantized:
                    n_cols = B[0]._rowwise_data.size(1) * 2
                else:
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
            accumulate=accumulate,
        )

        if grad and layout == "NT":
            return out
        return out[0]