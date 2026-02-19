import math
import torch
import fp4_gemm
from typing import Optional
from scipy.linalg import hadamard

from miniTransformer.ops.nvfp4_qgemm import swizzle_scale
from miniTransformer.ops.quarter_quant import quant_fp4, dequant_tp_had_eden, quant_had_eden
from miniTransformer.quantiztion import quantization
from miniTransformer.utils import utils

from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm


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


@torch.library.custom_op("clover::fp4_mm", mutates_args=())
def _fp4_mm(y: torch.Tensor, x_fp4: torch.Tensor, w_fp4: torch.Tensor, x_mx: torch.Tensor, w_mx: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return fp4_gemm.cutlass_gemm(
        y, x_fp4, w_fp4,
        x_mx, w_mx, alpha)


@_fp4_mm.register_fake
def _fp4_mm_fake(y: torch.Tensor, x_fp4: torch.Tensor, w_fp4: torch.Tensor, x_mx: torch.Tensor, w_mx: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return torch.empty((x_fp4.shape[0], w_fp4.shape[0]), device=x_fp4.device, dtype=torch.bfloat16)


@torch.compile(dynamic=False)
def abs_max(x):
    return x.abs().max().to(torch.float32)


class Nvfp4QuarterQuantizeOp(quantization.QuantizeOpBase):

    def __init__(self):
        super().__init__()

        self.forward_scale_override = 1.0
        self.backward_scale_override = (17 / 16) * 0.93
        self.four_over_six = True
        self.scratch_amax = torch.empty((), dtype=torch.int32, device='cuda')

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
        global_amax = abs_max(x)
        had = get_hadamard_matrix(128, x.dtype, x.device)
        # Re-randomize the rotation
        had_t = rerotate_hadamard(had)

        with utils.nvtx_annotate("Quant", color="yellow"):
            if qparams.param_type == quantization.ParamType.Y:
                x_ht_fp4, x_ht_ms, x_ht_ts = quant_had_eden(
                    x=x,
                    h=had_t,
                    scale_override=self.backward_scale_override,
                    scratch_amax=self.scratch_amax
                )
            else:
                quant_weight = quant_fp4(x, amax=global_amax, scale_override=self.forward_scale_override, four_over_six=self.four_over_six)

            if return_transpose:
                if qparams.param_type == quantization.ParamType.Y:
                    xt_ht_fp4, xt_ht_ms, xt_ht_ts = quant_had_eden(
                        x=x,
                        h=had_t,
                        scale_override=self.backward_scale_override,
                        transpose=True,
                        scratch_amax=self.scratch_amax
                    )
                else:
                    xt_ht_fp4, xt_ht_ms, xt_ht_ts = dequant_tp_had_eden(
                        x=quant_weight.fp4,
                        x_group_scales=quant_weight.micro_scales,
                        x_tensor_scale=quant_weight.tensor_scale,
                        h=had_t,
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
        scales_dq = scales_dq.unflatten(dim=-1, sizes=(-1, 1))
        x_dq = (x_fp4_dq.unflatten(dim=-1, sizes=(-1, 16)) * scales_dq).flatten(
            start_dim=-2
        ) * alpha
        return x_dq

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
    ) -> torch.Tensor:

        # forward qx qw  [m, k] [n, k] TN   => out [m, n]
        # dgrad qdy qw_t [m, n] [k, n] NN   => out [m, k]
        # wgrad qdy_t qx_t [n, m] [k, m] NT => out [n, k]
        m, n = qx.shape[0], qw.shape[0]

        y = torch.empty([m, n], dtype=torch.bfloat16, device=qx.device)

        # GEMM for FP4
        sx_sl = swizzle_scale(sx, qx.shape[0], qx.shape[1] * 2)
        sw_sl = swizzle_scale(sw, qw.shape[0], qw.shape[1] * 2)
        _fp4_mm(y, qx, qw, sx_sl, sw_sl, global_amax_qx * global_amax_qw)

        #TODO: fuse accumulate into GEMM
        if accumulate:
            assert out is not None, "Output tensor must be provided for accumulation."
            out = (out.to(torch.float32) + y).to(out_dtype)
            return out
        else:
            y = y.to(out_dtype)
            return y