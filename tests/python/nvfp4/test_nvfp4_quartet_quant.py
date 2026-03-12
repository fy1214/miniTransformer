from collections import namedtuple
import os
import sys
import time
import pytest
import torch

import miniTransformer.ops.quartet_quant as quartet2
from miniTransformer.quantiztion.nvfp4_quartet_op import Nvfp4QuartetQuantizeOp
from miniTransformer.quantiztion.quantization import QParams, ParamType

torch.random.manual_seed(42)


def unblock(blocked_scales, rows, cols):
    n_row_blocks, n_col_blocks = rows // 128, (cols // 16) // 4
    rearranged = blocked_scales.reshape(-1, 32, 4, 4)
    rearranged = rearranged.permute(0, 2, 1, 3).reshape(-1, n_col_blocks, 128, 4)
    rearranged = rearranged.permute(0, 2, 1, 3)
    # Reverse: view(n_row_blocks, 128, n_col_blocks, 4)
    return rearranged.reshape(n_row_blocks * 128, n_col_blocks * 4)


def _dq_fp4(x_e2m1: torch.Tensor, x_e4m3: torch.Tensor, alpha: float):
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


@torch.compile(fullgraph=True)
def quant_fp4(x, four_over_six=False):
    amax = torch.max(torch.abs(x)).to(torch.float32)
    q, s, t = quartet2.quant_fp4(x, scale_override=1.0, amax=amax, mode=four_over_six)
    return _dq_fp4(q, s, t)

def quant_fp4_ref(x, four_over_six=False):
    ref_op = Nvfp4QuartetQuantizeOp()
    param = QParams(
        output_dtype=torch.uint8,
        scaling_dtype=torch.float8_e4m3fn,
        quant_tile_shape=(1, 16),
        with_2d_quantization=False,
        with_rht=True,
        param_type=ParamType.X
    )
    result = ref_op.quantize(
        x,
        param,
        True,
        False,
    )
    return _dq_fp4(result.data, result.scale, result.tensor_scale)


@torch.compile(fullgraph=True)
def quant_with_transform(x, transpose=False):
    e = torch.eye(128, dtype=torch.bfloat16, device=x.device)
    h = torch.empty((128, 128), dtype=torch.bfloat16, device=x.device)
    #  output is swizzled, so we need to prepare h accordingly
    for j in range(0, 128, 16):
        order = [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]
        for i in range(16):
            h[:, j + i] = e[:, j+order[i]]
    q, s, g = quartet2.quant_had_eden(x=x, h=h, transpose=transpose)
    return _dq_fp4(q, s, g)


@pytest.mark.parametrize("m,n", [(1024, 1024), (1024, 4096), (4096, 2048)])
@pytest.mark.parametrize("four_over_six,expected", [(False, 3.38), (True, 3.52)])
def test_nvfp4_quarter_fp4(m, n, four_over_six, expected):
    # Create a sample input tensor
    x = torch.randn((m, n), dtype=torch.bfloat16, device="cuda")

    dq = quant_fp4(x.to(torch.bfloat16), four_over_six)
    dq_ref = quant_fp4_ref(x.to(torch.bfloat16), four_over_six)
    quad_err = ((x - dq).pow(2).sum(dim=-1) / x.pow(2).sum(dim=-1)).mean()
    eff_bitwidth = (-torch.log2(quad_err) / 2).item()

    assert eff_bitwidth > expected


@pytest.mark.parametrize("m,n", [(1024, 1024), (1024, 4096), (4096, 2048)])
@pytest.mark.parametrize("transpose,expected", [(False, 3.37), (True, 3.37)])
def test_eden_identity(m, n, transpose, expected):
    x = torch.randn((4096, 2048), device="cuda", dtype=torch.bfloat16)
    dq = quant_with_transform(x.to(torch.bfloat16), transpose=transpose)
    if transpose:
        x = x.transpose(-1, -2)
    quad_err = ((x - dq).pow(2).sum(dim=-1) / x.pow(2).sum(dim=-1)).mean()
    eff_bitwidth = (-torch.log2(quad_err) / 2).item()

    assert eff_bitwidth > expected


if __name__ == "__main__":
   #pytest.main([__file__, "-v", "-s"])
   test_nvfp4_quarter_fp4(1024, 4096, True, 3.52)