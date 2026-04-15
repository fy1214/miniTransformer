"""
TE Linear NVFP4 / MXFP8 Tensor Parallel (TP) Accuracy Tests
=============================================================

Tests te.Linear with NVFP4 and MXFP8 recipes under Tensor Parallel, split into
individual forward / backward test functions for easy diagnosis.

Test matrix:
  test_column_parallel_forward
  test_column_parallel_backward_dgrad
  test_column_parallel_backward_wgrad
  test_row_parallel_forward
  test_row_parallel_backward_dgrad
  test_row_parallel_backward_wgrad

Strategy:
  - Build baseline te.Linear (tp=1), broadcast weight from rank 0 to all ranks.
  - Build TP te.Linear (tp>1), copy local weight shard from the synced baseline.
  - Run under te.autocast(recipe=NVFP4BlockScaling).
  - Compare each rank's local result against the matching slice of the baseline.

Run with:
    torchrun --nproc_per_node=2 test_te_linear_nvfp4_tp.py            # SP on
    torchrun --nproc_per_node=2 test_te_linear_nvfp4_tp.py --no-sp    # SP off
"""

import argparse
import os
import traceback

import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.quantization import is_mxfp8_available


# ==================== Recipe ====================

def make_nvfp4_recipe() -> recipe.NVFP4BlockScaling:
    """标准 NVFP4 recipe，input/weight/grad 均使用默认 QParams。"""
    r = recipe.NVFP4BlockScaling()
    r.fp4_quant_fwd_inp    = recipe.QParams()
    r.fp4_quant_fwd_weight = recipe.QParams()
    r.fp4_quant_bwd_grad   = recipe.QParams()
    return r


def make_mxfp8_recipe(fp8_format: recipe.Format = recipe.Format.E4M3) -> recipe.MXFP8BlockScaling:
    """
    标准 MXFP8 recipe。

    MXFP8 采用 block-wise（每 32 个元素共享一个 E8M0 scale）量化策略，
    不需要像 NVFP4 那样手动配置 QParams，直接实例化即可。

    参数
    ----
    fp8_format : recipe.Format
        FP8 数据格式，支持 Format.E4M3（默认，精度更高）
        和 Format.HYBRID（前向 E4M3，反向 E5M2，与 DelayedScaling 一致）。

    用法
    ----
    mxfp8_recipe = make_mxfp8_recipe()
    with te.autocast(enabled=True, recipe=mxfp8_recipe):
        y = linear(x)
    """
    return recipe.MXFP8BlockScaling(fp8_format=fp8_format)


# 可用性检查（供调用方跳过不支持的测试）
nvfp4_available, nvfp4_unavail_reason = te.is_nvfp4_available(return_reason=True)
mxfp8_available, mxfp8_unavail_reason = is_mxfp8_available(return_reason=True)


# ==================== Distributed Utilities ====================

def initialize_distributed():
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
        )
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def create_tp_group(tp_size: int):
    world_size = dist.get_world_size()
    if tp_size > world_size:
        raise ValueError(f"tp_size ({tp_size}) > world_size ({world_size})")
    tp_group = dist.new_group(ranks=list(range(tp_size)))
    return tp_group if dist.get_rank() < tp_size else None


def broadcast_weight(linear, tp_group):
    """Broadcast weight (and bias) from rank 0 to all ranks in tp_group."""
    dist.broadcast(linear.weight.data, src=0, group=tp_group)
    if hasattr(linear, "bias") and linear.bias is not None and linear.bias.numel() > 0:
        dist.broadcast(linear.bias.data, src=0, group=tp_group)


def broadcast_tensor(t: torch.Tensor, tp_group) -> torch.Tensor:
    dist.broadcast(t, src=0, group=tp_group)
    return t


# ==================== Module Builders ====================

def build_baseline(
    in_features, out_features, parallel_mode, bias, params_dtype, seed
) -> te.Linear:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return te.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        params_dtype=params_dtype,
        parallel_mode=parallel_mode,
        tp_size=1,
        device="cuda",
    )


def build_tp(
    in_features, out_features, parallel_mode, bias, params_dtype,
    tp_size, tp_group, sequence_parallel, seed
) -> te.Linear:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return te.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        params_dtype=params_dtype,
        parallel_mode=parallel_mode,
        tp_size=tp_size,
        tp_group=tp_group,
        sequence_parallel=sequence_parallel,
        device="cuda",
    )


# ==================== Weight Shard Copy ====================

def copy_column_shard(linear_tp, linear_baseline, rank, tp_size):
    """Column parallel: weight[out/tp, in]. Copy row-slice from baseline.

    Use nn.Parameter replacement to avoid QuantizedTensor cache issues.
    """
    out_full = linear_baseline.weight.shape[0]
    chunk = out_full // tp_size
    shard = linear_baseline.weight.data[rank * chunk:(rank + 1) * chunk, :].contiguous().clone()
    linear_tp.weight = torch.nn.Parameter(shard)
    # Sanity check: verify the weight was actually set
    assert linear_tp.weight.data.norm() > 0, \
        f"rank={rank} copy_column_shard produced zero weight! " \
        f"baseline norm={linear_baseline.weight.data.norm():.4e} " \
        f"shard shape={shard.shape} shard norm={shard.norm():.4e}"


def copy_row_shard(linear_tp, linear_baseline, rank, tp_size):
    """Row parallel: weight[out, in/tp]. Copy col-slice from baseline.

    Use nn.Parameter replacement to avoid QuantizedTensor cache issues.
    """
    in_full = linear_baseline.weight.shape[1]
    chunk = in_full // tp_size
    shard = linear_baseline.weight.data[:, rank * chunk:(rank + 1) * chunk].contiguous().clone()
    linear_tp.weight = torch.nn.Parameter(shard)
    # Sanity check: verify the weight was actually set
    assert linear_tp.weight.data.norm() > 0, \
        f"rank={rank} copy_row_shard produced zero weight! " \
        f"baseline norm={linear_baseline.weight.data.norm():.4e} " \
        f"shard shape={shard.shape} shard norm={shard.norm():.4e}"


# ==================== Shared Setup Helper ====================

def _setup_column(tp_size, sequence_parallel, in_features, out_features,
                  total_batch, bias, params_dtype, seed, tp_group):
    """
    Build modules, sync weights, prepare inputs and grad tensors for
    column parallel tests. Returns a dict of all needed objects.
    """
    rank = dist.get_rank()

    baseline = build_baseline(in_features, out_features, "column", bias, params_dtype, seed)
    if tp_size > 1:
        broadcast_weight(baseline, tp_group)

    tp_linear = build_tp(in_features, out_features, "column", bias, params_dtype,
                         tp_size, tp_group, sequence_parallel, seed)
    if tp_size > 1:
        copy_column_shard(tp_linear, baseline, rank, tp_size)

    torch.manual_seed(seed + 100)
    inp_full = torch.randn(total_batch, in_features, device="cuda", dtype=params_dtype)
    if tp_size > 1:
        broadcast_tensor(inp_full, tp_group)

    torch.manual_seed(seed + 200)
    grad_full = torch.randn(total_batch, out_features, device="cuda", dtype=params_dtype)
    if tp_size > 1:
        broadcast_tensor(grad_full, tp_group)

    chunk_out = out_features // tp_size

    # inp for TP: sequence shard (SP) or full (no SP)
    if sequence_parallel and tp_size > 1:
        chunk_seq = total_batch // tp_size
        inp_tp_data = inp_full[rank * chunk_seq:(rank + 1) * chunk_seq, :].contiguous()
    else:
        inp_tp_data = inp_full

    # grad for TP output
    # column parallel 无论 SP 是否开启，out_tp shape 均为 [total_batch, out//tp]
    # （SP 影响 input AllGather，不影响 output shape）
    grad_tp = grad_full[:, rank * chunk_out:(rank + 1) * chunk_out].contiguous()

    # grad for baseline (tp=1): out_baseline shape [total_batch, out_features]
    # 只激活 col_shard 对应的列，其余置 0，保证与 tp dgrad 数学等价
    grad_baseline = torch.zeros_like(grad_full)
    grad_baseline[:, rank * chunk_out:(rank + 1) * chunk_out] = \
        grad_full[:, rank * chunk_out:(rank + 1) * chunk_out]

    return dict(
        baseline=baseline, tp_linear=tp_linear,
        inp_full=inp_full, inp_tp_data=inp_tp_data,
        grad_full=grad_full, grad_tp=grad_tp, grad_baseline=grad_baseline,
        chunk_out=chunk_out,
    )


def _setup_row(tp_size, sequence_parallel, in_features, out_features,
               total_batch, bias, params_dtype, seed, tp_group):
    """
    Build modules, sync weights, prepare inputs and grad tensors for
    row parallel tests.
    """
    rank = dist.get_rank()

    baseline = build_baseline(in_features, out_features, "row", bias, params_dtype, seed)
    if tp_size > 1:
        broadcast_weight(baseline, tp_group)

    tp_linear = build_tp(in_features, out_features, "row", bias, params_dtype,
                         tp_size, tp_group, sequence_parallel, seed)
    if tp_size > 1:
        copy_row_shard(tp_linear, baseline, rank, tp_size)

    torch.manual_seed(seed + 100)
    inp_full = torch.randn(total_batch, in_features, device="cuda", dtype=params_dtype)
    if tp_size > 1:
        broadcast_tensor(inp_full, tp_group)

    torch.manual_seed(seed + 200)
    grad_full = torch.randn(total_batch, out_features, device="cuda", dtype=params_dtype)
    if tp_size > 1:
        broadcast_tensor(grad_full, tp_group)

    chunk_in = in_features // tp_size

    # inp for TP: always shard along in_features (dim 1)
    if tp_size > 1:
        inp_tp_data = inp_full[:, rank * chunk_in:(rank + 1) * chunk_in].contiguous()
    else:
        inp_tp_data = inp_full

    # grad for TP output
    if sequence_parallel and tp_size > 1:
        chunk_seq = total_batch // tp_size
        grad_tp = grad_full[rank * chunk_seq:(rank + 1) * chunk_seq, :].contiguous()
    else:
        grad_tp = grad_full.contiguous()

    return dict(
        baseline=baseline, tp_linear=tp_linear,
        inp_full=inp_full, inp_tp_data=inp_tp_data,
        grad_full=grad_full, grad_tp=grad_tp,
        chunk_in=chunk_in,
    )


# ==================== Column Parallel Tests ====================

def test_column_parallel_forward(
    tp_size: int, sequence_parallel: bool,
    in_features: int = 1024, out_features: int = 1024,
    batch_size: int = 64, seq_len: int = 32,
    bias: bool = False, params_dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    active_recipe=None,
):
    """
    Column parallel forward: each rank computes output columns [rank*chunk_out:(rank+1)*chunk_out].

    SP=True : inp_tp = inp_full[seq_shard]  -> out_tp shape [total_batch/tp, out/tp]
              compare out_tp vs out_baseline[seq_shard, col_shard]
    SP=False: inp_tp = inp_full             -> out_tp shape [total_batch, out/tp]
              compare out_tp vs out_baseline[:, col_shard]
    """
    tp_group = create_tp_group(tp_size) if tp_size > 1 else None
    rank = dist.get_rank()
    total_batch = batch_size * seq_len
    active_recipe = active_recipe or make_nvfp4_recipe()

    s = _setup_column(tp_size, sequence_parallel, in_features, out_features,
                      total_batch, bias, params_dtype, seed, tp_group)

    inp_baseline = s["inp_full"].clone().requires_grad_(False)
    inp_tp = s["inp_tp_data"].clone().requires_grad_(False)

    with torch.no_grad():
        with te.autocast(enabled=True, recipe=active_recipe):
            out_baseline = s["baseline"](inp_baseline, is_first_microbatch=True)
        with te.autocast(enabled=True, recipe=active_recipe):
            out_tp = s["tp_linear"](inp_tp, is_first_microbatch=True)

    if rank < tp_size:
        chunk_out = s["chunk_out"]
        if sequence_parallel and tp_size > 1:
            chunk_seq = total_batch // tp_size
            ref = out_baseline[rank * chunk_seq:(rank + 1) * chunk_seq,
                               rank * chunk_out:(rank + 1) * chunk_out]
        else:
            ref = out_baseline[:, rank * chunk_out:(rank + 1) * chunk_out]

        # Debug: print weight norm, ref norm, out_tp norm
        print(f"  [col fwd debug] rank={rank} "
              f"weight_norm={s['tp_linear'].weight.data.norm():.4e} "
              f"ref_shape={ref.shape} ref_norm={ref.detach().norm():.4e} "
              f"out_tp_norm={out_tp.norm():.4e}")

        diff = (ref - out_tp).abs()
        print(f"  [col fwd] rank={rank} out_tp={out_tp.shape} "
              f"max={diff.max():.4e} mean={diff.mean():.4e}")
        torch.testing.assert_close(ref, out_tp, rtol=1e-1, atol=1e-1,
                                   msg=f"[col fwd] rank {rank} mismatch")
        print(f"  ✓ test_column_parallel_forward rank={rank} passed")


def test_column_parallel_backward_dgrad(
    tp_size: int, sequence_parallel: bool,
    in_features: int = 1024, out_features: int = 1024,
    batch_size: int = 64, seq_len: int = 32,
    bias: bool = False, params_dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    active_recipe=None,
):
    """
    Column parallel dgrad: dx = dy * w.

    SP=True : dgrad is ReduceScattered -> inp_tp.grad shape [total_batch/tp, in_features]
              compare vs inp_baseline.grad[seq_shard, :]
    SP=False: dgrad is AllReduced -> inp_tp.grad shape [total_batch, in_features]
              compare vs inp_baseline.grad
    """
    tp_group = create_tp_group(tp_size) if tp_size > 1 else None
    rank = dist.get_rank()
    total_batch = batch_size * seq_len
    active_recipe = active_recipe or make_nvfp4_recipe()

    s = _setup_column(tp_size, sequence_parallel, in_features, out_features,
                      total_batch, bias, params_dtype, seed, tp_group)

    inp_baseline = s["inp_full"].clone().requires_grad_(True)
    inp_tp = s["inp_tp_data"].clone().requires_grad_(True)

    with te.autocast(enabled=True, recipe=active_recipe):
        out_baseline = s["baseline"](inp_baseline, is_first_microbatch=True)
    out_baseline.backward(s["grad_baseline"])

    with te.autocast(enabled=True, recipe=active_recipe):
        out_tp = s["tp_linear"](inp_tp, is_first_microbatch=True)
    out_tp.backward(s["grad_tp"])

    if rank < tp_size and inp_baseline.grad is not None and inp_tp.grad is not None:
        if sequence_parallel and tp_size > 1:
            chunk_seq = total_batch // tp_size
            ref = inp_baseline.grad[rank * chunk_seq:(rank + 1) * chunk_seq, :]
        else:
            ref = inp_baseline.grad

        diff = (ref - inp_tp.grad).abs()
        print(f"  [col bwd dgrad] rank={rank} grad_shape={inp_tp.grad.shape} "
              f"max={diff.max():.4e} mean={diff.mean():.4e}")
        torch.testing.assert_close(ref, inp_tp.grad, rtol=1e-1, atol=1e-1,
                                   msg=f"[col bwd dgrad] rank {rank} mismatch")
        print(f"  ✓ test_column_parallel_backward_dgrad rank={rank} passed")


def test_column_parallel_backward_wgrad(
    tp_size: int, sequence_parallel: bool,
    in_features: int = 1024, out_features: int = 1024,
    batch_size: int = 64, seq_len: int = 32,
    bias: bool = False, params_dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    active_recipe=None,
):
    """
    Column parallel wgrad: dw = dy^T * x.

    tp_linear.weight.grad shape: [out/tp, in_features]
    compare vs baseline.weight.grad[col_shard, :]
    """
    tp_group = create_tp_group(tp_size) if tp_size > 1 else None
    rank = dist.get_rank()
    total_batch = batch_size * seq_len
    active_recipe = active_recipe or make_nvfp4_recipe()

    s = _setup_column(tp_size, sequence_parallel, in_features, out_features,
                      total_batch, bias, params_dtype, seed, tp_group)

    inp_baseline = s["inp_full"].clone().requires_grad_(True)
    inp_tp = s["inp_tp_data"].clone().requires_grad_(True)

    with te.autocast(enabled=True, recipe=active_recipe):
        out_baseline = s["baseline"](inp_baseline, is_first_microbatch=True)
    out_baseline.backward(s["grad_baseline"])

    with te.autocast(enabled=True, recipe=active_recipe):
        out_tp = s["tp_linear"](inp_tp, is_first_microbatch=True)
    out_tp.backward(s["grad_tp"])

    if rank < tp_size:
        if s["baseline"].weight.grad is not None and s["tp_linear"].weight.grad is not None:
            chunk_out = s["chunk_out"]
            ref = s["baseline"].weight.grad[rank * chunk_out:(rank + 1) * chunk_out, :]
            diff = (ref - s["tp_linear"].weight.grad).abs()
            print(f"  [col bwd wgrad] rank={rank} wgrad_shape={s['tp_linear'].weight.grad.shape} "
                  f"max={diff.max():.4e} mean={diff.mean():.4e}")
            torch.testing.assert_close(ref, s["tp_linear"].weight.grad, rtol=1e-1, atol=1e-1,
                                       msg=f"[col bwd wgrad] rank {rank} mismatch")
            print(f"  ✓ test_column_parallel_backward_wgrad rank={rank} passed")


# ==================== Row Parallel Tests ====================

def test_row_parallel_forward(
    tp_size: int, sequence_parallel: bool,
    in_features: int = 1024, out_features: int = 1024,
    batch_size: int = 64, seq_len: int = 32,
    bias: bool = False, params_dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    active_recipe=None,
):
    """
    Row parallel forward: each rank holds inp[:, in_shard], computes partial GEMM.

    SP=True : ReduceScatter -> out_tp shape [total_batch/tp, out_features]
              compare vs out_baseline[seq_shard, :]
    SP=False: AllReduce -> out_tp shape [total_batch, out_features]
              compare vs out_baseline
    """
    tp_group = create_tp_group(tp_size) if tp_size > 1 else None
    rank = dist.get_rank()
    total_batch = batch_size * seq_len
    active_recipe = active_recipe or make_nvfp4_recipe()

    s = _setup_row(tp_size, sequence_parallel, in_features, out_features,
                   total_batch, bias, params_dtype, seed, tp_group)

    inp_baseline = s["inp_full"].clone().requires_grad_(False)
    inp_tp = s["inp_tp_data"].clone().requires_grad_(False)

    with torch.no_grad():
        with te.autocast(enabled=True, recipe=active_recipe):
            out_baseline = s["baseline"](inp_baseline, is_first_microbatch=True)
        with te.autocast(enabled=True, recipe=active_recipe):
            out_tp = s["tp_linear"](inp_tp, is_first_microbatch=True)

    if rank < tp_size:
        if sequence_parallel and tp_size > 1:
            chunk_seq = total_batch // tp_size
            ref = out_baseline[rank * chunk_seq:(rank + 1) * chunk_seq, :]
        else:
            ref = out_baseline

        diff = (ref - out_tp).abs()
        print(f"  [row fwd] rank={rank} out_tp={out_tp.shape} "
              f"max={diff.max():.4e} mean={diff.mean():.4e}")
        torch.testing.assert_close(ref, out_tp, rtol=1e-1, atol=1e-1,
                                   msg=f"[row fwd] rank {rank} mismatch")
        print(f"  ✓ test_row_parallel_forward rank={rank} passed")


def test_row_parallel_backward_dgrad(
    tp_size: int, sequence_parallel: bool,
    in_features: int = 1024, out_features: int = 1024,
    batch_size: int = 64, seq_len: int = 32,
    bias: bool = False, params_dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    active_recipe=None,
):
    """
    Row parallel dgrad: inp_tp.grad shape [total_batch, in_features/tp].
    compare vs inp_baseline.grad[:, in_shard]
    """
    tp_group = create_tp_group(tp_size) if tp_size > 1 else None
    rank = dist.get_rank()
    total_batch = batch_size * seq_len
    active_recipe = active_recipe or make_nvfp4_recipe()

    s = _setup_row(tp_size, sequence_parallel, in_features, out_features,
                   total_batch, bias, params_dtype, seed, tp_group)

    inp_baseline = s["inp_full"].clone().requires_grad_(True)
    inp_tp = s["inp_tp_data"].clone().requires_grad_(True)

    with te.autocast(enabled=True, recipe=active_recipe):
        out_baseline = s["baseline"](inp_baseline, is_first_microbatch=True)
    out_baseline.backward(s["grad_full"])

    with te.autocast(enabled=True, recipe=active_recipe):
        out_tp = s["tp_linear"](inp_tp, is_first_microbatch=True)
    out_tp.backward(s["grad_tp"])

    if rank < tp_size and inp_baseline.grad is not None and inp_tp.grad is not None:
        chunk_in = s["chunk_in"]
        ref = inp_baseline.grad[:, rank * chunk_in:(rank + 1) * chunk_in]
        diff = (ref - inp_tp.grad).abs()
        print(f"  [row bwd dgrad] rank={rank} grad_shape={inp_tp.grad.shape} "
              f"max={diff.max():.4e} mean={diff.mean():.4e}")
        torch.testing.assert_close(ref, inp_tp.grad, rtol=1e-1, atol=1e-1,
                                   msg=f"[row bwd dgrad] rank {rank} mismatch")
        print(f"  ✓ test_row_parallel_backward_dgrad rank={rank} passed")


def test_row_parallel_backward_wgrad(
    tp_size: int, sequence_parallel: bool,
    in_features: int = 1024, out_features: int = 1024,
    batch_size: int = 64, seq_len: int = 32,
    bias: bool = False, params_dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    active_recipe=None,
):
    """
    Row parallel wgrad: tp_linear.weight.grad shape [out_features, in/tp].
    compare vs baseline.weight.grad[:, in_shard]
    """
    tp_group = create_tp_group(tp_size) if tp_size > 1 else None
    rank = dist.get_rank()
    total_batch = batch_size * seq_len
    active_recipe = active_recipe or make_nvfp4_recipe()

    s = _setup_row(tp_size, sequence_parallel, in_features, out_features,
                   total_batch, bias, params_dtype, seed, tp_group)

    inp_baseline = s["inp_full"].clone().requires_grad_(True)
    inp_tp = s["inp_tp_data"].clone().requires_grad_(True)

    with te.autocast(enabled=True, recipe=active_recipe):
        out_baseline = s["baseline"](inp_baseline, is_first_microbatch=True)
    out_baseline.backward(s["grad_full"])

    with te.autocast(enabled=True, recipe=active_recipe):
        out_tp = s["tp_linear"](inp_tp, is_first_microbatch=True)
    out_tp.backward(s["grad_tp"])

    if rank < tp_size:
        if s["baseline"].weight.grad is not None and s["tp_linear"].weight.grad is not None:
            chunk_in = s["chunk_in"]
            ref = s["baseline"].weight.grad[:, rank * chunk_in:(rank + 1) * chunk_in]
            diff = (ref - s["tp_linear"].weight.grad).abs()
            print(f"  [row bwd wgrad] rank={rank} wgrad_shape={s['tp_linear'].weight.grad.shape} "
                  f"max={diff.max():.4e} mean={diff.mean():.4e}")
            torch.testing.assert_close(ref, s["tp_linear"].weight.grad, rtol=1e-1, atol=1e-1,
                                       msg=f"[row bwd wgrad] rank {rank} mismatch")
            print(f"  ✓ test_row_parallel_backward_wgrad rank={rank} passed")


# ==================== Main ====================

ALL_TESTS = [
    ("Column Parallel Forward",         test_column_parallel_forward),
    ("Column Parallel Backward dgrad",  test_column_parallel_backward_dgrad),
    ("Column Parallel Backward wgrad",  test_column_parallel_backward_wgrad),
    ("Row Parallel Forward",            test_row_parallel_forward),
    ("Row Parallel Backward dgrad",     test_row_parallel_backward_dgrad),
    ("Row Parallel Backward wgrad",     test_row_parallel_backward_wgrad),
]


def run_tests(tp_size: int = 2, sequence_parallel: bool = True, recipe_name: str = "nvfp4"):
    rank = dist.get_rank()

    # 选择 recipe
    if recipe_name == "nvfp4":
        if not nvfp4_available:
            if rank == 0:
                print(f"跳过：NVFP4 不可用（{nvfp4_unavail_reason}）")
            return
        active_recipe = make_nvfp4_recipe()
    elif recipe_name == "mxfp8":
        if not mxfp8_available:
            if rank == 0:
                print(f"跳过：MXFP8 不可用（{mxfp8_unavail_reason}）")
            return
        active_recipe = make_mxfp8_recipe()
    else:
        raise ValueError(f"未知 recipe：{recipe_name}，支持 nvfp4 / mxfp8")

    if rank == 0:
        print("\n" + "=" * 60)
        print("TE Linear TP Tests")
        print("=" * 60)
        print(f"  recipe:            {recipe_name}")
        print(f"  tp_size:           {tp_size}")
        print(f"  sequence_parallel: {sequence_parallel}")
        print("=" * 60)

    for i, (name, fn) in enumerate(ALL_TESTS, 1):
        if rank == 0:
            print(f"\n[{i}/{len(ALL_TESTS)}] {name}")
        try:
            fn(tp_size=tp_size, sequence_parallel=sequence_parallel,
               active_recipe=active_recipe)
            dist.barrier()
        except Exception as e:
            print(f"\n  ✗ FAILED at rank {rank}: {e}")
            traceback.print_exc()
            raise

    if rank == 0:
        print("\n" + "=" * 60)
        print(f"✓ All {len(ALL_TESTS)} tests passed!")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--sp",    dest="sequence_parallel", action="store_true",  default=True)
    parser.add_argument("--no-sp", dest="sequence_parallel", action="store_false")
    parser.add_argument(
        "--recipe", choices=["nvfp4", "mxfp8"], default="nvfp4",
        help="量化 recipe：nvfp4（默认）或 mxfp8"
    )
    args = parser.parse_args()

    if "RANK" not in os.environ:
        print("ERROR: 请用 torchrun 启动")
        print("  torchrun --nproc_per_node=2 test_te_linear_nvfp4_tp.py --recipe nvfp4")
        print("  torchrun --nproc_per_node=2 test_te_linear_nvfp4_tp.py --recipe mxfp8 --no-sp")
        raise SystemExit(1)

    initialize_distributed()
    try:
        run_tests(tp_size=args.tp_size, sequence_parallel=args.sequence_parallel,
                  recipe_name=args.recipe)
    finally:
        cleanup_distributed()
