"""
_QuantizedLinear TP 通信正确性测试
=====================================

验证目标：在给定完全相同的量化 weight 和量化 input 的前提下，
tp=n 的分块 GEMM + TP 通信（AllGather/ReduceScatter）的结果
是否与 tp=1 的全量 GEMM 结果 bit-exact 一致。

验证思路：
  1. tp=1 baseline 先做一次 forward，触发 weight 量化并写入 weight_qresult_cache
  2. 将 baseline 的量化 weight (qw/sw) 对应切片直接注入到各 tp rank 的 weight_qresult_cache
  3. tp linear 使用 is_first_microbatch=False（跳过重新量化，直接读 cache）
  4. 用 allgather_quantize=False 的 recipe（先 AllGather BF16 再量化），
     保证 input 量化结果也与 tp=1 一致
  5. 对比输出，bit-exact 则说明 TP 通信逻辑正确

测试矩阵：
  test_column_parallel_forward
  test_column_parallel_backward_dgrad
  test_column_parallel_backward_wgrad
  test_row_parallel_forward
  test_row_parallel_backward_dgrad
  test_row_parallel_backward_wgrad

运行：
  torchrun --nproc_per_node=2 test_linear_tp_quantized.py
  torchrun --nproc_per_node=2 test_linear_tp_quantized.py --no-sp
"""

import argparse
import os
import traceback

import torch
import torch.distributed as dist

from miniTransformer.module.config import QuantizeRecipe, get_qlinear_params_from_predefined
from miniTransformer.module.linear import Linear
from miniTransformer.quantiztion import quantization


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


def broadcast_tensor(t: torch.Tensor, tp_group) -> torch.Tensor:
    dist.broadcast(t, src=0, group=tp_group)
    return t


def sync_weight(linear, tp_group):
    """将 rank 0 的 weight broadcast 到所有 rank，保证 baseline weight 一致。"""
    dist.broadcast(linear.weight.data, src=0, group=tp_group)


# ==================== Module Builders ====================

def build_linear(
    in_features, out_features, parallel_mode, tp_size, tp_group,
    sequence_parallel, seed, params_dtype=torch.bfloat16,
    qparam=QuantizeRecipe.FP4_SUB_CHANNEL,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    qlinear_params = get_qlinear_params_from_predefined(qparam)
    return Linear(
        in_features=in_features,
        out_features=out_features,
        sequence_parallel=sequence_parallel,
        fuse_wgrad_accumulation=False,
        tp_group=tp_group,
        tp_size=tp_size,
        get_rng_state_tracker=None,
        rng_tracker_name=None,
        init_method=None,
        bias=False,
        return_bias=False,
        params_dtype=params_dtype,
        parallel_mode=parallel_mode,
        parameters_split=None,
        device="cuda",
        ub_overlap_rs=False,
        ub_overlap_ag=False,
        ub_name=f"linear_{parallel_mode}_tp{tp_size}",
        layer_number=0,
        qlinear_params=qlinear_params,
    )


# ==================== 量化 Weight Cache 注入 ====================

def inject_column_weight_cache(linear_tp, linear_baseline, rank, tp_size):
    """
    Column parallel：baseline weight_qresult_cache 的 qw/sw 按行切片，
    注入到 linear_tp 的 weight_qresult_cache，并标记 should_update=False。

    这样 linear_tp forward 时用 is_first_microbatch=False 直接读 cache，
    不会重新量化，保证 qw 与 baseline 完全一致。
    """
    cache = linear_baseline.weight_qresult_cache
    assert cache.data is not None, "baseline weight cache 为空，请先跑一次 forward"

    out_full = cache.data.shape[0]  # qw shape: [out_features, in_features/16] (packed)
    # 注意 qw 是 packed uint8，shape[0] 对应 out_features
    chunk = out_full // tp_size

    # scale shape: [out_features, in_features // quant_tile_k]
    scale_full = cache.scale.shape[0]
    scale_chunk = scale_full // tp_size

    tp_cache = linear_tp.weight_qresult_cache
    tp_cache.data   = cache.data  [rank * chunk       : (rank + 1) * chunk,       :].contiguous()
    tp_cache.scale  = cache.scale [rank * scale_chunk : (rank + 1) * scale_chunk, :].contiguous()

    # 转置方向（columnwise）：qw_t shape [in_features/16, out_features]，按列切
    if cache.data_t is not None:
        out_t_full  = cache.data_t.shape[1]
        chunk_t     = out_t_full // tp_size
        scale_t_full = cache.scale_t.shape[1]
        chunk_st    = scale_t_full // tp_size
        tp_cache.data_t  = cache.data_t [:, rank * chunk_t  : (rank + 1) * chunk_t ].contiguous()
        tp_cache.scale_t = cache.scale_t[:, rank * chunk_st : (rank + 1) * chunk_st].contiguous()

    tp_cache.global_amax_row = cache.global_amax_row
    tp_cache.global_amax_col = cache.global_amax_col

    # 让 quantize_op 不再重新量化 weight
    linear_tp.weight_qresult_cache = tp_cache


def inject_row_weight_cache(linear_tp, linear_baseline, rank, tp_size):
    """
    Row parallel：baseline weight_qresult_cache 的 qw/sw 按列切片（in_features 方向），
    注入到 linear_tp 的 weight_qresult_cache。
    """
    cache = linear_baseline.weight_qresult_cache
    assert cache.data is not None, "baseline weight cache 为空，请先跑一次 forward"

    # qw shape: [out_features, in_features/16 packed]，按列（in 方向）切
    in_packed_full = cache.data.shape[1]
    chunk = in_packed_full // tp_size

    scale_in_full = cache.scale.shape[1]
    scale_chunk   = scale_in_full // tp_size

    tp_cache = linear_tp.weight_qresult_cache
    tp_cache.data   = cache.data [:, rank * chunk       : (rank + 1) * chunk      ].contiguous()
    tp_cache.scale  = cache.scale[:, rank * scale_chunk : (rank + 1) * scale_chunk].contiguous()

    if cache.data_t is not None:
        in_t_full   = cache.data_t.shape[0]
        chunk_t     = in_t_full // tp_size
        scale_t_in  = cache.scale_t.shape[0]
        chunk_st    = scale_t_in // tp_size
        tp_cache.data_t  = cache.data_t [rank * chunk_t  : (rank + 1) * chunk_t , :].contiguous()
        tp_cache.scale_t = cache.scale_t[rank * chunk_st : (rank + 1) * chunk_st, :].contiguous()

    tp_cache.global_amax_row = cache.global_amax_row
    tp_cache.global_amax_col = cache.global_amax_col

    linear_tp.weight_qresult_cache = tp_cache


# ==================== 共享 Setup ====================

def _setup(
    parallel_mode, tp_size, sequence_parallel,
    in_features, out_features, total_batch,
    seed, tp_group,
    qparam=QuantizeRecipe.FP4_SUB_CHANNEL,
    params_dtype=torch.bfloat16,
):
    rank = dist.get_rank()

    # 1. 构建 baseline (tp=1, sp=False)
    baseline = build_linear(
        in_features, out_features, parallel_mode,
        tp_size=1, tp_group=None, sequence_parallel=False,
        seed=seed, params_dtype=params_dtype, qparam=qparam,
    )
    sync_weight(baseline, tp_group)

    # 2. 构建 tp linear
    linear_tp = build_linear(
        in_features, out_features, parallel_mode,
        tp_size=tp_size, tp_group=tp_group, sequence_parallel=sequence_parallel,
        seed=seed, params_dtype=params_dtype, qparam=qparam,
    )

    # 3. 同步 tp linear 的 BF16 weight（从 baseline 切片），保证初始值一致
    #    （weight cache 注入在 baseline warmup 之后再做）
    if parallel_mode == "column":
        chunk = out_features // tp_size
        with torch.no_grad():
            linear_tp.weight.data.copy_(
                baseline.weight.data[rank * chunk:(rank + 1) * chunk, :]
            )
    else:  # row
        chunk = in_features // tp_size
        with torch.no_grad():
            linear_tp.weight.data.copy_(
                baseline.weight.data[:, rank * chunk:(rank + 1) * chunk]
            )

    # 4. 构造 input 和 grad，broadcast 保证一致
    torch.manual_seed(seed + 100)
    inp_full = torch.randn(total_batch, in_features, device="cuda", dtype=params_dtype)
    broadcast_tensor(inp_full, tp_group)

    torch.manual_seed(seed + 200)
    grad_full = torch.randn(total_batch, out_features, device="cuda", dtype=params_dtype)
    broadcast_tensor(grad_full, tp_group)

    return dict(
        baseline=baseline, linear_tp=linear_tp,
        inp_full=inp_full, grad_full=grad_full,
        rank=rank,
    )


def _warmup_baseline_and_inject_cache(s, parallel_mode, tp_size, total_batch,
                                       in_features, out_features, sequence_parallel):
    """
    用 inp_full 跑一次 baseline forward（触发 weight 量化写入 cache），
    然后将量化后的 weight cache 切片注入到 linear_tp。
    """
    baseline  = s["baseline"]
    linear_tp = s["linear_tp"]
    inp_full  = s["inp_full"]
    rank      = s["rank"]

    with torch.no_grad():
        # is_first_microbatch=True：强制重新量化 weight，写入 cache
        _ = baseline(inp_full, is_first_microbatch=True)

    # 注入量化 weight cache 到 linear_tp
    if parallel_mode == "column":
        inject_column_weight_cache(linear_tp, baseline, rank, tp_size)
    else:
        inject_row_weight_cache(linear_tp, baseline, rank, tp_size)


# ==================== Column Parallel Tests ====================

def test_column_parallel_forward(
    tp_size=2, sequence_parallel=True,
    in_features=1024, out_features=1024,
    batch_size=64, seq_len=32,
    params_dtype=torch.bfloat16, seed=42,
    qparam=QuantizeRecipe.FP4_SUB_CHANNEL,
):
    """
    Column parallel forward 通信正确性验证。

    两边使用完全相同的量化 weight（通过 cache 注入），
    input 用 allgather_quantize=False（先 AllGather BF16 再量化），
    所以量化结果完全一致，输出差异只来自 TP 通信逻辑。
    """
    tp_group    = create_tp_group(tp_size)
    total_batch = batch_size * seq_len

    s = _setup("column", tp_size, sequence_parallel,
                in_features, out_features, total_batch,
                seed, tp_group, qparam=qparam, params_dtype=params_dtype)

    _warmup_baseline_and_inject_cache(s, "column", tp_size, total_batch,
                                       in_features, out_features, sequence_parallel)

    rank      = s["rank"]
    baseline  = s["baseline"]
    linear_tp = s["linear_tp"]
    inp_full  = s["inp_full"]

    chunk_out = out_features // tp_size

    with torch.no_grad():
        # baseline：tp=1，完整 forward
        out_baseline = baseline(inp_full, is_first_microbatch=False)

        # tp linear：is_first_microbatch=False，直接用注入的 cache，不重新量化 weight
        if sequence_parallel and tp_size > 1:
            chunk_seq = total_batch // tp_size
            inp_tp = inp_full[rank * chunk_seq:(rank + 1) * chunk_seq, :].contiguous()
        else:
            inp_tp = inp_full
        out_tp = linear_tp(inp_tp, is_first_microbatch=False)

    if rank < tp_size:
        # baseline 对应切片
        if sequence_parallel and tp_size > 1:
            chunk_seq = total_batch // tp_size
            ref = out_baseline[rank * chunk_seq:(rank + 1) * chunk_seq,
                               rank * chunk_out:(rank + 1) * chunk_out]
        else:
            ref = out_baseline[:, rank * chunk_out:(rank + 1) * chunk_out]

        diff = (ref - out_tp).abs()
        print(f"  [col fwd] rank={rank} out_tp={out_tp.shape} "
              f"max={diff.max():.4e} mean={diff.mean():.4e}")
        torch.testing.assert_close(ref, out_tp, rtol=0, atol=0,
                                   msg=f"[col fwd] rank {rank} mismatch")
        print(f"  ✓ test_column_parallel_forward rank={rank} passed")


def test_column_parallel_backward_dgrad(
    tp_size=2, sequence_parallel=True,
    in_features=1024, out_features=1024,
    batch_size=64, seq_len=32,
    params_dtype=torch.bfloat16, seed=42,
    qparam=QuantizeRecipe.FP4_SUB_CHANNEL,
):
    """Column parallel dgrad 通信正确性验证。"""
    tp_group    = create_tp_group(tp_size)
    total_batch = batch_size * seq_len

    s = _setup("column", tp_size, sequence_parallel,
                in_features, out_features, total_batch,
                seed, tp_group, qparam=qparam, params_dtype=params_dtype)

    _warmup_baseline_and_inject_cache(s, "column", tp_size, total_batch,
                                       in_features, out_features, sequence_parallel)

    rank      = s["rank"]
    baseline  = s["baseline"]
    linear_tp = s["linear_tp"]
    inp_full  = s["inp_full"]
    grad_full = s["grad_full"]
    chunk_out = out_features // tp_size

    # baseline backward
    inp_b = inp_full.clone().requires_grad_(True)
    out_b = baseline(inp_b, is_first_microbatch=False)
    grad_b = grad_full[:, rank * chunk_out:(rank + 1) * chunk_out].contiguous()
    out_b.backward(grad_b)

    # tp backward
    if sequence_parallel and tp_size > 1:
        chunk_seq = total_batch // tp_size
        inp_tp = inp_full[rank * chunk_seq:(rank + 1) * chunk_seq, :].contiguous().requires_grad_(True)
        grad_tp = grad_full[rank * chunk_seq:(rank + 1) * chunk_seq,
                            rank * chunk_out:(rank + 1) * chunk_out].contiguous()
    else:
        inp_tp = inp_full.clone().requires_grad_(True)
        grad_tp = grad_b
    out_tp = linear_tp(inp_tp, is_first_microbatch=False)
    out_tp.backward(grad_tp)

    if rank < tp_size and inp_b.grad is not None and inp_tp.grad is not None:
        if sequence_parallel and tp_size > 1:
            chunk_seq = total_batch // tp_size
            ref = inp_b.grad[rank * chunk_seq:(rank + 1) * chunk_seq, :]
        else:
            ref = inp_b.grad

        diff = (ref - inp_tp.grad).abs()
        print(f"  [col bwd dgrad] rank={rank} shape={inp_tp.grad.shape} "
              f"max={diff.max():.4e} mean={diff.mean():.4e}")
        torch.testing.assert_close(ref, inp_tp.grad, rtol=0, atol=0,
                                   msg=f"[col bwd dgrad] rank {rank} mismatch")
        print(f"  ✓ test_column_parallel_backward_dgrad rank={rank} passed")


def test_column_parallel_backward_wgrad(
    tp_size=2, sequence_parallel=True,
    in_features=1024, out_features=1024,
    batch_size=64, seq_len=32,
    params_dtype=torch.bfloat16, seed=42,
    qparam=QuantizeRecipe.FP4_SUB_CHANNEL,
):
    """Column parallel wgrad 正确性验证。"""
    tp_group    = create_tp_group(tp_size)
    total_batch = batch_size * seq_len

    s = _setup("column", tp_size, sequence_parallel,
                in_features, out_features, total_batch,
                seed, tp_group, qparam=qparam, params_dtype=params_dtype)

    _warmup_baseline_and_inject_cache(s, "column", tp_size, total_batch,
                                       in_features, out_features, sequence_parallel)

    rank      = s["rank"]
    baseline  = s["baseline"]
    linear_tp = s["linear_tp"]
    inp_full  = s["inp_full"]
    grad_full = s["grad_full"]
    chunk_out = out_features // tp_size

    inp_b = inp_full.clone().requires_grad_(True)
    out_b = baseline(inp_b, is_first_microbatch=False)
    grad_b = grad_full[:, rank * chunk_out:(rank + 1) * chunk_out].contiguous()
    out_b.backward(grad_b)

    if sequence_parallel and tp_size > 1:
        chunk_seq = total_batch // tp_size
        inp_tp = inp_full[rank * chunk_seq:(rank + 1) * chunk_seq, :].contiguous().requires_grad_(True)
        grad_tp = grad_full[rank * chunk_seq:(rank + 1) * chunk_seq,
                            rank * chunk_out:(rank + 1) * chunk_out].contiguous()
    else:
        inp_tp = inp_full.clone().requires_grad_(True)
        grad_tp = grad_b
    out_tp = linear_tp(inp_tp, is_first_microbatch=False)
    out_tp.backward(grad_tp)

    if rank < tp_size:
        if baseline.weight.grad is not None and linear_tp.weight.grad is not None:
            ref = baseline.weight.grad[rank * chunk_out:(rank + 1) * chunk_out, :]
            diff = (ref - linear_tp.weight.grad).abs()
            print(f"  [col bwd wgrad] rank={rank} shape={linear_tp.weight.grad.shape} "
                  f"max={diff.max():.4e} mean={diff.mean():.4e}")
            torch.testing.assert_close(ref, linear_tp.weight.grad, rtol=0, atol=0,
                                       msg=f"[col bwd wgrad] rank {rank} mismatch")
            print(f"  ✓ test_column_parallel_backward_wgrad rank={rank} passed")


# ==================== Row Parallel Tests ====================

def test_row_parallel_forward(
    tp_size=2, sequence_parallel=True,
    in_features=1024, out_features=1024,
    batch_size=64, seq_len=32,
    params_dtype=torch.bfloat16, seed=42,
    qparam=QuantizeRecipe.FP4_SUB_CHANNEL,
):
    """Row parallel forward 通信正确性验证。"""
    tp_group    = create_tp_group(tp_size)
    total_batch = batch_size * seq_len

    s = _setup("row", tp_size, sequence_parallel,
                in_features, out_features, total_batch,
                seed, tp_group, qparam=qparam, params_dtype=params_dtype)

    _warmup_baseline_and_inject_cache(s, "row", tp_size, total_batch,
                                       in_features, out_features, sequence_parallel)

    rank      = s["rank"]
    baseline  = s["baseline"]
    linear_tp = s["linear_tp"]
    inp_full  = s["inp_full"]
    chunk_in  = in_features // tp_size

    with torch.no_grad():
        out_baseline = baseline(inp_full, is_first_microbatch=False)

        inp_tp = inp_full[:, rank * chunk_in:(rank + 1) * chunk_in].contiguous()
        out_tp = linear_tp(inp_tp, is_first_microbatch=False)

    if rank < tp_size:
        if sequence_parallel and tp_size > 1:
            chunk_seq = total_batch // tp_size
            ref = out_baseline[rank * chunk_seq:(rank + 1) * chunk_seq, :]
        else:
            ref = out_baseline

        diff = (ref - out_tp).abs()
        print(f"  [row fwd] rank={rank} out_tp={out_tp.shape} "
              f"max={diff.max():.4e} mean={diff.mean():.4e}")
        torch.testing.assert_close(ref, out_tp, rtol=0, atol=0,
                                   msg=f"[row fwd] rank {rank} mismatch")
        print(f"  ✓ test_row_parallel_forward rank={rank} passed")


def test_row_parallel_backward_dgrad(
    tp_size=2, sequence_parallel=True,
    in_features=1024, out_features=1024,
    batch_size=64, seq_len=32,
    params_dtype=torch.bfloat16, seed=42,
    qparam=QuantizeRecipe.FP4_SUB_CHANNEL,
):
    """Row parallel dgrad 通信正确性验证。"""
    tp_group    = create_tp_group(tp_size)
    total_batch = batch_size * seq_len

    s = _setup("row", tp_size, sequence_parallel,
                in_features, out_features, total_batch,
                seed, tp_group, qparam=qparam, params_dtype=params_dtype)

    _warmup_baseline_and_inject_cache(s, "row", tp_size, total_batch,
                                       in_features, out_features, sequence_parallel)

    rank      = s["rank"]
    baseline  = s["baseline"]
    linear_tp = s["linear_tp"]
    inp_full  = s["inp_full"]
    grad_full = s["grad_full"]
    chunk_in  = in_features // tp_size

    inp_b = inp_full.clone().requires_grad_(True)
    out_b = baseline(inp_b, is_first_microbatch=False)
    out_b.backward(grad_full)

    inp_tp = inp_full[:, rank * chunk_in:(rank + 1) * chunk_in].contiguous().requires_grad_(True)
    if sequence_parallel and tp_size > 1:
        chunk_seq = total_batch // tp_size
        grad_tp = grad_full[rank * chunk_seq:(rank + 1) * chunk_seq, :].contiguous()
    else:
        grad_tp = grad_full.contiguous()
    out_tp = linear_tp(inp_tp, is_first_microbatch=False)
    out_tp.backward(grad_tp)

    if rank < tp_size and inp_b.grad is not None and inp_tp.grad is not None:
        ref = inp_b.grad[:, rank * chunk_in:(rank + 1) * chunk_in]
        diff = (ref - inp_tp.grad).abs()
        print(f"  [row bwd dgrad] rank={rank} shape={inp_tp.grad.shape} "
              f"max={diff.max():.4e} mean={diff.mean():.4e}")
        torch.testing.assert_close(ref, inp_tp.grad, rtol=0, atol=0,
                                   msg=f"[row bwd dgrad] rank {rank} mismatch")
        print(f"  ✓ test_row_parallel_backward_dgrad rank={rank} passed")


def test_row_parallel_backward_wgrad(
    tp_size=2, sequence_parallel=True,
    in_features=1024, out_features=1024,
    batch_size=64, seq_len=32,
    params_dtype=torch.bfloat16, seed=42,
    qparam=QuantizeRecipe.FP4_SUB_CHANNEL,
):
    """Row parallel wgrad 通信正确性验证。"""
    tp_group    = create_tp_group(tp_size)
    total_batch = batch_size * seq_len

    s = _setup("row", tp_size, sequence_parallel,
                in_features, out_features, total_batch,
                seed, tp_group, qparam=qparam, params_dtype=params_dtype)

    _warmup_baseline_and_inject_cache(s, "row", tp_size, total_batch,
                                       in_features, out_features, sequence_parallel)

    rank      = s["rank"]
    baseline  = s["baseline"]
    linear_tp = s["linear_tp"]
    inp_full  = s["inp_full"]
    grad_full = s["grad_full"]
    chunk_in  = in_features // tp_size

    inp_b = inp_full.clone().requires_grad_(True)
    out_b = baseline(inp_b, is_first_microbatch=False)
    out_b.backward(grad_full)

    inp_tp = inp_full[:, rank * chunk_in:(rank + 1) * chunk_in].contiguous().requires_grad_(True)
    if sequence_parallel and tp_size > 1:
        chunk_seq = total_batch // tp_size
        grad_tp = grad_full[rank * chunk_seq:(rank + 1) * chunk_seq, :].contiguous()
    else:
        grad_tp = grad_full.contiguous()
    out_tp = linear_tp(inp_tp, is_first_microbatch=False)
    out_tp.backward(grad_tp)

    if rank < tp_size:
        if baseline.weight.grad is not None and linear_tp.weight.grad is not None:
            ref = baseline.weight.grad[:, rank * chunk_in:(rank + 1) * chunk_in]
            diff = (ref - linear_tp.weight.grad).abs()
            print(f"  [row bwd wgrad] rank={rank} shape={linear_tp.weight.grad.shape} "
                  f"max={diff.max():.4e} mean={diff.mean():.4e}")
            torch.testing.assert_close(ref, linear_tp.weight.grad, rtol=0, atol=0,
                                       msg=f"[row bwd wgrad] rank {rank} mismatch")
            print(f"  ✓ test_row_parallel_backward_wgrad rank={rank} passed")


# ==================== Main ====================

ALL_TESTS = [
    ("Column Parallel Forward",        test_column_parallel_forward),
    ("Column Parallel Backward dgrad", test_column_parallel_backward_dgrad),
    ("Column Parallel Backward wgrad", test_column_parallel_backward_wgrad),
    ("Row Parallel Forward",           test_row_parallel_forward),
    ("Row Parallel Backward dgrad",    test_row_parallel_backward_dgrad),
    ("Row Parallel Backward wgrad",    test_row_parallel_backward_wgrad),
]

QPARAM_MAP = {
    "fp4":     QuantizeRecipe.FP4_SUB_CHANNEL,
    "fp4_ref": QuantizeRecipe.FP4_SUB_CHANNEL_REF,
    "fp4_q":   QuantizeRecipe.FP4_SUB_CHANNEL_QUARTET,
    "int4":    QuantizeRecipe.INT4_SUB_CHANNEL,
}


def run_tests(tp_size=2, sequence_parallel=True, qparam_name="fp4"):
    rank = dist.get_rank()
    qparam = QPARAM_MAP[qparam_name]

    if rank == 0:
        print("\n" + "=" * 60)
        print("_QuantizedLinear TP 通信正确性测试")
        print("=" * 60)
        print(f"  tp_size:           {tp_size}")
        print(f"  sequence_parallel: {sequence_parallel}")
        print(f"  qparam:            {qparam_name}")
        print("=" * 60)

    for i, (name, fn) in enumerate(ALL_TESTS, 1):
        if rank == 0:
            print(f"\n[{i}/{len(ALL_TESTS)}] {name}")
        try:
            fn(tp_size=tp_size, sequence_parallel=sequence_parallel, qparam=qparam)
            dist.barrier()
        except Exception as e:
            print(f"\n  ✗ FAILED at rank {rank}: {e}")
            traceback.print_exc()
            raise

    if rank == 0:
        print("\n" + "=" * 60)
        print(f"✓ 全部 {len(ALL_TESTS)} 项测试通过！")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--sp",    dest="sequence_parallel", action="store_true",  default=True)
    parser.add_argument("--no-sp", dest="sequence_parallel", action="store_false")
    parser.add_argument(
        "--qparam", choices=list(QPARAM_MAP.keys()), default="fp4",
        help="量化 recipe（默认 fp4）"
    )
    args = parser.parse_args()

    if "RANK" not in os.environ:
        print("请用 torchrun 启动：")
        print("  torchrun --nproc_per_node=2 test_linear_tp_quantized.py")
        print("  torchrun --nproc_per_node=2 test_linear_tp_quantized.py --no-sp --qparam fp4")
        raise SystemExit(1)

    initialize_distributed()
    try:
        run_tests(tp_size=args.tp_size, sequence_parallel=args.sequence_parallel,
                  qparam_name=args.qparam)
    finally:
        cleanup_distributed()
