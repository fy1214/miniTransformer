"""
Linear TP-SP 正确性测试
=======================

验证 miniTransformer Linear 在 Tensor Parallel (TP) + Sequence Parallel (SP) 下的
forward/backward 数学等价性，覆盖 Column Parallel 和 Row Parallel 两种模式。

验证策略：
  - 整数初始化（x, w ∈ {-1,0,1}，dy = 整数），乘积和累加均为整数
  - bf16 在 [-2048, 2048] 可精确表示整数，bit-exact 验证（rtol=0, atol=0）

运行：
  torchrun --nproc_per_node=2 test_linear_tp.py --qparam 6 --tp-size 2
  torchrun --nproc_per_node=2 test_linear_tp.py --qparam 6 --tp-size 2 --no-sp
"""
import argparse
import os
import traceback

import torch
import torch.distributed as dist

from miniTransformer.module.config import QuantizeRecipe, get_qlinear_params_from_predefined
from miniTransformer.module.linear import Linear


# ── Distributed utilities ─────────────────────────────────────────────────────

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


def broadcast_tensor(tensor: torch.Tensor, tp_group) -> torch.Tensor:
    dist.broadcast(tensor, src=0, group=tp_group)
    return tensor


def sync_linear_weight(linear, tp_group) -> None:
    """Broadcast rank-0 weight to all ranks so every rank starts from the same baseline."""
    dist.broadcast(linear.weight.data, src=0, group=tp_group)


# ── Linear builder ────────────────────────────────────────────────────────────

def build_linear(
    in_features: int,
    out_features: int,
    parallel_mode: str,
    tp_size: int = 1,
    tp_group=None,
    qparam: QuantizeRecipe = None,
    seed: int = 42,
    sequence_parallel: bool = False,
) -> Linear:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return Linear(
        in_features,
        out_features,
        sequence_parallel,
        fuse_wgrad_accumulation=True,
        tp_group=tp_group,
        tp_size=tp_size,
        get_rng_state_tracker=None,
        rng_tracker_name=None,
        init_method=None,
        bias=False,
        return_bias=False,
        params_dtype=torch.bfloat16,
        parallel_mode=parallel_mode,
        parameters_split=None,
        device="cuda",
        ub_overlap_rs=False,
        ub_overlap_ag=False,
        ub_name=f"linear_tp{tp_size}_{parallel_mode}",
        layer_number=0,
        qlinear_params=get_qlinear_params_from_predefined(qparam),
    )


# ── Column parallel tests ─────────────────────────────────────────────────────

def test_column_parallel_forward(
    qparam: QuantizeRecipe, tp_size: int = 2, sequence_parallel: bool = True
):
    """
    Column parallel forward 数学等价性验证。

    out_features 沿 TP ranks 分割；每个 rank 计算 [batch, out//tp] 的输出分片。

    SP=False：每个 rank 收到完整 input [batch, in]
              out_tp [batch, out//tp]  ==  out_baseline[:, col_shard]

    SP=True ：每个 rank 收到 seq shard [batch//tp, in]
              linear 内部 AllGather 还原完整 input 再 GEMM
              out_tp [batch, out//tp]  ==  out_baseline[:, col_shard]
    """
    print(f"\n{'='*60}")
    print(f"Column Parallel Forward  (tp={tp_size}, sp={sequence_parallel})")
    print(f"{'='*60}")

    in_features, out_features, batch_size, seed = 4096, 4096, 1024, 42
    tp_group = create_tp_group(tp_size) if tp_size > 1 else None
    rank = dist.get_rank() if dist.is_initialized() else 0

    baseline = build_linear(in_features, out_features, "column", 1, None, qparam, seed, False)
    if tp_size > 1:
        sync_linear_weight(baseline, tp_group)

    linear_tp = build_linear(in_features, out_features, "column", tp_size, tp_group, qparam, seed, sequence_parallel)
    if tp_size > 1:
        chunk_out = out_features // tp_size
        with torch.no_grad():
            linear_tp.weight.data.copy_(
                baseline.weight.data[rank * chunk_out:(rank + 1) * chunk_out, :]
            )

    torch.manual_seed(seed)
    inp_full = torch.randn(batch_size, in_features, device="cuda", dtype=torch.bfloat16)
    if tp_size > 1:
        broadcast_tensor(inp_full, tp_group)

    if sequence_parallel and tp_size > 1:
        chunk_seq = batch_size // tp_size
        inp_tp = inp_full[rank * chunk_seq:(rank + 1) * chunk_seq, :].contiguous()
    else:
        inp_tp = inp_full

    with torch.no_grad():
        out_baseline = baseline(inp_full, is_first_microbatch=True)
        out_tp       = linear_tp(inp_tp, is_first_microbatch=True)

    if rank < tp_size:
        chunk_out = out_features // tp_size
        ref = out_baseline[:, rank * chunk_out:(rank + 1) * chunk_out]
        diff = (ref - out_tp).abs()
        print(f"  Rank {rank}:  out_tp={out_tp.shape}  "
              f"max_diff={diff.max():.4e}  mean_diff={diff.mean():.4e}")
        torch.testing.assert_close(ref, out_tp, rtol=1e-2, atol=1e-2,
                                   msg=f"Column parallel forward mismatch at rank {rank}")
        print(f"  ✓ Column parallel forward verified")

    print(f"{'='*60}\n")


def test_column_parallel_backward(
    qparam: QuantizeRecipe, tp_size: int = 2, sequence_parallel: bool = True
):
    """
    Column parallel backward 数学等价性验证（bit-exact）。

    dgrad  dx = sum_r(dy_r @ W_r.T)，经 AllReduce 后每个 rank 得完整 dx
      SP=False: inp_tp.grad  ==  inp_baseline.grad
      SP=True : inp_tp.grad  ==  inp_baseline.grad[seq_shard, :]

    wgrad  dw_r = dy_r.T @ x_full，每个 rank 独立计算，无需通信
      linear_tp.weight.main_grad  ==  baseline.weight.main_grad[col_shard, :]
    """
    print(f"\n{'='*60}")
    print(f"Column Parallel Backward (tp={tp_size}, sp={sequence_parallel})")
    print(f"{'='*60}")

    in_features, out_features, batch_size, seed = 4096, 4096, 1024, 42
    tp_group = create_tp_group(tp_size) if tp_size > 1 else None
    rank = dist.get_rank() if dist.is_initialized() else 0

    baseline = build_linear(in_features, out_features, "column", 1, None, qparam, seed, False)
    if tp_size > 1:
        sync_linear_weight(baseline, tp_group)
    baseline.weight.main_grad = torch.zeros_like(baseline.weight.data, dtype=torch.float32)

    linear_tp = build_linear(in_features, out_features, "column", tp_size, tp_group, qparam, seed, sequence_parallel)
    linear_tp.weight.main_grad = torch.zeros_like(linear_tp.weight.data, dtype=torch.float32)

    # 整数初始化 weight
    torch.manual_seed(seed)
    with torch.no_grad():
        baseline.weight.data.copy_(
            torch.randint(-1, 2, baseline.weight.shape, device="cuda").to(torch.bfloat16)
        )
    if tp_size > 1:
        sync_linear_weight(baseline, tp_group)
        chunk_out = out_features // tp_size
        linear_tp.weight.data.copy_(
            baseline.weight.data[rank * chunk_out:(rank + 1) * chunk_out, :]
        )
        linear_tp.weight.main_grad = torch.zeros_like(linear_tp.weight.data, dtype=torch.float32)

    # 整数初始化 input
    inp_full = torch.randint(-1, 2, (batch_size, in_features), device="cuda").to(torch.bfloat16)
    if tp_size > 1:
        broadcast_tensor(inp_full, tp_group)

    inp_baseline = inp_full.clone().requires_grad_(True)

    if sequence_parallel and tp_size > 1:
        chunk_seq = batch_size // tp_size
        inp_tp = inp_full[rank * chunk_seq:(rank + 1) * chunk_seq, :].contiguous().requires_grad_(True)
    else:
        inp_tp = inp_full.clone().requires_grad_(True)

    # 整数初始化 dy
    dy_full = torch.randint(-1, 2, (batch_size, out_features), device="cuda").to(torch.bfloat16)
    if tp_size > 1:
        broadcast_tensor(dy_full, tp_group)

    # column parallel output shape 始终为 [batch, out//tp]，与 SP 无关
    chunk_out = out_features // tp_size
    dy_tp = dy_full[:, rank * chunk_out:(rank + 1) * chunk_out].contiguous()

    out_baseline = baseline(inp_baseline, is_first_microbatch=True)
    out_tp       = linear_tp(inp_tp, is_first_microbatch=True)
    out_baseline.backward(dy_full)
    out_tp.backward(dy_tp)

    if rank < tp_size:
        # dgrad
        if inp_baseline.grad is not None and inp_tp.grad is not None:
            if sequence_parallel and tp_size > 1:
                chunk_seq = batch_size // tp_size
                ref_dgrad = inp_baseline.grad[rank * chunk_seq:(rank + 1) * chunk_seq, :]
            else:
                ref_dgrad = inp_baseline.grad

            diff = (ref_dgrad - inp_tp.grad).abs()
            print(f"  Rank {rank} dgrad:  shape={inp_tp.grad.shape}  "
                  f"max_diff={diff.max():.4e}  mean_diff={diff.mean():.4e}")
            torch.testing.assert_close(ref_dgrad, inp_tp.grad, rtol=0, atol=0,
                                       msg=f"Column parallel dgrad mismatch at rank {rank}")

        # wgrad
        if baseline.weight.main_grad is not None and linear_tp.weight.main_grad is not None:
            ref_wgrad = baseline.weight.main_grad[rank * chunk_out:(rank + 1) * chunk_out, :]
            diff = (ref_wgrad - linear_tp.weight.main_grad).abs()
            print(f"  Rank {rank} wgrad:  shape={linear_tp.weight.main_grad.shape}  "
                  f"max_diff={diff.max():.4e}  mean_diff={diff.mean():.4e}")
            torch.testing.assert_close(ref_wgrad, linear_tp.weight.main_grad, rtol=0, atol=0,
                                       msg=f"Column parallel wgrad mismatch at rank {rank}")

        print(f"  ✓ Column parallel backward verified")

    print(f"{'='*60}\n")


# ── Row parallel tests ────────────────────────────────────────────────────────

def test_row_parallel_forward(
    qparam: QuantizeRecipe, tp_size: int = 2, sequence_parallel: bool = True
):
    """
    Row parallel forward 数学等价性验证（bit-exact）。

    in_features 沿 TP ranks 分割：inp_r = inp[:, in_shard]，W_r = W[:, in_shard]

    SP=False：AllReduce 后 out_tp [batch, out]  ==  out_baseline
    SP=True ：ReduceScatter 后 out_tp [batch//tp, out]  ==  out_baseline[seq_shard, :]
    """
    print(f"\n{'='*60}")
    print(f"Row Parallel Forward     (tp={tp_size}, sp={sequence_parallel})")
    print(f"{'='*60}")

    in_features, out_features, batch_size, seed = 4096, 4096, 1024, 42
    tp_group = create_tp_group(tp_size) if tp_size > 1 else None
    rank = dist.get_rank() if dist.is_initialized() else 0

    baseline = build_linear(in_features, out_features, "row", 1, None, qparam, seed, False)
    if tp_size > 1:
        sync_linear_weight(baseline, tp_group)

    linear_tp = build_linear(in_features, out_features, "row", tp_size, tp_group, qparam, seed, sequence_parallel)

    # 整数初始化 input 和 weight
    torch.manual_seed(seed)
    inp_full = torch.randint(-1, 2, (batch_size, in_features), device="cuda").to(torch.bfloat16)
    if tp_size > 1:
        broadcast_tensor(inp_full, tp_group)

    with torch.no_grad():
        baseline.weight.data.copy_(
            torch.randint(-1, 2, baseline.weight.shape, device="cuda").to(torch.bfloat16)
        )
    if tp_size > 1:
        sync_linear_weight(baseline, tp_group)
        chunk_in = in_features // tp_size
        linear_tp.weight.data.copy_(
            baseline.weight.data[:, rank * chunk_in:(rank + 1) * chunk_in]
        )

    # row parallel: 按 in_features 维度切分 input
    if tp_size > 1:
        chunk_in = in_features // tp_size
        inp_tp = inp_full[:, rank * chunk_in:(rank + 1) * chunk_in].contiguous()
    else:
        inp_tp = inp_full

    with torch.no_grad():
        out_baseline = baseline(inp_full, is_first_microbatch=True)
        out_tp       = linear_tp(inp_tp, is_first_microbatch=True)

    if rank < tp_size:
        if sequence_parallel and tp_size > 1:
            chunk_seq = batch_size // tp_size
            ref = out_baseline[rank * chunk_seq:(rank + 1) * chunk_seq, :]
        else:
            ref = out_baseline

        diff = (ref - out_tp).abs()
        print(f"  Rank {rank}:  out_tp={out_tp.shape}  "
              f"max_diff={diff.max():.4e}  mean_diff={diff.mean():.4e}")
        torch.testing.assert_close(ref, out_tp, rtol=0, atol=0,
                                   msg=f"Row parallel forward mismatch at rank {rank}")
        print(f"  ✓ Row parallel forward verified")

    print(f"{'='*60}\n")


def test_row_parallel_backward(
    qparam: QuantizeRecipe, tp_size: int = 2, sequence_parallel: bool = True
):
    """
    Row parallel backward 数学等价性验证（bit-exact）。

    dgrad  dx_r = dy @ W_r.T，每个 rank 直接算自己的 in_features 切片，无需通信
      inp_tp.grad [batch, in//tp]  ==  inp_baseline.grad[:, in_shard]

    wgrad  dw_r = dy.T @ x_r，SP 时 dy 先 AllGather，其余与非 SP 一致
      linear_tp.weight.main_grad [out, in//tp]  ==  baseline.weight.main_grad[:, in_shard]
    """
    print(f"\n{'='*60}")
    print(f"Row Parallel Backward    (tp={tp_size}, sp={sequence_parallel})")
    print(f"{'='*60}")

    in_features, out_features, batch_size, seed = 4096, 4096, 1024, 42
    tp_group = create_tp_group(tp_size) if tp_size > 1 else None
    rank = dist.get_rank() if dist.is_initialized() else 0

    baseline = build_linear(in_features, out_features, "row", 1, None, qparam, seed, False)
    if tp_size > 1:
        sync_linear_weight(baseline, tp_group)
    baseline.weight.main_grad = torch.zeros_like(baseline.weight.data, dtype=torch.float32)

    linear_tp = build_linear(in_features, out_features, "row", tp_size, tp_group, qparam, seed, sequence_parallel)
    linear_tp.weight.main_grad = torch.zeros_like(linear_tp.weight.data, dtype=torch.float32)

    # 整数初始化 weight
    torch.manual_seed(seed)
    with torch.no_grad():
        baseline.weight.data.copy_(
            torch.randint(-1, 2, baseline.weight.shape, device="cuda").to(torch.bfloat16)
        )
    if tp_size > 1:
        sync_linear_weight(baseline, tp_group)
        chunk_in = in_features // tp_size
        linear_tp.weight.data.copy_(
            baseline.weight.data[:, rank * chunk_in:(rank + 1) * chunk_in]
        )
        linear_tp.weight.main_grad = torch.zeros_like(linear_tp.weight.data, dtype=torch.float32)

    # 整数初始化 input
    inp_full = torch.randint(-1, 2, (batch_size, in_features), device="cuda").to(torch.bfloat16)
    if tp_size > 1:
        broadcast_tensor(inp_full, tp_group)

    inp_baseline = inp_full.clone().requires_grad_(True)

    if tp_size > 1:
        chunk_in = in_features // tp_size
        inp_tp = inp_full[:, rank * chunk_in:(rank + 1) * chunk_in].contiguous().requires_grad_(True)
    else:
        inp_tp = inp_full.clone().requires_grad_(True)

    # 整数初始化 dy
    dy_full = torch.randint(-1, 2, (batch_size, out_features), device="cuda").to(torch.bfloat16)
    if tp_size > 1:
        broadcast_tensor(dy_full, tp_group)

    # SP=True : out_tp shape [batch//tp, out]（ReduceScatter）→ dy_tp = dy_full[seq_shard]
    # SP=False: out_tp shape [batch, out]（AllReduce）→ dy_tp = dy_full
    if sequence_parallel and tp_size > 1:
        chunk_seq = batch_size // tp_size
        dy_tp = dy_full[rank * chunk_seq:(rank + 1) * chunk_seq, :].contiguous()
    else:
        dy_tp = dy_full.contiguous()

    out_baseline = baseline(inp_baseline, is_first_microbatch=True)
    out_tp       = linear_tp(inp_tp, is_first_microbatch=True)
    out_baseline.backward(dy_full)
    out_tp.backward(dy_tp)

    if rank < tp_size:
        chunk_in = in_features // tp_size

        # dgrad
        if inp_baseline.grad is not None and inp_tp.grad is not None:
            ref_dgrad = inp_baseline.grad[:, rank * chunk_in:(rank + 1) * chunk_in]
            diff = (ref_dgrad - inp_tp.grad).abs()
            print(f"  Rank {rank} dgrad:  shape={inp_tp.grad.shape}  "
                  f"max_diff={diff.max():.4e}  mean_diff={diff.mean():.4e}")
            torch.testing.assert_close(ref_dgrad, inp_tp.grad, rtol=0, atol=0,
                                       msg=f"Row parallel dgrad mismatch at rank {rank}")

        # wgrad
        if baseline.weight.main_grad is not None and linear_tp.weight.main_grad is not None:
            ref_wgrad = baseline.weight.main_grad[:, rank * chunk_in:(rank + 1) * chunk_in]
            diff = (ref_wgrad - linear_tp.weight.main_grad).abs()
            print(f"  Rank {rank} wgrad:  shape={linear_tp.weight.main_grad.shape}  "
                  f"max_diff={diff.max():.4e}  mean_diff={diff.mean():.4e}")
            torch.testing.assert_close(ref_wgrad, linear_tp.weight.main_grad, rtol=0, atol=0,
                                       msg=f"Row parallel wgrad mismatch at rank {rank}")

        print(f"  ✓ Row parallel backward verified")

    print(f"{'='*60}\n")


# ── Runner ────────────────────────────────────────────────────────────────────

_QPARAM_MAP = {
    1: QuantizeRecipe.NON_QUANTIZE,
    2: QuantizeRecipe.INT4_SUB_CHANNEL,
    3: QuantizeRecipe.FP4_SUB_CHANNEL_REF,
    4: QuantizeRecipe.FP4_SUB_CHANNEL,
    5: QuantizeRecipe.FP4_SUB_CHANNEL_QUARTET,
    6: QuantizeRecipe.BF16_REF,
}


def run_tests(qparam_id: int = 6, tp_size: int = 2, sequence_parallel: bool = True):
    qparam = _QPARAM_MAP[qparam_id]
    rank = dist.get_rank() if dist.is_initialized() else 0

    if rank == 0:
        print("\n" + "=" * 60)
        print("Linear TP-SP Correctness Tests")
        print("=" * 60)
        print(f"  quantization  : {qparam}")
        print(f"  tp_size       : {tp_size}")
        print(f"  sequence_par  : {sequence_parallel}")
        print("=" * 60)

    try:
        test_column_parallel_forward(qparam, tp_size, sequence_parallel)
        test_row_parallel_forward(qparam, tp_size, sequence_parallel)
        test_column_parallel_backward(qparam, tp_size, sequence_parallel)
        test_row_parallel_backward(qparam, tp_size, sequence_parallel)

        if rank == 0:
            print("\n" + "=" * 60)
            print("✓ All tests passed!")
            print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ Test failed at rank {rank}: {e}")
        print("=" * 60)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear TP-SP Correctness Tests")
    parser.add_argument("--qparam", type=int, choices=list(_QPARAM_MAP), default=6,
                        help="Quantization recipe (1=NON, 2=INT4, 3=FP4_REF, 4=FP4, 5=FP4_QUARTET, 6=BF16_REF)")
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--sp",    dest="sequence_parallel", action="store_true",  default=True)
    parser.add_argument("--no-sp", dest="sequence_parallel", action="store_false")
    parser.add_argument("--local-rank", type=int, default=0)
    args = parser.parse_args()

    if "RANK" not in os.environ:
        print("ERROR: run with torchrun")
        print("  torchrun --nproc_per_node=2 test_linear_tp.py --qparam 6 --tp-size 2")
        raise SystemExit(1)

    initialize_distributed()
    try:
        run_tests(qparam_id=args.qparam, tp_size=args.tp_size, sequence_parallel=args.sequence_parallel)
    finally:
        cleanup_distributed()
