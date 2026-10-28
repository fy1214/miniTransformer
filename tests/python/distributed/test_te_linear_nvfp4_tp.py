"""
TE Linear TP bf16 communication correctness tests.

Validates Transformer Engine Linear AllGather / ReduceScatter / AllReduce under
tensor parallel (column and row) without quantization. Inputs, weights, and grads
use integer values in {-1, 0, 1} so bf16 GEMMs are bit-exact (rtol=0, atol=0).

Run:
  torchrun --nproc_per_node=2 test_te_linear_nvfp4_tp.py
  torchrun --nproc_per_node=2 test_te_linear_nvfp4_tp.py --no-sp
"""

import argparse
import traceback

import torch
import torch.distributed as dist
import transformer_engine.pytorch as te

from helpers import (
    broadcast_tensor,
    cleanup_distributed,
    create_tp_group,
    initialize_distributed,
    randint_bf16,
    require_torchrun,
)


def broadcast_weight(linear, tp_group):
    """Broadcast rank-0 weight (and bias if present) to all ranks in the TP group."""
    dist.broadcast(linear.weight.data, src=0, group=tp_group)
    if hasattr(linear, "bias") and linear.bias is not None and linear.bias.numel() > 0:
        dist.broadcast(linear.bias.data, src=0, group=tp_group)


def build_baseline(in_features, out_features, parallel_mode, bias, seed) -> te.Linear:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return te.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        params_dtype=torch.bfloat16,
        parallel_mode=parallel_mode,
        tp_size=1,
        device="cuda",
    )


def build_tp(
    in_features, out_features, parallel_mode, bias,
    tp_size, tp_group, sequence_parallel, seed,
) -> te.Linear:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return te.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        params_dtype=torch.bfloat16,
        parallel_mode=parallel_mode,
        tp_size=tp_size,
        tp_group=tp_group,
        sequence_parallel=sequence_parallel,
        device="cuda",
    )


def copy_column_shard(linear_tp, linear_baseline, rank, tp_size):
    """Install out-feature shard of baseline weight onto the column-parallel module."""
    chunk = linear_baseline.weight.shape[0] // tp_size
    shard = linear_baseline.weight.data[rank * chunk:(rank + 1) * chunk, :].contiguous().clone()
    linear_tp.weight = torch.nn.Parameter(shard)


def copy_row_shard(linear_tp, linear_baseline, rank, tp_size):
    """Install in-feature shard of baseline weight onto the row-parallel module."""
    chunk = linear_baseline.weight.shape[1] // tp_size
    shard = linear_baseline.weight.data[:, rank * chunk:(rank + 1) * chunk].contiguous().clone()
    linear_tp.weight = torch.nn.Parameter(shard)


def _setup_column(
    tp_size, sequence_parallel, in_features, out_features,
    total_batch, bias, seed, tp_group,
):
    rank = dist.get_rank()

    baseline = build_baseline(in_features, out_features, "column", bias, seed)
    if tp_size > 1:
        broadcast_weight(baseline, tp_group)

    tp_linear = build_tp(
        in_features, out_features, "column", bias,
        tp_size, tp_group, sequence_parallel, seed,
    )
    if tp_size > 1:
        copy_column_shard(tp_linear, baseline, rank, tp_size)

    torch.manual_seed(seed + 10)
    with torch.no_grad():
        baseline.weight.data.copy_(randint_bf16(*baseline.weight.shape))
    if tp_size > 1:
        broadcast_weight(baseline, tp_group)
        copy_column_shard(tp_linear, baseline, rank, tp_size)

    torch.manual_seed(seed + 100)
    inp_full = randint_bf16(total_batch, in_features)
    if tp_size > 1:
        broadcast_tensor(inp_full, tp_group)

    torch.manual_seed(seed + 200)
    grad_full = randint_bf16(total_batch, out_features)
    if tp_size > 1:
        broadcast_tensor(grad_full, tp_group)

    chunk_out = out_features // tp_size

    if sequence_parallel and tp_size > 1:
        chunk_seq = total_batch // tp_size
        inp_tp_data = inp_full[rank * chunk_seq:(rank + 1) * chunk_seq, :].contiguous()
    else:
        inp_tp_data = inp_full

    # Column parallel output is always [total_batch, out//tp].
    grad_tp = grad_full[:, rank * chunk_out:(rank + 1) * chunk_out].contiguous()

    return dict(
        baseline=baseline, tp_linear=tp_linear,
        inp_full=inp_full, inp_tp_data=inp_tp_data,
        grad_full=grad_full, grad_tp=grad_tp,
        chunk_out=chunk_out,
    )


def _setup_row(
    tp_size, sequence_parallel, in_features, out_features,
    total_batch, bias, seed, tp_group,
):
    rank = dist.get_rank()

    baseline = build_baseline(in_features, out_features, "row", bias, seed)
    if tp_size > 1:
        broadcast_weight(baseline, tp_group)

    tp_linear = build_tp(
        in_features, out_features, "row", bias,
        tp_size, tp_group, sequence_parallel, seed,
    )
    if tp_size > 1:
        copy_row_shard(tp_linear, baseline, rank, tp_size)

    torch.manual_seed(seed + 10)
    with torch.no_grad():
        baseline.weight.data.copy_(randint_bf16(*baseline.weight.shape))
    if tp_size > 1:
        broadcast_weight(baseline, tp_group)
        copy_row_shard(tp_linear, baseline, rank, tp_size)

    torch.manual_seed(seed + 100)
    inp_full = randint_bf16(total_batch, in_features)
    if tp_size > 1:
        broadcast_tensor(inp_full, tp_group)

    torch.manual_seed(seed + 200)
    grad_full = randint_bf16(total_batch, out_features)
    if tp_size > 1:
        broadcast_tensor(grad_full, tp_group)

    chunk_in = in_features // tp_size

    if tp_size > 1:
        inp_tp_data = inp_full[:, rank * chunk_in:(rank + 1) * chunk_in].contiguous()
    else:
        inp_tp_data = inp_full

    # SP: ReduceScatter -> [total_batch//tp, out]; else AllReduce -> full batch.
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


def test_column_parallel_forward(
    tp_size: int, sequence_parallel: bool,
    in_features: int = 512, out_features: int = 512,
    batch_size: int = 32, seq_len: int = 32,
    bias: bool = False, seed: int = 42,
):
    """Column parallel forward: out_tp vs baseline[:, col_shard] (bit-exact)."""
    tp_group = create_tp_group(tp_size) if tp_size > 1 else None
    rank = dist.get_rank()
    total_batch = batch_size * seq_len

    s = _setup_column(
        tp_size, sequence_parallel, in_features, out_features,
        total_batch, bias, seed, tp_group,
    )

    with torch.no_grad():
        out_baseline = s["baseline"](s["inp_full"].clone(), is_first_microbatch=True)
        out_tp = s["tp_linear"](s["inp_tp_data"].clone(), is_first_microbatch=True)

    if rank < tp_size:
        chunk_out = s["chunk_out"]
        ref = out_baseline[:, rank * chunk_out:(rank + 1) * chunk_out]

        diff = (ref - out_tp).abs()
        print(
            f"  [col fwd] rank={rank} out_tp={out_tp.shape} "
            f"max={diff.max():.4e} mean={diff.mean():.4e}"
        )
        torch.testing.assert_close(
            ref, out_tp, rtol=0, atol=0,
            msg=f"[col fwd] rank {rank} mismatch",
        )
        print(f"  ✓ test_column_parallel_forward rank={rank} passed")


def test_column_parallel_backward_dgrad(
    tp_size: int, sequence_parallel: bool,
    in_features: int = 512, out_features: int = 512,
    batch_size: int = 32, seq_len: int = 32,
    bias: bool = False, seed: int = 42,
):
    """
    Column parallel dgrad: TP AllReduce/ReduceScatter of per-shard dy @ W.T
    must match baseline dgrad (full or seq shard under SP).
    """
    tp_group = create_tp_group(tp_size) if tp_size > 1 else None
    rank = dist.get_rank()
    total_batch = batch_size * seq_len

    s = _setup_column(
        tp_size, sequence_parallel, in_features, out_features,
        total_batch, bias, seed, tp_group,
    )

    inp_baseline = s["inp_full"].clone().requires_grad_(True)
    inp_tp = s["inp_tp_data"].clone().requires_grad_(True)

    out_baseline = s["baseline"](inp_baseline, is_first_microbatch=True)
    out_baseline.backward(s["grad_full"])

    out_tp = s["tp_linear"](inp_tp, is_first_microbatch=True)
    out_tp.backward(s["grad_tp"])

    if rank < tp_size and inp_baseline.grad is not None and inp_tp.grad is not None:
        if sequence_parallel and tp_size > 1:
            chunk_seq = total_batch // tp_size
            ref = inp_baseline.grad[rank * chunk_seq:(rank + 1) * chunk_seq, :]
        else:
            ref = inp_baseline.grad

        diff = (ref - inp_tp.grad).abs()
        print(
            f"  [col bwd dgrad] rank={rank} grad_shape={inp_tp.grad.shape} "
            f"max={diff.max():.4e} mean={diff.mean():.4e}"
        )
        torch.testing.assert_close(
            ref, inp_tp.grad, rtol=0, atol=0,
            msg=f"[col bwd dgrad] rank {rank} mismatch",
        )
        print(f"  ✓ test_column_parallel_backward_dgrad rank={rank} passed")


def test_column_parallel_backward_wgrad(
    tp_size: int, sequence_parallel: bool,
    in_features: int = 512, out_features: int = 512,
    batch_size: int = 32, seq_len: int = 32,
    bias: bool = False, seed: int = 42,
):
    """Column parallel wgrad: local out-shard vs baseline.weight.grad[col_shard]."""
    tp_group = create_tp_group(tp_size) if tp_size > 1 else None
    rank = dist.get_rank()
    total_batch = batch_size * seq_len

    s = _setup_column(
        tp_size, sequence_parallel, in_features, out_features,
        total_batch, bias, seed, tp_group,
    )

    inp_baseline = s["inp_full"].clone().requires_grad_(True)
    inp_tp = s["inp_tp_data"].clone().requires_grad_(True)

    out_baseline = s["baseline"](inp_baseline, is_first_microbatch=True)
    out_baseline.backward(s["grad_full"])

    out_tp = s["tp_linear"](inp_tp, is_first_microbatch=True)
    out_tp.backward(s["grad_tp"])

    if rank < tp_size:
        if s["baseline"].weight.grad is not None and s["tp_linear"].weight.grad is not None:
            chunk_out = s["chunk_out"]
            ref = s["baseline"].weight.grad[rank * chunk_out:(rank + 1) * chunk_out, :]
            diff = (ref - s["tp_linear"].weight.grad).abs()
            print(
                f"  [col bwd wgrad] rank={rank} wgrad_shape={s['tp_linear'].weight.grad.shape} "
                f"max={diff.max():.4e} mean={diff.mean():.4e}"
            )
            torch.testing.assert_close(
                ref, s["tp_linear"].weight.grad, rtol=0, atol=0,
                msg=f"[col bwd wgrad] rank {rank} mismatch",
            )
            print(f"  ✓ test_column_parallel_backward_wgrad rank={rank} passed")


def test_row_parallel_forward(
    tp_size: int, sequence_parallel: bool,
    in_features: int = 512, out_features: int = 512,
    batch_size: int = 32, seq_len: int = 32,
    bias: bool = False, seed: int = 42,
):
    """Row parallel forward: ReduceScatter (SP) or AllReduce vs baseline."""
    tp_group = create_tp_group(tp_size) if tp_size > 1 else None
    rank = dist.get_rank()
    total_batch = batch_size * seq_len

    s = _setup_row(
        tp_size, sequence_parallel, in_features, out_features,
        total_batch, bias, seed, tp_group,
    )

    with torch.no_grad():
        out_baseline = s["baseline"](s["inp_full"].clone(), is_first_microbatch=True)
        out_tp = s["tp_linear"](s["inp_tp_data"].clone(), is_first_microbatch=True)

    if rank < tp_size:
        if sequence_parallel and tp_size > 1:
            chunk_seq = total_batch // tp_size
            ref = out_baseline[rank * chunk_seq:(rank + 1) * chunk_seq, :]
        else:
            ref = out_baseline

        diff = (ref - out_tp).abs()
        print(
            f"  [row fwd] rank={rank} out_tp={out_tp.shape} "
            f"max={diff.max():.4e} mean={diff.mean():.4e}"
        )
        torch.testing.assert_close(
            ref, out_tp, rtol=0, atol=0,
            msg=f"[row fwd] rank {rank} mismatch",
        )
        print(f"  ✓ test_row_parallel_forward rank={rank} passed")


def test_row_parallel_backward_dgrad(
    tp_size: int, sequence_parallel: bool,
    in_features: int = 512, out_features: int = 512,
    batch_size: int = 32, seq_len: int = 32,
    bias: bool = False, seed: int = 42,
):
    """Row parallel dgrad: local in-shard vs baseline.grad[:, in_shard]."""
    tp_group = create_tp_group(tp_size) if tp_size > 1 else None
    rank = dist.get_rank()
    total_batch = batch_size * seq_len

    s = _setup_row(
        tp_size, sequence_parallel, in_features, out_features,
        total_batch, bias, seed, tp_group,
    )

    inp_baseline = s["inp_full"].clone().requires_grad_(True)
    inp_tp = s["inp_tp_data"].clone().requires_grad_(True)

    out_baseline = s["baseline"](inp_baseline, is_first_microbatch=True)
    out_baseline.backward(s["grad_full"])

    out_tp = s["tp_linear"](inp_tp, is_first_microbatch=True)
    out_tp.backward(s["grad_tp"])

    if rank < tp_size and inp_baseline.grad is not None and inp_tp.grad is not None:
        chunk_in = s["chunk_in"]
        ref = inp_baseline.grad[:, rank * chunk_in:(rank + 1) * chunk_in]
        diff = (ref - inp_tp.grad).abs()
        print(
            f"  [row bwd dgrad] rank={rank} grad_shape={inp_tp.grad.shape} "
            f"max={diff.max():.4e} mean={diff.mean():.4e}"
        )
        torch.testing.assert_close(
            ref, inp_tp.grad, rtol=0, atol=0,
            msg=f"[row bwd dgrad] rank {rank} mismatch",
        )
        print(f"  ✓ test_row_parallel_backward_dgrad rank={rank} passed")


def test_row_parallel_backward_wgrad(
    tp_size: int, sequence_parallel: bool,
    in_features: int = 512, out_features: int = 512,
    batch_size: int = 32, seq_len: int = 32,
    bias: bool = False, seed: int = 42,
):
    """Row parallel wgrad: local in-shard vs baseline.weight.grad[:, in_shard]."""
    tp_group = create_tp_group(tp_size) if tp_size > 1 else None
    rank = dist.get_rank()
    total_batch = batch_size * seq_len

    s = _setup_row(
        tp_size, sequence_parallel, in_features, out_features,
        total_batch, bias, seed, tp_group,
    )

    inp_baseline = s["inp_full"].clone().requires_grad_(True)
    inp_tp = s["inp_tp_data"].clone().requires_grad_(True)

    out_baseline = s["baseline"](inp_baseline, is_first_microbatch=True)
    out_baseline.backward(s["grad_full"])

    out_tp = s["tp_linear"](inp_tp, is_first_microbatch=True)
    out_tp.backward(s["grad_tp"])

    if rank < tp_size:
        if s["baseline"].weight.grad is not None and s["tp_linear"].weight.grad is not None:
            chunk_in = s["chunk_in"]
            ref = s["baseline"].weight.grad[:, rank * chunk_in:(rank + 1) * chunk_in]
            diff = (ref - s["tp_linear"].weight.grad).abs()
            print(
                f"  [row bwd wgrad] rank={rank} wgrad_shape={s['tp_linear'].weight.grad.shape} "
                f"max={diff.max():.4e} mean={diff.mean():.4e}"
            )
            torch.testing.assert_close(
                ref, s["tp_linear"].weight.grad, rtol=0, atol=0,
                msg=f"[row bwd wgrad] rank {rank} mismatch",
            )
            print(f"  ✓ test_row_parallel_backward_wgrad rank={rank} passed")


ALL_TESTS = [
    ("Column Parallel Forward", test_column_parallel_forward),
    ("Column Parallel Backward dgrad", test_column_parallel_backward_dgrad),
    ("Column Parallel Backward wgrad", test_column_parallel_backward_wgrad),
    ("Row Parallel Forward", test_row_parallel_forward),
    ("Row Parallel Backward dgrad", test_row_parallel_backward_dgrad),
    ("Row Parallel Backward wgrad", test_row_parallel_backward_wgrad),
]


def run_tests(tp_size: int = 2, sequence_parallel: bool = True):
    rank = dist.get_rank()

    if rank == 0:
        print("\n" + "=" * 60)
        print("TE Linear TP bf16 Correctness (integer inputs, bit-exact)")
        print("=" * 60)
        print(f"  tp_size:           {tp_size}")
        print(f"  sequence_parallel: {sequence_parallel}")
        print("=" * 60)

    for i, (name, fn) in enumerate(ALL_TESTS, 1):
        if rank == 0:
            print(f"\n[{i}/{len(ALL_TESTS)}] {name}")
        try:
            fn(tp_size=tp_size, sequence_parallel=sequence_parallel)
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
    parser.add_argument("--sp", dest="sequence_parallel", action="store_true", default=True)
    parser.add_argument("--no-sp", dest="sequence_parallel", action="store_false")
    args = parser.parse_args()

    require_torchrun(
        "torchrun --nproc_per_node=2 test_te_linear_nvfp4_tp.py",
        "torchrun --nproc_per_node=2 test_te_linear_nvfp4_tp.py --no-sp",
    )

    initialize_distributed()
    try:
        run_tests(tp_size=args.tp_size, sequence_parallel=args.sequence_parallel)
    finally:
        cleanup_distributed()
