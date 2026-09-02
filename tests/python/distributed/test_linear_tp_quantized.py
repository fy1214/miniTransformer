"""
Quantized Linear TP communication correctness tests.

With identical quantized weights and inputs, tp=n sharded GEMM + TP collectives
must match tp=1 full GEMM bit-exactly.

Approach:
  1. Warm up tp=1 baseline so weight_qresult_cache is filled.
  2. Inject the matching qw/sw shard into each TP rank's cache.
  3. Run TP Linear with is_first_microbatch=False (read cache, skip requantize).
  4. Use allgather_quantize=False so input quant matches tp=1 after AllGather.

Run:
  torchrun --nproc_per_node=2 test_linear_tp_quantized.py
  torchrun --nproc_per_node=2 test_linear_tp_quantized.py --no-sp
"""

import argparse
import traceback

import torch
import torch.distributed as dist

from helpers import (
    broadcast_tensor,
    cleanup_distributed,
    create_tp_group,
    initialize_distributed,
    require_torchrun,
)
from miniTransformer.module.config import QuantizeRecipe, get_qlinear_params_from_predefined
from miniTransformer.module.linear import Linear


def sync_weight(linear, tp_group):
    """Broadcast rank-0 weight so all ranks share the same baseline."""
    dist.broadcast(linear.weight.data, src=0, group=tp_group)


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


def inject_column_weight_cache(linear_tp, linear_baseline, rank, tp_size):
    """
    Slice baseline weight_qresult_cache along out_features and install on linear_tp.
    With is_first_microbatch=False, TP ranks reuse this cache without requantizing.
    """
    cache = linear_baseline.weight_qresult_cache
    assert cache.data is not None, "baseline weight cache empty; run a warmup forward first"

    # qw is packed uint8; dim0 is out_features
    out_full = cache.data.shape[0]
    chunk = out_full // tp_size
    scale_full = cache.scale.shape[0]
    scale_chunk = scale_full // tp_size

    tp_cache = linear_tp.weight_qresult_cache
    tp_cache.data = cache.data[rank * chunk:(rank + 1) * chunk, :].contiguous()
    tp_cache.scale = cache.scale[rank * scale_chunk:(rank + 1) * scale_chunk, :].contiguous()

    # Columnwise (transposed) tensors: shard on out_features (dim1)
    if cache.data_t is not None:
        out_t_full = cache.data_t.shape[1]
        chunk_t = out_t_full // tp_size
        scale_t_full = cache.scale_t.shape[1]
        chunk_st = scale_t_full // tp_size
        tp_cache.data_t = cache.data_t[:, rank * chunk_t:(rank + 1) * chunk_t].contiguous()
        tp_cache.scale_t = cache.scale_t[:, rank * chunk_st:(rank + 1) * chunk_st].contiguous()

    tp_cache.global_amax_row = cache.global_amax_row
    tp_cache.global_amax_col = cache.global_amax_col
    linear_tp.weight_qresult_cache = tp_cache


def inject_row_weight_cache(linear_tp, linear_baseline, rank, tp_size):
    """
    Slice baseline weight_qresult_cache along in_features and install on linear_tp.
    """
    cache = linear_baseline.weight_qresult_cache
    assert cache.data is not None, "baseline weight cache empty; run a warmup forward first"

    # qw: [out_features, in_features/16 packed]; shard on packed in-dim
    in_packed_full = cache.data.shape[1]
    chunk = in_packed_full // tp_size
    scale_in_full = cache.scale.shape[1]
    scale_chunk = scale_in_full // tp_size

    tp_cache = linear_tp.weight_qresult_cache
    tp_cache.data = cache.data[:, rank * chunk:(rank + 1) * chunk].contiguous()
    tp_cache.scale = cache.scale[:, rank * scale_chunk:(rank + 1) * scale_chunk].contiguous()

    if cache.data_t is not None:
        in_t_full = cache.data_t.shape[0]
        chunk_t = in_t_full // tp_size
        scale_t_in = cache.scale_t.shape[0]
        chunk_st = scale_t_in // tp_size
        tp_cache.data_t = cache.data_t[rank * chunk_t:(rank + 1) * chunk_t, :].contiguous()
        tp_cache.scale_t = cache.scale_t[rank * chunk_st:(rank + 1) * chunk_st, :].contiguous()

    tp_cache.global_amax_row = cache.global_amax_row
    tp_cache.global_amax_col = cache.global_amax_col
    linear_tp.weight_qresult_cache = tp_cache


def _setup(
    parallel_mode, tp_size, sequence_parallel,
    in_features, out_features, total_batch,
    seed, tp_group,
    qparam=QuantizeRecipe.FP4_SUB_CHANNEL,
    params_dtype=torch.bfloat16,
):
    rank = dist.get_rank()

    baseline = build_linear(
        in_features, out_features, parallel_mode,
        tp_size=1, tp_group=None, sequence_parallel=False,
        seed=seed, params_dtype=params_dtype, qparam=qparam,
    )
    sync_weight(baseline, tp_group)

    linear_tp = build_linear(
        in_features, out_features, parallel_mode,
        tp_size=tp_size, tp_group=tp_group, sequence_parallel=sequence_parallel,
        seed=seed, params_dtype=params_dtype, qparam=qparam,
    )

    # Shard BF16 weights from baseline; quantized cache is injected after warmup.
    if parallel_mode == "column":
        chunk = out_features // tp_size
        with torch.no_grad():
            linear_tp.weight.data.copy_(
                baseline.weight.data[rank * chunk:(rank + 1) * chunk, :]
            )
    else:
        chunk = in_features // tp_size
        with torch.no_grad():
            linear_tp.weight.data.copy_(
                baseline.weight.data[:, rank * chunk:(rank + 1) * chunk]
            )

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


def _warmup_baseline_and_inject_cache(s, parallel_mode, tp_size):
    """Warm up baseline weight quant cache, then inject the TP shard into linear_tp."""
    baseline = s["baseline"]
    linear_tp = s["linear_tp"]
    inp_full = s["inp_full"]
    rank = s["rank"]

    with torch.no_grad():
        _ = baseline(inp_full, is_first_microbatch=True)

    if parallel_mode == "column":
        inject_column_weight_cache(linear_tp, baseline, rank, tp_size)
    else:
        inject_row_weight_cache(linear_tp, baseline, rank, tp_size)


def test_column_parallel_forward(
    tp_size=2, sequence_parallel=True,
    in_features=1024, out_features=1024,
    batch_size=64, seq_len=32,
    params_dtype=torch.bfloat16, seed=42,
    qparam=QuantizeRecipe.FP4_SUB_CHANNEL,
):
    """Column parallel forward: identical quantized weights/inputs, compare out shards."""
    tp_group = create_tp_group(tp_size)
    total_batch = batch_size * seq_len

    s = _setup(
        "column", tp_size, sequence_parallel,
        in_features, out_features, total_batch,
        seed, tp_group, qparam=qparam, params_dtype=params_dtype,
    )
    _warmup_baseline_and_inject_cache(s, "column", tp_size)

    rank = s["rank"]
    baseline = s["baseline"]
    linear_tp = s["linear_tp"]
    inp_full = s["inp_full"]
    chunk_out = out_features // tp_size

    with torch.no_grad():
        out_baseline = baseline(inp_full, is_first_microbatch=False)

        if sequence_parallel and tp_size > 1:
            chunk_seq = total_batch // tp_size
            inp_tp = inp_full[rank * chunk_seq:(rank + 1) * chunk_seq, :].contiguous()
        else:
            inp_tp = inp_full
        out_tp = linear_tp(inp_tp, is_first_microbatch=False)

    if rank < tp_size:
        if sequence_parallel and tp_size > 1:
            chunk_seq = total_batch // tp_size
            ref = out_baseline[
                rank * chunk_seq:(rank + 1) * chunk_seq,
                rank * chunk_out:(rank + 1) * chunk_out,
            ]
        else:
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
    tp_size=2, sequence_parallel=True,
    in_features=1024, out_features=1024,
    batch_size=64, seq_len=32,
    params_dtype=torch.bfloat16, seed=42,
    qparam=QuantizeRecipe.FP4_SUB_CHANNEL,
):
    """Column parallel dgrad communication check."""
    tp_group = create_tp_group(tp_size)
    total_batch = batch_size * seq_len

    s = _setup(
        "column", tp_size, sequence_parallel,
        in_features, out_features, total_batch,
        seed, tp_group, qparam=qparam, params_dtype=params_dtype,
    )
    _warmup_baseline_and_inject_cache(s, "column", tp_size)

    rank = s["rank"]
    baseline = s["baseline"]
    linear_tp = s["linear_tp"]
    inp_full = s["inp_full"]
    grad_full = s["grad_full"]
    chunk_out = out_features // tp_size

    inp_b = inp_full.clone().requires_grad_(True)
    out_b = baseline(inp_b, is_first_microbatch=False)
    grad_b = grad_full[:, rank * chunk_out:(rank + 1) * chunk_out].contiguous()
    out_b.backward(grad_b)

    if sequence_parallel and tp_size > 1:
        chunk_seq = total_batch // tp_size
        inp_tp = inp_full[rank * chunk_seq:(rank + 1) * chunk_seq, :].contiguous().requires_grad_(True)
        grad_tp = grad_full[
            rank * chunk_seq:(rank + 1) * chunk_seq,
            rank * chunk_out:(rank + 1) * chunk_out,
        ].contiguous()
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
        print(
            f"  [col bwd dgrad] rank={rank} shape={inp_tp.grad.shape} "
            f"max={diff.max():.4e} mean={diff.mean():.4e}"
        )
        torch.testing.assert_close(
            ref, inp_tp.grad, rtol=0, atol=0,
            msg=f"[col bwd dgrad] rank {rank} mismatch",
        )
        print(f"  ✓ test_column_parallel_backward_dgrad rank={rank} passed")


def test_column_parallel_backward_wgrad(
    tp_size=2, sequence_parallel=True,
    in_features=1024, out_features=1024,
    batch_size=64, seq_len=32,
    params_dtype=torch.bfloat16, seed=42,
    qparam=QuantizeRecipe.FP4_SUB_CHANNEL,
):
    """Column parallel wgrad check vs baseline out-shard."""
    tp_group = create_tp_group(tp_size)
    total_batch = batch_size * seq_len

    s = _setup(
        "column", tp_size, sequence_parallel,
        in_features, out_features, total_batch,
        seed, tp_group, qparam=qparam, params_dtype=params_dtype,
    )
    _warmup_baseline_and_inject_cache(s, "column", tp_size)

    rank = s["rank"]
    baseline = s["baseline"]
    linear_tp = s["linear_tp"]
    inp_full = s["inp_full"]
    grad_full = s["grad_full"]
    chunk_out = out_features // tp_size

    inp_b = inp_full.clone().requires_grad_(True)
    out_b = baseline(inp_b, is_first_microbatch=False)
    grad_b = grad_full[:, rank * chunk_out:(rank + 1) * chunk_out].contiguous()
    out_b.backward(grad_b)

    if sequence_parallel and tp_size > 1:
        chunk_seq = total_batch // tp_size
        inp_tp = inp_full[rank * chunk_seq:(rank + 1) * chunk_seq, :].contiguous().requires_grad_(True)
        grad_tp = grad_full[
            rank * chunk_seq:(rank + 1) * chunk_seq,
            rank * chunk_out:(rank + 1) * chunk_out,
        ].contiguous()
    else:
        inp_tp = inp_full.clone().requires_grad_(True)
        grad_tp = grad_b
    out_tp = linear_tp(inp_tp, is_first_microbatch=False)
    out_tp.backward(grad_tp)

    if rank < tp_size:
        if baseline.weight.grad is not None and linear_tp.weight.grad is not None:
            ref = baseline.weight.grad[rank * chunk_out:(rank + 1) * chunk_out, :]
            diff = (ref - linear_tp.weight.grad).abs()
            print(
                f"  [col bwd wgrad] rank={rank} shape={linear_tp.weight.grad.shape} "
                f"max={diff.max():.4e} mean={diff.mean():.4e}"
            )
            torch.testing.assert_close(
                ref, linear_tp.weight.grad, rtol=0, atol=0,
                msg=f"[col bwd wgrad] rank {rank} mismatch",
            )
            print(f"  ✓ test_column_parallel_backward_wgrad rank={rank} passed")


def test_row_parallel_forward(
    tp_size=2, sequence_parallel=True,
    in_features=1024, out_features=1024,
    batch_size=64, seq_len=32,
    params_dtype=torch.bfloat16, seed=42,
    qparam=QuantizeRecipe.FP4_SUB_CHANNEL,
):
    """Row parallel forward communication check."""
    tp_group = create_tp_group(tp_size)
    total_batch = batch_size * seq_len

    s = _setup(
        "row", tp_size, sequence_parallel,
        in_features, out_features, total_batch,
        seed, tp_group, qparam=qparam, params_dtype=params_dtype,
    )
    _warmup_baseline_and_inject_cache(s, "row", tp_size)

    rank = s["rank"]
    baseline = s["baseline"]
    linear_tp = s["linear_tp"]
    inp_full = s["inp_full"]
    chunk_in = in_features // tp_size

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
    tp_size=2, sequence_parallel=True,
    in_features=1024, out_features=1024,
    batch_size=64, seq_len=32,
    params_dtype=torch.bfloat16, seed=42,
    qparam=QuantizeRecipe.FP4_SUB_CHANNEL,
):
    """Row parallel dgrad communication check."""
    tp_group = create_tp_group(tp_size)
    total_batch = batch_size * seq_len

    s = _setup(
        "row", tp_size, sequence_parallel,
        in_features, out_features, total_batch,
        seed, tp_group, qparam=qparam, params_dtype=params_dtype,
    )
    _warmup_baseline_and_inject_cache(s, "row", tp_size)

    rank = s["rank"]
    baseline = s["baseline"]
    linear_tp = s["linear_tp"]
    inp_full = s["inp_full"]
    grad_full = s["grad_full"]
    chunk_in = in_features // tp_size

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
        print(
            f"  [row bwd dgrad] rank={rank} shape={inp_tp.grad.shape} "
            f"max={diff.max():.4e} mean={diff.mean():.4e}"
        )
        torch.testing.assert_close(
            ref, inp_tp.grad, rtol=0, atol=0,
            msg=f"[row bwd dgrad] rank {rank} mismatch",
        )
        print(f"  ✓ test_row_parallel_backward_dgrad rank={rank} passed")


def test_row_parallel_backward_wgrad(
    tp_size=2, sequence_parallel=True,
    in_features=1024, out_features=1024,
    batch_size=64, seq_len=32,
    params_dtype=torch.bfloat16, seed=42,
    qparam=QuantizeRecipe.FP4_SUB_CHANNEL,
):
    """Row parallel wgrad communication check."""
    tp_group = create_tp_group(tp_size)
    total_batch = batch_size * seq_len

    s = _setup(
        "row", tp_size, sequence_parallel,
        in_features, out_features, total_batch,
        seed, tp_group, qparam=qparam, params_dtype=params_dtype,
    )
    _warmup_baseline_and_inject_cache(s, "row", tp_size)

    rank = s["rank"]
    baseline = s["baseline"]
    linear_tp = s["linear_tp"]
    inp_full = s["inp_full"]
    grad_full = s["grad_full"]
    chunk_in = in_features // tp_size

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
            print(
                f"  [row bwd wgrad] rank={rank} shape={linear_tp.weight.grad.shape} "
                f"max={diff.max():.4e} mean={diff.mean():.4e}"
            )
            torch.testing.assert_close(
                ref, linear_tp.weight.grad, rtol=0, atol=0,
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

QPARAM_MAP = {
    "fp4": QuantizeRecipe.FP4_SUB_CHANNEL,
    "fp4_ref": QuantizeRecipe.FP4_SUB_CHANNEL_REF,
    "fp4_q": QuantizeRecipe.FP4_SUB_CHANNEL_QUARTET,
    "int4": QuantizeRecipe.INT4_SUB_CHANNEL,
}


def run_tests(tp_size=2, sequence_parallel=True, qparam_name="fp4"):
    rank = dist.get_rank()
    qparam = QPARAM_MAP[qparam_name]

    if rank == 0:
        print("\n" + "=" * 60)
        print("Quantized Linear TP Communication Correctness")
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
        print(f"✓ All {len(ALL_TESTS)} tests passed!")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--sp", dest="sequence_parallel", action="store_true", default=True)
    parser.add_argument("--no-sp", dest="sequence_parallel", action="store_false")
    parser.add_argument(
        "--qparam", choices=list(QPARAM_MAP.keys()), default="fp4",
        help="Quantization recipe (default: fp4)",
    )
    args = parser.parse_args()

    require_torchrun(
        "torchrun --nproc_per_node=2 test_linear_tp_quantized.py",
        "torchrun --nproc_per_node=2 test_linear_tp_quantized.py --no-sp --qparam fp4",
    )

    initialize_distributed()
    try:
        run_tests(
            tp_size=args.tp_size,
            sequence_parallel=args.sequence_parallel,
            qparam_name=args.qparam,
        )
    finally:
        cleanup_distributed()
