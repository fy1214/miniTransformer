"""Row-parallel GEMM + AllReduce performance benchmark.

Schemes:
  A: bf16 GEMM + bf16 AllReduce
  B: bf16 GEMM + fp32 AllReduce -> bf16
  C: CuTeDSL tile-level fp32 AllReduce

Reports median GPU time, TFLOPS, and MFU vs a B200 bf16 peak constant.

Usage:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \\
    --master_addr=127.0.0.1 --master_port=29700 \\
    tests/python/distributed/cutedsl/benchmark/bench_gemm_allreduce.py
"""

import os
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from miniTransformer.distributed.cutedsl.gemm_allreduce import GemmAllReduceCtx

_CUTEDSL = Path(__file__).resolve().parents[1]
if str(_CUTEDSL) not in sys.path:
    sys.path.insert(0, str(_CUTEDSL))

from helpers import (  # noqa: E402
    cleanup_nvshmem_distributed,
    initialize_nvshmem_distributed,
    make_world_tp_group,
    scheme_a,
    scheme_b,
    scheme_c,
    shard_k,
)

SHAPES = [
    (1024, 1024, 1024),
]
WARMUP_ITERS = 2
BENCH_ITERS = 5
B200_BF16_PEAK_TFLOPS = 2500.0


def compute_flops(M: int, N: int, K: int, tp_size: int) -> float:
    """Total FLOP = local GEMM + ring AllReduce arithmetic model."""
    gemm_flops = 2.0 * M * N * (K // tp_size)
    ar_flops = 2.0 * (tp_size - 1) / tp_size * M * N
    return gemm_flops + ar_flops


class GPUTimer:
    """CUDA Event timer (ms)."""

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, *args):
        self.end_event.record()
        torch.cuda.synchronize()

    def elapsed_ms(self) -> float:
        return self.start_event.elapsed_time(self.end_event)


def benchmark_fn(fn, warmup: int, iters: int, pre_call=None) -> list:
    for _ in range(warmup):
        if pre_call is not None:
            pre_call()
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if pre_call is not None:
            pre_call()
        with GPUTimer() as t:
            fn()
        times.append(t.elapsed_ms())
    return times


def bench_shape(M: int, N: int, K: int, tp_group, ctx: GemmAllReduceCtx, seed: int = 42) -> dict:
    rank = dist.get_rank(tp_group)
    world_size = dist.get_world_size(tp_group)

    torch.manual_seed(seed)
    a_full = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b_full = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    dist.broadcast(a_full, src=0, group=tp_group)
    dist.broadcast(b_full, src=0, group=tp_group)
    a_local, b_local, _ = shard_k(a_full, b_full, rank, world_size)

    total_flops = compute_flops(M, N, K, world_size)

    def tflops(ms):
        return total_flops / (ms * 1e-3) / 1e12

    def mfu(tf):
        return tf / B200_BF16_PEAK_TFLOPS * 100

    med_a = statistics.median(
        benchmark_fn(lambda: scheme_a(a_local, b_local, tp_group), WARMUP_ITERS, BENCH_ITERS)
    )
    med_b = statistics.median(
        benchmark_fn(lambda: scheme_b(a_local, b_local, tp_group), WARMUP_ITERS, BENCH_ITERS)
    )
    med_c = statistics.median(
        benchmark_fn(
            lambda: scheme_c(a_local, b_local, ctx),
            WARMUP_ITERS,
            BENCH_ITERS,
            pre_call=ctx.barrier_before_reuse,
        )
    )

    return {
        "M": M,
        "N": N,
        "K": K,
        "flops": total_flops,
        "A_ms": med_a,
        "A_tflops": tflops(med_a),
        "A_mfu": mfu(tflops(med_a)),
        "B_ms": med_b,
        "B_tflops": tflops(med_b),
        "B_mfu": mfu(tflops(med_b)),
        "C_ms": med_c,
        "C_tflops": tflops(med_c),
        "C_mfu": mfu(tflops(med_c)),
        "C_vs_A_speedup": med_a / med_c,
    }


def main():
    if "RANK" not in os.environ:
        print("ERROR: run with torchrun")
        print(
            "  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 "
            "--master_addr=127.0.0.1 --master_port=29700 "
            "tests/python/distributed/cutedsl/benchmark/bench_gemm_allreduce.py"
        )
        raise SystemExit(1)

    initialize_nvshmem_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tp_group = make_world_tp_group()

    if rank == 0:
        print(f"\n{'='*78}")
        print(f"GEMM AllReduce Benchmark  (tp={world_size}, dtype=bf16)")
        print(f"B200 bf16 peak constant: {B200_BF16_PEAK_TFLOPS:.0f} TFLOPS")
        print(f"Warmup={WARMUP_ITERS}  Bench={BENCH_ITERS}  (median)")
        print(f"{'='*78}")
        print(
            f"{'M':>5} {'N':>5} {'K':>5} | "
            f"{'A ms':>8} {'A TFLOPS':>10} {'A MFU%':>7} | "
            f"{'B ms':>8} {'B TFLOPS':>10} {'B MFU%':>7} | "
            f"{'C ms':>8} {'C TFLOPS':>10} {'C MFU%':>7} | "
            f"{'C/A':>6}"
        )
        print(f"{'-'*78}")

    for M, N, K in SHAPES:
        if K % world_size != 0:
            if rank == 0:
                print(f"  skip ({M},{N},{K}): K not divisible by tp_size={world_size}")
            continue

        ctx = GemmAllReduceCtx(tp_group=tp_group)
        chunk = K // world_size
        torch.manual_seed(0)
        a_tmp = torch.randn(M, chunk, device="cuda", dtype=torch.bfloat16)
        b_tmp = torch.randn(N, chunk, device="cuda", dtype=torch.bfloat16)
        _ = ctx.forward(a_tmp, b_tmp)
        ctx.barrier_before_reuse()
        if rank == 0:
            print(f"  [{M},{N},{K}] compiled, benchmarking...")

        r = bench_shape(M, N, K, tp_group, ctx)
        if rank == 0:
            print(
                f"{r['M']:>5} {r['N']:>5} {r['K']:>5} | "
                f"{r['A_ms']:>8.2f} {r['A_tflops']:>10.2f} {r['A_mfu']:>6.1f}% | "
                f"{r['B_ms']:>8.2f} {r['B_tflops']:>10.2f} {r['B_mfu']:>6.1f}% | "
                f"{r['C_ms']:>8.2f} {r['C_tflops']:>10.2f} {r['C_mfu']:>6.1f}% | "
                f"{r['C_vs_A_speedup']:>5.2f}x"
            )

        ctx.barrier_before_reuse()
        ctx.free()
        dist.barrier(group=tp_group)

    if rank == 0:
        print(f"{'='*78}\n")
        print("Legend:")
        print("  A = bf16 GEMM + bf16 AllReduce")
        print("  B = bf16 GEMM + fp32 AllReduce -> bf16")
        print("  C = CuTeDSL tile-level fp32 AllReduce")
        print("  FLOP = 2*M*N*(K/tp) [GEMM] + 2*(tp-1)/tp*M*N [AllReduce ring]")
        print(f"  MFU  = TFLOPS / {B200_BF16_PEAK_TFLOPS:.0f} TFLOPS")

    cleanup_nvshmem_distributed()


if __name__ == "__main__":
    main()
