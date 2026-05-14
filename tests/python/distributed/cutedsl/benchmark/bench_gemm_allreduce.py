"""
bench_gemm_allreduce.py
=======================

row parallel GEMM + AllReduce 性能基准测试。

对比三种方案：
  A: bf16 GEMM + bf16 AllReduce（当前常见实现）
  B: bf16 GEMM + fp32 AllReduce → bf16
  C: CuTeDSL tile-level fp32 AllReduce（本实现）

指标：
  - 中位数 GPU 时间（ms）
  - TFLOPS（含 GEMM FLOP + AllReduce 通信折算 FLOP）
  - MFU（Model FLOP Utilization，相对 B200 bf16 理论峰值）

运行：
  CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=29700 \
    tests/python/distributed/cutedsl/benchmark/bench_gemm_allreduce.py
"""

import os
import math
import statistics
import argparse

import torch
import torch.distributed as dist

from miniTransformer.distributed.cutedsl.nvshmem_communicator import (
    init_nvshmem_with_dist,
    finalize_nvshmem,
)
from miniTransformer.distributed.cutedsl.gemm_allreduce import GemmAllReduceCtx


# ── 测试配置 ──────────────────────────────────────────────────────────────────

SHAPES = [
    (1024, 1024, 1024),
]

WARMUP_ITERS = 2
BENCH_ITERS  = 5

# B200 bf16 理论峰值（TFLOPS）
B200_BF16_PEAK_TFLOPS = 2500.0


# ── FLOP 计算 ─────────────────────────────────────────────────────────────────

def compute_flops(M: int, N: int, K: int, tp_size: int) -> float:
    """
    计算 row parallel GEMM + AllReduce 的总 FLOP。

    GEMM FLOP：每个 rank 做 M × (K//tp) × N 的矩阵乘
      = 2 × M × N × (K // tp_size)

    AllReduce 通信折算 FLOP（ring allreduce）：
      每个元素经过 2×(tp-1)/tp 次加法
      = 2 × (tp_size - 1) / tp_size × M × N

    总 FLOP = GEMM + AllReduce
    """
    gemm_flops = 2.0 * M * N * (K // tp_size)
    ar_flops   = 2.0 * (tp_size - 1) / tp_size * M * N
    return gemm_flops + ar_flops


# ── 计时工具 ──────────────────────────────────────────────────────────────────

class GPUTimer:
    """用 CUDA Event 精确测量 GPU 时间（ms）。"""

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event   = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, *args):
        self.end_event.record()
        torch.cuda.synchronize()

    def elapsed_ms(self) -> float:
        return self.start_event.elapsed_time(self.end_event)


def benchmark_fn(fn, warmup: int, iters: int, pre_call=None) -> list:
    """运行 fn 若干次，返回每次的 GPU 时间（ms）列表。"""
    # warmup
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


# ── 三种方案的实现 ────────────────────────────────────────────────────────────

def scheme_a(a_local, b_local, tp_group):
    """A: bf16 GEMM + bf16 AllReduce"""
    out = a_local @ b_local.T
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=tp_group)
    return out


def scheme_b(a_local, b_local, tp_group):
    """B: bf16 GEMM + fp32 AllReduce → bf16"""
    out = (a_local @ b_local.T).float()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=tp_group)
    return out.to(torch.bfloat16)


def scheme_c(a_local, b_local, ctx):
    """C: CuTeDSL tile-level fp32 AllReduce"""
    return ctx.forward(a_local, b_local)


# ── 单个 shape 的 benchmark ───────────────────────────────────────────────────

def bench_shape(
    M: int, N: int, K: int,
    tp_group,
    ctx: GemmAllReduceCtx,
    seed: int = 42,
) -> dict:
    rank       = dist.get_rank(tp_group)
    world_size = dist.get_world_size(tp_group)
    chunk      = K // world_size

    # 生成输入
    torch.manual_seed(seed)
    a_full = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b_full = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    dist.broadcast(a_full, src=0, group=tp_group)
    dist.broadcast(b_full, src=0, group=tp_group)

    a_local = a_full[:, rank * chunk:(rank + 1) * chunk].contiguous()
    b_local = b_full[:, rank * chunk:(rank + 1) * chunk].contiguous()

    total_flops = compute_flops(M, N, K, world_size)

    def tflops(ms):
        return total_flops / (ms * 1e-3) / 1e12

    def mfu(tf):
        return tf / B200_BF16_PEAK_TFLOPS * 100

    # 方案 A
    times_a = benchmark_fn(
        lambda: scheme_a(a_local, b_local, tp_group),
        WARMUP_ITERS, BENCH_ITERS
    )
    med_a = statistics.median(times_a)

    # 方案 B
    times_b = benchmark_fn(
        lambda: scheme_b(a_local, b_local, tp_group),
        WARMUP_ITERS, BENCH_ITERS
    )
    med_b = statistics.median(times_b)

    # 方案 C（ctx 已在外部 compile 好）
    times_c = benchmark_fn(
        lambda: scheme_c(a_local, b_local, ctx),
        WARMUP_ITERS, BENCH_ITERS,
        pre_call=ctx.barrier_before_reuse,
    )
    med_c = statistics.median(times_c)

    return {
        "M": M, "N": N, "K": K,
        "flops": total_flops,
        "A_ms": med_a, "A_tflops": tflops(med_a), "A_mfu": mfu(tflops(med_a)),
        "B_ms": med_b, "B_tflops": tflops(med_b), "B_mfu": mfu(tflops(med_b)),
        "C_ms": med_c, "C_tflops": tflops(med_c), "C_mfu": mfu(tflops(med_c)),
        "C_vs_A_speedup": med_a / med_c,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if "RANK" not in os.environ:
        print("请用 torchrun 启动：")
        print("  CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 "
              "--master_addr=127.0.0.1 --master_port=29700 "
              "tests/python/distributed/cutedsl/benchmark/bench_gemm_allreduce.py")
        raise SystemExit(1)

    # 初始化
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dev = init_nvshmem_with_dist()

    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    tp_group   = dist.new_group(ranks=list(range(world_size)))

    if rank == 0:
        print(f"\n{'='*78}")
        print(f"GEMM AllReduce Benchmark  (tp={world_size}, dtype=bf16, B200)")
        print(f"B200 bf16 理论峰值: {B200_BF16_PEAK_TFLOPS:.0f} TFLOPS")
        print(f"Warmup={WARMUP_ITERS}  Bench={BENCH_ITERS}  取中位数")
        print(f"{'='*78}")
        print(f"{'M':>5} {'N':>5} {'K':>5} | "
              f"{'A ms':>8} {'A TFLOPS':>10} {'A MFU%':>7} | "
              f"{'B ms':>8} {'B TFLOPS':>10} {'B MFU%':>7} | "
              f"{'C ms':>8} {'C TFLOPS':>10} {'C MFU%':>7} | "
              f"{'C/A':>6}")
        print(f"{'-'*78}")

    results = []

    # ctx 只创建一次并复用——nvshmem alloc 很慢，每次 shape 都重建会累积显存泄漏
    # 注意：ctx 的 M/N 在 _compile 时固定，不同 shape 需要不同 ctx
    for M, N, K in SHAPES:
        if K % world_size != 0:
            if rank == 0:
                print(f"  跳过 ({M},{N},{K})：K 不能被 tp_size={world_size} 整除")
            continue

        ctx = GemmAllReduceCtx(tp_group=tp_group)

        # warmup compile（第一次 forward 触发 JIT）
        chunk = K // world_size
        torch.manual_seed(0)
        a_tmp = torch.randn(M, chunk, device="cuda", dtype=torch.bfloat16)
        b_tmp = torch.randn(N, chunk, device="cuda", dtype=torch.bfloat16)
        _ = ctx.forward(a_tmp, b_tmp)   # 触发 compile
        torch.cuda.synchronize()
        if rank == 0:
            print(f"  [{M},{N},{K}] compiled, benchmarking...")

        r = bench_shape(M, N, K, tp_group, ctx)
        results.append(r)

        if rank == 0:
            print(f"{r['M']:>5} {r['N']:>5} {r['K']:>5} | "
                  f"{r['A_ms']:>8.2f} {r['A_tflops']:>10.2f} {r['A_mfu']:>6.1f}% | "
                  f"{r['B_ms']:>8.2f} {r['B_tflops']:>10.2f} {r['B_mfu']:>6.1f}% | "
                  f"{r['C_ms']:>8.2f} {r['C_tflops']:>10.2f} {r['C_mfu']:>6.1f}% | "
                  f"{r['C_vs_A_speedup']:>5.2f}x")

        ctx.free()
        dist.barrier(group=tp_group)

    if rank == 0:
        print(f"{'='*78}\n")
        print("说明：")
        print("  A = bf16 GEMM + bf16 AllReduce")
        print("  B = bf16 GEMM + fp32 AllReduce → bf16")
        print("  C = CuTeDSL tile-level fp32 AllReduce")
        print(f"  FLOP = 2×M×N×(K/tp) [GEMM] + 2×(tp-1)/tp×M×N [AllReduce ring]")
        print(f"  MFU  = TFLOPS / {B200_BF16_PEAK_TFLOPS:.0f} TFLOPS (B200 bf16 peak)")

    finalize_nvshmem()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
