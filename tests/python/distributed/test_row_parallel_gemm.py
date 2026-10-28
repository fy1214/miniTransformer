"""Row-parallel GEMM + AllReduce math check (integer and random bf16).

tp=1: o = x @ w.T
tp=2: each rank GEMMs a K-shard, then AllReduce; sum must match o.

Integer path uses x,w in {-1,0,1} for bit-exact bf16 checks.
Random path compares bf16/fp32 GEMM+AllReduce variants vs tp=1 bf16.

Usage:
  torchrun --nproc_per_node=2 test_row_parallel_gemm.py
"""

import traceback

import torch
import torch.distributed as dist

from helpers import (
    cleanup_distributed,
    initialize_distributed,
    require_torchrun,
)


def test_row_parallel_gemm(M: int = 1024, K: int = 4096, N: int = 1024, seed: int = 42):
    """x @ w.T == sum_r(x_r @ w_r.T) after AllReduce (bit-exact on integer inputs)."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, f"this test requires 2 ranks, got world_size={world_size}"
    assert K % world_size == 0, f"K={K} must be divisible by tp_size={world_size}"

    chunk = K // world_size
    x = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(N, K, device="cuda", dtype=torch.bfloat16)
    if rank == 0:
        torch.manual_seed(seed)
        x = torch.randint(-1, 2, (M, K), device="cuda").to(torch.bfloat16)
        w = torch.randint(-1, 2, (N, K), device="cuda").to(torch.bfloat16)
    dist.broadcast(x, src=0)
    dist.broadcast(w, src=0)

    o_baseline = x @ w.T
    x_local = x[:, rank * chunk : (rank + 1) * chunk].contiguous()
    w_local = w[:, rank * chunk : (rank + 1) * chunk].contiguous()
    o_local = x_local @ w_local.T

    o_allreduce = o_local.clone()
    dist.all_reduce(o_allreduce, op=dist.ReduceOp.SUM)

    diff = (o_allreduce - o_baseline).abs()
    print(f"  Rank {rank}:")
    print(f"    x shape:           {x.shape}  dtype={x.dtype}  values in {{-1,0,1}}")
    print(f"    w shape:           {w.shape}")
    print(f"    x_local shape:     {x_local.shape}  (rank {rank} shard)")
    print(f"    w_local shape:     {w_local.shape}  (rank {rank} shard)")
    print(f"    o_baseline range:  [{o_baseline.min():.0f}, {o_baseline.max():.0f}]")
    print(f"    o_allreduce range: [{o_allreduce.min():.0f}, {o_allreduce.max():.0f}]")
    print(f"    max abs diff:      {diff.max():.4e}")
    print(f"    mean abs diff:     {diff.mean():.4e}")

    torch.testing.assert_close(
        o_allreduce,
        o_baseline,
        rtol=0,
        atol=0,
        msg=f"Rank {rank}: AllReduce result != full GEMM",
    )
    if rank == 0:
        print("  o0 + o1 == o (bf16 integer bit-exact)")


def test_fp32_allreduce_precision(
    M: int = 1024, K: int = 4096, N: int = 1024, seed: int = 42
):
    """Compare row-parallel GEMM+AllReduce precision variants vs tp=1 bf16."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2
    assert K % world_size == 0

    chunk = K // world_size
    x = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(N, K, device="cuda", dtype=torch.bfloat16)
    if rank == 0:
        torch.manual_seed(seed)
        x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
        w = torch.randn(N, K, device="cuda").to(torch.bfloat16)
    dist.broadcast(x, src=0)
    dist.broadcast(w, src=0)

    o_baseline = x @ w.T
    x_local = x[:, rank * chunk : (rank + 1) * chunk].contiguous()
    w_local = w[:, rank * chunk : (rank + 1) * chunk].contiguous()

    o_a = (x_local @ w_local.T).clone()
    dist.all_reduce(o_a, op=dist.ReduceOp.SUM)

    o_b = (x_local @ w_local.T).float()
    dist.all_reduce(o_b, op=dist.ReduceOp.SUM)
    o_b = o_b.to(torch.bfloat16)

    o_c = x_local.float() @ w_local.float().T
    dist.all_reduce(o_c, op=dist.ReduceOp.SUM)
    o_c = o_c.to(torch.bfloat16)

    o_d = (x_local.float() @ w_local.float().T).to(torch.bfloat16)
    dist.all_reduce(o_d, op=dist.ReduceOp.SUM)

    if rank == 0:
        diff_a = (o_a.float() - o_baseline.float()).abs()
        diff_b = (o_b.float() - o_baseline.float()).abs()
        diff_c = (o_c.float() - o_baseline.float()).abs()
        diff_d = (o_d.float() - o_baseline.float()).abs()

        print("\n  Precision vs tp=1 bf16 GEMM:")
        print(f"  {'scheme':<42} {'max diff':>12} {'mean diff':>12}")
        print(f"  {'-'*66}")
        print(
            f"  {'A: bf16 GEMM + bf16 AllReduce':<42} "
            f"{diff_a.max().item():>12.4e} {diff_a.mean().item():>12.4e}"
        )
        print(
            f"  {'B: bf16 GEMM + fp32 AllReduce -> bf16':<42} "
            f"{diff_b.max().item():>12.4e} {diff_b.mean().item():>12.4e}"
        )
        print(
            f"  {'C: fp32 GEMM + fp32 AllReduce -> bf16':<42} "
            f"{diff_c.max().item():>12.4e} {diff_c.mean().item():>12.4e}"
        )
        print(
            f"  {'D: fp32 GEMM + bf16 AllReduce':<42} "
            f"{diff_d.max().item():>12.4e} {diff_d.mean().item():>12.4e}"
        )
        print()
        print("  Deltas:")
        print(
            f"    A vs B (AllReduce upcast gain): "
            f"{(diff_a.mean() - diff_b.mean()).item():+.4e}"
        )
        print(
            f"    A vs D (GEMM upcast gain):      "
            f"{(diff_a.mean() - diff_d.mean()).item():+.4e}"
        )
        print(
            f"    D vs C (extra AllReduce upcast on fp32 GEMM): "
            f"{(diff_d.mean() - diff_c.mean()).item():+.4e}"
        )

        assert diff_b.mean() <= diff_a.mean() + 1e-6, (
            "scheme B should be at least as accurate as A (fp32 AllReduce)"
        )
        print("\n  Scheme B (fp32 AllReduce) precision >= scheme A (bf16 AllReduce)")


if __name__ == "__main__":
    require_torchrun("torchrun --nproc_per_node=2 test_row_parallel_gemm.py")
    initialize_distributed()
    rank = dist.get_rank()

    if rank == 0:
        print("\n" + "=" * 64)
        print("Row Parallel GEMM AllReduce correctness")
        print("=" * 64)
        print("  M=1024, K=4096, N=1024")
        print("=" * 64)

    try:
        if rank == 0:
            print("\n[1] Integer input bit-exact check")
        test_row_parallel_gemm(M=1024, K=4096, N=1024)
        dist.barrier()

        if rank == 0:
            print("\n[2] Random bf16: GEMM+AllReduce precision variants")
        test_fp32_allreduce_precision(M=1024, K=4096, N=1024)
        dist.barrier()

        if rank == 0:
            print("\nAll tests completed.\n" + "=" * 64)
    except Exception as e:
        print(f"Rank {rank} failed: {e}")
        traceback.print_exc()
        raise
    finally:
        cleanup_distributed()
