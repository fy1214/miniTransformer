"""2-rank AllReduce smoke: complementary shards sum to the full tensor.

Rank 0 holds t0. Each rank keeps a non-overlapping row shard (zeros elsewhere).
After AllReduce(SUM), every rank must match t0 bit-exactly.

Usage:
  torchrun --nproc_per_node=2 test_allreduce.py
"""

import torch
import torch.distributed as dist

from helpers import (
    cleanup_distributed,
    initialize_distributed,
    require_torchrun,
)


def test_allreduce_2rank(M: int = 1024, N: int = 1024, seed: int = 42):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, f"this test requires 2 ranks, got world_size={world_size}"

    t0 = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    if rank == 0:
        torch.manual_seed(seed)
        t0 = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    dist.broadcast(t0, src=0)

    half = M // 2
    local_t = torch.zeros(M, N, device="cuda", dtype=t0.dtype)
    if rank == 0:
        local_t[:half, :] = t0[:half, :]
    else:
        local_t[half:, :] = t0[half:, :]

    gathered = [torch.zeros_like(local_t) for _ in range(world_size)]
    dist.all_gather(gathered, local_t)
    if rank == 0:
        t_sum_local = gathered[0] + gathered[1]
        assert torch.allclose(t_sum_local, t0, atol=0, rtol=0), (
            "shard check failed: t1 + t2 != t0"
        )
        print("  [shard check] t1 + t2 == t0")

    dist.barrier()

    result = local_t.clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)

    diff = (result - t0).abs()
    print(
        f"  Rank {rank}: result shape={result.shape}  "
        f"max_diff={diff.max():.4e}  mean_diff={diff.mean():.4e}"
    )
    assert torch.allclose(result, t0, atol=0, rtol=0), (
        f"Rank {rank}: AllReduce result != t0; max_diff={diff.max():.4e}"
    )
    if rank == 0:
        print("  AllReduce result == t0 (bit-exact)")


if __name__ == "__main__":
    require_torchrun("torchrun --nproc_per_node=2 test_allreduce.py")
    initialize_distributed()
    rank = dist.get_rank()

    if rank == 0:
        print("\n" + "=" * 50)
        print("AllReduce 2-rank correctness")
        print("=" * 50)

    try:
        test_allreduce_2rank(M=1024, N=1024)
        dist.barrier()
        if rank == 0:
            print("\nAll tests passed.\n" + "=" * 50)
    except Exception as e:
        print(f"Rank {rank} failed: {e}")
        raise
    finally:
        cleanup_distributed()
