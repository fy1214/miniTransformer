"""
AllReduce 2-rank 正确性测试
===========================

验证：
  rank 0 持有完整张量 t0 [M, N]
  rank 0 取 t1 = t0 的前半部分（或任意切片）
  rank 1 取 t2 = t0 的后半部分（或补集），保证 t1 + t2 = t0
  AllReduce 后每个 rank 得到 t1 + t2，检查是否等于 t0

运行：
  torchrun --nproc_per_node=2 test_allreduce.py
"""

import os
import torch
import torch.distributed as dist


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


def test_allreduce_2rank(M: int = 1024, N: int = 1024, seed: int = 42):
    """
    rank 0: t0 [M, N]（完整张量，broadcast 到所有 rank）
            t1 = t0 的前 M//2 行（或任意分割，只要 t1 + t2 = t0）
    rank 1: t2 = t0 - t1 = t0 的后 M//2 行对应补集

    这里用最简单的分割方式：
      t1 = t0 的前 M//2 行，其余行填 0
      t2 = t0 的后 M//2 行，其余行填 0
    则 t1 + t2 = t0（行不重叠，互补）

    AllReduce 后验证结果 == t0。
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, f"此测试仅支持 2 个 rank，当前 world_size={world_size}"

    # --- rank 0 生成 t0，broadcast 到所有 rank ---
    t0 = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    if rank == 0:
        torch.manual_seed(seed)
        t0 = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    dist.broadcast(t0, src=0)

    # --- 每个 rank 持有自己的分片（行切分，互补） ---
    half = M // 2
    local_t = torch.zeros(M, N, device="cuda", dtype=t0.dtype)
    if rank == 0:
        # t1：前 half 行保留，后半行填 0
        local_t[:half, :] = t0[:half, :]
    else:
        # t2：后 half 行保留，前半行填 0
        local_t[half:, :] = t0[half:, :]

    # 验证切分正确：t1 + t2 应等于 t0（在单 rank 上本地验证）
    # 先 AllGather 两个 local_t 到 rank 0 验证
    gathered = [torch.zeros_like(local_t) for _ in range(world_size)]
    dist.all_gather(gathered, local_t)
    if rank == 0:
        t_sum_local = gathered[0] + gathered[1]
        assert torch.allclose(t_sum_local, t0, atol=0, rtol=0), \
            "切分验证失败：t1 + t2 ≠ t0（AllGather 检查）"
        print(f"  [切分验证] t1 + t2 == t0  ✓")

    dist.barrier()

    # --- AllReduce ---
    result = local_t.clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)

    # --- 每个 rank 验证 AllReduce 结果 == t0 ---
    diff = (result - t0).abs()
    print(f"  Rank {rank}: result shape={result.shape}  "
          f"max_diff={diff.max():.4e}  mean_diff={diff.mean():.4e}")

    assert torch.allclose(result, t0, atol=0, rtol=0), \
        f"Rank {rank}: AllReduce 结果不等于 t0！max_diff={diff.max():.4e}"

    if rank == 0:
        print(f"  ✓ AllReduce 结果 == t0（bit-exact）")


if __name__ == "__main__":
    if "RANK" not in os.environ:
        print("请用 torchrun 启动：")
        print("  torchrun --nproc_per_node=2 test_allreduce.py")
        raise SystemExit(1)

    initialize_distributed()
    rank = dist.get_rank()

    if rank == 0:
        print("\n" + "=" * 50)
        print("AllReduce 2-rank 正确性测试")
        print("=" * 50)

    try:
        test_allreduce_2rank(M=1024, N=1024)
        dist.barrier()
        if rank == 0:
            print("\n✓ 测试通过！\n" + "=" * 50)
    except Exception as e:
        print(f"✗ Rank {rank} 失败: {e}")
        raise
    finally:
        cleanup_distributed()
