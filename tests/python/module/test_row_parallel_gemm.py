"""
Row Parallel GEMM AllReduce 正确性测试
=======================================

验证 row parallel linear 的数学等价性：

tp=1:
  rank 0: x [m, k], w [n, k]
  o = x @ w.T  [m, n]

tp=2:
  rank 0: x0 = x[:, 0:k//2] [m, k//2],  w0 = w[:, 0:k//2] [n, k//2]
  rank 1: x1 = x[:, k//2:k] [m, k//2],  w1 = w[:, k//2:k] [n, k//2]
  o0 = x0 @ w0.T  [m, n]
  o1 = x1 @ w1.T  [m, n]
  AllReduce: o0 + o1 == o ?

精度设计：
  x 和 w 初始化为小整数（0/1/-1），乘法和累加结果都是整数。
  bf16 在 [-2048, 2048] 范围内可以精确表示整数，
  只要 K 次累加结果不超过此范围，就能 bit-exact。
  用 x,w ∈ {0,1,-1}，每项乘积 ∈ {-1,0,1}，
  K=4096 次累加最坏 ±4096，在 bf16 精确整数范围内（2^12=4096 < 2^14）。

运行：
  torchrun --nproc_per_node=2 test_row_parallel_gemm.py
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


def test_row_parallel_gemm(M: int = 1024, K: int = 4096, N: int = 1024, seed: int = 42):
    """
    验证：x @ w.T == sum_r(x_r @ w_r.T) after AllReduce

    x, w ∈ {-1, 0, 1}，乘积和累加均为整数，bf16 可精确表示，
    无需禁用 TF32，直接 bit-exact 验证。
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, f"此测试仅支持 2 个 rank，当前 world_size={world_size}"
    assert K % world_size == 0, f"K={K} 必须能被 tp_size={world_size} 整除"

    chunk = K // world_size

    # rank 0 生成整数值 x, w ∈ {-1, 0, 1}，broadcast 到所有 rank
    x = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(N, K, device="cuda", dtype=torch.bfloat16)
    if rank == 0:
        torch.manual_seed(seed)
        # randint(-1, 2) → {-1, 0, 1}，转 bfloat16（整数值精确表示）
        x = torch.randint(-1, 2, (M, K), device="cuda").to(torch.bfloat16)
        w = torch.randint(-1, 2, (N, K), device="cuda").to(torch.bfloat16)
    dist.broadcast(x, src=0)
    dist.broadcast(w, src=0)

    # tp=1 baseline：完整 GEMM
    o_baseline = x @ w.T  # [m, n]，结果为整数，bf16 精确

    # tp=2：每个 rank 取自己的 k 切片，做分块 GEMM
    x_local = x[:, rank * chunk:(rank + 1) * chunk].contiguous()  # [m, k//2]
    w_local = w[:, rank * chunk:(rank + 1) * chunk].contiguous()  # [n, k//2]
    o_local = x_local @ w_local.T                                  # [m, n]，整数

    # AllReduce: o_local_0 + o_local_1
    o_allreduce = o_local.clone()
    dist.all_reduce(o_allreduce, op=dist.ReduceOp.SUM)

    # 验证
    diff = (o_allreduce - o_baseline).abs()

    print(f"  Rank {rank}:")
    print(f"    x shape:           {x.shape}  dtype={x.dtype}  values∈{{-1,0,1}}")
    print(f"    w shape:           {w.shape}")
    print(f"    x_local shape:     {x_local.shape}  (rank {rank} shard)")
    print(f"    w_local shape:     {w_local.shape}  (rank {rank} shard)")
    print(f"    o_baseline range:  [{o_baseline.min():.0f}, {o_baseline.max():.0f}]")
    print(f"    o_allreduce range: [{o_allreduce.min():.0f}, {o_allreduce.max():.0f}]")
    print(f"    max abs diff:      {diff.max():.4e}")
    print(f"    mean abs diff:     {diff.mean():.4e}")

    torch.testing.assert_close(
        o_allreduce, o_baseline,
        rtol=0, atol=0,
        msg=f"Rank {rank}: AllReduce 结果与完整 GEMM 不一致"
    )

    if rank == 0:
        print(f"  ✓ o0 + o1 == o（bf16 整数值 bit-exact）")


if __name__ == "__main__":
    if "RANK" not in os.environ:
        print("请用 torchrun 启动：")
        print("  torchrun --nproc_per_node=2 test_row_parallel_gemm.py")
        raise SystemExit(1)

    initialize_distributed()
    rank = dist.get_rank()

    if rank == 0:
        print("\n" + "=" * 55)
        print("Row Parallel GEMM AllReduce 正确性测试")
        print("=" * 55)
        print(f"  M=1024, K=4096, N=1024, dtype=bfloat16, x/w∈{{-1,0,1}}")
        print("=" * 55)

    try:
        test_row_parallel_gemm(M=1024, K=4096, N=1024)
        dist.barrier()
        if rank == 0:
            print("\n✓ 测试通过！\n" + "=" * 55)
    except Exception as e:
        print(f"✗ Rank {rank} 失败: {e}")
        import traceback; traceback.print_exc()
        raise
    finally:
        cleanup_distributed()

