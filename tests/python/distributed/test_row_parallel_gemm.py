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


def test_fp32_allreduce_precision(
    M: int = 1024, K: int = 4096, N: int = 1024, seed: int = 42
):
    """
    对比三种 row parallel 方案在随机 bf16 输入下的精度差异：

    方案 A：bf16 GEMM → bf16 AllReduce（当前常见实现）
    方案 B：bf16 GEMM → fp32 AllReduce → cast bf16（升精度 reduce）
    方案 C：fp32 GEMM → fp32 AllReduce → cast bf16（全程高精度）

    baseline：tp=1 完整 GEMM（bf16）

    理论分析：
      - 方案 A：GEMM 内部 bf16 累加误差 + AllReduce bf16 加法截断
      - 方案 B：GEMM 内部 bf16 累加误差，但 AllReduce 在 fp32 精度下求和，减少截断
      - 方案 C：GEMM 内部 fp32 精度（或 TF32）累加，AllReduce fp32 精度，误差最小
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2
    assert K % world_size == 0

    chunk = K // world_size

    # rank 0 生成随机 bf16，broadcast
    x = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(N, K, device="cuda", dtype=torch.bfloat16)
    if rank == 0:
        torch.manual_seed(seed)
        x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
        w = torch.randn(N, K, device="cuda").to(torch.bfloat16)
    dist.broadcast(x, src=0)
    dist.broadcast(w, src=0)

    # baseline：tp=1，bf16 完整 GEMM
    o_baseline = x @ w.T   # [M, N]，bf16

    x_local = x[:, rank * chunk:(rank + 1) * chunk].contiguous()
    w_local = w[:, rank * chunk:(rank + 1) * chunk].contiguous()

    # ── 方案 A：bf16 GEMM + bf16 AllReduce ──────────────────────────────────
    o_a = (x_local @ w_local.T).clone()                     # bf16 GEMM → bf16
    dist.all_reduce(o_a, op=dist.ReduceOp.SUM)              # bf16 AllReduce

    # ── 方案 B：bf16 GEMM + fp32 AllReduce → cast bf16 ──────────────────────
    o_b = (x_local @ w_local.T).float()                     # bf16 GEMM → cast fp32
    dist.all_reduce(o_b, op=dist.ReduceOp.SUM)              # fp32 AllReduce
    o_b = o_b.to(torch.bfloat16)                            # cast 回 bf16

    # ── 方案 C：fp32 GEMM + fp32 AllReduce → cast bf16 ──────────────────────
    o_c = (x_local.float() @ w_local.float().T)             # fp32 GEMM
    dist.all_reduce(o_c, op=dist.ReduceOp.SUM)              # fp32 AllReduce
    o_c = o_c.to(torch.bfloat16)                            # cast 回 bf16

    # ── 方案 D：fp32 GEMM + bf16 AllReduce ──────────────────────────────────
    o_d = (x_local.float() @ w_local.float().T).to(torch.bfloat16)  # fp32 GEMM → cast bf16
    dist.all_reduce(o_d, op=dist.ReduceOp.SUM)              # bf16 AllReduce

    if rank == 0:
        diff_a = (o_a.float() - o_baseline.float()).abs()
        diff_b = (o_b.float() - o_baseline.float()).abs()
        diff_c = (o_c.float() - o_baseline.float()).abs()
        diff_d = (o_d.float() - o_baseline.float()).abs()

        print(f"\n  精度对比（baseline = tp=1 bf16 GEMM）：")
        print(f"  {'方案':<42} {'max diff':>12} {'mean diff':>12}")
        print(f"  {'-'*66}")
        print(f"  {'A: bf16 GEMM + bf16 AllReduce':<42} "
              f"{diff_a.max().item():>12.4e} {diff_a.mean().item():>12.4e}")
        print(f"  {'B: bf16 GEMM + fp32 AllReduce → bf16':<42} "
              f"{diff_b.max().item():>12.4e} {diff_b.mean().item():>12.4e}")
        print(f"  {'C: fp32 GEMM + fp32 AllReduce → bf16':<42} "
              f"{diff_c.max().item():>12.4e} {diff_c.mean().item():>12.4e}")
        print(f"  {'D: fp32 GEMM + bf16 AllReduce':<42} "
              f"{diff_d.max().item():>12.4e} {diff_d.mean().item():>12.4e}")
        print()
        print(f"  分析：")
        print(f"    A vs B：AllReduce 升精度的收益  = {(diff_a.mean() - diff_b.mean()).item():+.4e}")
        print(f"    A vs D：GEMM 升精度的收益       = {(diff_a.mean() - diff_d.mean()).item():+.4e}")
        print(f"    D vs C：AllReduce 升精度在fp32GEMM下的额外收益 = {(diff_d.mean() - diff_c.mean()).item():+.4e}")

        # 检验 B 比 A 更精确
        assert diff_b.mean() <= diff_a.mean() + 1e-6, \
            "方案 B 应比方案 A 更精确（fp32 AllReduce 减少截断误差）"
        print(f"\n  ✓ 方案 B（fp32 AllReduce）精度 ≥ 方案 A（bf16 AllReduce）")


if __name__ == "__main__":
    if "RANK" not in os.environ:
        print("请用 torchrun 启动：")
        print("  torchrun --nproc_per_node=2 test_row_parallel_gemm.py")
        raise SystemExit(1)

    initialize_distributed()
    rank = dist.get_rank()

    if rank == 0:
        print("\n" + "=" * 64)
        print("Row Parallel GEMM AllReduce 正确性测试")
        print("=" * 64)
        print(f"  M=1024, K=4096, N=1024")
        print("=" * 64)

    try:
        if rank == 0:
            print("\n[1] 整数输入 bit-exact 验证")
        test_row_parallel_gemm(M=1024, K=4096, N=1024)
        dist.barrier()

        if rank == 0:
            print("\n[2] 随机 bf16 输入：三种方案精度对比")
        test_fp32_allreduce_precision(M=1024, K=4096, N=1024)
        dist.barrier()

        if rank == 0:
            print("\n✓ 全部测试完成！\n" + "=" * 64)
    except Exception as e:
        print(f"✗ Rank {rank} 失败: {e}")
        import traceback; traceback.print_exc()
        raise
    finally:
        cleanup_distributed()

