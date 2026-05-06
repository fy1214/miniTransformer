"""
test_gemm_allreduce.py
======================

GemmAllReduceCtx 正确性和精度对比测试。

运行：
    torchrun --nproc_per_node=2 \
        miniTransformer/distributed/cutedsl/tests/test_gemm_allreduce.py

测试内容：
  1. test_correctness：验证 ctx.forward(a, b) 结果与 torch 参考实现一致
  2. test_precision_comparison：对比四种 AllReduce 方案的精度
"""

import os
import traceback

import torch
import torch.distributed as dist

from miniTransformer.distributed.cutedsl.nvshmem_init import (
    init_nvshmem_with_dist,
    finalize_nvshmem,
)
from miniTransformer.distributed.cutedsl.gemm_allreduce import GemmAllReduceCtx


# ── 分布式初始化 ──────────────────────────────────────────────────────────────

def initialize():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dev = init_nvshmem_with_dist()
    return dev


def cleanup():
    finalize_nvshmem()
    if dist.is_initialized():
        dist.destroy_process_group()


# ── 参考实现（torch baseline）────────────────────────────────────────────────

def torch_gemm_allreduce(a_local, b_local, tp_group):
    """
    分块 GEMM + fp32 AllReduce 的 torch 参考实现。

    a_local: [M, K//tp] bfloat16（本 rank 持有的 K 分片）
    b_local: [N, K//tp] bfloat16（本 rank 持有的 W 分片）
    返回: [M, N] bfloat16
    """
    # fp32 分块 GEMM
    partial = a_local.float() @ b_local.float().T   # [M, N] fp32

    # fp32 AllReduce
    dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=tp_group)

    return partial.to(torch.bfloat16)


# ── Test 1: 正确性验证 ────────────────────────────────────────────────────────

def test_correctness(
    M: int = 1024,
    N: int = 1024,
    K: int = 1024,
    seed: int = 42,
):
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    tp_group   = dist.new_group(ranks=list(range(world_size)))

    assert K % world_size == 0, f"K={K} 必须被 world_size={world_size} 整除"
    chunk = K // world_size

    # 生成随机 bf16 输入（rank 0 生成，broadcast 保证一致）
    torch.manual_seed(seed)
    a_full = torch.zeros(M, K, dtype=torch.bfloat16, device="cuda")
    b_full = torch.zeros(N, K, dtype=torch.bfloat16, device="cuda")
    if rank == 0:
        a_full = torch.randn(M, K, device="cuda").to(torch.bfloat16)
        b_full = torch.randn(N, K, device="cuda").to(torch.bfloat16)
    dist.broadcast(a_full, src=0, group=tp_group)
    dist.broadcast(b_full, src=0, group=tp_group)

    # 本 rank 的 K 分片
    a_local = a_full[:, rank * chunk:(rank + 1) * chunk].contiguous()
    b_local = b_full[:, rank * chunk:(rank + 1) * chunk].contiguous()

    # ── torch 参考结果 ──────────────────────────────────────────────────────
    ref = torch_gemm_allreduce(a_local, b_local, tp_group)

    # ── CuTeDSL 结果 ────────────────────────────────────────────────────────
    ctx = GemmAllReduceCtx(tp_group=tp_group)
    result = ctx.forward(a_local, b_local)
    ctx.free()

    # ── 比较 ────────────────────────────────────────────────────────────────
    diff = (result.float() - ref.float()).abs()
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()

    # bf16 精度下允许的容差（两边量化误差不同）
    atol = 2.0

    if rank == 0:
        print(f"  [correctness] M={M} N={N} K={K}")
        print(f"    max_diff={max_diff:.4e}  mean_diff={mean_diff:.4e}  atol={atol}")

    torch.testing.assert_close(
        result.float(), ref.float(),
        rtol=0.1, atol=atol,
        msg=f"rank {rank}: CuTeDSL AllReduce mismatch"
    )

    if rank == 0:
        print(f"  ✓ test_correctness passed")


# ── Test 2: 精度对比 ──────────────────────────────────────────────────────────

def test_precision_comparison(
    M: int = 1024,
    N: int = 1024,
    K: int = 1024,
    seed: int = 42,
):
    """
    对比四种 AllReduce 方案 vs fp32 baseline 的精度：
      A: bf16 GEMM + bf16 AllReduce
      B: bf16 GEMM + fp32 AllReduce → bf16
      C: fp32 GEMM + fp32 AllReduce → bf16（torch 版本）
      D: CuTeDSL tile-level fp32 AllReduce（本实现）
    """
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    tp_group   = dist.new_group(ranks=list(range(world_size)))

    assert K % world_size == 0
    chunk = K // world_size

    torch.manual_seed(seed)
    a_full = torch.zeros(M, K, dtype=torch.bfloat16, device="cuda")
    b_full = torch.zeros(N, K, dtype=torch.bfloat16, device="cuda")
    if rank == 0:
        a_full = torch.randn(M, K, device="cuda").to(torch.bfloat16)
        b_full = torch.randn(N, K, device="cuda").to(torch.bfloat16)
    dist.broadcast(a_full, src=0, group=tp_group)
    dist.broadcast(b_full, src=0, group=tp_group)

    a_local = a_full[:, rank * chunk:(rank + 1) * chunk].contiguous()
    b_local = b_full[:, rank * chunk:(rank + 1) * chunk].contiguous()

    # fp32 baseline（tp=1 等价，最精确）
    baseline_fp32 = (a_full.float() @ b_full.float().T).to(torch.bfloat16)

    # A: bf16 GEMM + bf16 AllReduce
    o_a = (a_local @ b_local.T).clone()
    dist.all_reduce(o_a, op=dist.ReduceOp.SUM, group=tp_group)

    # B: bf16 GEMM + fp32 AllReduce
    o_b = (a_local @ b_local.T).float()
    dist.all_reduce(o_b, op=dist.ReduceOp.SUM, group=tp_group)
    o_b = o_b.to(torch.bfloat16)

    # C: fp32 GEMM + fp32 AllReduce（torch）
    o_c = torch_gemm_allreduce(a_local, b_local, tp_group)

    # D: CuTeDSL tile-level AllReduce
    ctx = GemmAllReduceCtx(tp_group=tp_group)
    o_d = ctx.forward(a_local, b_local)
    ctx.free()

    if rank == 0:
        ref = baseline_fp32.float()
        da = (o_a.float() - ref).abs()
        db = (o_b.float() - ref).abs()
        dc = (o_c.float() - ref).abs()
        dd = (o_d.float() - ref).abs()

        scale = ref.abs().max().clamp(min=1.0)

        print(f"\n  精度对比 (M={M} N={N} K={K}, baseline=fp32):")
        print(f"  {'方案':<48} {'max_diff':>12} {'mean_diff':>12} {'max_rel':>10}")
        print(f"  {'-'*82}")
        print(f"  {'A: bf16 GEMM + bf16 AllReduce':<48} "
              f"{da.max():>12.4e} {da.mean():>12.4e} {(da.max()/scale):>10.4e}")
        print(f"  {'B: bf16 GEMM + fp32 AllReduce → bf16':<48} "
              f"{db.max():>12.4e} {db.mean():>12.4e} {(db.max()/scale):>10.4e}")
        print(f"  {'C: fp32 GEMM + fp32 AllReduce → bf16 (torch)':<48} "
              f"{dc.max():>12.4e} {dc.mean():>12.4e} {(dc.max()/scale):>10.4e}")
        print(f"  {'D: CuTeDSL tile-level fp32 AllReduce':<48} "
              f"{dd.max():>12.4e} {dd.mean():>12.4e} {(dd.max()/scale):>10.4e}")
        print()

        # D 应该和 C 精度相当（都是 fp32 AllReduce）
        assert dd.mean() <= da.mean() * 1.5 + 1e-4, \
            "CuTeDSL (D) 精度应接近或优于 bf16 AllReduce (A)"
        print(f"  ✓ test_precision_comparison passed")


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    ("correctness",          test_correctness),
    ("precision_comparison", test_precision_comparison),
]


def main():
    if "RANK" not in os.environ:
        print("请用 torchrun 启动：")
        print("  torchrun --nproc_per_node=2 "
              "miniTransformer/distributed/cutedsl/tests/test_gemm_allreduce.py")
        raise SystemExit(1)

    dev = initialize()
    rank = dist.get_rank()

    if rank == 0:
        print("\n" + "=" * 70)
        print("GemmAllReduceCtx Tests (CuTeDSL NVLS tile-level AllReduce)")
        print("=" * 70)

    for i, (name, fn) in enumerate(ALL_TESTS, 1):
        if rank == 0:
            print(f"\n[{i}/{len(ALL_TESTS)}] {name}")
        try:
            fn()
            dist.barrier()
        except Exception as e:
            print(f"\n  ✗ FAILED at rank {rank}: {e}")
            traceback.print_exc()
            cleanup()
            raise

    if rank == 0:
        print(f"\n✓ 全部 {len(ALL_TESTS)} 项测试通过！\n" + "=" * 70 + "\n")

    cleanup()


if __name__ == "__main__":
    main()
