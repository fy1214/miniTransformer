"""
test_gemm_allreduce.py
======================

GemmAllReduceCtx 正确性和精度对比测试。

运行：
    torchrun --nproc_per_node=2 \
        tests/python/distributed/cutedsl/test_gemm_allreduce.py

设计：两个测试共享同一个 ctx 和 tp_group，只 JIT 编译一次。
"""

import os
import traceback

import torch
import torch.distributed as dist

from miniTransformer.distributed.cutedsl.nvshmem_communicator import (
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


# ── 参考实现 ──────────────────────────────────────────────────────────────────

def torch_gemm_allreduce(a_local, b_local, tp_group):
    """fp32 GEMM + fp32 AllReduce 参考实现。"""
    partial = a_local.float() @ b_local.float().T
    dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=tp_group)
    return partial.to(torch.bfloat16)


# ── 单一主测试（共享 ctx，只编译一次）────────────────────────────────────────

def run_tests(
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

    a_local = a_full[:, rank * chunk:(rank + 1) * chunk].contiguous()
    b_local = b_full[:, rank * chunk:(rank + 1) * chunk].contiguous()

    # ── 初始化 ctx（JIT 编译发生在第一次 forward）────────────────────────────
    if rank == 0:
        print(f"\n[1/1] correctness + precision_comparison (shared ctx, compile once)")
        print(f"  Compiling CuTeDSL kernel (M={M} N={N} K={K})...")

    ctx = GemmAllReduceCtx(tp_group=tp_group)

    # ── 第一次 forward（触发 JIT 编译）──────────────────────────────────────
    o_d = ctx.forward(a_local, b_local)

    if rank == 0:
        print(f"  Compilation done.")

    # ── Test 1: 正确性验证 ───────────────────────────────────────────────────
    ref = torch_gemm_allreduce(a_local, b_local, tp_group)
    diff = (o_d.float() - ref.float()).abs()

    atol = 2.0
    if rank == 0:
        print(f"\n  [correctness]")
        print(f"    max_diff={diff.max():.4e}  mean_diff={diff.mean():.4e}  atol={atol}")

    torch.testing.assert_close(
        o_d.float(), ref.float(),
        rtol=0.1, atol=atol,
        msg=f"rank {rank}: CuTeDSL AllReduce mismatch"
    )
    if rank == 0:
        print(f"  ✓ correctness passed")

    # ── Test 2: 精度对比（复用已编译的 ctx，无需再编译）─────────────────────
    # fp32 baseline
    baseline_fp32 = (a_full.float() @ b_full.float().T).to(torch.bfloat16)

    # A: bf16 GEMM + bf16 AllReduce
    o_a = (a_local @ b_local.T).clone()
    dist.all_reduce(o_a, op=dist.ReduceOp.SUM, group=tp_group)

    # B: bf16 GEMM + fp32 AllReduce → bf16
    o_b = (a_local @ b_local.T).float()
    dist.all_reduce(o_b, op=dist.ReduceOp.SUM, group=tp_group)
    o_b = o_b.to(torch.bfloat16)

    # C: fp32 GEMM + fp32 AllReduce（torch）
    o_c = torch_gemm_allreduce(a_local, b_local, tp_group)

    # D: CuTeDSL（已编译，直接 forward）
    o_d2 = ctx.forward(a_local, b_local)

    if rank == 0:
        ref_f = baseline_fp32.float()
        da = (o_a.float() - ref_f).abs()
        db = (o_b.float() - ref_f).abs()
        dc = (o_c.float() - ref_f).abs()
        dd = (o_d2.float() - ref_f).abs()
        scale = ref_f.abs().max().clamp(min=1.0)

        print(f"\n  [precision_comparison] (baseline=fp32 full GEMM)")
        print(f"  {'方案':<50} {'max_diff':>12} {'mean_diff':>12} {'max_rel':>10}")
        print(f"  {'-'*84}")
        print(f"  {'A: bf16 GEMM + bf16 AllReduce':<50} "
              f"{da.max():>12.4e} {da.mean():>12.4e} {(da.max()/scale):>10.4e}")
        print(f"  {'B: bf16 GEMM + fp32 AllReduce → bf16':<50} "
              f"{db.max():>12.4e} {db.mean():>12.4e} {(db.max()/scale):>10.4e}")
        print(f"  {'C: fp32 GEMM + fp32 AllReduce (torch)':<50} "
              f"{dc.max():>12.4e} {dc.mean():>12.4e} {(dc.max()/scale):>10.4e}")
        print(f"  {'D: CuTeDSL fp32 tile-level AllReduce':<50} "
              f"{dd.max():>12.4e} {dd.mean():>12.4e} {(dd.max()/scale):>10.4e}")
        print()

        assert dd.mean() <= da.mean() * 1.5 + 1e-4, \
            "CuTeDSL (D) 精度应接近或优于 bf16 AllReduce (A)"
        print(f"  ✓ precision_comparison passed")

    ctx.free()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if "RANK" not in os.environ:
        print("请用 torchrun 启动：")
        print("  torchrun --nproc_per_node=2 "
              "tests/python/distributed/cutedsl/test_gemm_allreduce.py")
        raise SystemExit(1)

    dev = initialize()
    rank = dist.get_rank()

    if rank == 0:
        print("\n" + "=" * 70)
        print("GemmAllReduceCtx Tests (CuTeDSL NVLS tile-level AllReduce)")
        print("=" * 70)

    try:
        run_tests()
        dist.barrier()
        if rank == 0:
            print(f"\n✓ All tests passed!\n" + "=" * 70 + "\n")
    except Exception as e:
        print(f"\n  ✗ FAILED at rank {rank}: {e}")
        traceback.print_exc()
        cleanup()
        raise

    cleanup()


if __name__ == "__main__":
    main()
