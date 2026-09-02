"""GemmAllReduceCtx correctness + precision vs torch GEMM+AllReduce variants.

Both checks share one ctx / tp_group so the CuTeDSL kernel JITs once.

Usage:
  torchrun --nproc_per_node=2 \\
      tests/python/distributed/cutedsl/test_gemm_allreduce.py
"""

import os
import sys
import traceback
from pathlib import Path

import torch
import torch.distributed as dist

from miniTransformer.distributed.cutedsl.gemm_allreduce import GemmAllReduceCtx

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from helpers import (  # noqa: E402
    cleanup_nvshmem_distributed,
    initialize_nvshmem_distributed,
    make_world_tp_group,
    scheme_a,
    scheme_b,
    shard_k,
    torch_gemm_allreduce,
)


def run_tests(M: int = 1024, N: int = 1024, K: int = 1024, seed: int = 42):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tp_group = make_world_tp_group()

    assert K % world_size == 0, f"K={K} must be divisible by world_size={world_size}"

    torch.manual_seed(seed)
    a_full = torch.zeros(M, K, dtype=torch.bfloat16, device="cuda")
    b_full = torch.zeros(N, K, dtype=torch.bfloat16, device="cuda")
    if rank == 0:
        a_full = torch.randn(M, K, device="cuda").to(torch.bfloat16)
        b_full = torch.randn(N, K, device="cuda").to(torch.bfloat16)
    dist.broadcast(a_full, src=0, group=tp_group)
    dist.broadcast(b_full, src=0, group=tp_group)

    a_local, b_local, _ = shard_k(a_full, b_full, rank, world_size)

    if rank == 0:
        print("\n[1/1] correctness + precision_comparison (shared ctx, compile once)")
        print(f"  Compiling CuTeDSL kernel (M={M} N={N} K={K})...")

    ctx = GemmAllReduceCtx(tp_group=tp_group)
    o_d = ctx.forward(a_local, b_local)
    if rank == 0:
        print("  Compilation done.")

    ref = torch_gemm_allreduce(a_local, b_local, tp_group)
    diff = (o_d.float() - ref.float()).abs()
    atol = 2.0
    if rank == 0:
        print("\n  [correctness]")
        print(f"    max_diff={diff.max():.4e}  mean_diff={diff.mean():.4e}  atol={atol}")

    torch.testing.assert_close(
        o_d.float(),
        ref.float(),
        rtol=0.1,
        atol=atol,
        msg=f"rank {rank}: CuTeDSL AllReduce mismatch",
    )
    if rank == 0:
        print("  correctness passed")

    baseline_fp32 = (a_full.float() @ b_full.float().T).to(torch.bfloat16)
    o_a = scheme_a(a_local, b_local, tp_group)
    o_b = scheme_b(a_local, b_local, tp_group)
    o_c = torch_gemm_allreduce(a_local, b_local, tp_group)
    o_d2 = ctx.forward(a_local, b_local)

    if rank == 0:
        ref_f = baseline_fp32.float()
        da = (o_a.float() - ref_f).abs()
        db = (o_b.float() - ref_f).abs()
        dc = (o_c.float() - ref_f).abs()
        dd = (o_d2.float() - ref_f).abs()
        scale = ref_f.abs().max().clamp(min=1.0)

        print("\n  [precision_comparison] (baseline=fp32 full GEMM)")
        print(f"  {'scheme':<50} {'max_diff':>12} {'mean_diff':>12} {'max_rel':>10}")
        print(f"  {'-'*84}")
        print(
            f"  {'A: bf16 GEMM + bf16 AllReduce':<50} "
            f"{da.max():>12.4e} {da.mean():>12.4e} {(da.max()/scale):>10.4e}"
        )
        print(
            f"  {'B: bf16 GEMM + fp32 AllReduce -> bf16':<50} "
            f"{db.max():>12.4e} {db.mean():>12.4e} {(db.max()/scale):>10.4e}"
        )
        print(
            f"  {'C: fp32 GEMM + fp32 AllReduce (torch)':<50} "
            f"{dc.max():>12.4e} {dc.mean():>12.4e} {(dc.max()/scale):>10.4e}"
        )
        print(
            f"  {'D: CuTeDSL fp32 tile-level AllReduce':<50} "
            f"{dd.max():>12.4e} {dd.mean():>12.4e} {(dd.max()/scale):>10.4e}"
        )
        print()
        assert dd.mean() <= da.mean() * 1.5 + 1e-4, (
            "CuTeDSL (D) should be close to or better than bf16 AllReduce (A)"
        )
        print("  precision_comparison passed")

    ctx.free()


def main():
    if "RANK" not in os.environ:
        print("ERROR: run with torchrun")
        print(
            "  torchrun --nproc_per_node=2 "
            "tests/python/distributed/cutedsl/test_gemm_allreduce.py"
        )
        raise SystemExit(1)

    initialize_nvshmem_distributed()
    rank = dist.get_rank()

    if rank == 0:
        print("\n" + "=" * 70)
        print("GemmAllReduceCtx Tests (CuTeDSL NVLS tile-level AllReduce)")
        print("=" * 70)

    try:
        run_tests()
        dist.barrier()
        if rank == 0:
            print("\nAll tests passed!\n" + "=" * 70 + "\n")
    except Exception as e:
        print(f"\n  FAILED at rank {rank}: {e}")
        traceback.print_exc()
        cleanup_nvshmem_distributed()
        raise

    cleanup_nvshmem_distributed()


if __name__ == "__main__":
    main()
