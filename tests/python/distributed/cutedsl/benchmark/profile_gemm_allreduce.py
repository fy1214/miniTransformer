"""Profile GEMM+AllReduce schemes with nsys (multi-rank) or ncu (single-rank GEMM).

Schemes in nsys mode:
  A: bf16 GEMM + bf16 AllReduce
  B: bf16 GEMM + fp32 AllReduce -> bf16
  C: CuTeDSL tile-level fp32 AllReduce
"""

import argparse
import os
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

try:
    import nvtx

    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False


class NvtxRange:
    """NVTX range; no-op when nvtx is unavailable."""

    def __init__(self, name: str, color: str = "blue"):
        self.name = name
        self.color = color

    def __enter__(self):
        if HAS_NVTX:
            nvtx.push_range(self.name, color=self.color)
        return self

    def __exit__(self, *args):
        if HAS_NVTX:
            nvtx.pop_range()


def run_ncu_mode(args):
    """Single-process bf16 GEMM loop for ncu attach (no dist / NVSHMEM)."""
    torch.cuda.set_device(0)
    M, N, K = args.M, args.N, args.K
    print(f"[ncu mode] M={M} N={N} K={K}  warmup={args.warmup}  iters={args.iters}")
    print("[ncu mode] measuring local bf16 GEMM only (no AllReduce)")

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    for _ in range(args.warmup):
        _ = a @ b.T
    torch.cuda.synchronize()

    print(f"[ncu mode] profiling ({args.iters} iters)...")
    for i in range(args.iters):
        with NvtxRange(f"bf16_gemm_iter{i}", color="green"):
            _ = a @ b.T
            torch.cuda.synchronize()
    print("[ncu mode] done")


def run_nsys_mode(args):
    """Multi-rank timeline with NVTX ranges around schemes A/B/C."""
    initialize_nvshmem_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tp_group = make_world_tp_group()

    M, N, K = args.M, args.N, args.K
    if rank == 0:
        print(f"[nsys mode] M={M} N={N} K={K}  tp={world_size}")
        print(f"[nsys mode] warmup={args.warmup}  iters={args.iters}")

    torch.manual_seed(42)
    a_full = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b_full = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    dist.broadcast(a_full, src=0, group=tp_group)
    dist.broadcast(b_full, src=0, group=tp_group)
    a_local, b_local, _ = shard_k(a_full, b_full, rank, world_size)

    ctx = GemmAllReduceCtx(tp_group=tp_group)
    _ = ctx.forward(a_local, b_local)
    torch.cuda.synchronize()
    ctx.barrier_before_reuse()

    for _ in range(args.warmup):
        scheme_a(a_local, b_local, tp_group)
        scheme_b(a_local, b_local, tp_group)
        ctx.barrier_before_reuse()
        scheme_c(a_local, b_local, ctx)
    torch.cuda.synchronize()
    dist.barrier(group=tp_group)

    if rank == 0:
        print("[nsys mode] warmup done, profiling...")

    for i in range(args.iters):
        with NvtxRange(f"A_bf16_gemm_bf16_ar_iter{i}", color="red"):
            scheme_a(a_local, b_local, tp_group)
        torch.cuda.synchronize()

        with NvtxRange(f"B_bf16_gemm_fp32_ar_iter{i}", color="yellow"):
            scheme_b(a_local, b_local, tp_group)
        torch.cuda.synchronize()

        ctx.barrier_before_reuse()
        with NvtxRange(f"C_cutedsl_fp32_ar_iter{i}", color="green"):
            scheme_c(a_local, b_local, ctx)
        torch.cuda.synchronize()
        dist.barrier(group=tp_group)

    if rank == 0:
        print("[nsys mode] profile done")

    ctx.free()
    cleanup_nvshmem_distributed()


def main():
    parser = argparse.ArgumentParser(description="GEMM AllReduce Profile")
    parser.add_argument("--mode", choices=["ncu", "nsys"], default="nsys")
    parser.add_argument("--M", type=int, default=2048)
    parser.add_argument("--N", type=int, default=2048)
    parser.add_argument("--K", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=3, help="warmup iterations")
    parser.add_argument("--iters", type=int, default=5, help="profile iterations")
    args = parser.parse_args()

    if args.mode == "ncu":
        run_ncu_mode(args)
        return

    if "RANK" not in os.environ:
        print("ERROR: nsys mode requires torchrun")
        raise SystemExit(1)
    run_nsys_mode(args)


if __name__ == "__main__":
    main()
