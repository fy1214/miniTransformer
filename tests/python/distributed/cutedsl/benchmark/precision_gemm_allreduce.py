"""Row-parallel GEMM + AllReduce precision table vs fp32 full GEMM.

Schemes:
  A: bf16 GEMM + bf16 AllReduce
  B: bf16 GEMM + fp32 AllReduce -> bf16
  C: CuTeDSL tile-level fp32 AllReduce

Usage:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \\
    --master_addr=127.0.0.1 --master_port=29701 \\
    tests/python/distributed/cutedsl/benchmark/precision_gemm_allreduce.py
"""

import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

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

SHAPES = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]


def compute_metrics(pred: torch.Tensor, ref: torch.Tensor) -> dict:
    pred_f = pred.float()
    ref_f = ref.float()
    err = (pred_f - ref_f).abs()
    signal_power = (ref_f ** 2).mean()
    noise_power = (err ** 2).mean()
    snr_db = 10 * math.log10(signal_power.item() / (noise_power.item() + 1e-30))
    cos_sim = F.cosine_similarity(
        pred_f.flatten().unsqueeze(0),
        ref_f.flatten().unsqueeze(0),
    ).item()
    return dict(
        max_abs=err.max().item(),
        mean_abs=err.mean().item(),
        snr_db=snr_db,
        cos_sim=cos_sim,
    )


def precision_shape(M: int, N: int, K: int, tp_group, ctx: GemmAllReduceCtx, seed: int = 42):
    rank = dist.get_rank(tp_group)
    world_size = dist.get_world_size(tp_group)

    torch.manual_seed(seed)
    a_full = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
    b_full = torch.zeros(N, K, device="cuda", dtype=torch.bfloat16)
    if rank == 0:
        a_full = torch.randn(M, K, device="cuda").to(torch.bfloat16)
        b_full = torch.randn(N, K, device="cuda").to(torch.bfloat16)
    dist.broadcast(a_full, src=0, group=tp_group)
    dist.broadcast(b_full, src=0, group=tp_group)
    a_local, b_local, _ = shard_k(a_full, b_full, rank, world_size)

    with torch.no_grad():
        baseline = (a_full.float() @ b_full.float().T).to(torch.bfloat16)
        o_a = scheme_a(a_local, b_local, tp_group)
        o_b = scheme_b(a_local, b_local, tp_group)
        ctx.barrier_before_reuse()
        o_c = scheme_c(a_local, b_local, ctx)

    if rank == 0:
        return {
            "M": M,
            "N": N,
            "K": K,
            "A": compute_metrics(o_a, baseline),
            "B": compute_metrics(o_b, baseline),
            "C": compute_metrics(o_c, baseline),
        }
    return None


def print_header():
    print(f"\n{'='*110}")
    print("GEMM AllReduce Precision  (tp=2, baseline=fp32 full GEMM)")
    print(f"{'='*110}")
    print(
        f"{'M':>5} {'N':>5} {'K':>5} | "
        f"{'':^3} {'max_abs':>10} {'mean_abs':>10} {'SNR(dB)':>9} {'cos_sim':>9} | "
        f"{'':^3} {'max_abs':>10} {'mean_abs':>10} {'SNR(dB)':>9} {'cos_sim':>9} | "
        f"{'':^3} {'max_abs':>10} {'mean_abs':>10} {'SNR(dB)':>9} {'cos_sim':>9}"
    )
    print(
        f"{'':>16} | "
        f"{'A':^3} {'':>10} {'':>10} {'':>9} {'':>9} | "
        f"{'B':^3} {'':>10} {'':>10} {'':>9} {'':>9} | "
        f"{'C':^3} {'':>10} {'':>10} {'':>9} {'':>9}"
    )
    print(f"{'-'*110}")


def print_row(r: dict):
    def fmt(m):
        return (
            f"{m['max_abs']:>10.4e} {m['mean_abs']:>10.4e} "
            f"{m['snr_db']:>9.2f} {m['cos_sim']:>9.6f}"
        )

    print(
        f"{r['M']:>5} {r['N']:>5} {r['K']:>5} | "
        f"{'A':^3} {fmt(r['A'])} | "
        f"{'B':^3} {fmt(r['B'])} | "
        f"{'C':^3} {fmt(r['C'])}"
    )


def main():
    if "RANK" not in os.environ:
        print("ERROR: run with torchrun")
        print(
            "  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 "
            "--master_addr=127.0.0.1 --master_port=29701 "
            "tests/python/distributed/cutedsl/benchmark/precision_gemm_allreduce.py"
        )
        raise SystemExit(1)

    initialize_nvshmem_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tp_group = make_world_tp_group()

    if rank == 0:
        print_header()

    for M, N, K in SHAPES:
        if K % world_size != 0:
            if rank == 0:
                print(f"  skip ({M},{N},{K}): K not divisible by tp_size={world_size}")
            continue

        ctx = GemmAllReduceCtx(tp_group=tp_group)
        chunk = K // world_size
        a_tmp = torch.randn(M, chunk, device="cuda", dtype=torch.bfloat16)
        b_tmp = torch.randn(N, chunk, device="cuda", dtype=torch.bfloat16)
        _ = ctx.forward(a_tmp, b_tmp)
        ctx.barrier_before_reuse()

        r = precision_shape(M, N, K, tp_group, ctx)
        if rank == 0 and r is not None:
            print_row(r)

        ctx.barrier_before_reuse()
        ctx.free()
        dist.barrier(group=tp_group)

    if rank == 0:
        print(f"{'='*110}")
        print("\nLegend:")
        print("  A = bf16 GEMM + bf16 AllReduce")
        print("  B = bf16 GEMM + fp32 AllReduce -> bf16")
        print("  C = CuTeDSL bf16 GEMM + tile-level fp32 AllReduce -> bf16")
        print("  SNR(dB): higher is better; C should approach B")
        print("  baseline = fp32 full GEMM (no TP)\n")

    cleanup_nvshmem_distributed()


if __name__ == "__main__":
    main()
