"""
precision_gemm_allreduce.py
============================

row parallel GEMM + AllReduce 精度对比测试。

baseline：fp32 完整 GEMM（单卡，无 TP，最高精度参考值）

对比方案：
  A: bf16 GEMM + bf16 AllReduce
  B: bf16 GEMM + fp32 AllReduce → bf16
  C: CuTeDSL tile-level fp32 AllReduce（本实现）

指标：
  - max abs diff
  - mean abs diff
  - SNR (dB)：信噪比，越高越好
  - cosine similarity：方向相似度，越接近 1 越好

运行：
  CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 \\
    --master_addr=127.0.0.1 --master_port=29701 \\
    tests/python/distributed/cutedsl/benchmark/precision_gemm_allreduce.py
"""

import os
import math
import argparse

import torch
import torch.distributed as dist
import torch.nn.functional as F

from miniTransformer.distributed.cutedsl.nvshmem_communicator import (
    init_nvshmem_with_dist,
    finalize_nvshmem,
)
from miniTransformer.distributed.cutedsl.gemm_allreduce import GemmAllReduceCtx


# ── 测试配置 ──────────────────────────────────────────────────────────────────

SHAPES = [
    (512,   512,   512),
    (1024,  1024,  1024),
    (2048,  2048,  2048),
    (4096,  4096,  4096),
]


# ── 精度指标计算 ──────────────────────────────────────────────────────────────

def compute_metrics(pred: torch.Tensor, ref: torch.Tensor) -> dict:
    """
    计算 pred 相对于 ref 的精度指标。

    Args:
        pred: 待评估的结果 tensor
        ref:  fp32 baseline（参考值）

    Returns:
        dict with max_abs, mean_abs, snr_db, cos_sim
    """
    pred_f = pred.float()
    ref_f  = ref.float()
    err    = (pred_f - ref_f).abs()

    max_abs  = err.max().item()
    mean_abs = err.mean().item()

    # SNR (dB) = 10 × log10(signal_power / noise_power)
    signal_power = (ref_f ** 2).mean()
    noise_power  = (err ** 2).mean()
    snr_db = 10 * math.log10(
        signal_power.item() / (noise_power.item() + 1e-30)
    )

    # cosine similarity
    cos_sim = F.cosine_similarity(
        pred_f.flatten().unsqueeze(0),
        ref_f.flatten().unsqueeze(0),
    ).item()

    return dict(
        max_abs=max_abs,
        mean_abs=mean_abs,
        snr_db=snr_db,
        cos_sim=cos_sim,
    )


# ── 三种方案的实现 ────────────────────────────────────────────────────────────

def scheme_a(a_local, b_local, tp_group):
    """A: bf16 GEMM + bf16 AllReduce"""
    out = (a_local @ b_local.T).clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=tp_group)
    return out


def scheme_b(a_local, b_local, tp_group):
    """B: bf16 GEMM + fp32 AllReduce → bf16"""
    out = (a_local @ b_local.T).float()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=tp_group)
    return out.to(torch.bfloat16)


def scheme_c(a_local, b_local, ctx):
    """C: CuTeDSL tile-level fp32 AllReduce"""
    return ctx.forward(a_local, b_local)


# ── 单个 shape 的精度测试 ─────────────────────────────────────────────────────

def precision_shape(
    M: int, N: int, K: int,
    tp_group,
    ctx: GemmAllReduceCtx,
    seed: int = 42,
) -> dict:
    rank       = dist.get_rank(tp_group)
    world_size = dist.get_world_size(tp_group)
    chunk      = K // world_size

    # 生成输入（rank 0 生成，broadcast 保证一致）
    torch.manual_seed(seed)
    a_full = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
    b_full = torch.zeros(N, K, device="cuda", dtype=torch.bfloat16)
    if rank == 0:
        a_full = torch.randn(M, K, device="cuda").to(torch.bfloat16)
        b_full = torch.randn(N, K, device="cuda").to(torch.bfloat16)
    dist.broadcast(a_full, src=0, group=tp_group)
    dist.broadcast(b_full, src=0, group=tp_group)

    a_local = a_full[:, rank * chunk:(rank + 1) * chunk].contiguous()
    b_local = b_full[:, rank * chunk:(rank + 1) * chunk].contiguous()

    # fp32 baseline（完整 GEMM，无 TP，最高精度参考）
    with torch.no_grad():
        baseline = (a_full.float() @ b_full.float().T).to(torch.bfloat16)

    # 三种方案
    with torch.no_grad():
        o_a = scheme_a(a_local, b_local, tp_group)
        o_b = scheme_b(a_local, b_local, tp_group)
        ctx.barrier_before_reuse()
        o_c = scheme_c(a_local, b_local, ctx)

    # 计算指标（只在 rank 0 计算和打印）
    if rank == 0:
        m_a = compute_metrics(o_a, baseline)
        m_b = compute_metrics(o_b, baseline)
        m_c = compute_metrics(o_c, baseline)
        return {"M": M, "N": N, "K": K, "A": m_a, "B": m_b, "C": m_c}
    return None


# ── 打印表格 ──────────────────────────────────────────────────────────────────

def print_header():
    print(f"\n{'='*110}")
    print(f"GEMM AllReduce Precision  (tp=2, baseline=fp32 full GEMM)")
    print(f"{'='*110}")
    print(f"{'M':>5} {'N':>5} {'K':>5} | "
          f"{'':^3} {'max_abs':>10} {'mean_abs':>10} {'SNR(dB)':>9} {'cos_sim':>9} | "
          f"{'':^3} {'max_abs':>10} {'mean_abs':>10} {'SNR(dB)':>9} {'cos_sim':>9} | "
          f"{'':^3} {'max_abs':>10} {'mean_abs':>10} {'SNR(dB)':>9} {'cos_sim':>9}")
    print(f"{'':>16} | "
          f"{'A':^3} {'':>10} {'':>10} {'':>9} {'':>9} | "
          f"{'B':^3} {'':>10} {'':>10} {'':>9} {'':>9} | "
          f"{'C':^3} {'':>10} {'':>10} {'':>9} {'':>9}")
    print(f"{'-'*110}")


def print_row(r: dict):
    def fmt(m):
        return (f"{m['max_abs']:>10.4e} {m['mean_abs']:>10.4e} "
                f"{m['snr_db']:>9.2f} {m['cos_sim']:>9.6f}")
    print(f"{r['M']:>5} {r['N']:>5} {r['K']:>5} | "
          f"{'A':^3} {fmt(r['A'])} | "
          f"{'B':^3} {fmt(r['B'])} | "
          f"{'C':^3} {fmt(r['C'])}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if "RANK" not in os.environ:
        print("请用 torchrun 启动：")
        print("  CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 "
              "--master_addr=127.0.0.1 --master_port=29701 "
              "tests/python/distributed/cutedsl/benchmark/precision_gemm_allreduce.py")
        raise SystemExit(1)

    # 初始化
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dev = init_nvshmem_with_dist()

    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    tp_group   = dist.new_group(ranks=list(range(world_size)))

    if rank == 0:
        print_header()

    for M, N, K in SHAPES:
        if K % world_size != 0:
            if rank == 0:
                print(f"  跳过 ({M},{N},{K})：K 不能被 tp_size={world_size} 整除")
            continue

        ctx = GemmAllReduceCtx(tp_group=tp_group)

        # compile warmup
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
        print("\n说明：")
        print("  A = bf16 GEMM + bf16 AllReduce")
        print("  B = bf16 GEMM + fp32 AllReduce → bf16")
        print("  C = CuTeDSL bf16 GEMM + tile-level fp32 AllReduce → bf16")
        print("  SNR(dB)：越高越好，C 应接近 B（fp32 AllReduce 精度）")
        print("  baseline = fp32 完整 GEMM（单卡，无 TP，最高精度参考）\n")

    finalize_nvshmem()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
