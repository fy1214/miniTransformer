"""Shared NVSHMEM + GEMM-AllReduce helpers for cutedsl distributed scripts."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from miniTransformer.distributed.cutedsl.nvshmem_communicator import (
    init_nvshmem_with_dist,
    finalize_nvshmem,
)


def initialize_nvshmem_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return init_nvshmem_with_dist()


def cleanup_nvshmem_distributed() -> None:
    finalize_nvshmem()
    if dist.is_initialized():
        dist.destroy_process_group()


def make_world_tp_group():
    world_size = dist.get_world_size()
    return dist.new_group(ranks=list(range(world_size)))


def scheme_a(a_local, b_local, tp_group):
    """bf16 GEMM + bf16 AllReduce."""
    out = (a_local @ b_local.T).clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=tp_group)
    return out


def scheme_b(a_local, b_local, tp_group):
    """bf16 GEMM + fp32 AllReduce -> bf16."""
    out = (a_local @ b_local.T).float()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=tp_group)
    return out.to(torch.bfloat16)


def scheme_c(a_local, b_local, ctx):
    """CuTeDSL tile-level fp32 AllReduce."""
    return ctx.forward(a_local, b_local)


def torch_gemm_allreduce(a_local, b_local, tp_group):
    """fp32 GEMM + fp32 AllReduce reference."""
    partial = a_local.float() @ b_local.float().T
    dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=tp_group)
    return partial.to(torch.bfloat16)


def shard_k(a_full, b_full, rank: int, world_size: int):
    chunk = a_full.shape[1] // world_size
    a_local = a_full[:, rank * chunk : (rank + 1) * chunk].contiguous()
    b_local = b_full[:, rank * chunk : (rank + 1) * chunk].contiguous()
    return a_local, b_local, chunk
