"""Shared torchrun helpers for distributed correctness scripts."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def initialize_distributed() -> None:
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
        )
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def create_tp_group(tp_size: int):
    world_size = dist.get_world_size()
    if tp_size > world_size:
        raise ValueError(f"tp_size ({tp_size}) > world_size ({world_size})")
    tp_group = dist.new_group(ranks=list(range(tp_size)))
    return tp_group if dist.get_rank() < tp_size else None


def broadcast_tensor(tensor: torch.Tensor, group=None, src: int = 0) -> torch.Tensor:
    dist.broadcast(tensor, src=src, group=group)
    return tensor


def require_torchrun(*example_cmds: str) -> None:
    if "RANK" in os.environ:
        return
    print("ERROR: run with torchrun")
    for cmd in example_cmds:
        print(f"  {cmd}")
    raise SystemExit(1)


def randint_bf16(*shape, device="cuda") -> torch.Tensor:
    """Integer values in {-1, 0, 1} as bfloat16 (exact in bf16 for small K)."""
    return torch.randint(-1, 2, shape, device=device).to(torch.bfloat16)
