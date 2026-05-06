"""
nvshmem_init.py
===============

独立的 nvshmem 初始化模块。

在 torch.distributed.init_process_group() 之后调用 init_nvshmem_with_dist()，
通过 torch.distributed.broadcast 交换 nvshmem UniqueID，完成 nvshmem 初始化。

用法：
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")   # 或任意 backend

    from miniTransformer.distributed.cutedsl.nvshmem_init import init_nvshmem_with_dist
    dev = init_nvshmem_with_dist()            # 在其后调用一次

    # ... 之后可使用 nvshmem.core.tensor() 等 API ...

    from miniTransformer.distributed.cutedsl.nvshmem_init import finalize_nvshmem
    finalize_nvshmem()                        # 进程退出前调用
"""

import os

import numpy as np
import torch
import torch.distributed as dist

try:
    import nvshmem.core
except ImportError:
    raise ImportError(
        "nvshmem4py is required. Install with:\n"
        "  pip install nvshmem4py-cu12 nvidia-nvshmem-cu12"
    )

try:
    from cuda.pathfinder import load_nvidia_dynamic_lib
    load_nvidia_dynamic_lib("nvshmem_host")
except (ImportError, RuntimeError):
    raise ImportError(
        "nvidia-nvshmem is required. Install with:\n"
        "  pip install nvidia-nvshmem-cu12"
    )

try:
    from cuda.core.experimental import Device
except ImportError:
    from cuda.core import Device


def init_nvshmem_with_dist(tp_group=None):
    """
    在 torch.distributed 已初始化后，初始化 nvshmem。

    通过 torch.distributed.broadcast 在所有 rank 间交换 nvshmem UniqueID，
    然后调用 nvshmem.core.init() 完成初始化。

    不需要修改现有的 dist.init_process_group 调用，
    也不需要双 backend（cpu:gloo,cuda:nccl），直接复用已有的 dist。

    Args:
        tp_group: torch.distributed.ProcessGroup，None 则使用 default group

    Returns:
        cuda.core.Device: 当前 rank 的 CUDA device 对象
    """
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    # 创建 cuda.core Device
    dev = Device(local_rank)
    dev.set_current()

    rank = dist.get_rank(tp_group)
    world_size = dist.get_world_size(tp_group)

    # rank 0 生成 UniqueID，其余 rank 生成空的占位
    uid = nvshmem.core.get_unique_id(empty=(rank != 0))

    # 通过 torch.distributed broadcast 同步 UID
    uid_bytes = uid._data.view(np.uint8).copy()
    uid_tensor = torch.from_numpy(uid_bytes).cuda()
    dist.broadcast(uid_tensor, src=0, group=tp_group)
    dist.barrier(group=tp_group)
    uid._data[:] = uid_tensor.cpu().numpy().view(uid._data.dtype)

    # 初始化 nvshmem
    nvshmem.core.init(
        device=dev,
        uid=uid,
        rank=rank,
        nranks=world_size,
        initializer_method="uid",
    )

    return dev


def finalize_nvshmem():
    """
    清理 nvshmem 资源，在进程退出前调用。
    必须在所有 nvshmem.core.free_tensor() 调用之后执行。
    """
    try:
        nvshmem.core.finalize()
    except Exception:
        pass
