import torch
from typing import Any, Optional, Tuple

from miniTransformer.utils.utils import FP8_DTYPES

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    "tensor_model_parallel": False,
    "partition_dim": -1,
    "partition_stride": 1,
}


def get_distributed_world_size(
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> int:
    """Return world size for the distributed group."""
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group=group)


def get_distributed_rank(
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> int:
    """Return rank for the distributed group."""
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank(group=group)


def set_tensor_model_parallel_attributes(
    tensor: torch.Tensor, is_parallel: bool, dim: int, stride: int
) -> None:
    """set attributes needed for TP"""
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, "tensor_model_parallel", is_parallel)
    setattr(tensor, "partition_dim", dim)
    setattr(tensor, "partition_stride", stride)


def reduce_scatter_along_first_dim(
    input: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup,
    async_op: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.distributed.Work]]:
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = get_distributed_world_size(tp_group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input, None

    dim_size = list(input.size())
    assert (
        dim_size[0] % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"

    dim_size[0] = dim_size[0] // world_size

    output = torch.empty(
        dim_size, dtype=input.dtype, device=torch.cuda.current_device()
    )
    handle = torch.distributed.reduce_scatter_tensor(
        output, input.contiguous(), group=tp_group, async_op=async_op
    )
    return output, handle


def gather_along_first_dim(
    input: torch.Tensor,
    process_group: torch.distributed.ProcessGroup,
    async_op: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.distributed.Work]]:
    """All-gather tensors and concatenate along first dimension."""

    # Return immediately if no communication is required
    world_size = get_distributed_world_size(process_group)
    if world_size == 1:
        return input, None

    # Allocate output tensor
    output_shape = list(input.size())
    output_shape[0] *= world_size
    output = torch.empty(
        output_shape,
        dtype=input.dtype,
        device=input.device,
        memory_format=torch.contiguous_format,
    )
    src = input.contiguous()
    dst = output
    if input.dtype in FP8_DTYPES:
        src = src.view(torch.uint8)
        dst = dst.view(torch.uint8)

    # Launch all-gather
    handle = torch.distributed.all_gather_into_tensor(
        dst,
        src,
        group=process_group,
        async_op=async_op,
    )
    return output, handle


def allreduce(
    input_: torch.Tensor,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    async_op: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.distributed.Work]]:
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_distributed_world_size(tp_group) == 1:
        return input_, None

    # All-reduce.
    handle = torch.distributed.all_reduce(input_, group=tp_group, async_op=async_op)

    return input_, handle
