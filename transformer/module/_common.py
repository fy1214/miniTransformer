"""Internal function used by multiple modules."""

from typing import Any, List, Optional, Tuple, Union

import torch

class _NoopCatFunc(torch.autograd.Function):
    """Concatenate tensors, doing a no-op if possible

    See _noop_cat.

    """

    @staticmethod
    def forward(
        ctx: Any,
        dim: int,
        *tensors: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring

        # Check first tensor
        if not tensors:
            raise ValueError("Attempted to concatenate 0 tensors")
        num_dims = tensors[0].dim()
        if not -num_dims <= dim < num_dims:
            raise ValueError(
                "Attempted to concatenate tensor "
                f"with shape {list(tensors[0].size())} along dim {dim}"
            )
        dim %= num_dims

        # Check remaining tensors
        out_shape = list(tensors[0].size())
        split_ranges = [(0, tensors[0].size(dim))]
        for tensor in tensors[1:]:
            in_shape = list(tensor.size())
            if (
                len(in_shape) != num_dims
                or in_shape[:dim] != out_shape[:dim]
                or in_shape[dim + 1 :] != out_shape[dim + 1 :]
            ):
                raise ValueError(
                    "Attempted to concatenate tensors with shapes "
                    f"{[list(tensor.size()) for tensor in tensors]} "
                    f"along dim {dim}"
                )
            split_start = out_shape[dim]
            split_end = split_start + in_shape[dim]
            out_shape[dim] = split_end
            split_ranges.append((split_start, split_end))

        # Save state for backward
        ctx.dim = dim
        ctx.split_ranges = split_ranges

        # Out-of-place concatenation if needed
        dtype = tensors[0].dtype
        device = tensors[0].device
        strides = tensors[0].stride()
        data_ptr_stride = strides[dim] * tensors[0].element_size()
        data_ptr = tensors[0].data_ptr() + tensors[0].size(dim) * data_ptr_stride
        for tensor in tensors[1:]:
            if (
                tensor.dtype != dtype
                or tensor.device != device
                or tensor.stride() != strides
                or tensor.data_ptr() != data_ptr
            ):
                return torch.cat(tensors, dim=dim)
            data_ptr += tensor.size(dim) * data_ptr_stride

        # No-op concatenation
        out = tensors[0].new()
        out.set_(
            tensors[0].untyped_storage(),
            tensors[0].storage_offset(),
            out_shape,
            strides,
        )
        out.requires_grad = any(tensor.requires_grad for tensor in tensors)
        return out

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        grad_inputs = []
        for split_start, split_end in ctx.split_ranges:
            slices = [slice(None)] * grad_output.dim()
            slices[ctx.dim] = slice(split_start, split_end)
            grad_inputs.append(grad_output[tuple(slices)])
        return None, *grad_inputs


def noop_cat(
    tensors: List[torch.Tensor],
    dim: int = 0,
) -> torch.Tensor:
    """Concatenate tensors, doing a no-op if possible

    If tensors are already concatenated in memory, a tensor view of
    that memory region will be returned. Otherwise the tensors will be
    concatenated out-of-place, as usual.

    """
    if not tensors:
        raise ValueError("Attempted to concatenate 0 tensors")
    if len(tensors) == 1:
        return tensors[0]
    return _NoopCatFunc.apply(dim, *tensors)