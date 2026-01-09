import logging
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from miniTransformer.module._common import noop_cat
from miniTransformer.module.base import LayerLinearBaseModule
import miniTransformer.module.config as config

from miniTransformer.distributed.distributed import (
    allreduce,
    get_distributed_rank,
    gather_along_first_dim,
    reduce_scatter_along_first_dim,
    set_tensor_model_parallel_attributes,
)
from miniTransformer.ops import quantization
from miniTransformer.utils.utils import (
    is_rank_0,
    init_method_constant,
    cast_if_needed,
    divide,
)


logger = logging.getLogger(__name__)

class _NonQuantizedLinear(torch.autograd.Function):
    """Non-quantized Linear function (FP32, FP16, BF16)

    1) Row Major GEMM Equation
    Suppose x is [m, k], w is [n, k], and Y is [m, n]. Note that the
    [weight](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
    tensor in PyTorch is initialized with shape [n, k] instead of [k, n].

    fwd:    Y = x @ w.t()  # [m, k] x [k, n] = [m, n]
    dgrad: dX = dY @ w     # [m, n] x [n, k] = [m, k]
    wgrad: dW = dY.t() @ x # [n, m] x [m, k] = [n, k]

    2) GEMM Equation in cuBLAS terms

    The GEMM Layout in cuBLAS terms (TN, NN, NT) uses letters T/N to indicate whether
    the tensor is transposed in memory. The first letter refers to argument A, and
    the second to argument B. Note that in cuBLAS, the order of A and B is switched
    compared to the above equations.

    BF16/FP32 GEMM supports different GEMM layouts without requiring explicit
    transposition in memory. And torch.matmul automatically selects the most efficient
    GEMM layout.

    fwd   (TN):  Y = w @ x      # [n, k] x [m, k] = [m, n]
    dgrad (NN): dX = w @ dY     # [n, k] x [m, n] = [m, k]
    wgrad (NT): dW = x @ dY     # [m, k] x [m, n] = [n, k]
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        x_meta: Optional[quantization.QuantizeInputMetaBase],
        w: torch.Tensor,
        weight_qresult_cache: quantization.QuantizeResult,
        bias: Union[torch.Tensor, None],
        sequence_parallel: bool,
        tp_group: Optional[torch.distributed.ProcessGroup],
        tp_size: int,
        parallel_mode: Optional[str],
        qlinear_params: config.QLinearParams,
        activation_dtype: torch.dtype,
        is_grad_enabled: bool,
        is_first_microbatch: bool | None,
        fuse_wgrad_accumulation: bool,
    ) -> torch.Tensor:
        # Set dtypes, shapes and bias
        x_shape = x.shape
        x = x.view(-1, x.shape[-1])
        x = cast_if_needed(x, activation_dtype)
        use_bias = bias is not None
        if use_bias:
            # Make mypy happy with assert on Noneness.
            assert bias is not None
            bias = cast_if_needed(bias, activation_dtype)

        # AllGather for column and sequence parallel
        if parallel_mode == "column" and sequence_parallel:
            assert tp_group is not None
            x, _ = gather_along_first_dim(x, tp_group)

        # Forward pass GEMM
        # y = torch.matmul(x, w.t())
        # if use_bias and bias.numel():
        #     y = y + bias
        y = torch.nn.functional.linear(x, w, bias)

        # Save context for backward pass
        if is_grad_enabled:
            ctx.use_bias = use_bias
            ctx.sequence_parallel = sequence_parallel
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size
            ctx.parallel_mode = parallel_mode
            ctx.x_shape = x_shape

            ctx.save_for_backward(
                x,
                w,
            )

        # Reduce scatter for row and sequence parallel
        if parallel_mode == "row" and sequence_parallel:
            assert tp_group is not None
            y, _ = reduce_scatter_along_first_dim(y, tp_group)
        # Allreduce for rowwise parallel
        elif parallel_mode == "row" and tp_size > 1:
            y, _ = allreduce(y, tp_group)

        return y.view(-1, *x_shape[1:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, *dy_tup: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        # Get saved tensors and sanity checks
        assert len(dy_tup) == 1
        dy = dy_tup[0]
        (x, w) = ctx.saved_tensors
        assert x is not None and w is not None

        # Set dtypes, shapes
        dy = dy.contiguous()
        dy = dy.view(-1, dy.shape[-1])

        # Bias gradient
        b_grad = torch.sum(dy, dim=0) if ctx.use_bias else None

        # AllGather for row and sequence parallel
        if ctx.parallel_mode == "row" and ctx.sequence_parallel:
            dy, _ = gather_along_first_dim(dy, ctx.tp_group)

        # GEMM dgrad
        dgrad = torch.matmul(dy, w)

        # ReduceScatter / Allreduce for column parallel (Overlap with wgrad)
        if ctx.parallel_mode == "column" and ctx.sequence_parallel:
            dgrad, handle = reduce_scatter_along_first_dim(dgrad, ctx.tp_group, True)
        elif ctx.parallel_mode == "column" and ctx.tp_size > 1:
            dgrad, handle = allreduce(dgrad, ctx.tp_group, True)
        else:
            handle = None

        # GEMM wgrad
        # TODO need to support wgrad accumulation
        wgrad = torch.matmul(dy.t(), x)

        # Wait for communication to finish
        if handle is not None:
            handle.wait()

        return (
            dgrad.view(ctx.x_shape),
            None,
            wgrad,
            None,
            b_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

class _QuantizedLinear(torch.autograd.Function):
    """Optimized Linear Function with quantization support.

    The tensors x, w, bias, y, dy, wgrad, dgrad and bgrad should
    have the same dtype (FP32, BF16 or FP16). The GEMM output
    (y, wgrad, dgrad, bgrad) dtype could be further controlled
    by the qlinear_params.mm_(fprop|dgrad|wgrad).out_dtype.

    GEMM Equation in cuBLAS terms for TN layout

    Hopper FP8 GEMM only supports TN layout. So, the equations become:

    fwd   (TN):  Y = w @ x            # [n, k] x [m, k] = [m, n]
    dgrad (TN): dX = w.t() @ dY       # [k, n] x [m, n] = [m, k]
    wgrad (TN): dW = x.t() @ dY.t()   # [k, m] x [n, m] = [n, k]
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        x_meta: Optional[quantization.QuantizeInputMetaBase],
        w: torch.Tensor,
        weight_qresult_cache: quantization.QuantizeResultCache,
        bias: Union[torch.Tensor, None],
        sequence_parallel: bool,
        tp_group: Optional[torch.distributed.ProcessGroup],
        tp_size: int,
        parallel_mode: Optional[str],
        qlinear_params: config.QLinearParams,
        activation_dtype: torch.dtype,
        is_grad_enabled: bool,
        is_first_microbatch: bool | None,
        fuse_wgrad_accumulation: bool,
    ) -> torch.Tensor:

        # Get quantization op
        assert qlinear_params.quantize_op is not None
        quantize_op: quantization.QuantizeOpBase = qlinear_params.quantize_op

        # Set dtypes, shapes and bias
        x_shape = x.shape
        out_dtype = (
            x.dtype
            if qlinear_params.mm_fprop.out_dtype is None
            else qlinear_params.mm_fprop.out_dtype
        )
        x = x.view(-1, x.shape[-1])
        x = cast_if_needed(x, activation_dtype)
        use_bias = bias is not None
        if use_bias:
            assert bias is not None
            bias = cast_if_needed(bias, activation_dtype)

        # Quantize x (with optional transpose).
        # Will also perform AllGather if needed.
        gather_required = parallel_mode == "column" and sequence_parallel
        qresult_x = _QuantizedLinear.quantize_optionally_allgather(
            quantize_op,
            x,
            qlinear_params.x_params,
            transpose_required=is_grad_enabled,
            allgather_required=gather_required,
            allgather_quantize=qlinear_params.allgather_quantize,
            tp_group=tp_group,
        )

        qx, sx, qx_global_amax_row = qresult_x.data, qresult_x.scale, qresult_x.global_amax_row

        # Quantize w only for is_first_microbatch is True or None
        update_quantize_weights = is_first_microbatch or is_first_microbatch is None
        if update_quantize_weights:
            # Quantize w (with optional transpose)
            qresult_w = quantize_op.quantize(
                w, qlinear_params.w_params, return_transpose=is_grad_enabled
            )
            qw, sw, qw_global_amax_row = qresult_w.data, qresult_w.scale, qresult_w.global_amax_row
            # save quantized weight cache between microbatches
            weight_qresult_cache.data = qresult_w.data
            weight_qresult_cache.scale = qresult_w.scale
            weight_qresult_cache.data_t = qresult_w.data_t
            weight_qresult_cache.scale_t = qresult_w.scale_t
            weight_qresult_cache.global_amax_row = qresult_w.global_amax_row
            weight_qresult_cache.global_amax_col = qresult_w.global_amax_col
        else:
            qresult_w = weight_qresult_cache.to_qresult()
            qw, sw, qw_global_amax_row = qresult_w.data, qresult_w.scale, qresult_w.global_amax_row
            assert qw is not None, "quantized weight cache should not be None"

        # Forward pass GEMM
        y = quantize_op.qgemm(
            qx,
            qw,
            qlinear_params.mm_fprop,
            out_dtype,
            sx,
            sw,
            bias,
            accumulate=False,
            global_amax_qx=qx_global_amax_row,
            global_amax_qw=qw_global_amax_row,
            qparams_x=qlinear_params.x_params,
            qparams_w=qlinear_params.w_params,
        )

        # Save context for backward pass
        if is_grad_enabled:
            ctx.use_bias = use_bias
            ctx.sequence_parallel = sequence_parallel
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size
            ctx.tp_rank = get_distributed_rank(tp_group)
            ctx.x_gather_required = gather_required
            ctx.parallel_mode = parallel_mode
            ctx.x_shape = x_shape
            ctx.qlinear_params = qlinear_params
            ctx.quantize_op = quantize_op
            ctx.is_first_microbatch = is_first_microbatch
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.activation_dtype = activation_dtype

            qx_t, sx_t, qx_global_amax_col, qw_t, sw_t, qw_global_amax_col = (
                qresult_x.data_t,
                qresult_x.scale_t,
                qresult_x.global_amax_col,
                qresult_w.data_t,
                qresult_w.scale_t,
                qresult_w.global_amax_col,
            )

            # If memory saving is enabled, we only save a shard of the qx_t tensor.
            # The full tensor will need to be AllGathered in the backward pass.
            if qlinear_params.memory_saving and gather_required:
                # we need an explicit clone here, otherwise PyTorch is not able to free
                # the original full qx_t tensor because of reference counting
                qx_t = torch.chunk(qx_t, tp_size, dim=0)[ctx.tp_rank].detach().clone()

            ctx.save_for_backward(
                qx_t,
                sx_t,
                qx_global_amax_col,
                qw_t,
                sw_t,
                qw_global_amax_col,
                w,
            )

        # Reduce scatter for row and sequence parallel
        if parallel_mode == "row" and sequence_parallel:
            assert tp_group is not None
            y, _ = reduce_scatter_along_first_dim(y, tp_group)
        # Allreduce for rowwise parallel
        elif parallel_mode == "row" and tp_size > 1:
            y, _ = allreduce(y, tp_group)

        return y.view(-1, *x_shape[1:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, *dy_tup: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        # Get saved tensors and sanity checks
        assert len(dy_tup) == 1
        dy = dy_tup[0]
        (qx_t, sx_t, qx_global_amax_col, qw_t, sw_t, qw_global_amax_col, w) = ctx.saved_tensors

        assert qw_t is not None and sw_t is not None
        assert qx_t is not None and sx_t is not None

        # Set dtypes, shapes
        out_dtype_dgrad = (
            dy.dtype
            if ctx.qlinear_params.mm_dgrad.out_dtype is None
            else ctx.qlinear_params.mm_dgrad.out_dtype
        )
        dy = dy.contiguous()
        dy = dy.view(-1, dy.shape[-1])

        # Bias gradient
        b_grad = torch.sum(dy, dim=0) if ctx.use_bias else None

        # Quantize dy (with transpose)
        # Will also perform AllGather if needed
        gather_required = ctx.parallel_mode == "row" and ctx.sequence_parallel
        qresult_dy = _QuantizedLinear.quantize_optionally_allgather(
            ctx.quantize_op,
            dy,
            ctx.qlinear_params.g_params,
            transpose_required=True,
            allgather_required=gather_required,
            allgather_quantize=ctx.qlinear_params.allgather_quantize,
            tp_group=ctx.tp_group,
        )
        qdy, sdy, qdy_t, sdy_t, qdy_global_amax_row, qdy_global_amax_col = (
            qresult_dy.data,
            qresult_dy.scale,
            qresult_dy.data_t,
            qresult_dy.scale_t,
            qresult_dy.global_amax_row,
            qresult_dy.global_amax_col,
        )

        # If memory saving is enabled, we need to AllGather the full
        # qx_t tensor for wgrad GEMM. This is overlapped with dgrad GEMM.
        qx_t_handle = None
        if ctx.qlinear_params.memory_saving and ctx.x_gather_required:
            assert ctx.tp_group is not None
            qx_t, qx_t_handle = gather_along_first_dim(
                qx_t, ctx.tp_group, async_op=True
            )

        # GEMM dgrad
        dgrad = ctx.quantize_op.qgemm(
            qdy,
            qw_t,
            ctx.qlinear_params.mm_dgrad,
            out_dtype_dgrad,
            sdy,
            sw_t,
            None,
            None,
            accumulate=False,
            global_amax_qx=qdy_global_amax_row,
            global_amax_qw=qw_global_amax_col,
            qparams_x=ctx.qlinear_params.g_params,
            qparams_w=ctx.qlinear_params.w_params,
        )

        # If memory saving is enabled and we have a qx_t handle, Wait for qx_t AllGather to finish
        if qx_t_handle is not None:
            qx_t_handle.wait()
        # ReduceScatter / Allreduce for column parallel (Overlap with wgrad)
        if ctx.parallel_mode == "column" and ctx.sequence_parallel:
            dgrad, handle = reduce_scatter_along_first_dim(dgrad, ctx.tp_group, True)
        elif ctx.parallel_mode == "column" and ctx.tp_size > 1:
            dgrad, handle = allreduce(dgrad, ctx.tp_group, True)
        else:
            handle = None

        if ctx.is_first_microbatch is not None:
            accumulate_wgrad_into_param_main_grad = (
                ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
            )
        else:
            accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

        # For wgrad, we will set the dtype to float32
        out_dtype_wgrad = w.main_grad.dtype if ctx.fuse_wgrad_accumulation else torch.float32
        # GEMM wgrad
        wgrad = ctx.quantize_op.qgemm(
            qdy_t,
            qx_t,
            ctx.qlinear_params.mm_wgrad,
            out_dtype_wgrad,
            sdy_t,
            sx_t,
            None,
            out=w.main_grad if ctx.fuse_wgrad_accumulation else None,  # main_grad is fp32 dtype
            accumulate=accumulate_wgrad_into_param_main_grad,
            global_amax_qx=qdy_global_amax_col,
            global_amax_qw=qx_global_amax_col,
            qparams_x=ctx.qlinear_params.g_params,
            qparams_w=ctx.qlinear_params.x_params,
        )
        # Handle wgrad accumulation for mcore if needed
        wgrad = _QuantizedLinear.reset_wgrad_if_needed(ctx, w, wgrad)

        # Wait for communication to finish
        if handle is not None:
            handle.wait()

        return (
            dgrad.view(ctx.x_shape),
            None,
            wgrad,
            None,
            b_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def quantize_optionally_allgather(
        quantize_op: quantization.QuantizeOpBase,
        x: torch.Tensor,
        qparams: quantization.QParams,
        transpose_required: bool,
        allgather_required: bool,
        allgather_quantize: bool,
        tp_group: Optional[torch.distributed.ProcessGroup],
    ):
        """
        Quantize the input with optional AllGather.
        If AllGather is required, it support two paths:
            1) Quantize the local tensor shard (with amax reduction if needed), then AllGather in quantize type
            2) AllGather in BF16, then quantize the full tensor
        If AllGather is not required, it will just quantize the local tensor shard.
        """
        if allgather_required:
            # 1) AllGather in quantize
            if allgather_quantize:
                assert quantize_op.supports_allgather
                assert tp_group is not None
                qresult_x = quantize_op.quantize(
                    x,
                    qparams,
                    return_transpose=False,
                    reduce_amax=True,
                    tp_group=tp_group,
                )
                qresult_x.data, _ = gather_along_first_dim(qresult_x.data, tp_group)
            # 2) AllGather in BF16
            else:
                assert tp_group is not None
                x, _ = gather_along_first_dim(x, tp_group)
                qresult_x = quantize_op.quantize(
                    x,
                    qparams,
                    return_transpose=transpose_required,
                    reduce_amax=False,
                    tp_group=tp_group,
                )
        # AllGather not required
        else:
            qresult_x = quantize_op.quantize(
                x,
                qparams,
                return_transpose=transpose_required,
                reduce_amax=False,
                tp_group=tp_group,
            )
        return qresult_x

    @staticmethod
    def reset_wgrad_if_needed(ctx, w, wgrad):
        """
        When wgrad is already accumulated into w.main_grad, we need to either:
        1) Set wgrad to None so that the PyTorch autograd system does not handle wgrad anymore.
        2) For mcore, it has custom logic that requires resetting wgrad to an empty tensor or zeros to prevent deadlocks.
           See: https://github.com/NVIDIA/Megatron-LM/blob/215a2eb2c5cad942cd095976539bc74c6ae9d622/megatron/core/tensor_parallel/layers.py#L521-L544
        """
        if w.requires_grad:
            # Handle wgrad accumulation for mcore
            if ctx.fuse_wgrad_accumulation and hasattr(w, "grad_added_to_main_grad"):
                # Choose between torch.zeros or torch.empty based on zero_out_wgrad attribute.
                if getattr(w, "zero_out_wgrad", False):
                    wgrad = torch.zeros(
                        w.main_grad.shape,
                        dtype=w.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    wgrad = torch.empty(
                        w.main_grad.shape,
                        dtype=w.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                w.grad_added_to_main_grad = True
            # set wgrad to None if wgrad is already accumulated into main grad
            elif ctx.fuse_wgrad_accumulation:
                wgrad = None
            # else wgrad is unchanged
        else:
            # set wgrad to None if w does not requires grad
            wgrad = None

        return wgrad


class Linear(LayerLinearBaseModule):
    """Applies a linear transformation to the incoming data :math:`y = xA^T + b`

    On NVIDIA GPUs it is a drop-in replacement for `torch.nn.Linear`.

    Parameters
    ----------
    in_features : int
                 size of each input sample.
    out_features : int
                  size of each output sample.
    bias : bool, default = `True`
          if set to `False`, the layer will not learn an additive bias.
    init_method : Callable, default = `None`
                 used for initializing weights in the following way: `init_method(weight)`.
                 When set to `None`, defaults to `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    get_rng_state_tracker : Callable, default = `None`
                 used to get the random number generator state tracker for initializing weights.
    rng_tracker_name : str, default = `None`
                 the param passed to get_rng_state_tracker to get the specific rng tracker.
    parameters_split : Optional[Union[Tuple[str, ...], Dict[str, int]]], default = None
                      Configuration for splitting the weight and bias tensors along dim 0 into
                      multiple PyTorch parameters. If a list or tuple of strings is provided,
                      they are used to make the names of equally-sized parameters. If a dict
                      (preferably an OrderedDict) is provided, the keys are used as names and
                      values as split sizes along dim 0. The resulting parameters will have
                      names that end in `_weight` or `_bias`, so trailing underscores are
                      stripped from any provided names.
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will be allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.

    Parallelism parameters
    ----------------------
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    tp_size : int, default = 1
             used as TP (tensor parallel) world size when TP groups are not formed during
             initialization. In this case, users must call the
             `set_tensor_parallel_group(tp_group)` method on the initialized module before the
             forward pass to supply the tensor parallel group needed for tensor and sequence
             parallel collectives.
    parallel_mode : {None, 'column', 'row'}, default = `None`
                   used to decide whether this Linear layer is Column Parallel Linear or Row
                   Parallel Linear as described `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
                   When set to `None`, no communication is performed.
    ub_overlap_rs : bool, default = `False`
                    Reduce-Scatter overlap with GEMM by pipelining the GEMM and Reduce-Scatter.
    ub_overlap_ag : bool, default = `False`
                    All-Gather overlap with GEMM by pipelining the GEMM and All-Gather.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional `main_grad` attribute (used instead of the
                             regular `grad`) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in.
    return_bias : bool, default = `False`
                 when set to `True`, this module will not apply the additive bias itself, but
                 instead return the bias value during the forward pass together with the
                 output of the linear transformation :math:`y = xA^T`. This is useful when
                 the bias addition can be fused to subsequent operations.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.

    Quantization parameters
    -----------------------
    qlinear_params : Optional[config.QLinearParams], default = `None`
                     used to set quantization linear parameters for the linear layer.
                     If not provided, currently it will be determined from ENV Variable `QAT_PARAMS`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        rng_tracker_name: Optional[str] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        parallel_mode: Optional[str] = None,
        parameters_split: Optional[Union[Tuple[str, ...], Dict[str, int]]] = None,
        device: Union[torch.device, str] = "cuda",
        ub_overlap_rs: bool = False,
        ub_overlap_ag: bool = False,
        ub_name: Optional[str] = None,
        # JQ: hybrid addtional args
        layer_number: Optional[int] = None,
        qlinear_params: Optional[config.QLinearParams] = None,
    ) -> None:
        super().__init__(tp_group, tp_size, sequence_parallel)

        params_dtype = (
            torch.get_default_dtype() if params_dtype is None else params_dtype
        )
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.ub_overlap_rs = ub_overlap_rs
        self.ub_overlap_ag = ub_overlap_ag
        if ub_overlap_rs or ub_overlap_ag:
            assert ub_name is not None, "Userbuffer name [string] is not set."
        self.ub_name = ub_name
        self.get_rng_state_tracker = get_rng_state_tracker
        self.rng_tracker_name = rng_tracker_name
        # JQ: hybrid addtional
        self.layer_name = ub_name
        self.layer_number = layer_number

        # JQ: just set hybrid recipe
        # Set quantization parameters
        self.qlinear_params = self.set_qlinear_params(
            qlinear_params=qlinear_params,
            layer_number=self.layer_number,
            layer_name=self.layer_name,
        )

        if is_rank_0():
            logger.info(
                f"local_qlinear_params for layer {layer_number} linear module {self.layer_name} is set to {self.qlinear_params}"
            )

        if device == "meta":
            assert (
                parameters_split is None
            ), "Cannot split module parameters on 'meta' device."

        self.parallel_mode = parallel_mode
        assert self.parallel_mode in (
            "row",
            "column",
            None,
        ), f"parallel_mode {parallel_mode} not supported"

        if self.parallel_mode == "column":
            self.out_features = divide(self.out_features, self.tp_size)
        elif self.parallel_mode == "row":
            self.in_features = divide(self.in_features, self.tp_size)

        # JQ: not supported for now
        with_fp8_params = False
        # Initialize params in FP8
        # with_fp8_params = FP8GlobalStateManager.is_fp8_params_enabled()
        # self.fp8 = FP8GlobalStateManager.is_fp8_enabled()
        # assert self.fp8, "FP8 must be enabled for Linear module"
        assert with_fp8_params is False, "FP8 only parameters are not supported yet"
        # assert self.fsdp_group is None, "FSDP sharding is not supported yet"

        # Contiguous buffers for params
        weight_tensor = torch.empty(
            self.out_features,
            self.in_features,
            device=device,
            dtype=params_dtype,
        )
        bias_tensor = None
        if self.use_bias:
            bias_tensor = torch.empty(
                self.out_features,
                device=device,
                dtype=params_dtype,
            )

        # Configure parameter splits
        self.weight_names = []
        self.bias_names = []
        self.parameter_split_sizes = []
        if parameters_split is None:
            # Split into a single parameter by default
            self.weight_names = ["weight"]
            self.bias_names = ["bias"]
            self.parameter_split_sizes = [out_features]
        elif not parameters_split:
            raise ValueError("Cannot split weight buffer into 0 parameters")
        elif isinstance(parameters_split, dict):
            # Split parameters with provided sizes
            for name, split_size in parameters_split.items():
                self.weight_names.append(f"{name.rstrip('_')}_weight")
                self.bias_names.append(f"{name.rstrip('_')}_bias")
                self.parameter_split_sizes.append(split_size)
        elif all(isinstance(name, str) for name in parameters_split):
            # Split parameters evenly
            split_size = out_features // len(parameters_split)
            assert out_features % len(parameters_split) == 0, "Invalid split size"
            for name in parameters_split:
                self.weight_names.append(f"{name.rstrip('_')}_weight")
                self.bias_names.append(f"{name.rstrip('_')}_bias")
                self.parameter_split_sizes.append(split_size)
        else:
            raise TypeError("Invalid configuration for parameters split")

        # Make sure parameter splits are valid
        if sum(self.parameter_split_sizes) != out_features:
            raise ValueError(
                f"Trying to split weight buffer ({out_features=}) "
                f"with split sizes {self.parameter_split_sizes}"
            )

        # Adjust parameter splits for tensor-parallel distribution
        if self.parallel_mode == "column":
            for i, size in enumerate(self.parameter_split_sizes):
                if size % self.tp_size != 0:
                    raise RuntimeError(
                        f"Attempting to distribute a parameter with out_features={size} "
                        f"between {self.tp_size} tensor-parallel processes"
                    )
                self.parameter_split_sizes[i] = size // self.tp_size

        # Construct weight parameters
        # Note: Register weights together so that they are adjacent to
        # each other in Linear.parameters(). This makes it more likely
        # that they will stay contiguous if the weights are
        # manipulated externally, e.g. by FSDP.
        offset = 0
        for i, split_size in enumerate(self.parameter_split_sizes):
            split_start = offset
            offset += split_size
            split_end = offset

            # Check if parameters are subviews of buffers
            is_subview = (split_start, split_end) != (0, self.out_features)
            if is_subview and with_fp8_params:
                raise RuntimeError(
                    "Splitting Float8Tensor into multiple params is not supported"
                )

            # Construct weight parameter
            self.register_parameter(
                self.weight_names[i],
                torch.nn.Parameter(weight_tensor[split_start:split_end]),
                init_fn=init_method,
                get_rng_state_tracker=get_rng_state_tracker,
            )

        # Construct bias parameters if needed
        if self.use_bias:
            assert bias_tensor is not None
            offset = 0
            for i, split_size in enumerate(self.parameter_split_sizes):
                split_start = offset
                offset += split_size
                split_end = offset
                self.register_parameter(
                    self.bias_names[i],
                    torch.nn.Parameter(bias_tensor[split_start:split_end]),
                    init_fn=init_method_constant(0.0),
                )
        else:
            for name in self.bias_names:
                # Name "bias" would shadow bool parameter.
                bias_casted = torch.Tensor().to(dtype=params_dtype, device=device)
                setattr(self, name, bias_casted)

        self.reset_parameters(defer_init=(device == "meta"))

        # quantized weight cache between microbatches
        self.weight_qresult_cache = quantization.QuantizeResultCache(
            data=None, scale=None, data_t=None, scale_t=None
        )

        # For RPL, bias has to be added after TP collectives
        # So it cannot be fused with the GEMM
        if self.parallel_mode == "row" and self.apply_bias:
            self.gemm_bias_unfused_add = True
        else:
            self.gemm_bias_unfused_add = False

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)

        if not defer_init:
            # Set parallelism attributes for linear weights
            for weight in self.weight_names:
                set_tensor_model_parallel_attributes(
                    tensor=getattr(self, weight),
                    is_parallel=True,
                    dim=1 if self.parallel_mode == "row" else 0,
                    stride=1,
                )

            # Set parallelism attributes for linear biases
            if self.use_bias:
                for bias in self.bias_names:
                    if self.parallel_mode == "row":
                        setattr(
                            getattr(self, bias),
                            "sequence_parallel",
                            self.sequence_parallel,
                        )
                    elif self.parallel_mode == "column":
                        set_tensor_model_parallel_attributes(
                            getattr(self, bias), True, 0, 1
                        )

    @torch._dynamo.disable(recursive=True)
    def forward(
        self,
        inp: torch.Tensor,
        is_first_microbatch: Optional[bool] = None,
        fp8_output: Optional[bool] = False,
        inp_meta: Optional[quantization.QuantizeInputMetaBase] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply the linear transformation to the input.

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
        is_first_microbatch : {True, False, None}, default = None
                             During training using either gradient accumulation or
                             pipeline parallelism a minibatch of data is further split
                             into microbatches. Between the microbatches of the same minibatch
                             the model weights are not updated. Setting this parameter indicates
                             whether the current microbatch is the first in a minibatch or not.
                             When set, this parameter enables additional optimizations:

                             * during FP8 training, it allows caching of the FP8 versions of
                               the weights
                             * it also allows skipping gradient accumulation during the
                               first microbatch (since it is the first gradient being
                               produced)
        """

        self.set_activation_dtype(inp)
        inp = inp.contiguous()
        # Sanity checks
        assert self.tp_group_initialized, "TP group not initialized"
        input_dtypes = (torch.float, torch.float16, torch.bfloat16, torch.float32)
        assert inp.dtype in input_dtypes, f"Input dtype {inp.dtype} not supported"
        assert isinstance(
            self.weight_qresult_cache, quantization.QuantizeResultCache
        ), "weight_qresult_cache must be QuantizeResultCache"
        assert fp8_output is False, "FP8 output is not supported yet"

        weight_tensor = noop_cat(
            [getattr(self, name) for name in self.weight_names]
        )
        if self.use_bias:
            bias_tensor = noop_cat(
                [getattr(self, name) for name in self.bias_names],
            )
        else:
            bias_tensor = getattr(self, self.bias_names[0])  # Unused

        # Choose the appropriate linear function
        LinearFunction = (
            _QuantizedLinear if self.qlinear_params.quantize else _NonQuantizedLinear
        )

        if torch.is_grad_enabled():
            linear_fn = LinearFunction.apply
            args: List[Any] = []
        else:
            linear_fn = LinearFunction.forward
            args = [None]
        args += (
            inp,
            inp_meta,
            weight_tensor,
            self.weight_qresult_cache,
            bias_tensor if self.apply_bias and not self.gemm_bias_unfused_add else None,
            self.sequence_parallel,
            self.tp_group,
            self.tp_size,
            self.parallel_mode,
            self.qlinear_params,
            self.activation_dtype,
            torch.is_grad_enabled(),
            is_first_microbatch,
            self.fuse_wgrad_accumulation,
        )
        out = linear_fn(*args)

        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(bias_tensor, self.activation_dtype)

        if self.return_bias:
            return out, cast_if_needed(bias_tensor, self.activation_dtype)

        return out
