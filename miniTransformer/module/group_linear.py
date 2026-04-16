import logging
import torch
from typing import Callable, List, Optional, Tuple, Union

from . import config
from .base import LayerLinearBaseModule, get_dummy_wgrad
from ..distributed.distributed import get_distributed_rank, allreduce, set_tensor_model_parallel_attributes
from ..quantiztion import quantization
from ..quantiztion.quantization import GroupQuantizeResultCache
from ..utils.utils import is_rank_0, cast_if_needed, divide, init_method_constant

logger = logging.getLogger(__name__)


def log_rank_0(msg):
    if is_rank_0():
        logger.info(msg)


class _QuantizedGroupedLinear(torch.autograd.Function):
    """Optimized Linear Function with quantization support.

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
        m_splits: List[int],
        weight_qresult_cache: quantization.QuantizeResultCache,
        tp_group: Optional[torch.distributed.ProcessGroup],
        tp_size: int,
        parallel_mode: Optional[str],
        qlinear_params: config.QLinearParams,
        activation_dtype: torch.dtype,
        is_grad_enabled: bool,
        is_first_microbatch: bool | None,
        fuse_wgrad_accumulation: bool,
        *weights,
    ) -> torch.Tensor:
        num_gemms = len(m_splits)

        assert qlinear_params.quantize_op is not None
        # Use DeepGEMM quantize_op for x, w, w_t, dy (not dy_t nor x_t)
        quantize_op: quantization.QuantizeOpBase = qlinear_params.quantize_op

        # Set dtypes, shapes
        x_shape = x.shape
        out_dtype = (
            x.dtype
            if qlinear_params.mm_fprop.out_dtype is None
            else qlinear_params.mm_fprop.out_dtype
        )
        x = x.view(-1, x.shape[-1])  # to 2D
        x = cast_if_needed(x, activation_dtype)

        # Quantize x (x_t is handled in backward)
        # x: [m_sum, k], scale shape: [m_sum, k/128]
        qresult_x_list = quantize_op.group_quantize(
            x,
            qlinear_params.x_params,
            m_splits=m_splits,
            return_transpose=is_grad_enabled
        )
        qx, sx, tx, qx_t, sx_t, tx_t = (qresult_x_list.data,
                                          qresult_x_list.scale,
                                          qresult_x_list.tensor_scale,
                                          qresult_x_list.data_t,
                                          qresult_x_list.scale_t,
                                          qresult_x_list.tensor_scale_t)


        # Quantize w only for is_first_microbatch is True or None
        update_fp8_weights = (is_first_microbatch or (is_first_microbatch is None))
        if update_fp8_weights:
            w = torch.stack(weights, dim=0)  # [num_gemms, n, k]
            qresult_w_list = quantize_op.group_quantize(
                w,
                qlinear_params.w_params,
                m_splits=m_splits,
                return_transpose=is_grad_enabled
            )
            # save FP8 weight cache between microbatches
            qw = weight_qresult_cache.data = qresult_w_list.data
            sw = weight_qresult_cache.scale = qresult_w_list.scale
            qw_t = weight_qresult_cache.data_t = qresult_w_list.data_t
            sw_t = weight_qresult_cache.scale_t = qresult_w_list.scale_t
            tw = weight_qresult_cache.global_amax_row = qresult_w_list.tensor_scale
            tw_t = weight_qresult_cache.global_amax_col = qresult_w_list.tensor_scale_t
        else:
            w = None
            qw, sw, tw, qw_t, sw_t, tw_t  = (weight_qresult_cache.data,
                                             weight_qresult_cache.scale,
                                             weight_qresult_cache.global_amax_row,
                                             weight_qresult_cache.data_t,
                                             weight_qresult_cache.scale_t,
                                             weight_qresult_cache.global_amax_col)

        assert qw is not None, "FP8 weight cache should not be None"

        y = quantize_op.grouped_gemm(
            A=qx,
            B=qw,
            sA=sx,
            sB=sw,
            global_amax_A=tx,
            global_amax_B=tw,
            out=None,
            out_dtype=out_dtype,
            m_splits=m_splits,
            qparams_x=qlinear_params.x_params,
            qparams_w=qlinear_params.w_params
        )
        logger.debug(f"[Hybrid] After Grouped QGEMM, y shape [m, n]: {list(y.shape)}")

        # Save context for backward pass
        if is_grad_enabled:
            ctx.use_bias = False    # not use bias inside this OP, but may apply bias outside
            # ctx.sequence_parallel = sequence_parallel
            ctx.num_gemms = num_gemms
            ctx.m_splits = m_splits
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size
            ctx.tp_rank = get_distributed_rank(tp_group) if tp_group is not None else 0
            # ctx.x_gather_required = gather_required
            ctx.parallel_mode = parallel_mode
            ctx.x_shape = x_shape
            ctx.qlinear_params = qlinear_params
            ctx.quantize_op = quantize_op
            ctx.is_first_microbatch = is_first_microbatch
            ctx.activation_dtype = activation_dtype
            ctx.device = x.device  # Save device for backward
            ctx.weights_requires_grad = weights[0].requires_grad
            if fuse_wgrad_accumulation and ctx.weights_requires_grad:
                # This check is needed to ensure that main_grad is not created
                # during the forward pass when using MCore FSDP as it creates
                # the main_grad buffer lazily before backprop
                if hasattr(weights[0], "__fsdp_param__"):
                    # MCore FSDP creates main_grad lazily before backward
                    ctx.main_grad_funcs = [weights[i].get_main_grad for i in range(num_gemms)]
                else:
                    ctx.main_grad_funcs = [
                        lambda j=i: weights[j].main_grad for i in range(num_gemms)
                    ]
            else:
                ctx.main_grad_funcs = [lambda: None for i in range(num_gemms)]

            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation

            # save_for_backward only accepts plain torch.Tensor.
            # NVFP4TensorStorage / NVFP4Tensor are TE custom objects, store them on ctx directly.
            # Only plain weight tensors go into save_for_backward.
            ctx.qx_t  = qx_t
            ctx.sx_t  = sx_t
            ctx.tx_t  = tx_t
            ctx.qw_t  = qw_t
            ctx.sw_t  = sw_t
            ctx.tw_t  = tw_t
            ctx.save_for_backward(*weights)

        # JQ: sequence parallel not ready yet
        # if parallel_mode == "row" and sequence_parallel:
        #     assert tp_group is not None
        #     y, _ = reduce_scatter_along_first_dim(y, tp_group)

        # TODO:
        if (parallel_mode == "row") and (tp_size > 1):
            y, _ = allreduce(y, tp_group)

        return y.view(-1, *x_shape[1:-1], y.shape[-1])


    @staticmethod
    def backward(ctx, *dy_tup: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        logger.debug(f"[Hybrid] To run GroupedLinear backward")
        N = ctx.num_gemms
        main_grads = [main_grad_func() for main_grad_func in ctx.main_grad_funcs]

        # Get saved tensors and sanity checks
        assert len(dy_tup) == 1
        dy = dy_tup[0]

        # Restore quantized tensors stored directly on ctx (not via save_for_backward)
        qx_t = ctx.qx_t
        sx_t = ctx.sx_t
        tx_t = ctx.tx_t
        qw_t = ctx.qw_t
        sw_t = ctx.sw_t
        tw_t = ctx.tw_t
        weights = list(ctx.saved_tensors)   # plain weight tensors

        assert qw_t is not None and sw_t is not None

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
        # dy: [m_sum, h_out]
        # dy_t: [groups, h_out, m]
        qresult_dy_list = ctx.quantize_op.group_quantize(
            dy,
            ctx.qlinear_params.g_params,
            m_splits=ctx.m_splits,
            return_transpose=True,  # Need transpose for wgrad computation
        )
        qdy, sdy, tdy, qdy_t, sdy_t, tdy_t = (qresult_dy_list.data,
                                              qresult_dy_list.scale,
                                              qresult_dy_list.tensor_scale,
                                              qresult_dy_list.data_t,
                                              qresult_dy_list.scale_t,
                                              qresult_dy_list.tensor_scale_t)

        dgrad = ctx.quantize_op.grouped_gemm(
            A=qdy,
            B=qw_t,
            sA=sdy,
            sB=sw_t,
            global_amax_A=tdy,
            global_amax_B=tw_t,
            out=None,
            out_dtype=out_dtype_dgrad,
            layout="NN",
            m_splits=ctx.m_splits,
            grad=True,
            qparams_x=ctx.qlinear_params.g_params,
            qparams_w=ctx.qlinear_params.w_params
        )

        # AllReduce for TP
        handle = None
        if (ctx.parallel_mode == "column") and (ctx.tp_size > 1):
            dgrad, handle = allreduce(dgrad, ctx.tp_group, True)

        # TODO: support it?
        if ctx.is_first_microbatch is not None:
            accumulate_wgrad_into_param_main_grad = (
                ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
            )
        else:
            accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

        if ctx.fuse_wgrad_accumulation:
            wgrad_list = main_grads
        else:
            wgrad_list = [
                torch.empty(w.size(), dtype=ctx.activation_dtype, device=ctx.device)
                for w in weights
            ]

        # GEMM wgrad
        # TODO: single wgrad or list?
        wgrad = ctx.quantize_op.grouped_gemm(
            A=qdy_t,
            B=qx_t,
            sA=sdy_t,
            sB=sx_t,
            global_amax_A=tdy_t,
            global_amax_B=tx_t,
            out=wgrad_list,
            out_dtype=ctx.activation_dtype,
            layout="NT",
            m_splits=ctx.m_splits,
            grad=True,
            accumulate=accumulate_wgrad_into_param_main_grad,
            qparams_x=ctx.qlinear_params.g_params,
            qparams_w=ctx.qlinear_params.x_params
        )

        # TODO: Handle wgrad accumulation for mcore if needed
        def handle_custom_ddp_from_mcore(weight, wgrad):
            if ctx.weights_requires_grad:
                # Handle custom DDP from mcore.
                if ctx.fuse_wgrad_accumulation and hasattr(
                        weight, "grad_added_to_main_grad"
                ):
                    weight.grad_added_to_main_grad = True
                    if getattr(weight, "zero_out_wgrad", False):
                        wgrad = get_dummy_wgrad(
                            list(weight.main_grad.shape),
                            weight.dtype,
                            zero=True,
                        )
                    else:
                        wgrad = get_dummy_wgrad(
                            list(weight.main_grad.shape),
                            weight.dtype,
                        )
                elif ctx.fuse_wgrad_accumulation:
                    wgrad = None
            else:
                wgrad = None
            return wgrad

        wgrad_list = [
            handle_custom_ddp_from_mcore(weight, wgrad)
            for weight, wgrad in zip(weights, wgrad_list)
        ]

        # Wait for communication to finish
        if handle is not None:
            handle.wait()

        return (
            dgrad.view(ctx.x_shape),
            None,        # m_splits
            None,        # weight_qresult_cache
            None,        # tp_group
            None,        # tp_size
            None,        # parallel_model
            None,        # qlinear_params
            None,        # activation_dtype
            None,        # is_grad_enabled
            None,        # is_first_microbatch
            None,        # fuse_wgrad_accumulation
            *wgrad_list,      # *weights
        )

class GroupedLinear(LayerLinearBaseModule):
    def __init__(
        self,
        num_gemms: int,
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
        self.num_gemms = num_gemms
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = (bias and not return_bias)   # not used yet
        self.ub_overlap_rs = ub_overlap_rs
        self.ub_overlap_ag = ub_overlap_ag
        self.ub_name = ub_name
        assert (
            not ub_overlap_rs and not ub_overlap_ag
        ), "GroupedLinear doesn't support Userbuffer overlap."
        self.get_rng_state_tracker = get_rng_state_tracker
        self.rng_tracker_name = rng_tracker_name
        # JQ: hybrid addtional
        self.layer_name = ub_name
        self.layer_number = layer_number

        log_rank_0(f"[Hybrid] Grouped Linear init:"
                   f" num_gemms: {num_gemms}"
                   f", in_features: {in_features}, out_features: {out_features}"
                   f", fuse wgrad accum: {fuse_wgrad_accumulation}"
                   f", use_bias: {bias}")

        # JQ: just set hybrid recipe
        # Set quantization parameters
        self.qlinear_params = self.set_qlinear_params(
            qlinear_params=qlinear_params,
            layer_number=self.layer_number,
            layer_name=self.layer_name,
        )
        log_rank_0(f"[Hybrid] local_qlinear_params for layer {layer_number} Grouped Linear module {self.layer_name} is set to {self.qlinear_params}")

        log_rank_0(f"[Hybrid] GroupedLinear will not perform TP allgather, double-check by yourself!")
        self.parallel_mode = parallel_mode
        assert (
            self.parallel_mode in ("row", "column", None,)
        ), f"parallel_mode {parallel_mode} not supported"

        if self.parallel_mode == "column":
            self.out_features = divide(self.out_features, self.tp_size)
        elif self.parallel_mode == "row":
            self.in_features = divide(self.in_features, self.tp_size)

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel
        assert (
            not self.sequence_parallel
        ), f"Grouped Linear does not support sequence_parallel!"

        # Reshape w to [groups*h_out, h_in] before 2D quantization
        # thus h_out must be multiple of 128, otherwise different groups are mixed together.
        assert self.out_features % 128 == 0, "W's out_features {self.out_features} must be multiple of 128!"

        self._offsets = {
            "input": 0,
            "weight": num_gemms,
            "output": 2 * num_gemms,
            "grad_output": 0
        }
        for i in range(self.num_gemms):
            # Construct weight parameter
            self.register_parameter(
                f"weight{i}",
                torch.nn.Parameter(
                    torch.empty(
                        self.out_features,
                        self.in_features,
                        device=device,
                        dtype=params_dtype,
                    ),
                ),
                init_fn=init_method,
                get_rng_state_tracker=get_rng_state_tracker,
                # JQ: not needed in hybrid, see _ParameterInitMeta
                # fp8_meta_index=self._offsets["weight"] + i,
            )

            # Construct bias parameters if needed
            if self.use_bias:
                self.register_parameter(
                    f"bias{i}",
                    torch.nn.Parameter(
                        torch.empty(
                            self.out_features,
                            device=device,
                            dtype=params_dtype,
                        ),
                    ),
                    init_fn=init_method_constant(0.0),
                )
            else:
                bias = torch.Tensor().to(dtype=params_dtype, device=device)
                setattr(self, f"bias{i}", bias)

        # if self.primary_weights_in_fp8:
        #     self.init_fp8_metadata(num_gemms=self.num_gemms)

        self.reset_parameters(defer_init=(device=="meta"))

        # quantized weight cache between microbatches
        self.weight_qresult_cache = GroupQuantizeResultCache(
            data=None, scale=None, data_t=None, scale_t=None
        )

        # JQ: DeepGEMM Grouped GEMM not support bias yet, thus apply bias at the end.
        self.gemm_bias_unfused_add = (self.use_bias and not return_bias)

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)

        if not defer_init:
            # Set parallelism attributes for linear weights
            for i in range(self.num_gemms):
                set_tensor_model_parallel_attributes(
                    tensor=getattr(self, f"weight{i}"),
                    is_parallel=True,
                    dim=1 if self.parallel_mode == "row" else 0,
                    stride=1,
                )

            # Set parallelism attributes for linear biases
            if self.use_bias:
                for i in range(self.num_gemms):
                    if self.parallel_mode == "row":
                        setattr(
                            getattr(self, f"bias{i}"),
                            "sequence_parallel",
                            self.sequence_parallel,
                        )
                    elif self.parallel_mode == "column":
                        set_tensor_model_parallel_attributes(getattr(self, f"bias{i}"), True, 0, 1)

    @torch._dynamo.disable(recursive=True)
    def forward(
        self,
        inp: torch.Tensor,
        m_splits: List[int],
        is_first_microbatch: Optional[bool] = None,
        # JQ: Linear has these two, do we need them here?
        # fp8_output: Optional[bool] = False,
        # inp_meta: Optional[quantization.QuantizeInputMetaBase] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply the linear transformation to the input.

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
        m_splits : List[int]
                 List of integers representing the split of the input tensor.
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
        assert len(m_splits) == self.num_gemms, "Number of splits should match number of GEMMs."

        # skip_fp8_weight_update = FP8GlobalStateManager.get_skip_fp8_weight_update_tensor()
        # if skip_fp8_weight_update is not None:
        #     is_first_microbatch = False

        # with self.prepare_forward(inp, num_gemms=self.num_gemms) as inp:
        self.set_activation_dtype(inp)
        inp = inp.contiguous()

        # Prepare weight and bias
        weight_tensors = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
        bias_tensors = [getattr(self, f"bias{i}") for i in range(self.num_gemms)]

        if torch.is_grad_enabled(): # Training
            linear_fn = _QuantizedGroupedLinear.apply
            args = []
        else: # Validation
            linear_fn = _QuantizedGroupedLinear.forward
            args = [None]
        args += (
            inp,
            m_splits,
            self.weight_qresult_cache,
            self.tp_group,
            self.tp_size,
            self.parallel_mode,
            self.qlinear_params,
            self.activation_dtype,
            torch.is_grad_enabled(),
            is_first_microbatch,
            self.fuse_wgrad_accumulation,
            *weight_tensors,
        )
        out = linear_fn(*args)

        if self.gemm_bias_unfused_add:
            out_shape = out.shape
            out = torch.cat(
                [
                    o + cast_if_needed(b, self.activation_dtype)
                    for o, b in zip(
                        torch.split(out.view(-1, self.out_features), m_splits), bias_tensors
                    )
                ]
            ).view(out_shape)

        if self.return_bias:
            return out, [cast_if_needed(b, self.activation_dtype) for b in bias_tensors]

        return out
