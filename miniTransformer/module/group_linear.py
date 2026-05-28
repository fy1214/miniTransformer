import logging
import os
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from typing import Callable, List, Optional, Tuple, Union

from . import config
from .base import LayerLinearBaseModule, get_dummy_wgrad
from ..distributed.distributed import get_distributed_rank, allreduce, set_tensor_model_parallel_attributes
from ..quantiztion import quantization
from ..quantiztion.quantization import GroupQuantizeResultCache
from ..utils.utils import is_rank_0, cast_if_needed, divide, init_method_constant
from ..utils.env import should_keep_backward_in_bf16

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
        disable_backward_quant: bool,
        *weights,
    ) -> torch.Tensor:
        num_gemms = len(m_splits)

        assert qlinear_params.quantize_op is not None
        quantize_op: quantization.QuantizeOpBase = qlinear_params.quantize_op

        x_shape = x.shape
        out_dtype = (
            x.dtype
            if qlinear_params.mm_fprop.out_dtype is None
            else qlinear_params.mm_fprop.out_dtype
        )
        x = x.view(-1, x.shape[-1])  # to 2D
        x = cast_if_needed(x, activation_dtype)

        # ── DUMP: save first forward call's raw bf16 x/w/y for offline analysis ──
        _dump_dir = os.environ.get("NVFP4_DUMP_DIR", "")
        _dump_call_limit = int(os.environ.get("NVFP4_DUMP_CALLS", "0"))
        _dump_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        if not hasattr(_QuantizedGroupedLinear, "_dump_count"):
            _QuantizedGroupedLinear._dump_count = 0
        _do_dump = _dump_dir and _dump_rank == 0 and _QuantizedGroupedLinear._dump_count < _dump_call_limit

        _prof_dir = os.environ.get("NVFP4_PROFILE_DIR", "")
        _prof_calls = int(os.environ.get("NVFP4_PROFILE_CALLS", "0"))
        _prof_grad = os.environ.get("NVFP4_PROFILE_GRAD", "0") == "1"
        if not hasattr(_QuantizedGroupedLinear, "_prof_count"):
            _QuantizedGroupedLinear._prof_count = 0
        # When NVFP4_PROFILE_GRAD=1, profiling is driven from GroupedLinear.forward
        # to capture backward as well; skip starting the profiler here.
        _do_profile = (
            bool(_prof_dir)
            and _dump_rank == 0
            and _QuantizedGroupedLinear._prof_count < _prof_calls
            and not _prof_grad
        )
        if _do_profile:
            _prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            _prof.__enter__()

        # disable_backward_quant skips columnwise quantization since x_t is unused in BF16 backward
        with record_function("group_quantize_x"):
            qresult_x_list = quantize_op.group_quantize(
                x,
                qlinear_params.x_params,
                m_splits=m_splits,
                return_transpose=(is_grad_enabled and not disable_backward_quant)
            )
        qx, sx, tx, qx_t, sx_t, tx_t = (qresult_x_list.data,
                                          qresult_x_list.scale,
                                          qresult_x_list.tensor_scale,
                                          qresult_x_list.data_t,
                                          qresult_x_list.scale_t,
                                          qresult_x_list.tensor_scale_t)


        update_fp8_weights = (is_first_microbatch or (is_first_microbatch is None))
        # If the cache was filled during a no_grad forward (data_t=[]) but this
        # forward needs transpose for backward, force a re-quantize so qw_t is valid.
        need_transpose = is_grad_enabled and not disable_backward_quant
        if not update_fp8_weights and need_transpose and not weight_qresult_cache.data_t:
            update_fp8_weights = True

        if quantize_op.should_update_quantize_weights:
            w = torch.stack(weights, dim=0)  # [num_gemms, n, k]
            with record_function("group_quantize_w"):
                qresult_w_list = quantize_op.group_quantize(
                    w,
                    qlinear_params.w_params,
                    m_splits=m_splits,
                    return_transpose=(is_grad_enabled and not disable_backward_quant)
                )
            qw = qresult_w_list.data
            sw = qresult_w_list.scale
            qw_t = qresult_w_list.data_t
            sw_t = qresult_w_list.scale_t
            tw = qresult_w_list.tensor_scale
            tw_t = qresult_w_list.tensor_scale_t
        elif update_fp8_weights:
            w = torch.stack(weights, dim=0)  # [num_gemms, n, k]
            with record_function("group_quantize_w"):
                qresult_w_list = quantize_op.group_quantize(
                    w,
                    qlinear_params.w_params,
                    m_splits=m_splits,
                    return_transpose=(is_grad_enabled and not disable_backward_quant)
                )
            qw = weight_qresult_cache.data = qresult_w_list.data
            sw = weight_qresult_cache.scale = qresult_w_list.scale
            qw_t = weight_qresult_cache.data_t = qresult_w_list.data_t
            sw_t = weight_qresult_cache.scale_t = qresult_w_list.scale_t
            tw = weight_qresult_cache.global_amax_row = qresult_w_list.tensor_scale
            tw_t = weight_qresult_cache.global_amax_col = qresult_w_list.tensor_scale_t
            weight_qresult_cache.extra_info = qresult_w_list.extra_info
        else:
            w = None
            qw, sw, tw, qw_t, sw_t, tw_t  = (weight_qresult_cache.data,
                                             weight_qresult_cache.scale,
                                             weight_qresult_cache.global_amax_row,
                                             weight_qresult_cache.data_t,
                                             weight_qresult_cache.scale_t,
                                             weight_qresult_cache.global_amax_col)
            qresult_w_list = quantization.GroupQuantizeResult(
                data=qw, scale=sw, data_t=qw_t, scale_t=sw_t,
                tensor_scale=tw, tensor_scale_t=tw_t,
                extra_info=weight_qresult_cache.extra_info,
            )

        assert qw is not None, "FP8 weight cache should not be None"

        with record_function("grouped_gemm"):
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
                qparams_w=qlinear_params.w_params,
                extra_info_A=qresult_x_list.extra_info,
                extra_info_B=qresult_w_list.extra_info
            )
        if _do_profile:
            _prof.__exit__(None, None, None)
            print(_prof.key_averages().table(sort_by="cuda_time_total", row_limit=30), flush=True)
            os.makedirs(_prof_dir, exist_ok=True)
            _trace = os.path.join(_prof_dir, f"gl_trace_{_QuantizedGroupedLinear._prof_count:03d}.json")
            _prof.export_chrome_trace(_trace)
            print(f"[PROFILE] Trace saved → {_trace}", flush=True)
            _QuantizedGroupedLinear._prof_count += 1
        logger.debug(f"[Hybrid] After Grouped QGEMM, y shape [m, n]: {list(y.shape)}")

        # ── DUMP: save x / w(stacked) / y / m_splits for offline single-op replay ──
        if _do_dump:
            os.makedirs(_dump_dir, exist_ok=True)
            save_path = os.path.join(_dump_dir, f"gemm_call_{_QuantizedGroupedLinear._dump_count:03d}.pt")
            # w may be None when using cached weight (is_first_microbatch=False); skip those
            w_save = w.detach().cpu() if w is not None else None
            torch.save({
                "x": x.detach().cpu(),       # bf16 [total_m, k]
                "w": w_save,                 # bf16 [num_gemms, n, k] or None
                "m_splits": m_splits,
                "y_fp4": y.detach().cpu(),   # FP4 GEMM output [total_m, n]
                "out_dtype": str(out_dtype),
            }, save_path)
            print(f"[DUMP] call={_QuantizedGroupedLinear._dump_count}  saved={save_path}  "
                  f"x={tuple(x.shape)}  m_splits={m_splits}  "
                  f"w={'x'.join(str(d) for d in w_save.shape) if w_save is not None else 'cached'}  "
                  f"y={tuple(y.shape)}", flush=True)
            _QuantizedGroupedLinear._dump_count += 1

        # Save context for backward pass
        if is_grad_enabled:
            ctx.use_bias = False    # bias is applied outside this op
            ctx.num_gemms = num_gemms
            ctx.m_splits = m_splits
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size
            ctx.tp_rank = get_distributed_rank(tp_group) if tp_group is not None else 0
            ctx.parallel_mode = parallel_mode
            ctx.x_shape = x_shape
            ctx.qlinear_params = qlinear_params
            ctx.quantize_op = quantize_op
            ctx.is_first_microbatch = is_first_microbatch
            ctx.activation_dtype = activation_dtype
            ctx.device = x.device  # Save device for backward
            ctx.weights_requires_grad = weights[0].requires_grad
            if fuse_wgrad_accumulation and ctx.weights_requires_grad:
                # MCore FSDP creates main_grad lazily before backward — don't access it during forward.
                if hasattr(weights[0], "__fsdp_param__"):
                    # MCore FSDP: retrieve main_grad via accessor to avoid premature materialization
                    ctx.main_grad_funcs = [weights[i].get_main_grad for i in range(num_gemms)]
                else:
                    ctx.main_grad_funcs = [
                        lambda j=i: weights[j].main_grad for i in range(num_gemms)
                    ]
            else:
                ctx.main_grad_funcs = [lambda: None for i in range(num_gemms)]

            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.disable_backward_quant = disable_backward_quant

            # save_for_backward only accepts plain torch.Tensor; TE custom objects (NVFP4TensorStorage etc.)
            # must be stored directly on ctx.
            if disable_backward_quant:
                # BF16 backward path: save raw input slices; dgrad = dy @ w.T, wgrad = dy.T @ x
                ctx.x_bf16_list = list(torch.split(x, m_splits, dim=0))
                ctx.qx_t = ctx.sx_t = ctx.tx_t = None
                ctx.qw_t = ctx.sw_t = ctx.tw_t = None
            else:
                ctx.qx_t  = qx_t
                ctx.sx_t  = sx_t
                ctx.tx_t  = tx_t
                ctx.qw_t  = qw_t
                ctx.sw_t  = sw_t
                ctx.tw_t  = tw_t
                ctx.x_bf16_list = None
            ctx.save_for_backward(*weights)

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

        if not ctx.disable_backward_quant:
            assert qw_t is not None and sw_t is not None

        out_dtype_dgrad = (
            dy.dtype
            if ctx.qlinear_params.mm_dgrad.out_dtype is None
            else ctx.qlinear_params.mm_dgrad.out_dtype
        )
        dy = dy.contiguous()
        dy = dy.view(-1, dy.shape[-1])

        b_grad = torch.sum(dy, dim=0) if ctx.use_bias else None

        dy_splits = None  # reused by both dgrad and wgrad BF16 paths below

        if ctx.disable_backward_quant:
            # BF16 path: dgrad = dy @ w.T (layout NN)
            dy_splits = list(torch.split(dy, ctx.m_splits, dim=0))
            dgrad = ctx.quantize_op.grouped_gemm(
                A=dy_splits,
                B=weights,
                sA=None, sB=None,
                global_amax_A=None, global_amax_B=None,
                out=None,
                out_dtype=out_dtype_dgrad,
                layout="NN",
                m_splits=ctx.m_splits,
                grad=True,
            )
        else:
            # Quantized path: quantize dy and compute dgrad via quantized grouped GEMM
            qresult_dy_list = ctx.quantize_op.group_quantize(
                dy,
                ctx.qlinear_params.g_params,
                m_splits=ctx.m_splits,
                return_transpose=True,
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

        if ctx.disable_backward_quant:
            # BF16 path: wgrad = dy.T @ x (layout NT); dy_splits already computed in dgrad path
            wgrad = ctx.quantize_op.grouped_gemm(
                A=dy_splits,
                B=ctx.x_bf16_list,
                sA=None, sB=None,
                global_amax_A=None, global_amax_B=None,
                out=wgrad_list,
                out_dtype=ctx.activation_dtype,
                layout="NT",
                m_splits=ctx.m_splits,
                grad=True,
                accumulate=accumulate_wgrad_into_param_main_grad,
            )
        else:
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

        def handle_custom_ddp_from_mcore(weight, wgrad):
            if ctx.weights_requires_grad:
                # Megatron-Core custom DDP accumulates gradients into main_grad instead of .grad.
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
            None,
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
        self.layer_name = ub_name
        self.layer_number = layer_number

        log_rank_0(f"[Hybrid] Grouped Linear init:"
                   f" num_gemms: {num_gemms}"
                   f", in_features: {in_features}, out_features: {out_features}"
                   f", fuse wgrad accum: {fuse_wgrad_accumulation}"
                   f", use_bias: {bias}")

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
            )

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

        # 当 DISABLE_BACKWARD_QUANT=1 时，backward 不量化 dy/x，直接用 BF16 GEMM
        self.disable_backward_quant = should_keep_backward_in_bf16()

        # Megatron's _split_extra_state (sharded_state_dict path) accesses these
        # attributes that are normally provided by TE's TransformerEngineBaseModule.
        # Since miniTransformer does not use TE's FP8 pipeline, set them to safe
        # defaults so the checkpoint save/load path does not raise AttributeError.
        self.fp8 = False
        self.fp8_calibration = False
        self.fp8_meta: dict = {"fp8_checkpoint": False}

        # Grouped GEMM backend doesn't support bias fusion; apply bias after GEMM.
        self.gemm_bias_unfused_add = (self.use_bias and not return_bias)

    def init_fp8_metadata(self, num_gemms: int = 1) -> None:
        """Stub for Megatron compatibility (merge_extra_states pre-hook calls this on load).
        miniTransformer uses its own quantization pipeline, not TE's FP8 metadata.
        """
        self.fp8_meta["num_gemms"] = num_gemms

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)

        if not defer_init:
            for i in range(self.num_gemms):
                set_tensor_model_parallel_attributes(
                    tensor=getattr(self, f"weight{i}"),
                    is_parallel=True,
                    dim=1 if self.parallel_mode == "row" else 0,
                    stride=1,
                )

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

        self.set_activation_dtype(inp)
        inp = inp.contiguous()

        weight_tensors = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
        bias_tensors = [getattr(self, f"bias{i}") for i in range(self.num_gemms)]

        if torch.is_grad_enabled():
            linear_fn = _QuantizedGroupedLinear.apply
            args = []
        else:
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
            self.disable_backward_quant,
            *weight_tensors,
        )

        _prof_dir = os.environ.get("NVFP4_PROFILE_DIR", "")
        _prof_calls = int(os.environ.get("NVFP4_PROFILE_CALLS", "0"))
        _prof_grad = os.environ.get("NVFP4_PROFILE_GRAD", "0") == "1"
        _prof_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        if not hasattr(GroupedLinear, "_grad_prof_count"):
            GroupedLinear._grad_prof_count = 0
        _do_grad_profile = (
            _prof_grad
            and bool(_prof_dir)
            and _prof_rank == 0
            and torch.is_grad_enabled()
            and GroupedLinear._grad_prof_count < _prof_calls
        )

        if _do_grad_profile:
            _gprof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            _gprof.__enter__()

        out = linear_fn(*args)

        if _do_grad_profile:
            # Register backward hook to stop profiler after backward completes.
            def _stop_prof(grad):
                _gprof.__exit__(None, None, None)
                print(_gprof.key_averages().table(sort_by="cuda_time_total", row_limit=30), flush=True)
                os.makedirs(_prof_dir, exist_ok=True)
                _trace = os.path.join(_prof_dir, f"gl_grad_trace_{GroupedLinear._grad_prof_count:03d}.json")
                _gprof.export_chrome_trace(_trace)
                print(f"[PROFILE] grad trace saved → {_trace}", flush=True)
                GroupedLinear._grad_prof_count += 1
                return grad
            out.register_hook(_stop_prof)

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
