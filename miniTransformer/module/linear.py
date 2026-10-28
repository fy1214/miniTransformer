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
from miniTransformer.quantiztion import quantization
from miniTransformer.quantiztion.int4_quantize_op import Int4MarlinForwardQuantizeOp
from miniTransformer.utils.env import should_keep_backward_in_bf16, get_linear_profiler_config, get_linear_tensor_stats_config
from miniTransformer.utils.utils import (
    is_rank_0,
    init_method_constant,
    cast_if_needed,
    divide,
)


logger = logging.getLogger(__name__)


class _NonQuantizedLinear(torch.autograd.Function):
    """Dense linear in BF16/FP16/FP32 via ``torch.nn.functional.linear``.

    PyTorch weight layout is [n, k]. Equivalent cuBLAS-style layouts:

    fwd   (TN):  Y = w @ x      # [n, k] x [m, k] -> [m, n]
    dgrad (NN): dX = w @ dY     # [n, k] x [m, n] -> [m, k]
    wgrad (NT): dW = x @ dY     # [m, k] x [m, n] -> [n, k]
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
        disable_backward_quant: bool
    ) -> torch.Tensor:
        x_shape = x.shape
        x = cast_if_needed(x.view(-1, x.shape[-1]), activation_dtype)
        use_bias = bias is not None
        if use_bias:
            bias = cast_if_needed(bias, activation_dtype)

        # Column + sequence parallel: AllGather activation before GEMM.
        if parallel_mode == "column" and sequence_parallel:
            assert tp_group is not None
            x, _ = gather_along_first_dim(x, tp_group)

        # Dense path: no quantize_op; standard fused linear in activation_dtype.
        y = torch.nn.functional.linear(x, w, bias)

        if is_grad_enabled:
            ctx.use_bias = use_bias
            ctx.sequence_parallel = sequence_parallel
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size
            ctx.parallel_mode = parallel_mode
            ctx.x_shape = x_shape
            ctx.is_first_microbatch = is_first_microbatch
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.activation_dtype = activation_dtype

            ctx.save_for_backward(
                x,
                w,
            )

        if parallel_mode == "row" and sequence_parallel:
            assert tp_group is not None
            y, _ = reduce_scatter_along_first_dim(y, tp_group)
        elif parallel_mode == "row" and tp_size > 1:
            y, _ = allreduce(y, tp_group)

        return y.view(-1, *x_shape[1:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, *dy_tup: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        assert len(dy_tup) == 1
        dy = dy_tup[0]
        (x, w) = ctx.saved_tensors
        assert x is not None and w is not None

        dy = dy.contiguous()
        dy = dy.view(-1, dy.shape[-1])

        b_grad = torch.sum(dy, dim=0) if ctx.use_bias else None

        if ctx.parallel_mode == "row" and ctx.sequence_parallel:
            dy, _ = gather_along_first_dim(dy, ctx.tp_group)

        dgrad = torch.matmul(dy, w)

        # Launch column-parallel dgrad comm async; overlap with wgrad below.
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

        out_dtype_wgrad = w.main_grad.dtype if ctx.fuse_wgrad_accumulation else ctx.activation_dtype

        wgrad = torch.matmul(dy.t(), x)

        if accumulate_wgrad_into_param_main_grad:
            if ctx.fuse_wgrad_accumulation:
                w.main_grad.add_(wgrad)
                wgrad = w.main_grad

        wgrad = _QuantizedLinear.reset_wgrad_if_needed(ctx, w, wgrad)

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
    """Linear autograd function with quantization support.

    Quantize and GEMM are delegated to ``qlinear_params.quantize_op``
    (``quantize`` + ``qgemm``). High-precision activations use
    ``qlinear_params.mm_(fprop|dgrad|wgrad).out_dtype`` when set.

    Quantized GEMM layouts (cuBLAS notation):

    fwd   (TN):  Y = w @ x
    dgrad (NN): dX = w.t() @ dY       # via qgemm layout="NN"
    wgrad (NT): dW = x.t() @ dY.t()   # via qgemm layout="NT"
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
        disable_backward_quant: bool
    ) -> torch.Tensor:

        assert qlinear_params.quantize_op is not None
        quantize_op: quantization.QuantizeOpBase = qlinear_params.quantize_op

        x_shape = x.shape
        out_dtype = (
            x.dtype
            if qlinear_params.mm_fprop.out_dtype is None
            else qlinear_params.mm_fprop.out_dtype
        )
        x = cast_if_needed(x.view(-1, x.shape[-1]), activation_dtype)
        use_bias = bias is not None
        if use_bias:
            bias = cast_if_needed(bias, activation_dtype)

        # --- op: quantize activation (x_params) ---------------------------------
        # Input [m, k] bf16/fp16. return_transpose=True also builds columnwise qx_t
        # for backward NN/NT qgemm unless disable_backward_quant skips that path.
        gather_required = parallel_mode == "column" and sequence_parallel
        qresult_x = _QuantizedLinear.quantize_optionally_allgather(
            quantize_op,
            x,
            qlinear_params.x_params,
            transpose_required=True,
            allgather_required=gather_required,
            allgather_quantize=qlinear_params.allgather_quantize,
            tp_group=tp_group,
        )
        # qx/sx/tensor_scale are op-specific (NVFP4 tensor, FP8 wrapper, etc.).
        qx, sx, qx_global_amax_row = (
            qresult_x.data,
            qresult_x.scale,
            qresult_x.tensor_scale,
        )

        # --- op: quantize weight (w_params) -------------------------------------
        # Weight quant cache: first microbatch quantizes and stores; later ones reuse.
        refresh_quantized_weights = (
            is_first_microbatch is None or is_first_microbatch
        )
        if quantize_op.should_update_quantize_weights or refresh_quantized_weights:
            # return_transpose=True produces qw_t used by dgrad qgemm (layout NN).
            qresult_w = quantize_op.quantize(
                w, qlinear_params.w_params, return_transpose=True
            )
            qw, sw, qw_global_amax_row = (
                qresult_w.data,
                qresult_w.scale,
                qresult_w.tensor_scale,
            )
            if not quantize_op.should_update_quantize_weights:
                # Cache lives on the module; reused until next first microbatch.
                weight_qresult_cache.data = qresult_w.data
                weight_qresult_cache.scale = qresult_w.scale
                weight_qresult_cache.data_t = qresult_w.data_t
                weight_qresult_cache.scale_t = qresult_w.scale_t
                weight_qresult_cache.global_amax_row = qresult_w.tensor_scale
                weight_qresult_cache.global_amax_col = qresult_w.tensor_scale_t
        else:
            qresult_w = weight_qresult_cache.to_qresult()
            qw, sw, qw_global_amax_row = (
                qresult_w.data,
                qresult_w.scale,
                qresult_w.tensor_scale,
            )
            assert qw is not None, "quantized weight cache must not be None"

        # --- op: forward qgemm (TN, mm_fprop) -----------------------------------
        # y = qw @ qx (+ bias when fused). Kernel choice is inside quantize_op.
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

        if is_grad_enabled:
            ctx.use_bias = use_bias
            ctx.sequence_parallel = sequence_parallel
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size
            ctx.tp_rank = get_distributed_rank(tp_group) if tp_group is not None else 0
            ctx.x_gather_required = gather_required
            ctx.parallel_mode = parallel_mode
            ctx.x_shape = x_shape
            ctx.qlinear_params = qlinear_params
            ctx.quantize_op = quantize_op
            ctx.is_first_microbatch = is_first_microbatch
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.activation_dtype = activation_dtype
            ctx.disable_backward_quant = disable_backward_quant

            # Columnwise quant from forward; consumed by backward qgemm unless STE path.
            qx_t, sx_t, qx_global_amax_col, qw_t, sw_t, qw_global_amax_col = (
                qresult_x.data_t,
                qresult_x.scale_t,
                qresult_x.tensor_scale_t,
                qresult_w.data_t,
                qresult_w.scale_t,
                qresult_w.tensor_scale_t,
            )

            # memory_saving: store only this rank's qx_t shard; AllGather in backward.
            if qlinear_params.memory_saving and gather_required:
                qx_t = torch.chunk(qx_t, tp_size, dim=0)[ctx.tp_rank].detach().clone()

            if disable_backward_quant:
                # STE path: save rowwise qx/qw for dequant in backward; keep high-precision w.
                ctx.save_for_backward(
                    qx,
                    sx,
                    qx_global_amax_row,
                    qw,
                    sw,
                    qw_global_amax_row,
                    w,
                )
            else:
                # Quant backward: save columnwise qx_t/qw_t for NN/NT qgemm.
                ctx.save_for_backward(
                    qx_t,
                    sx_t,
                    qx_global_amax_col,
                    qw_t,
                    sw_t,
                    qw_global_amax_col,
                    w,
                )

        # Row parallel: ReduceScatter (sequence parallel) or AllReduce output.
        if parallel_mode == "row" and sequence_parallel:
            assert tp_group is not None
            y, _ = reduce_scatter_along_first_dim(y, tp_group)
        elif parallel_mode == "row" and tp_size > 1:
            y, _ = allreduce(y, tp_group)

        # Restore leading dims stripped before [m, k] quant/gemm.
        return y.view(-1, *x_shape[1:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, *dy_tup: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        assert len(dy_tup) == 1
        dy = dy_tup[0]
        # Same seven slots for both backward modes; names map to rowwise or columnwise tensors.
        (qx_t, sx_t, qx_global_amax_col, qw_t, sw_t, qw_global_amax_col, w) = (
            ctx.saved_tensors
        )

        assert qw_t is not None and sw_t is not None
        assert qx_t is not None and sx_t is not None

        out_dtype_dgrad = (
            dy.dtype
            if ctx.qlinear_params.mm_dgrad.out_dtype is None
            else ctx.qlinear_params.mm_dgrad.out_dtype
        )
        dy = dy.contiguous().view(-1, dy.shape[-1])

        b_grad = torch.sum(dy, dim=0) if ctx.use_bias else None

        # --- op: quantize grad_output (g_params) --------------------------------
        # Builds qdy + qdy_t for dgrad/wgrad qgemm. AllGather dy first when row+SP.
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
            qresult_dy.tensor_scale,
            qresult_dy.tensor_scale_t,
        )

        qx_t_handle = None
        if ctx.qlinear_params.memory_saving and ctx.x_gather_required:
            # Restore full qx_t shard saved during forward before wgrad qgemm.
            assert ctx.tp_group is not None
            qx_t, qx_t_handle = gather_along_first_dim(
                qx_t, ctx.tp_group, async_op=True
            )

        if ctx.disable_backward_quant:
            # --- op: dequantize + BF16 dgrad ------------------------------------
            # Saved slots hold rowwise qx/qw; dequantize then high-precision matmul.
            dq_w_t = ctx.quantize_op.dequantize(qw_t, sw_t, qw_global_amax_col)
            dgrad = dy @ dq_w_t
        else:
            # --- op: dgrad qgemm (NN, mm_dgrad) ---------------------------------
            # dgrad = qdy @ qw_t using g_params for dy and w_params for weight side.
            dgrad = ctx.quantize_op.qgemm(
                qdy,
                qw_t,
                ctx.qlinear_params.mm_dgrad,
                out_dtype_dgrad,
                sdy,
                sw_t,
                None,
                None,
                grad=True,
                accumulate=False,
                layout="NN",
                global_amax_qx=qdy_global_amax_row,
                global_amax_qw=qw_global_amax_col,
                qparams_x=ctx.qlinear_params.g_params,
                qparams_w=ctx.qlinear_params.w_params,
            )

        if qx_t_handle is not None:
            qx_t_handle.wait()

        # Overlap column-parallel dgrad comm with wgrad qgemm.
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

        out_dtype_wgrad = (
            w.main_grad.dtype if ctx.fuse_wgrad_accumulation else ctx.activation_dtype
        )
        if ctx.disable_backward_quant:
            # --- op: dequantize + BF16 wgrad ------------------------------------
            dq_x_t = ctx.quantize_op.dequantize(qx_t, sx_t, qx_global_amax_col)
            wgrad = dy.T @ dq_x_t
            if accumulate_wgrad_into_param_main_grad and ctx.fuse_wgrad_accumulation:
                w.main_grad.add_(wgrad)
                wgrad = w.main_grad
        else:
            # --- op: wgrad qgemm (NT, mm_wgrad) ---------------------------------
            # wgrad = qdy_t @ qx_t; out=main_grad when accumulating across microbatches.
            wgrad = ctx.quantize_op.qgemm(
                qdy_t,
                qx_t,
                ctx.qlinear_params.mm_wgrad,
                out_dtype_wgrad,
                sdy_t,
                sx_t,
                None,
                out=w.main_grad if ctx.fuse_wgrad_accumulation else None,
                grad=True,
                accumulate=accumulate_wgrad_into_param_main_grad,
                layout="NT",
                global_amax_qx=qdy_global_amax_col,
                global_amax_qw=qx_global_amax_col,
                qparams_x=ctx.qlinear_params.g_params,
                qparams_w=ctx.qlinear_params.x_params,
            )

        wgrad = _QuantizedLinear.reset_wgrad_if_needed(ctx, w, wgrad)
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
        """Call ``quantize_op.quantize`` with optional sequence-parallel AllGather.

        When AllGather is required:
          1) quantize local shard (with amax reduction), then AllGather quant tensors, or
          2) AllGather in high precision, then quantize the full tensor.
        """
        if allgather_required:
            if allgather_quantize:
                # Path 1: quantize local shard (reduce amax across TP), then AllGather qx.
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
            else:
                # Path 2: AllGather high-precision x, then quantize the full tensor.
                assert tp_group is not None
                x, _ = gather_along_first_dim(x, tp_group)
                qresult_x = quantize_op.quantize(
                    x,
                    qparams,
                    return_transpose=transpose_required,
                    reduce_amax=False,
                    tp_group=tp_group,
                )
        else:
            # No sequence-parallel gather: quantize the local [m, k] tensor directly.
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
        """Return wgrad for autograd after optional main_grad fusion."""
        if w.requires_grad:
            if ctx.fuse_wgrad_accumulation and hasattr(w, "grad_added_to_main_grad"):
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
            elif ctx.fuse_wgrad_accumulation:
                wgrad = None
        else:
            wgrad = None

        return wgrad


class Linear(LayerLinearBaseModule):
    """Linear layer ``y = x @ W.T + b`` with optional quantization and tensor parallelism."""

    _profiler_instance_count: int = 0
    _stats_instance_count: int = 0

    @classmethod
    def _try_attach_profiler(cls, module: "Linear") -> None:
        cfg = get_linear_profiler_config()
        if cfg is None:
            return
        if cfg.pop("target_class") != "linear":
            return
        target = cfg.pop("target_instance")
        idx = cls._profiler_instance_count
        cls._profiler_instance_count += 1
        if idx != target:
            return
        from miniTransformer.profiler import LinearProfiler
        label = module.layer_name or f"linear_{idx}"
        LinearProfiler(label=label, **cfg).attach(module)
        print(f"[LinearProfiler] attached to {cls.__name__} instance {idx} (label={label})", flush=True)

    @classmethod
    def _try_attach_tensor_stats(cls, module: "Linear") -> None:
        cfg = get_linear_tensor_stats_config()
        if cfg is None:
            return
        if cfg.pop("target_class") != "linear":
            return
        target = cfg.pop("target_instance")
        idx = cls._stats_instance_count
        cls._stats_instance_count += 1
        if idx != target:
            return
        from miniTransformer.profiler import LinearStatsHook
        label = module.layer_name or f"linear_{idx}"
        LinearStatsHook(label=label, **cfg).attach(module)
        print(f"[TensorStats] attached to {cls.__name__} instance {idx} (label={label})", flush=True)

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
        layer_number: Optional[int] = None,
        save_original_input: bool = False,
        qlinear_params: Optional[config.QLinearParams] = None,
    ) -> None:
        super().__init__(tp_group, tp_size, sequence_parallel)
        del save_original_input  # reserved for API compatibility

        params_dtype = (
            torch.get_default_dtype() if params_dtype is None else params_dtype
        )
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        if ub_overlap_rs or ub_overlap_ag:
            assert ub_name is not None, "Userbuffer name [string] is not set."
        self.get_rng_state_tracker = get_rng_state_tracker
        self.rng_tracker_name = rng_tracker_name
        self.layer_name = ub_name
        self.layer_number = layer_number

        self.disable_backward_quant = should_keep_backward_in_bf16()

        # Resolve quant recipe (env QAT_PARAMS or caller override) and pick quantize_op.
        self.qlinear_params = self.set_qlinear_params(
            qlinear_params=qlinear_params,
            layer_number=self.layer_number,
            layer_name=self.layer_name,
        )
        if isinstance(self.qlinear_params.quantize_op, Int4MarlinForwardQuantizeOp):
            logger.info(
                "INT4 Marlin forward is GroupedLinear-only; keeping dense Linear in BF16."
            )
            self.qlinear_params = config.QLinearParams(quantize=False)

        if is_rank_0():
            logger.info(
                f"Linear layer={layer_number} name={self.layer_name} "
                f"qlinear_params={self.qlinear_params}"
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

        with_fp8_params = False
        assert with_fp8_params is False, "FP8 only parameters are not supported yet"

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

        self.weight_names = []
        self.bias_names = []
        self.parameter_split_sizes = []
        if parameters_split is None:
            self.weight_names = ["weight"]
            self.bias_names = ["bias"]
            self.parameter_split_sizes = [out_features]
        elif not parameters_split:
            raise ValueError("Cannot split weight buffer into 0 parameters")
        elif isinstance(parameters_split, dict):
            for name, split_size in parameters_split.items():
                self.weight_names.append(f"{name.rstrip('_')}_weight")
                self.bias_names.append(f"{name.rstrip('_')}_bias")
                self.parameter_split_sizes.append(split_size)
        elif all(isinstance(name, str) for name in parameters_split):
            split_size = out_features // len(parameters_split)
            assert out_features % len(parameters_split) == 0, "Invalid split size"
            for name in parameters_split:
                self.weight_names.append(f"{name.rstrip('_')}_weight")
                self.bias_names.append(f"{name.rstrip('_')}_bias")
                self.parameter_split_sizes.append(split_size)
        else:
            raise TypeError("Invalid configuration for parameters split")

        if sum(self.parameter_split_sizes) != out_features:
            raise ValueError(
                f"Trying to split weight buffer ({out_features=}) "
                f"with split sizes {self.parameter_split_sizes}"
            )

        if self.parallel_mode == "column":
            for i, size in enumerate(self.parameter_split_sizes):
                if size % self.tp_size != 0:
                    raise RuntimeError(
                        f"Attempting to distribute a parameter with out_features={size} "
                        f"between {self.tp_size} tensor-parallel processes"
                    )
                self.parameter_split_sizes[i] = size // self.tp_size

        # Register weights contiguously so FSDP external manipulation is less likely to break contiguity.
        offset = 0
        for i, split_size in enumerate(self.parameter_split_sizes):
            split_start = offset
            offset += split_size
            split_end = offset

            is_subview = (split_start, split_end) != (0, self.out_features)
            if is_subview and with_fp8_params:
                raise RuntimeError(
                    "Splitting Float8Tensor into multiple params is not supported"
                )

            self.register_parameter(
                self.weight_names[i],
                torch.nn.Parameter(weight_tensor[split_start:split_end]),
                init_fn=init_method,
                get_rng_state_tracker=get_rng_state_tracker,
            )

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

        # Per-layer cache shared across microbatches in the same minibatch.
        self.weight_qresult_cache = quantization.QuantizeResultCache(
            data=None, scale=None, data_t=None, scale_t=None
        )

        # Row parallel + bias: add bias after TP collectives, not inside qgemm.
        self.gemm_bias_unfused_add = self.parallel_mode == "row" and self.apply_bias

        Linear._try_attach_profiler(self)
        Linear._try_attach_tensor_stats(self)

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)

        if not defer_init:
            for weight in self.weight_names:
                set_tensor_model_parallel_attributes(
                    tensor=getattr(self, weight),
                    is_parallel=True,
                    dim=1 if self.parallel_mode == "row" else 0,
                    stride=1,
                )

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
        """Apply linear transform; dispatches to quant or dense autograd path."""

        self.set_activation_dtype(inp)
        inp = inp.contiguous()
        assert self.tp_group_initialized, "TP group not initialized"
        input_dtypes = (torch.float, torch.float16, torch.bfloat16, torch.float32)
        assert inp.dtype in input_dtypes, f"Input dtype {inp.dtype} not supported"
        assert isinstance(
            self.weight_qresult_cache, quantization.QuantizeResultCache
        ), "weight_qresult_cache must be QuantizeResultCache"
        assert fp8_output is False, "quantized fp8 output is not supported"

        # noop_cat merges split QKV weights without copy when already contiguous.
        weight_tensor = noop_cat(
            [getattr(self, name) for name in self.weight_names]
        )
        if self.use_bias:
            bias_tensor = noop_cat(
                [getattr(self, name) for name in self.bias_names],
            )
        else:
            bias_tensor = getattr(self, self.bias_names[0])

        linear_cls = (
            _QuantizedLinear if self.qlinear_params.quantize else _NonQuantizedLinear
        )

        # _QuantizedLinear routes to qlinear_params.quantize_op (quantize + qgemm).
        # _NonQuantizedLinear uses torch.nn.functional.linear (BF16/FP16 path).
        if torch.is_grad_enabled():
            linear_fn = linear_cls.apply
            args: List[Any] = []
        else:
            linear_fn = linear_cls.forward
            args = [None]
        args += (
            inp,
            inp_meta,  # optional QuantizeInputMeta for future x-side op hooks
            weight_tensor,
            self.weight_qresult_cache,  # microbatch weight-quant cache for quant path
            bias_tensor if self.apply_bias and not self.gemm_bias_unfused_add else None,
            self.sequence_parallel,
            self.tp_group,
            self.tp_size,
            self.parallel_mode,
            self.qlinear_params,  # carries quantize_op + x/w/g QParams + mm_* settings
            self.activation_dtype,
            torch.is_grad_enabled(),
            is_first_microbatch,
            self.fuse_wgrad_accumulation,
            self.disable_backward_quant,  # DISABLE_BACKWARD_QUANT=1 -> STE backward
        )
        out = linear_fn(*args)

        # Row-parallel qgemm cannot fuse bias; add after TP collectives inside autograd.
        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(bias_tensor, self.activation_dtype)

        if self.return_bias:
            return out, cast_if_needed(bias_tensor, self.activation_dtype)

        return out
