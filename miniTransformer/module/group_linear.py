import logging
import torch
from torch.profiler import record_function
from typing import Callable, List, Optional, Tuple, Union

from . import config
from .base import LayerLinearBaseModule, get_dummy_wgrad
from ..quantiztion.int4_quantize_op import Int4MarlinForwardQuantizeOp
from ..distributed.distributed import allreduce, set_tensor_model_parallel_attributes
from ..quantiztion import quantization
from ..quantiztion.quantization import GroupQuantizeResult, GroupQuantizeResultCache
from ..utils.utils import is_rank_0, cast_if_needed, divide, init_method_constant
from ..utils.env import (
    should_keep_backward_in_bf16,
    get_linear_profiler_config,
    get_linear_tensor_stats_config,
)

logger = logging.getLogger(__name__)


def _log_rank_0(msg: str) -> None:
    if is_rank_0():
        logger.info(msg)


class _QuantizedGroupedLinear(torch.autograd.Function):
    """Grouped linear autograd function with quantization support.

    All quantize / GEMM work is delegated to ``qlinear_params.quantize_op``.
    The op must implement ``group_quantize`` and ``grouped_gemm``.

    GEMM layout (cuBLAS notation, one expert group at a time):

    fwd   (TN):  Y = w @ x            # [n, k] x [m, k] -> [m, n]
    dgrad (NN): dX = w @ dY             # [n, k] x [m, n] -> [m, k]
    wgrad (NT): dW = x @ dY             # [m, k] x [m, n] -> [n, k]
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        m_splits: List[int],
        weight_qresult_cache: GroupQuantizeResultCache,
        tp_group: Optional[torch.distributed.ProcessGroup],
        tp_size: int,
        parallel_mode: Optional[str],
        qlinear_params: config.QLinearParams,
        activation_dtype: torch.dtype,
        is_grad_enabled: bool,
        is_first_microbatch: bool | None,
        fuse_wgrad_accumulation: bool,
        disable_backward_quant: bool,
        input_row_amax: Optional[torch.Tensor],
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
        x = cast_if_needed(x.view(-1, x.shape[-1]), activation_dtype)
        # Columnwise quant output (x_t) is only needed when backward runs in quant mode.
        need_transpose = is_grad_enabled and not disable_backward_quant

        # Quantize activation: op receives [sum_M, k] and splits by m_splits.
        # x_params selects format (NVFP4 / MXFP8 / BF16, etc.) inside the op.
        x_quant_kwargs = {}
        if input_row_amax is not None:
            # Pass prefused per-row amax when upstream already computed it; supported
            # ops skip the row-amax kernel and reuse this tensor directly.
            x_quant_kwargs["row_amax"] = input_row_amax
        with record_function("group_quantize_x"):
            qresult_x_list = quantize_op.group_quantize(
                x,
                qlinear_params.x_params,
                m_splits=m_splits,
                return_transpose=need_transpose,
                **x_quant_kwargs,
            )
        # Unpack GroupQuantizeResult lists returned by the op (one entry per expert group).
        qx = qresult_x_list.data
        sx = qresult_x_list.scale
        tx = qresult_x_list.tensor_scale
        qx_t = qresult_x_list.data_t
        sx_t = qresult_x_list.scale_t
        tx_t = qresult_x_list.tensor_scale_t

        # Weight quant cache policy (mirrors dense Linear):
        # - first microbatch in a minibatch: quantize and store
        # - later microbatches: reuse cache when backward still needs w_t
        # - should_update_quantize_weights=True: always re-quantize, never cache
        refresh_quantized_weights = (
            is_first_microbatch is None or is_first_microbatch
        )
        if (
            not refresh_quantized_weights
            and is_grad_enabled
            and not disable_backward_quant
            and not quantize_op.should_update_quantize_weights
        ):
            cached_dt = weight_qresult_cache.data_t
            if cached_dt is None or (isinstance(cached_dt, list) and len(cached_dt) == 0):
                refresh_quantized_weights = True

        if quantize_op.should_update_quantize_weights or refresh_quantized_weights:
            # Stack per-expert weights to [num_gemms, n, k] for group_quantize.
            w = torch.stack(weights, dim=0)
            with record_function("group_quantize_w"):
                qresult_w_list = quantize_op.group_quantize(
                    w,
                    qlinear_params.w_params,
                    m_splits=m_splits,
                    return_transpose=need_transpose,
                )
            if quantize_op.should_update_quantize_weights:
                # Op owns weight quant lifetime (e.g. per-step refresh inside the op).
                qw = qresult_w_list.data
                sw = qresult_w_list.scale
                qw_t = qresult_w_list.data_t
                sw_t = qresult_w_list.scale_t
                tw = qresult_w_list.tensor_scale
                tw_t = qresult_w_list.tensor_scale_t
            else:
                # Save quantized weight cache between microbatches.
                qw = weight_qresult_cache.data = qresult_w_list.data
                sw = weight_qresult_cache.scale = qresult_w_list.scale
                qw_t = weight_qresult_cache.data_t = qresult_w_list.data_t
                sw_t = weight_qresult_cache.scale_t = qresult_w_list.scale_t
                tw = weight_qresult_cache.global_amax_row = qresult_w_list.tensor_scale
                tw_t = weight_qresult_cache.global_amax_col = qresult_w_list.tensor_scale_t
                weight_qresult_cache.extra_info = qresult_w_list.extra_info
        else:
            # Reuse cached quant tensors; rebuild a GroupQuantizeResult for extra_info handoff.
            qw = weight_qresult_cache.data
            sw = weight_qresult_cache.scale
            tw = weight_qresult_cache.global_amax_row
            qw_t = weight_qresult_cache.data_t
            sw_t = weight_qresult_cache.scale_t
            tw_t = weight_qresult_cache.global_amax_col
            qresult_w_list = GroupQuantizeResult(
                data=qw,
                scale=sw,
                data_t=qw_t,
                scale_t=sw_t,
                tensor_scale=tw,
                tensor_scale_t=tw_t,
                extra_info=weight_qresult_cache.extra_info,
            )

        assert qw is not None, "quantized weight cache must not be None"

        # Forward grouped GEMM (TN): op consumes per-group qx/qw lists plus scales.
        # extra_info_* carries layout metadata (offsets, swizzle, etc.) when the op needs it.
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
                extra_info_B=qresult_w_list.extra_info,
            )

        if is_grad_enabled:
            ctx.num_gemms = num_gemms
            ctx.m_splits = m_splits
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size
            ctx.parallel_mode = parallel_mode
            ctx.x_shape = x_shape
            ctx.qlinear_params = qlinear_params
            ctx.quantize_op = quantize_op
            ctx.is_first_microbatch = is_first_microbatch
            ctx.activation_dtype = activation_dtype
            ctx.device = x.device
            ctx.weights_requires_grad = weights[0].requires_grad
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.disable_backward_quant = disable_backward_quant

            # Defer main_grad lookup to backward so FSDP does not materialize it too early.
            if fuse_wgrad_accumulation and ctx.weights_requires_grad:
                if hasattr(weights[0], "__fsdp_param__"):
                    ctx.main_grad_funcs = [weights[i].get_main_grad for i in range(num_gemms)]
                else:
                    ctx.main_grad_funcs = [
                        lambda j=i: weights[j].main_grad for i in range(num_gemms)
                    ]
            else:
                ctx.main_grad_funcs = [lambda: None for _ in range(num_gemms)]

            # save_for_backward only accepts plain torch.Tensor; quant wrappers and
            # extra_info dicts are stored on ctx directly.
            if disable_backward_quant:
                # BF16 backward: keep raw x slices; dgrad/wgrad use high-precision grouped_gemm.
                ctx.x_bf16_list = list(torch.split(x, m_splits, dim=0))
                ctx.qx_t = ctx.sx_t = ctx.tx_t = None
                ctx.qw_t = ctx.sw_t = ctx.tw_t = None
            else:
                ctx.x_bf16_list = None
                ctx.qx_t = qx_t
                ctx.sx_t = sx_t
                ctx.tx_t = tx_t
                ctx.qw_t = qw_t
                ctx.sw_t = sw_t
                ctx.tw_t = tw_t
                ctx.w_extra_info = qresult_w_list.extra_info
            ctx.save_for_backward(*weights)

        # Row-parallel output AllReduce (GroupedLinear does not AllGather input).
        if parallel_mode == "row" and tp_size > 1:
            with record_function("allreduce_output"):
                y, _ = allreduce(y, tp_group)

        return y.view(-1, *x_shape[1:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, *dy_tup: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        assert len(dy_tup) == 1
        dy = dy_tup[0].contiguous().view(-1, dy_tup[0].shape[-1])

        main_grads = [fn() for fn in ctx.main_grad_funcs]
        weights = list(ctx.saved_tensors)

        # Quantized backward needs columnwise w_t / x_t saved on ctx during forward.
        if not ctx.disable_backward_quant:
            assert ctx.qw_t is not None and ctx.sw_t is not None

        out_dtype_dgrad = (
            dy.dtype
            if ctx.qlinear_params.mm_dgrad.out_dtype is None
            else ctx.qlinear_params.mm_dgrad.out_dtype
        )

        dy_splits = None
        if ctx.disable_backward_quant:
            # STE / QAT path: dgrad = dy @ w.T with BF16 weights, no dy quant.
            dy_splits = list(torch.split(dy, ctx.m_splits, dim=0))
            with record_function("grouped_dgrad"):
                dgrad = ctx.quantize_op.grouped_gemm(
                    A=dy_splits,
                    B=weights,
                    sA=None,
                    sB=None,
                    global_amax_A=None,
                    global_amax_B=None,
                    out=None,
                    out_dtype=out_dtype_dgrad,
                    layout="NN",
                    m_splits=ctx.m_splits,
                    grad=True,
                )
        else:
            # Quantized backward: quantize dy with g_params, then dgrad via grouped_gemm (NN).
            with record_function("group_quantize_dy"):
                qresult_dy_list = ctx.quantize_op.group_quantize(
                    dy,
                    ctx.qlinear_params.g_params,
                    m_splits=ctx.m_splits,
                    return_transpose=True,
                )
            qdy = qresult_dy_list.data
            sdy = qresult_dy_list.scale
            tdy = qresult_dy_list.tensor_scale
            qdy_t = qresult_dy_list.data_t
            sdy_t = qresult_dy_list.scale_t
            tdy_t = qresult_dy_list.tensor_scale_t

            # dgrad (NN): qdy @ qw_t using saved columnwise weight quant from forward.
            with record_function("grouped_dgrad"):
                dgrad = ctx.quantize_op.grouped_gemm(
                    A=qdy,
                    B=ctx.qw_t,
                    sA=sdy,
                    sB=ctx.sw_t,
                    global_amax_A=tdy,
                    global_amax_B=ctx.tw_t,
                    out=None,
                    out_dtype=out_dtype_dgrad,
                    layout="NN",
                    m_splits=ctx.m_splits,
                    grad=True,
                    qparams_x=ctx.qlinear_params.g_params,
                    qparams_w=ctx.qlinear_params.w_params,
                    extra_info_A=qresult_dy_list.extra_info,
                    extra_info_B=ctx.w_extra_info,
                )

        # Overlap column-parallel dgrad AllReduce with wgrad GEMM below.
        handle = None
        if ctx.parallel_mode == "column" and ctx.tp_size > 1:
            with record_function("allreduce_dgrad"):
                dgrad, handle = allreduce(dgrad, ctx.tp_group, True)

        if ctx.is_first_microbatch is not None:
            accumulate_wgrad = ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
        else:
            accumulate_wgrad = ctx.fuse_wgrad_accumulation

        # grouped_gemm NT writes one [n, k] wgrad per expert; use main_grad buffers when fusing.
        if ctx.fuse_wgrad_accumulation:
            wgrad_list = main_grads
        else:
            wgrad_list = [
                torch.empty(w.size(), dtype=ctx.activation_dtype, device=ctx.device)
                for w in weights
            ]

        if ctx.disable_backward_quant:
            # wgrad (NT): dy.T @ x with BF16 activations saved in forward.
            with record_function("grouped_wgrad"):
                ctx.quantize_op.grouped_gemm(
                    A=dy_splits,
                    B=ctx.x_bf16_list,
                    sA=None,
                    sB=None,
                    global_amax_A=None,
                    global_amax_B=None,
                    out=wgrad_list,
                    out_dtype=ctx.activation_dtype,
                    layout="NT",
                    m_splits=ctx.m_splits,
                    grad=True,
                    accumulate=accumulate_wgrad,
                )
        else:
            # wgrad (NT): qdy_t @ qx_t; accumulate into main_grad when fuse_wgrad is on.
            with record_function("grouped_wgrad"):
                ctx.quantize_op.grouped_gemm(
                    A=qdy_t,
                    B=ctx.qx_t,
                    sA=sdy_t,
                    sB=ctx.sx_t,
                    global_amax_A=tdy_t,
                    global_amax_B=ctx.tx_t,
                    out=wgrad_list,
                    out_dtype=ctx.activation_dtype,
                    layout="NT",
                    m_splits=ctx.m_splits,
                    grad=True,
                    accumulate=accumulate_wgrad,
                    qparams_x=ctx.qlinear_params.g_params,
                    qparams_w=ctx.qlinear_params.x_params,
                )

        def _finalize_wgrad(weight, wgrad):
            # When wgrad is already in main_grad, return dummy tensor or None for autograd.
            if not ctx.weights_requires_grad:
                return None
            if ctx.fuse_wgrad_accumulation and hasattr(weight, "grad_added_to_main_grad"):
                weight.grad_added_to_main_grad = True
                return get_dummy_wgrad(
                    list(weight.main_grad.shape),
                    weight.dtype,
                    zero=getattr(weight, "zero_out_wgrad", False),
                )
            if ctx.fuse_wgrad_accumulation:
                return None
            return wgrad

        wgrad_list = [
            _finalize_wgrad(weight, wgrad)
            for weight, wgrad in zip(weights, wgrad_list)
        ]

        if handle is not None:
            handle.wait()

        return (
            dgrad.view(ctx.x_shape),
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
            None,
            *wgrad_list,
        )


class GroupedLinear(LayerLinearBaseModule):

    _profiler_instance_count: int = 0
    _stats_instance_count: int = 0

    @classmethod
    def _try_attach_profiler(cls, module: "GroupedLinear") -> None:
        cfg = get_linear_profiler_config()
        if cfg is None:
            return
        if cfg.pop("target_class") != "grouped":
            return
        target = cfg.pop("target_instance")
        idx = cls._profiler_instance_count
        cls._profiler_instance_count += 1
        if idx != target:
            return
        from miniTransformer.profiler import LinearProfiler

        label = module.layer_name or f"grouped_linear_{idx}"
        LinearProfiler(label=label, **cfg).attach(module)
        print(
            f"[LinearProfiler] attached to GroupedLinear instance {idx} (label={label})",
            flush=True,
        )

    @classmethod
    def _try_attach_tensor_stats(cls, module: "GroupedLinear") -> None:
        cfg = get_linear_tensor_stats_config()
        if cfg is None:
            return
        if cfg.pop("target_class") != "grouped":
            return
        target = cfg.pop("target_instance")
        idx = cls._stats_instance_count
        cls._stats_instance_count += 1
        if idx != target:
            return
        from miniTransformer.profiler import LinearStatsHook

        label = module.layer_name or f"grouped_linear_{idx}"
        LinearStatsHook(label=label, **cfg).attach(module)
        print(
            f"[TensorStats] attached to GroupedLinear instance {idx} (label={label})",
            flush=True,
        )

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
        assert (
            not ub_overlap_rs and not ub_overlap_ag
        ), "GroupedLinear does not support userbuffer overlap."
        self.get_rng_state_tracker = get_rng_state_tracker
        self.rng_tracker_name = rng_tracker_name
        self.layer_name = ub_name
        self.layer_number = layer_number

        _log_rank_0(
            f"GroupedLinear init: num_gemms={num_gemms}, "
            f"in_features={in_features}, out_features={out_features}, "
            f"fuse_wgrad_accumulation={fuse_wgrad_accumulation}, use_bias={bias}"
        )

        # Resolve quant recipe (env QAT_PARAMS or caller override) and pick quantize_op.
        self.qlinear_params = self.set_qlinear_params(
            qlinear_params=qlinear_params,
            layer_number=self.layer_number,
            layer_name=self.layer_name,
        )
        _log_rank_0(
            f"GroupedLinear layer={layer_number} name={self.layer_name} "
            f"qlinear_params={self.qlinear_params}"
        )

        self.parallel_mode = parallel_mode
        assert self.parallel_mode in ("row", "column", None), (
            f"parallel_mode {parallel_mode} not supported"
        )

        if self.parallel_mode == "column":
            self.out_features = divide(self.out_features, self.tp_size)
        elif self.parallel_mode == "row":
            self.in_features = divide(self.in_features, self.tp_size)

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel
        assert not self.sequence_parallel, (
            "GroupedLinear does not support sequence parallelism"
        )

        # Each expert weight is quantized independently; out_features must be 128-aligned.
        assert self.out_features % 128 == 0, (
            f"out_features {self.out_features} must be a multiple of 128"
        )

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
                setattr(
                    self,
                    f"bias{i}",
                    torch.Tensor().to(dtype=params_dtype, device=device),
                )

        self.reset_parameters(defer_init=(device == "meta"))

        # Per-layer cache shared across microbatches in the same minibatch.
        self.weight_qresult_cache = GroupQuantizeResultCache(
            data=None, scale=None, data_t=None, scale_t=None
        )

        # DISABLE_BACKWARD_QUANT=1: backward grouped_gemm stays in BF16 (STE / QAT).
        self.disable_backward_quant = should_keep_backward_in_bf16()
        if (
            isinstance(self.qlinear_params.quantize_op, Int4MarlinForwardQuantizeOp)
            and not self.disable_backward_quant
        ):
            raise ValueError(
                "INT4 Marlin forward on GroupedLinear requires DISABLE_BACKWARD_QUANT=1 "
                "so backward can update weights in BF16."
            )

        # Checkpoint hooks expect these attributes on the module.
        self.fp8 = False
        self.fp8_calibration = False
        self.fp8_meta: dict = {"fp8_checkpoint": False}

        # grouped_gemm in quantize_op does not fuse bias; add per-expert bias after GEMM.
        self.gemm_bias_unfused_add = self.use_bias and not return_bias

        GroupedLinear._try_attach_profiler(self)
        GroupedLinear._try_attach_tensor_stats(self)

    def init_fp8_metadata(self, num_gemms: int = 1) -> None:
        """No-op metadata hook used by external checkpoint loaders."""
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
                        set_tensor_model_parallel_attributes(
                            getattr(self, f"bias{i}"), True, 0, 1
                        )

    @torch._dynamo.disable(recursive=True)
    def forward(
        self,
        inp: torch.Tensor,
        m_splits: List[int],
        is_first_microbatch: Optional[bool] = None,
        input_row_amax: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Run grouped linear over experts split by ``m_splits``.

        Args:
            inp: Input tensor ``[..., in_features]``.
            m_splits: Token count per expert; length must equal ``num_gemms``.
            is_first_microbatch: When set, enables weight-quant cache and wgrad
                accumulation behavior across microbatches in the same minibatch.
            input_row_amax: Optional prefused per-row abs-max ``(sum(m_splits),)`` in
                fp32, row-aligned with ``inp``. Lets supported quantize ops skip
                redundant row amax computation.
        """
        assert len(m_splits) == self.num_gemms, (
            "m_splits length must match num_gemms"
        )

        self.set_activation_dtype(inp)
        inp = inp.contiguous()

        weight_tensors = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
        bias_tensors = [getattr(self, f"bias{i}") for i in range(self.num_gemms)]

        # Dispatch to autograd Function; inference calls forward directly with ctx=None.
        if torch.is_grad_enabled():
            linear_fn = _QuantizedGroupedLinear.apply
            args: list = []
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
            input_row_amax,
            *weight_tensors,
        )

        out = linear_fn(*args)

        # Bias is applied outside quantize_op.grouped_gemm (one add per expert slice).
        if self.gemm_bias_unfused_add:
            out_shape = out.shape
            with record_function("bias_add"):
                out = torch.cat(
                    [
                        o + cast_if_needed(b, self.activation_dtype)
                        for o, b in zip(
                            torch.split(out.view(-1, self.out_features), m_splits),
                            bias_tensors,
                        )
                    ]
                ).view(out_shape)

        if self.return_bias:
            return out, [cast_if_needed(b, self.activation_dtype) for b in bias_tensors]

        return out
