import os
import torch
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

from . import config
from ..distributed.distributed import get_distributed_world_size
from ..utils.utils import get_default_init_method #, FP8GlobalStateManager


@dataclass
class _ParameterInitMeta:
    """
    Stores essential metadata needed to support deferred parameter initialization.
    """

    init_fn: Optional[Callable] = get_default_init_method()
    get_rng_state_tracker: Optional[Callable] = None

    def __post_init__(self):
        """Safeguard reference to the parameter's parent module and initialization function."""
        if self.init_fn is None:
            self.init_fn = get_default_init_method()


class LayerLinearBaseModule(torch.nn.Module, ABC):
    """Layer Linear Base module."""

    def __init__(self, tp_group=None, tp_size=1, sequence_parallel=False):
        super().__init__()
        assert torch.cuda.is_available(), "CUDA must be available"

        # Set tp_group and tp_size
        self.tp_group_initialized = False
        if tp_group is None:
            if tp_size == 1:
                self.tp_size = 1
                self.set_tensor_parallel_group(tp_group)  # Set tp_group to None
            else:
                # tp_group is not initialized and user must call `set_tensor_parallel_group(tp_group)`
                # method on the initialized module before the forward pass
                self.tp_size = tp_size
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.set_nccl_overlap_warning_if_tp()

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel
        self.param_init_meta = {}
        # JQ: not supported for now
        self.primary_weights_in_fp8 = None
        # self.primary_weights_in_fp8 = FP8GlobalStateManager.is_fp8_params_enabled()
        # assert (
        #     not self.primary_weights_in_fp8
        # ), "Primary weights in FP8 are not supported yet."
        self.fsdp_group = None

    def get_extra_state(self) -> None:
        """Keep compatibility with TE state dict."""
        return None

    def set_extra_state(self, state: Any) -> None:
        """Extra state is ignored"""

    def set_activation_dtype(self, inp: torch.Tensor) -> None:
        """Set activation data type for AMP."""
        # Native AMP (`torch.autocast`) gets highest priority
        if torch.is_autocast_enabled():
            self.activation_dtype = torch.get_autocast_gpu_dtype()
            return

        # All checks after this have already been performed once, thus skip
        if hasattr(self, "activation_dtype") and self.activation_dtype == inp.dtype:
            return

        dtype = inp.dtype
        for name, param in self.named_parameters():
            if param is not None:
                assert dtype == param.dtype, (
                    "Data types for parameters must match when outside of autocasted region. "
                    f" Found input dtype: {dtype} and {name!r} dtype: {param.dtype}"
                )
        self.activation_dtype = dtype

    def set_tensor_parallel_group(
        self, tp_group: Union[torch.distributed.ProcessGroup, None]
    ) -> None:
        """
        Set the tensor parallel group for the given
        module before executing the forward pass.

        Parameters
        ----------
        tp_group : ProcessGroup, default = `None`
                  tensor parallel process group.
        """
        self.tp_group = tp_group
        self.tp_group_initialized = True

    def set_nccl_overlap_warning_if_tp(self) -> None:
        """When using TP, the NCCL communication needs to be scheduled
        before the GEMM for there to be a guaranteed overlap. From the
        host side, the comm calls are always launched first, but
        to ensure that the GEMM isn't scheduled first, the environment
        variable `CUDA_DEVICE_MAX_CONNECTIONS` needs to be set to 1 to
        force a single channel.
        """
        if self.tp_size == 1:
            return
        num_cuda_work_queues = int(os.getenv("CUDA_DEVICE_MAX_CONNECTIONS", "0"))
        if num_cuda_work_queues != 1:
            warnings.warn(
                "To guarantee overlapping TP and SP collectives with the backward"
                "GEMMs, set environment variable CUDA_DEVICE_MAX_CONNECTIONS = 1"
            )

    def register_parameter(self, name, param, **kwargs):
        """
        Thin wrapper around PyTorch parameter registration to stash additional parameter
        metedata used in deferred initialization.
        """
        super().register_parameter(name, param)
        self.param_init_meta[name] = _ParameterInitMeta(**kwargs)

    def reset_parameters(self, defer_init: Optional[bool] = False) -> None:
        """
        Reset all module parameters to initial values. Unless deferred initialization
        is specified, all parameters on a 'meta' device are also materialized on a real cuda
        device before the values are reset to initial.
        """
        if defer_init:
            return

        for name, param in self.named_parameters(recurse=False):
            # Ensure parameter is on a real device
            if param.device == torch.device("meta"):
                param = torch.nn.Parameter(torch.empty_like(param, device="cuda"))

            # Initialize the parameter values on device
            init_fn = self.param_init_meta[name].init_fn
            get_rng_state_tracker = self.param_init_meta[name].get_rng_state_tracker
            if get_rng_state_tracker is None:
                init_fn(param)
            else:
                if hasattr(self, "rng_tracker_name") and self.rng_tracker_name:
                    with get_rng_state_tracker().fork(self.rng_tracker_name):
                        init_fn(param)
                else:
                    with get_rng_state_tracker().fork():
                        init_fn(param)

            # If primary weights are in fp8, wrap the parameter as Float8Tensor
            if self.primary_weights_in_fp8:
                raise NotImplementedError(
                    "Primary weights in FP8 are not supported yet."
                )

            # Redo parameter wrap in case we broke it above
            # NOTE: Currently this can only be broken when primary weights are in Fp8 but
            #       re-applying the nn.Parameter() wrap is a no-op when the input is already
            #       a parameter so we always re-apply it just for extra safety.
            setattr(self, name, torch.nn.Parameter(param))

    def set_qlinear_params(
        self,
        *,
        qlinear_params: Optional[config.QLinearParams],
        layer_number: Optional[int],
        layer_name: Optional[str],
    ):
        """Set local qlinear_params based on passed in qlinear_params, optionally disable FP8 quantization"""

        if qlinear_params is None:
            qlinear_params = config.get_qlinear_params_from_env_qat_params()
        if layer_number in config.get_keep_in_bf16_layer_numbers():
            return config.QLinearParams(quantize=False)
        else:
            if layer_name is not None:
                qlinear_params = qlinear_params.customize_for_layer_name(layer_name)
            elif len(qlinear_params.overrides_for_layer_name) != 0:
                raise ValueError(
                    "Failing fast where selected recipe has layer_name "
                    "overrides, but no layer_name provided."
                )
            return qlinear_params

    @abstractmethod
    def forward(self):
        """Needs override."""
