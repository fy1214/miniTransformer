import torch
import torch.nn.functional as F

from typing import Callable, Dict, Optional, Tuple, Union, Any
from miniTransformer.module import config
from miniTransformer.module.linear import Linear
from miniTransformer.utils.env import should_switch_to_hybrid_linear

try:
    import transformer_engine as te  # type: ignore

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


def make_te_norm(
    normalization: str,
    in_features: int,
    eps: float,
    sequence_parallel: bool,
    params_dtype: Optional[torch.dtype],
    zero_centered_gamma: bool,
    device: Union[torch.device, str],
) -> Any:
    """Build a norm module from the external TE package when available."""
    assert HAVE_TE, "transformer_engine is required for this norm backend."

    if normalization == "LayerNorm":
        return te.pytorch.LayerNorm(
            in_features,
            eps=eps,
            sequence_parallel=sequence_parallel,
            params_dtype=params_dtype,
            zero_centered_gamma=zero_centered_gamma,
            device=device,
        )
    if normalization == "RMSNorm":
        return te.pytorch.RMSNorm(
            in_features,
            eps=eps,
            sequence_parallel=sequence_parallel,
            params_dtype=params_dtype,
            zero_centered_gamma=zero_centered_gamma,
            device=device,
        )
    raise ValueError(f"Normalization type {normalization} is not supported.")


def make_torch_norm(
    normalization: str,
    in_features: int,
    eps: float,
    sequence_parallel: bool,
    params_dtype: Optional[torch.dtype],
    zero_centered_gamma: bool,
    device: Union[torch.device, str],
) -> Union[torch.nn.LayerNorm, torch.nn.RMSNorm]:
    """Build a norm module with native PyTorch ops."""
    assert not zero_centered_gamma, (
        "zero_centered_gamma is not supported for the torch norm backend."
    )
    assert not sequence_parallel, (
        "sequence_parallel is not supported for the torch norm backend."
    )

    if normalization == "LayerNorm":
        return torch.nn.LayerNorm(
            in_features,
            eps=eps,
            dtype=params_dtype,
            device=device,
        )
    if normalization == "RMSNorm":
        return torch.nn.RMSNorm(
            in_features,
            eps=eps,
            dtype=params_dtype,
            device=device,
        )
    raise ValueError(f"Normalization type {normalization} is not supported.")


class LayerNormLinear(Linear):
    """LayerNorm (or RMSNorm) followed by :class:`Linear`.

    Inherits from ``Linear`` so checkpoint keys stay ``layer_norm_weight`` /
    ``weight`` at this module level rather than under a nested submodule.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        normalization: str = "LayerNorm",
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        parallel_mode: Optional[str] = None,
        return_layernorm_output: bool = False,
        return_layernorm_output_gathered: bool = False,
        parameters_split: Optional[Union[Tuple[str, ...], Dict[str, int]]] = None,
        zero_centered_gamma: bool = False,
        device: Union[torch.device, str] = "cuda",
        ub_bulk_wgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_overlap_ag: bool = False,
        ub_overlap_rs_dgrad: bool = False,
        ub_name: Optional[str] = None,
        qlinear_params: Optional[config.QLinearParams] = None,
        layer_number: Optional[int] = None,
        use_te_norm: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            sequence_parallel=sequence_parallel,
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            tp_group=tp_group,
            tp_size=tp_size,
            get_rng_state_tracker=get_rng_state_tracker,
            init_method=init_method,
            bias=bias,
            return_bias=return_bias,
            params_dtype=params_dtype,
            parallel_mode=parallel_mode,
            parameters_split=parameters_split,
            device=device,
            ub_overlap_ag=ub_overlap_ag,
            ub_name=ub_name,
            layer_number=layer_number,
            qlinear_params=qlinear_params,  # forwarded; selects quantize_op on Linear
        )

        if any([ub_bulk_wgrad, ub_bulk_dgrad, ub_overlap_ag, ub_overlap_rs_dgrad]):
            assert ub_name is not None, "Userbuffer name [string] is not set."
        assert not (
            ub_bulk_wgrad or ub_bulk_dgrad or ub_overlap_rs_dgrad
        ), "bulk_wgrad, bulk_dgrad and overlap_rs_dgrad are not supported."

        self.return_layernorm_output = return_layernorm_output
        self.return_layernorm_output_gathered = return_layernorm_output_gathered
        assert not self.return_layernorm_output_gathered, (
            "return_layernorm_output_gathered is not supported yet."
        )

        assert normalization in ("LayerNorm", "RMSNorm"), (
            f"Unsupported normalization type: {normalization}"
        )
        self.normalization = normalization

        if use_te_norm is None:
            use_te_norm = should_switch_to_hybrid_linear()

        if use_te_norm:
            _norm = make_te_norm(
                normalization,
                in_features,
                eps=eps,
                sequence_parallel=sequence_parallel,
                params_dtype=params_dtype,
                zero_centered_gamma=zero_centered_gamma,
                device=device,
            )
        else:
            _norm = make_torch_norm(
                normalization,
                in_features,
                eps=eps,
                sequence_parallel=sequence_parallel,
                params_dtype=params_dtype,
                zero_centered_gamma=zero_centered_gamma,
                device=device,
            )

        # Hoist norm weights to this module for checkpoint name compatibility.
        self.register_parameter(
            "layer_norm_weight", _norm._parameters.pop("weight", None)
        )
        if normalization == "LayerNorm":
            self.register_parameter(
                "layer_norm_bias", _norm._parameters.pop("bias", None)
            )
        else:
            self.layer_norm_bias = None

        self._layer_norm_eps = _norm.eps
        self._layer_norm_normalized_shape = getattr(
            _norm, "normalized_shape", (in_features,)
        )

    @torch._dynamo.disable(recursive=True)
    def forward(
        self,
        inp: torch.Tensor,
        is_first_microbatch: Optional[bool] = None,
        fp8_output: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Apply norm, then delegate to :class:`Linear` (quant op path unchanged)."""
        inp = inp.contiguous()

        # Norm runs in high precision; output becomes Linear input (then x_params quant).
        if self.normalization == "LayerNorm":
            ln_out = F.layer_norm(
                inp,
                self._layer_norm_normalized_shape,
                weight=self.layer_norm_weight,
                bias=self.layer_norm_bias,
                eps=self._layer_norm_eps,
            )
        else:
            ln_out = F.rms_norm(
                inp,
                self._layer_norm_normalized_shape,
                weight=self.layer_norm_weight,
                eps=self._layer_norm_eps,
            )

        # --- op path starts in Linear -------------------------------------------
        # super().forward -> _QuantizedLinear or _NonQuantizedLinear
        #   quant path: quantize_op.quantize(ln_out, x_params) -> qgemm(TN)
        #   dense path: torch.nn.functional.linear(ln_out, weight)
        # qlinear_params / quantize_op come from Linear.__init__ (QAT_PARAMS or override).
        input_meta = None  # reserved for passing norm-side metadata into x quant
        out = super().forward(ln_out, is_first_microbatch, fp8_output, input_meta)

        if self.return_bias:
            out, returned_bias = out
            if self.return_layernorm_output:
                return out, returned_bias, ln_out
            return out, returned_bias

        if self.return_layernorm_output:
            return out, ln_out
        return out
