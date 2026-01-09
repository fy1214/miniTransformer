import torch

from typing import Callable, Dict, Optional, Tuple, Union, Any
from miniTransformer.module import config
from miniTransformer.module.linear import Linear

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
    assert HAVE_TE, "Transformer Engine is not available."

    if normalization == "LayerNorm":
        return te.pytorch.LayerNorm(
            in_features,
            eps=eps,
            sequence_parallel=sequence_parallel,
            params_dtype=params_dtype,
            zero_centered_gamma=zero_centered_gamma,
            device=device,
        )
    elif normalization == "RMSNorm":
        return te.pytorch.RMSNorm(
            in_features,
            eps=eps,
            sequence_parallel=sequence_parallel,
            params_dtype=params_dtype,
            zero_centered_gamma=zero_centered_gamma,
            device=device,
        )
    else:
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

    assert (
        not zero_centered_gamma
    ), "Zero centered gamma is not supported for TorchNorm."
    assert not sequence_parallel, "Sequence parallelism is not supported for TorchNorm."

    if normalization == "LayerNorm":
        return torch.nn.LayerNorm(
            in_features,
            eps=eps,
            dtype=params_dtype,
            device=device,
        )
    elif normalization == "RMSNorm":
        return torch.nn.RMSNorm(
            in_features,
            eps=eps,
            dtype=params_dtype,
            device=device,
        )
    else:
        raise ValueError(f"Normalization type {normalization} is not supported.")


class LayerNormLinear(Linear):
    r"""
    Applies layer normalization followed by linear transformation to the incoming data.

    We inherit from Linear to keep compatability with TE/Megatron checkpoints.
    If we inherit from LinearBaseModule, we will see Kitchen's weight parameter stored in following format: linear_fc1.linear.weight.
    MCore MLP's ShardedStateDict would expect linear_fc1.weight directly: https://github.com/NVIDIA/Megatron-LM/blob/05ac33c6aa5b7e7712907e10401fa4726b694655/megatron/core/transformer/mlp.py#L139

    Parameters
    ----------
    in_features : int
                 size of each input sample.
    out_features : int
                  size of each output sample.
    eps : float, default = 1e-5
         a value added to the denominator of layer normalization for numerical stability.
    bias : bool, default = `True`
          if set to `False`, the layer will not learn an additive bias.
    normalization : { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                   type of normalization applied.
    init_method : Callable, default = `None`
                 used for initializing weights in the following way: `init_method(weight)`.
                 When set to `None`, defaults to `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    return_layernorm_output : bool, default = `False`
                             if set to `True`, output of layernorm is returned from the forward
                             together with the output of the linear transformation.
                             Example use case: residual connection for transformer module is
                             taken post layernorm.
    return_layernorm_output_gathered : bool, default = `False`
                             if set to `True`, output of layernorm is returned after the all
                             gather operation. Ignored if return_layernorm_output is False.
                             Example use case: with sequence parallel, input to residual connection
                             for transformer module (e.g. LoRA) will need to be gathered.
                             Returning layernorm output gathered will prevent a redundant gather.
    parameters_split : Optional[Union[Tuple[str, ...], Dict[str, int]]], default = None
                      Configuration for splitting the weight and bias tensors along dim 0 into
                      multiple PyTorch parameters. If a list or tuple of strings is provided,
                      they are used to make the names of equally-sized parameters. If a dict
                      (preferably an OrderedDict) is provided, the keys are used as names and
                      values as split sizes along dim 0. The resulting parameters will have
                      names that end in `_weight` or `_bias`, so trailing underscores are
                      stripped from any provided names.
    zero_centered_gamma : bool, default = 'False'
                         if set to 'True', gamma parameter in LayerNorm is initialized to 0 and
                         the LayerNorm formula changes to

                         .. math::
                            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \varepsilon}} *
                            (1 + \gamma) + \beta
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
            qlinear_params=qlinear_params,
        )

        # userbuffer checks
        if any([ub_bulk_wgrad, ub_bulk_dgrad, ub_overlap_ag, ub_overlap_rs_dgrad]):
            assert ub_name is not None, "Userbuffer name [string] is not set."
        assert not (
            ub_bulk_wgrad or ub_bulk_dgrad or ub_overlap_rs_dgrad
        ), "Userbuffer optimizations for bulk_wgrad, bulk_dgrad and overlap_rs_dgrad are not supported yet."

        self.return_layernorm_output = return_layernorm_output
        self.return_layernorm_output_gathered = return_layernorm_output_gathered

        # does not support return_layernorm_output_gathered for now to keep the code simple
        assert (
            not self.return_layernorm_output_gathered
        ), "return_layernorm_output_gathered is not supported yet."

        # Setup normalization
        assert normalization in [
            "LayerNorm",
            "RMSNorm",
        ], "Unsupported normalization type!"
        self.normalization = normalization

        # use_te_norm is None by default, and will be set to True if we should switch to hybrid linear
        if use_te_norm is None:
            use_te_norm = config.should_switch_to_hybrid_linear()

        if use_te_norm:
            self.norm = make_te_norm(
                normalization,
                in_features,
                eps=eps,
                sequence_parallel=sequence_parallel,
                params_dtype=params_dtype,
                zero_centered_gamma=zero_centered_gamma,
                device=device,
            )
        else:
            self.norm = make_torch_norm(
                normalization,
                in_features,
                eps=eps,
                sequence_parallel=sequence_parallel,
                params_dtype=params_dtype,
                zero_centered_gamma=zero_centered_gamma,
                device=device,
            )

    @torch._dynamo.disable(recursive=True)
    def forward(
        self,
        inp: torch.Tensor,
        is_first_microbatch: Optional[bool] = None,
        fp8_output: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply layer normalization to the input followed by a linear transformation.

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
        inp = inp.contiguous()

        # Run normalization without fusion
        ln_out = self.norm(inp)
        input_meta = None

        # Run the linear layer by calling parent's forward
        out = super().forward(ln_out, is_first_microbatch, fp8_output, input_meta)

        if self.return_bias:
            out, returned_bias = out
            if self.return_layernorm_output:
                return out, returned_bias, ln_out
            else:
                return out, returned_bias
        else:
            if self.return_layernorm_output:
                return out, ln_out
            else:
                return out
