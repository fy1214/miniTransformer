# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import contextlib
import gc
import os
import sys
import time
import argparse
import inspect
import types

if os.environ.get("TE_FP4_PERTOKEN") == "1":
    import transformer_engine.common.recipe as _te_fp4_recipe_mod
    import megatron.core.fp4_utils as _megatron_fp4_utils
    from contextlib import nullcontext

    def _te_native_pertoken_fp4_recipe(config):
        del config
        return _te_fp4_recipe_mod.NVFP4PerTokenBlockScaling()

    def _te_native_pertoken_fp4_context(config, layer_no: int = -1, is_init: bool = False):
        """NVFP4 per-token recipe cannot use Megatron's amax-reduction process group."""
        num_bf16_layers_at_start = (
            config.num_layers_at_start_in_bf16 if config.first_last_layers_bf16 else 0
        )
        num_bf16_layers_at_end = (
            config.num_layers_at_end_in_bf16 if config.first_last_layers_bf16 else 0
        )
        is_first_layer = layer_no < num_bf16_layers_at_start
        is_last_layer = layer_no >= config.num_layers - num_bf16_layers_at_end
        need_fp4_context = config.fp4 if not is_init else config.fp4_param
        if (
            not need_fp4_context
            or (layer_no >= 0 and config.first_last_layers_bf16 and (is_first_layer or is_last_layer))
        ):
            return nullcontext()

        fp4_recipe = _te_native_pertoken_fp4_recipe(config)
        import transformer_engine.pytorch

        if not is_init:
            return transformer_engine.pytorch.fp8_autocast(
                enabled=True, fp8_recipe=fp4_recipe, fp8_group=None
            )
        import inspect as _inspect

        context_args = {"enabled": True}
        if "recipe" in _inspect.signature(transformer_engine.pytorch.fp8_model_init).parameters:
            context_args["recipe"] = fp4_recipe
        return transformer_engine.pytorch.fp8_model_init(**context_args)

    _megatron_fp4_utils.get_fp4_recipe = _te_native_pertoken_fp4_recipe
    _megatron_fp4_utils.get_fp4_context = _te_native_pertoken_fp4_context

import pytest
import torch
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.enums import ModelType
from megatron.core.fp8_utils import is_float8tensor
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.utils import is_te_min_version
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)
from megatron.training.training import get_model, setup_model_and_optimizer
from megatron.training.utils import get_device_arch_version
from megatron.core.transformer.moe.moe_utils import (
    apply_router_token_dropping,
    compute_routing_scores_for_aux_loss,
)
from megatron.core.transformer.moe.router import TopKRouter
from tests.unit_tests.test_utilities import Utils

_SEED = 1234
fp8_available, reason_for_no_fp8 = check_fp8_support()

cuda_graph_supported = False
reason_for_no_cuda_graph = ""
try:
    from transformer_engine.pytorch.tensor.utils import post_all_gather_processing

    cuda_graph_supported = True
except ImportError:
    reason_for_no_cuda_graph = "Need newer TransformerEngine"


def _add_lowbit_speed_args(parser):
    group = parser.add_argument_group(title='lowbit speed profiling')
    group.add_argument('--lowbit-warmup-steps', type=int, default=5,
                       help='Steps before speed timing starts.')
    group.add_argument('--lowbit-bench-steps', type=int, default=10,
                       help='Steps included in speed timing.')
    group.add_argument('--lowbit-profile-steps', type=int, default=0,
                       help='Steps captured by torch.profiler (0 disables).')
    group.add_argument('--lowbit-profile-dir', type=str, default='/tmp/lowbit_profile',
                       help='Output directory for torch.profiler traces.')
    return parser


def _capture_lowbit_argv():
    """Pick --lowbit-* flags out of sys.argv at import time so they survive create_test_args's argv reset."""
    known = {'--lowbit-warmup-steps', '--lowbit-bench-steps',
             '--lowbit-profile-steps', '--lowbit-profile-dir'}
    out, argv, i = [], sys.argv[1:], 0
    while i < len(argv):
        tok = argv[i]
        key = tok.split('=', 1)[0]
        if key in known:
            out.append(tok)
            if '=' not in tok and i + 1 < len(argv):
                out.append(argv[i + 1])
                i += 2
            else:
                i += 1
        else:
            i += 1
    return out


_LOWBIT_ARGV = _capture_lowbit_argv()


def _make_random_topk_probs_and_map(logits: torch.Tensor, topk: int):
    """Build routing outputs with uniform random top-k expert assignment."""
    num_tokens, num_experts = logits.shape
    if num_experts < topk:
        raise ValueError(
            f"random top-k routing requires num_experts >= topk, "
            f"got experts={num_experts}, topk={topk}"
        )

    scores = torch.rand(
        num_tokens, num_experts, device=logits.device, dtype=torch.float32
    )
    _, top_indices = torch.topk(scores, k=topk, dim=1)
    sel_probs = torch.full(
        (num_tokens, topk), 1.0 / topk, device=logits.device, dtype=logits.dtype
    )
    routing_probs = torch.zeros_like(logits).scatter(1, top_indices, sel_probs)
    routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    return routing_probs, routing_map


def _patch_router_random_topk(router: TopKRouter):
    original_routing = router.routing

    def random_routing(self, logits: torch.Tensor):
        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)
        logits = self.apply_z_loss(logits)

        probs, routing_map = _make_random_topk_probs_and_map(logits, self.topk)

        if self.config.moe_expert_capacity_factor is not None:
            probs, routing_map = apply_router_token_dropping(
                probs,
                routing_map,
                router_topk=self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                drop_policy=self.config.moe_token_drop_policy,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            )

        if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
            routing_map_for_aux_loss, scores_for_aux_loss = compute_routing_scores_for_aux_loss(
                logits,
                self.topk,
                self.score_function,
                fused=self.config.moe_router_fusion,
            )
            probs = self._apply_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)
            probs = self._apply_seq_aux_loss(
                probs, scores_for_aux_loss, routing_map_for_aux_loss, seq_length, bsz
            )
            probs = self._apply_global_aux_loss(
                probs, scores_for_aux_loss, routing_map_for_aux_loss
            )

        self._apply_expert_bias(routing_map)
        return probs, routing_map

    router.routing = types.MethodType(random_routing, router)
    return router, original_routing


def _patch_model_random_topk_routing(model_chunks):
    patches = []
    for model_chunk in model_chunks:
        for module in model_chunk.modules():
            if isinstance(module, TopKRouter):
                patches.append(_patch_router_random_topk(module))
    return patches


def _restore_random_topk_routing(patches):
    for router, original_routing in patches:
        router.routing = original_routing


def enable_forward_pre_hook(model_chunks):
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.enable_forward_pre_hook()


def disable_forward_pre_hook(model_chunks, param_sync=True):
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.disable_forward_pre_hook(param_sync=param_sync)


def should_disable_forward_pre_hook(args):
    """Block forward pre-hook for certain configurations."""
    return (
        not args.use_megatron_fsdp and args.use_distributed_optimizer and args.overlap_param_gather
    )


class TestFP8Param:

    # Populated by main() (standalone runner). pytest path leaves this empty
    # and falls back to per-method defaults.
    _extra_args: dict = {}

    def setup_method(self, method):
        self.seq_length = 512
        self.micro_batch_size = 2
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        gc.collect()

    def model_provider(
        self,
        pre_process=True,
        post_process=True,
        layer_spec_fn=get_gpt_layer_with_transformer_engine_spec,
        **config_kwargs,
    ):
        model_parallel_cuda_manual_seed(_SEED)
        args = get_args()
        config = core_transformer_config_from_args(args)
        num_experts = getattr(args, "num_experts", None)
        if num_experts and num_experts > 0:
            transformer_layer_spec = layer_spec_fn(
                num_experts=num_experts,
                moe_grouped_gemm=getattr(args, "moe_grouped_gemm", False),
            )
        else:
            transformer_layer_spec = layer_spec_fn()
        return GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.vocal_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
        )

    def create_test_args(
        self,
        tp,
        recipe,
        sequence_length,
        micro_batch_size,
        inference,
        fp8_param_gather,
        use_cuda_graph,
        **kwargs,
    ):
        destroy_global_vars()
        destroy_num_microbatches_calculator()

        sys.argv = ['test_fp8_param.py', *_LOWBIT_ARGV]
        args = parse_args(extra_args_provider=_add_lowbit_speed_args)
        args.num_layers = 4
        args.vocal_size = 128800
        args.hidden_size = 128
        args.num_attention_heads = 8
        args.max_position_embeddings = 512
        args.micro_batch_size = micro_batch_size
        args.create_attention_mask_in_dataloader = True
        args.seq_length = sequence_length
        args.tensor_model_parallel_size = tp
        args.sequence_parallel = True if tp > 1 else False
        args.pipeline_model_parallel_size = 1
        args.context_parallel_size = 1
        args.train_iters = 10
        args.lr = 3e-5
        args.bf16 = True
        args.add_bias_linear = False
        args.swiglu = True
        args.use_distributed_optimizer = not inference
        # NVFP4 goes through megatron's --fp4-format/--fp4-recipe path; fp4 and fp8
        # are mutually exclusive in validate_args, so clear fp8 when recipe="nvfp4".
        if recipe == "nvfp4":
            args.fp8 = None
            args.fp8_recipe = None
            args.fp8_param_gather = False
            args.fp4 = "e2m1"
            args.fp4_recipe = "nvfp4"
            # TE's replace_raw_data() does not support NVFP4Tensor yet, so megatron's
            # DDP _ParamAndGradBuffer.modify_underlying_storage() will crash if we
            # enable fp4_param. Keep main params in bf16 and quantize to NVFP4 on the
            # fly during forward — re-enable once TE adds NVFP4Tensor support.
            args.fp4_param = False
            # When QAT_PARAMS is set, miniTransformer.GroupedLinear owns fp4 via its
            # own kernels. We still want config.fp4=True so megatron applies the
            # hidden/seq padding required for fp4 alignment, but we DON'T want the
            # per-layer NVFP4 autocast (which would force TE Linear/LayerNormLinear
            # to quantize and trigger A/B scaling_mode mismatch in wgrad backward).
            # Trick: flag every layer as a "first-last bf16" layer, so get_fp4_context
            # returns nullcontext for the whole stack. Validation in transformer_config
            # only kicks in when config.fp8 is set, so this is safe for fp4-only.
            if os.environ.get("QAT_PARAMS"):
                args.first_last_layers_bf16 = True
                args.num_layers_at_start_in_bf16 = args.num_layers
                args.num_layers_at_end_in_bf16 = 0
        elif recipe == "bf16":
            args.fp8 = None
            args.fp8_recipe = None
            args.fp8_param_gather = False
            args.fp4 = None
            args.fp4_recipe = None
            args.fp4_param = False
        else:
            args.fp8 = "e4m3"
            args.fp8_recipe = recipe
            args.fp8_param_gather = fp8_param_gather
        args.ddp_bucket_size = 1024  # Create more buckets to test the rs/ag overlap.
        args.attention_backend = AttnBackend.flash
        # Container may lack apex fused_weight_gradient_mlp_cuda.
        args.gradient_accumulation_fusion = False
        # MXFP8 test settings
        if recipe == "mxfp8" and fp8_param_gather:
            args.reuse_grad_buf_for_mxfp8_param_ag = True

        if use_cuda_graph:
            args.cuda_graph_impl = "transformer_engine"
            args.cuda_graph_warmup_steps = 0

        # Merge class-level extras first (from CLI main()), then per-call kwargs
        # so kwargs win on conflict. Allow optional extension attrs that are not
        # always registered on older Megatron arg parsers.
        for key, value in type(self)._extra_args.items():
            if not hasattr(args, key):
                setattr(args, key, value)
                continue
            setattr(args, key, value)

        for key, value in kwargs.items():
            if not hasattr(args, key):
                setattr(args, key, value)
                continue
            setattr(args, key, value)

        validate_args(args)
        set_global_variables(args, False)
        return args

    def get_batch(self, seq_length, micro_batch_size):
        data = list(range(seq_length))
        input_ids =  torch.randint(1000, [len(data)], dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        labels = 1 + torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, seq_length, seq_length), dtype=bool
        ).cuda()
        loss_mask = torch.ones(seq_length).repeat((micro_batch_size, 1)).cuda()
        return input_ids, labels, position_ids, attention_mask, loss_mask

    def _run_test_helper(
        self,
        tp_size,
        recipe,
        inference: bool = False,
        fp8_param_gather: bool = True,
        use_cuda_graph: bool = False,
        **kwargs,
    ):
        """Test fp8_param with gpt_model."""
        args = self.create_test_args(
            tp_size,
            recipe,
            self.seq_length,
            self.micro_batch_size,
            inference,
            fp8_param_gather,
            use_cuda_graph,
            **kwargs,
        )

        # Speed profiling config: controlled by CLI args.
        #   --lowbit-warmup-steps   steps before timing starts (default 5)
        #   --lowbit-bench-steps    steps to time (default 10)
        #   --lowbit-profile-steps  steps to capture torch.profiler trace (0 = disabled)
        #   --lowbit-profile-dir    output dir for profiler traces (default /tmp/lowbit_profile)
        warmup_steps = args.lowbit_warmup_steps
        bench_steps  = args.lowbit_bench_steps
        profile_steps = args.lowbit_profile_steps
        profile_dir   = args.lowbit_profile_dir
        total_steps = warmup_steps + bench_steps

        if recipe == "blockwise" and args.sequence_parallel:
            assert (
                tp_size * 128 <= self.seq_length
            ), "Blockwise recipe and sequence parallelism requires tp_size * 128 <= seq_length"

        set_args(args)
        torch.manual_seed(_SEED)
        ep_size = getattr(args, "expert_model_parallel_size", 1) or 1
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
        )

        input_ids, labels, position_ids, attention_mask, loss_mask = self.get_batch(
            self.seq_length, self.micro_batch_size
        )
        if inference:
            gpt_model = get_model(
                self.model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False
            )
            gpt_model[0].eval()
            optimizer = None
        else:
            gpt_model, optimizer, _ = setup_model_and_optimizer(
                self.model_provider, ModelType.encoder_or_decoder
            )
        assert len(gpt_model) == 1  # Assume only one model in the model provider.

        moe_enabled = bool(getattr(args, "num_experts", 0))
        router_patches = []
        if getattr(self, "random_topk_routing", False):
            assert moe_enabled, "random_topk_routing requires MoE"
            router_patches = _patch_model_random_topk_routing(gpt_model)
            assert router_patches, "random_topk_routing enabled but no TopKRouter found"

        cuda_graph_helper = None
        # Hard coded to use cuda_graph_impl="transformer_engine"
        cuda_graph_impl = "transformer_engine"
        if use_cuda_graph and cuda_graph_impl == "transformer_engine":
            from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

            cuda_graph_helper = TECudaGraphHelper(
                model=gpt_model,
                config=gpt_model[0].config,
                seq_length=self.seq_length,
                micro_batch_size=self.micro_batch_size,
                optimizers=[optimizer],
            )

        num_fp8_params = 0
        for _, param in gpt_model[0].named_parameters():
            if not inference:
                assert param.requires_grad
                assert param.main_grad is not None
            if is_float8tensor(param):
                num_fp8_params += 1

        # Verify the number of fp8 params.
        fp8_layers = args.num_layers
        if kwargs.get("first_last_layers_bf16", False):
            fp8_layers -= kwargs["num_layers_at_start_in_bf16"]
            fp8_layers -= kwargs["num_layers_at_end_in_bf16"]
        # Each layer has 4 GEMM weights: qkv, proj, fc1, fc2.
        # Skip the strict count when MoE is on (different expert weight layout) or
        # when recipe is fp4 (is_float8tensor doesn't match fp4 param tensors).
        fp4_recipe = bool(getattr(args, "fp4", None))
        if fp8_param_gather and not moe_enabled and not fp4_recipe:
            assert num_fp8_params == 4 * fp8_layers

        loss_list = []

        # Build profiler context: active only when profile_steps > 0.
        if profile_steps > 0:
            prof_ctx = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=warmup_steps,
                    warmup=0,
                    active=profile_steps,
                    repeat=1,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
                record_shapes=True,
                with_stack=False,
            )
        else:
            prof_ctx = contextlib.nullcontext()

        step_times = []

        try:
            with prof_ctx as prof:
                for i in range(total_steps):
                    if not inference:
                        gpt_model[0].zero_grad_buffer()
                        optimizer.zero_grad()

                    # Capture CUDA graphs after warmup if helper is provided.
                    # Hard coded cuda_graph_warmup_steps = 0.
                    cuda_graph_warmup_steps = 0
                    if cuda_graph_helper is not None and i == cuda_graph_warmup_steps:
                        if should_disable_forward_pre_hook(args):
                            disable_forward_pre_hook(gpt_model, param_sync=False)
                        cuda_graph_helper.create_cudagraphs()
                        if should_disable_forward_pre_hook(args):
                            enable_forward_pre_hook(gpt_model)
                            cuda_graph_helper.cuda_graph_set_manual_hooks()

                    # For the mxfp8_param with reuse_grad_buf_for_mxfp8_param_ag and dp_ag_overlap,
                    # we need to call the _copy_main_params_to_param_buffer() after the grad buffer
                    # is zeroed by zero_grad_buffer() because param and grad buffer are shared.
                    if args.reuse_grad_buf_for_mxfp8_param_ag and args.overlap_param_gather:
                        for optim_instance in optimizer.chained_optimizers:
                            if hasattr(optim_instance, "_copy_main_params_to_param_buffer"):
                                optim_instance._copy_main_params_to_param_buffer()

                    gpt_model[0].set_is_first_microbatch()

                    torch.cuda.synchronize()
                    t0 = time.perf_counter()

                    output = gpt_model[0].forward(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        loss_mask=loss_mask,
                    )

                    # Check output shapes
                    assert output.shape[0] == self.micro_batch_size
                    assert output.shape[1] == self.seq_length

                    if inference:
                        torch.cuda.synchronize()
                        step_times.append(time.perf_counter() - t0)
                        if prof is not None:
                            prof.step()
                        continue

                    # Compute masked cross-entropy loss (same as pretrain_gpt.py loss_func)
                    losses = output.view(-1).float()
                    loss_mask_flat = loss_mask.view(-1).float()
                    loss = torch.sum(losses * loss_mask_flat) / loss_mask_flat.sum()

                    # Precision check: loss must be finite
                    assert torch.isfinite(loss), f"loss is not finite at step {i}: {loss.item()}"

                    loss.backward()

                    if args.overlap_grad_reduce:
                        gpt_model[0].finish_grad_sync()

                    # Precision check: gradients must be finite
                    for name, param in gpt_model[0].named_parameters():
                        assert param.main_grad is not None
                        assert torch.isfinite(param.main_grad).all(), \
                            f"non-finite grad in {name} at step {i}"

                    update_successful, _, _ = optimizer.step()
                    assert update_successful

                    torch.cuda.synchronize()
                    step_times.append(time.perf_counter() - t0)

                    loss_list.append(loss.item())

                    if prof is not None:
                        prof.step()
        finally:
            _restore_random_topk_routing(router_patches)

        # ── Speed report ──────────────────────────────────────────────────────
        if step_times:
            bench_times = step_times[warmup_steps:]
            if bench_times:
                avg_ms  = sum(bench_times) / len(bench_times) * 1e3
                min_ms  = min(bench_times) * 1e3
                max_ms  = max(bench_times) * 1e3
                _rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                print(
                    f"\n[rank={_rank}][speed] recipe={recipe} tp={tp_size} "
                    f"seq={self.seq_length} bs={self.micro_batch_size} "
                    f"| avg={avg_ms:.1f}ms  min={min_ms:.1f}ms  max={max_ms:.1f}ms "
                    f"over {len(bench_times)} steps"
                )

        # ── Precision report ──────────────────────────────────────────────────
        if loss_list and not inference:
            first, last = loss_list[0], loss_list[-1]
            _rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(
                f"[rank={_rank}][precision] loss: first={first:.4f}  last={last:.4f}  "
                f"delta={last - first:+.4f}"
            )
            # Loss should decrease (or at least not diverge) over the run.
            assert last < first * 2, \
                f"loss diverged: first={first:.4f} last={last:.4f}"

        return torch.tensor(loss_list)

    def run_test(self, tp_size, recipe, inference: bool = False, **kwargs):
        """Test fp8_param with gpt_model."""
        if inference:
            with torch.inference_mode():
                self._run_test_helper(tp_size, recipe, inference=True, **kwargs)
        else:
            loss_list = self._run_test_helper(tp_size, recipe, fp8_param_gather=True, **kwargs)

            # Before TE 2.2.0, we cannot guarantee that the main params are the same with/without
            # fp8-param-gather, so skip the checking of tensor values.
            if is_te_min_version("2.2.0"):
                loss_list_ref = self._run_test_helper(
                    tp_size, recipe, fp8_param_gather=False, **kwargs
                )
                torch.testing.assert_close(loss_list, loss_list_ref, atol=1e-4, rtol=1e-4)

    def run_test_with_cuda_graph(self, tp_size, recipe, **kwargs):
        loss = self._run_test_helper(
            tp_size, recipe, fp8_param_gather=True, use_cuda_graph=True, **kwargs
        )
        loss_ref = self._run_test_helper(
            tp_size, recipe, fp8_param_gather=True, use_cuda_graph=False, **kwargs
        )
        torch.testing.assert_close(loss, loss_ref, atol=0, rtol=0)

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(True, True)])
    def test_delayed_scaling(self, tp_size, dp_overlap):
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test(tp_size=tp_size, recipe="delayed", **kwargs)

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(True, True)])
    @pytest.mark.skipif(not cuda_graph_supported, reason=reason_for_no_cuda_graph)
    def test_delayed_scaling_with_cuda_graph(self, tp_size, dp_overlap):
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test_with_cuda_graph(tp_size, "delayed", **kwargs)

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.2.0"), reason="TE 2.2.0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(True, True)])
    def test_tensorwise_scaling(self, tp_size, dp_overlap):
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test(tp_size=tp_size, recipe="tensorwise", **kwargs)

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.2.0"), reason="TE 2.2.0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    def test_tensorwise_scaling_inference(self, tp_size):
        self.run_test(tp_size=tp_size, recipe="tensorwise", inference=True)

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.2.0"), reason="TE 2.2.0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(True, True)])
    @pytest.mark.skipif(not cuda_graph_supported, reason=reason_for_no_cuda_graph)
    def test_tensorwise_scaling_with_cuda_graph(self, tp_size, dp_overlap):
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test_with_cuda_graph(tp_size, "tensorwise", **kwargs)

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.2.0"), reason="TE 2.2.0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    def test_tensorwise_scaling_with_first_last_layers_bf16(self, tp_size):
        kwargs = {
            "first_last_layers_bf16": True,
            "num_layers_at_start_in_bf16": 1,
            "num_layers_at_end_in_bf16": 1,
        }
        self.run_test(tp_size=tp_size, recipe="tensorwise", **kwargs)

    @pytest.mark.skipif(
        get_device_arch_version() != 9, reason="blockwise is only supported on Hopper architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.4.0.dev0"), reason="TE 2.4.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(True, True)])
    def test_blockwise_scaling(self, tp_size, dp_overlap):
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test(tp_size=tp_size, recipe="blockwise")

    @pytest.mark.skipif(
        get_device_arch_version() != 9, reason="blockwise is only supported on Hopper architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.4.0.dev0"), reason="TE 2.4.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(True, True)])
    @pytest.mark.skipif(not cuda_graph_supported, reason=reason_for_no_cuda_graph)
    def test_blockwise_scaling_with_cuda_graph(self, tp_size, dp_overlap):
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test_with_cuda_graph(tp_size, "blockwise", **kwargs)

    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(False, False), (False, True), (True, True)])
    def test_mxfp8(self, tp_size, dp_overlap):
        """
        dp_overlap: (overlap_param_gather, overlap_grad_reduce)
        """
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test(tp_size=tp_size, recipe="mxfp8", **kwargs)

    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(False, False), (False, True), (True, True)])
    @pytest.mark.skipif(not cuda_graph_supported, reason=reason_for_no_cuda_graph)
    def test_mxfp8_with_cuda_graph(self, tp_size, dp_overlap):
        """
        dp_overlap: (overlap_param_gather, overlap_grad_reduce)
        """
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test_with_cuda_graph(tp_size=tp_size, recipe="mxfp8", **kwargs)

    @pytest.mark.skipif(
        get_device_arch_version() != 9, reason="blockwise is only supported on Hopper architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.4.0.dev0"), reason="TE 2.4.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    def test_blockwise_scaling_with_first_last_layers_bf16(self, tp_size):
        kwargs = {
            "first_last_layers_bf16": True,
            "num_layers_at_start_in_bf16": 1,
            "num_layers_at_end_in_bf16": 1,
        }
        self.run_test(tp_size=tp_size, recipe="blockwise", **kwargs)

    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="NVFP4 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.4.0.dev0"), reason="TE 2.4.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(False, False), (False, True), (True, True)])
    def test_nvfp4(self, tp_size, dp_overlap):
        """
        dp_overlap: (overlap_param_gather, overlap_grad_reduce)
        """
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test(tp_size=tp_size, recipe="nvfp4", **kwargs)

    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="NVFP4 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.4.0.dev0"), reason="TE 2.4.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(False, False), (False, True), (True, True)])
    @pytest.mark.skipif(not cuda_graph_supported, reason=reason_for_no_cuda_graph)
    def test_nvfp4_with_cuda_graph(self, tp_size, dp_overlap):
        """
        dp_overlap: (overlap_param_gather, overlap_grad_reduce)
        """
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test_with_cuda_graph(tp_size=tp_size, recipe="nvfp4", **kwargs)


def _build_cli_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone runner for TestFP8Param. Launch via torchrun:\n"
            "  torchrun --nproc_per_node=N tests/python/megatron/test_lowbit_param.py "
            "<method> [--tp-size N] [--overlap-param-gather 0|1] [--overlap-grad-reduce 0|1] "
            "[--lowbit-warmup-steps INT] [--lowbit-bench-steps INT] "
            "[--lowbit-profile-steps INT] [--lowbit-profile-dir DIR]"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("method", help="Test method on TestFP8Param, e.g. test_mxfp8")
    parser.add_argument("--tp-size", type=int, default=2,
                        help="Tensor-parallel size (when ep=1 must match --nproc_per_node).")
    parser.add_argument("--ep-size", type=int, default=1,
                        help="Expert-parallel size. world_size = tp_size * ep_size (pp=cp=1 here).")
    parser.add_argument("--moe", action="store_true",
                        help="Replace dense MLP with MoE layer.")
    parser.add_argument("--num-experts", type=int, default=8,
                        help="Number of MoE experts (used when --moe).")
    parser.add_argument("--moe-router-topk", type=int, default=2,
                        help="MoE router top-k (used when --moe).")
    parser.add_argument("--moe-grouped-gemm", action="store_true",
                        help="Enable grouped GEMM for MoE experts (used when --moe).")
    parser.add_argument("--moe-token-dispatcher-type", type=str, default="alltoall",
                        choices=["alltoall", "allgather", "alltoall_seq"],
                        help="MoE token dispatcher type (used when --moe).")
    parser.add_argument("--nvfp4-pertoken-amax-fuse", action="store_true",
                        help=("Enable the unified MoE NVFP4 per-token amax-fuse path: "
                              "fc1 sort_chunks row amax + skip-K1 handoff and "
                              "fc2 fused SwiGLU row amax + skip-K1 handoff. "
                              "Requires TE with moe_sort_chunks_by_index(compute_row_amax=) "
                              "(fy1214/TransformerEngine#8 on grouped-per-token-sf-swizzle)."))
    parser.add_argument("--random-topk-routing", action="store_true",
                        help=("MoE test-only: patch TopKRouter.routing() to assign each "
                              "token to random top-k experts with uniform probs."))
    parser.add_argument("--ffn-hidden-size", type=int, default=None,
                        help="Override args.ffn_hidden_size. Default lets megatron derive it.")
    parser.add_argument("--moe-ffn-hidden-size", type=int, default=None,
                        help=(
                            "Override args.moe_ffn_hidden_size (used when --moe). "
                            "Auto-default = max(512, 64*tp_size) so GroupedLinear's "
                            "out_features %% 128 == 0 holds for swiglu gate+up packed."
                        ))
    parser.add_argument("--qat-params", type=int, default=None,
                        help=(
                            "Select miniTransformer hybrid-linear recipe by index "
                            "(see miniTransformer/module/config.py:get_qlinear_params_from_env_qat_params). "
                            "Forwarded to QAT_PARAMS env var before model build. "
                            "Default None = leave env untouched."
                        ))
    parser.add_argument("--overlap-param-gather", type=int, default=1, choices=[0, 1],
                        help="dp_overlap[0]; ignored if method has no dp_overlap.")
    parser.add_argument("--overlap-grad-reduce", type=int, default=1, choices=[0, 1],
                        help="dp_overlap[1]; ignored if method has no dp_overlap.")
    parser.add_argument("--lowbit-warmup-steps", type=int, default=5)
    parser.add_argument("--lowbit-bench-steps", type=int, default=10)
    parser.add_argument("--lowbit-profile-steps", type=int, default=0)
    parser.add_argument("--lowbit-profile-dir", type=str, default="/tmp/lowbit_profile")
    return parser


def main():
    cli = _build_cli_parser().parse_args()

    # Forward --qat-params into QAT_PARAMS env var BEFORE megatron builds the model,
    # so miniTransformer.utils.env.should_switch_to_hybrid_linear() and
    # config.get_qlinear_params_from_env_qat_params() observe the chosen recipe.
    if cli.qat_params is not None:
        os.environ["QAT_PARAMS"] = str(cli.qat_params)

    # Forward lowbit values into megatron's parse_args path used by create_test_args.
    global _LOWBIT_ARGV
    _LOWBIT_ARGV = [
        f"--lowbit-warmup-steps={cli.lowbit_warmup_steps}",
        f"--lowbit-bench-steps={cli.lowbit_bench_steps}",
        f"--lowbit-profile-steps={cli.lowbit_profile_steps}",
        f"--lowbit-profile-dir={cli.lowbit_profile_dir}",
    ]

    extra = {"expert_model_parallel_size": cli.ep_size}
    if cli.ffn_hidden_size is not None:
        extra["ffn_hidden_size"] = cli.ffn_hidden_size
    if cli.moe:
        # GroupedLinear asserts out_features % 128 == 0 on the column-parallel
        # linear_fc1 (gate+up packed when swiglu). Post-TP that means
        #   2 * moe_ffn_hidden_size / tp_size  must be a multiple of 128
        # i.e. moe_ffn_hidden_size must be a multiple of 64 * tp_size.
        moe_ffn = cli.moe_ffn_hidden_size
        if moe_ffn is None:
            moe_ffn = max(512, 64 * cli.tp_size)
        assert moe_ffn % (64 * cli.tp_size) == 0, (
            f"--moe-ffn-hidden-size={moe_ffn} must be a multiple of "
            f"64*tp_size={64 * cli.tp_size} (GroupedLinear divisibility)."
        )
        extra.update({
            "num_experts": cli.num_experts,
            "moe_router_topk": cli.moe_router_topk,
            "moe_grouped_gemm": cli.moe_grouped_gemm,
            "moe_token_dispatcher_type": cli.moe_token_dispatcher_type,
            "moe_ffn_hidden_size": moe_ffn,
            "nvfp4_pertoken_amax_fuse": bool(cli.nvfp4_pertoken_amax_fuse),
        })
    TestFP8Param._extra_args = extra

    obj = TestFP8Param()
    obj.random_topk_routing = bool(cli.moe and cli.random_topk_routing)
    if not hasattr(obj, cli.method) or not cli.method.startswith("test_"):
        raise SystemExit(f"unknown test method: {cli.method!r}")
    method = getattr(obj, cli.method)
    sig = inspect.signature(method)
    call_kwargs = {}
    if "tp_size" in sig.parameters:
        call_kwargs["tp_size"] = cli.tp_size
    if "dp_overlap" in sig.parameters:
        call_kwargs["dp_overlap"] = (bool(cli.overlap_param_gather),
                                     bool(cli.overlap_grad_reduce))

    obj.setup_method(None)
    try:
        method(**call_kwargs)
    finally:
        obj.teardown_method(None)


if __name__ == "__main__":
    main()
