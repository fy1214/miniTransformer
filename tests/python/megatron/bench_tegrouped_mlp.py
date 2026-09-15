#!/usr/bin/env python3
"""Standard TEGroupedMLP microbench (fc1 + GLU + fc2; fwd / bwd / fwb).

Measures steady-state ``TEGroupedMLP`` with input already expert-permuted
(no router / dispatch). See design:

  docs/superpowers/specs/2026-09-09-tegrouped-mlp-forward-bench-design.md

Backends (implementation path):
  te_bf16              — no FP4 autocast
  te_nvfp4             — NVFP4BlockScaling(E2M1)
  te_nvfp4_pertoken    — NVFP4PerTokenBlockScaling
  mini_qat14           — SHOULD_REPLACE_TE_GROUPLINEAR + QAT_PARAMS=14
  mini_qat15           — QAT_PARAMS=15; default MINITE_NVFP4_PATH=dense
  mini_qat16           — QAT_PARAMS=16 (NVFP4 fwd + TE-style BF16 bwd); dense default

Modes:
  --bench-mode {forward,fwd,backward,bwd,fwb,both}
      forward|fwd   timed forward only (default)
      backward|bwd  untimed forward, timed backward only (no optimizer)
      fwb           timed forward + backward together
      both          run forward, backward, then fwb for each config

Feature:
  --amax-fuse {off,on,both}
      off   stock mlp(...)
      on    prefused input_row_amax outside timer (skip-K1); unsupported
            backends are skipped with a warning
      both  run off then on for each (backend, seq)

Examples:
  python3 tests/python/megatron/bench_tegrouped_mlp.py \\
      --backend mini_qat15 --backends mini_qat15 \\
      --bench-mode both --amax-fuse both --seq-lens 16384 32768 \\
      --warmup 15 --iters 40 --out-json profiles/tegrouped_mlp/mini_qat15.json

  OUT_DIR=profiles/tegrouped_mlp AMAX_FUSE=both BENCH_MODE=both \\
    bash tests/python/megatron/run_tegrouped_mlp_sweep.sh
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple


def parse_cli_flag_before_imports(
    flag: str, default: str | None = None
) -> str | None:
    """Read a CLI flag before heavy imports (needed for Megatron TE replace)."""
    for index, token in enumerate(sys.argv[1:], start=1):
        if token == flag and index < len(sys.argv) - 1:
            return sys.argv[index + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return default


def configure_backend_environment(backend_name: str) -> None:
    """Set env vars that Megatron/miniTE read at import or GroupedLinear build time."""
    os.environ.pop("SHOULD_REPLACE_TE_GROUPLINEAR", None)
    os.environ.pop("QAT_PARAMS", None)
    if backend_name.startswith("mini_"):
        os.environ["SHOULD_REPLACE_TE_GROUPLINEAR"] = "1"
        if backend_name == "mini_qat14":
            os.environ["QAT_PARAMS"] = "14"
        elif backend_name == "mini_qat16":
            os.environ["QAT_PARAMS"] = "16"
        else:
            os.environ["QAT_PARAMS"] = "15"
        # Dense NVFP4 packaging for QAT15/16 unless the caller already set a path.
        if (
            backend_name in ("mini_qat15", "mini_qat16")
            and "MINITE_NVFP4_PATH" not in os.environ
        ):
            os.environ["MINITE_NVFP4_PATH"] = "dense"


# Backend must be chosen before Megatron TE extension import (replace is import-time).
IMPORT_TIME_BACKEND = parse_cli_flag_before_imports("--backend", "te_bf16") or "te_bf16"
configure_backend_environment(IMPORT_TIME_BACKEND)

os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("NVTE_NVFP4_DISABLE_2D_QUANTIZATION", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29671")
os.environ.setdefault("LOCAL_RANK", "0")

import torch
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.common import recipe as te_recipe

MEGATRON_ROOT_PATH = os.environ.get("MEGATRON_ROOT", "/root/Megatron-LM")
if MEGATRON_ROOT_PATH not in sys.path:
    sys.path.insert(0, MEGATRON_ROOT_PATH)
MINITRANSFORMER_ROOT_PATH = str(Path(__file__).resolve().parents[3])
if MINITRANSFORMER_ROOT_PATH not in sys.path:
    sys.path.insert(0, MINITRANSFORMER_ROOT_PATH)

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

from tests.unit_tests.test_utilities import Utils  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402
from megatron.core.models.gpt.gpt_layer_specs import (  # noqa: E402
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection  # noqa: E402
from megatron.core.transformer.spec_utils import build_module  # noqa: E402


QWEN3_30B_A3B_MODEL_DIMS = {
    "hidden_size": 2048,
    "moe_ffn_hidden_size": 768,
    "num_experts_global": 128,
    "top_k": 8,
    "num_attention_heads": 32,
    "num_query_groups": 4,
    "kv_channels": 128,
    "ffn_hidden_size": 6144,
}

SUPPORTED_BACKENDS = (
    "te_bf16",
    "te_nvfp4",
    "te_nvfp4_pertoken",
    "mini_qat14",
    "mini_qat15",
    "mini_qat16",
)

# Backends where amax-fuse simulation is meaningful / API-supported.
BACKENDS_SUPPORTING_AMAX_FUSE = frozenset(
    {"mini_qat15", "mini_qat16", "te_nvfp4_pertoken"}
)


# Canonical timed modes written into results JSON / tables.
SUPPORTED_BENCH_MODES = ("forward", "backward", "fwb")


@dataclass
class BenchTimingResult:
    backend: str
    bench_mode: str
    amax_fuse_enabled: bool
    sequence_length: int
    num_local_experts: int
    num_tokens: int
    tokens_per_expert: int
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


def normalize_bench_mode_alias(bench_mode: str) -> str:
    """Map CLI aliases onto canonical mode names."""
    aliases = {
        "forward": "forward",
        "fwd": "forward",
        "backward": "backward",
        "bwd": "backward",
        "fwb": "fwb",
        "both": "both",
    }
    if bench_mode not in aliases:
        raise ValueError(f"unknown --bench-mode: {bench_mode!r}")
    return aliases[bench_mode]


def synchronize_cuda() -> None:
    torch.cuda.synchronize()


def get_transformer_engine_version() -> str:
    try:
        import transformer_engine as te_pkg

        return getattr(te_pkg, "__version__", "unknown")
    except Exception:
        return "unknown"


@contextlib.contextmanager
def nvfp4_autocast_context(recipe_kind: Optional[str]) -> Iterator[None]:
    """Enter TE NVFP4 autocast, or no-op when ``recipe_kind`` is None (BF16)."""
    if recipe_kind is None:
        yield
        return
    if recipe_kind == "block":
        recipe = te_recipe.NVFP4BlockScaling(fp4_format=te_recipe.Format.E2M1)
    elif recipe_kind == "pertoken":
        if not hasattr(te_recipe, "NVFP4PerTokenBlockScaling"):
            raise RuntimeError("TE missing NVFP4PerTokenBlockScaling")
        recipe = te_recipe.NVFP4PerTokenBlockScaling()
    else:
        raise ValueError(f"unknown nvfp4 recipe_kind={recipe_kind!r}")
    if hasattr(te, "autocast"):
        with te.autocast(enabled=True, recipe=recipe):
            yield
    else:
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            yield


def reload_megatron_transformer_engine_extension() -> None:
    """Re-import Megatron TE extension so SHOULD_REPLACE / QAT_PARAMS take effect."""
    extension_module_name = "megatron.core.extensions.transformer_engine"
    if extension_module_name in sys.modules:
        del sys.modules[extension_module_name]
    for module_name in list(sys.modules):
        if module_name.startswith("megatron.core.transformer.moe.experts"):
            del sys.modules[module_name]
        if module_name.startswith("megatron.core.models.gpt.gpt_layer_specs"):
            del sys.modules[module_name]
    importlib.import_module(extension_module_name)


def build_tegrouped_mlp(num_local_experts: int, *, enable_fp4_config: bool):
    """Construct one TEGroupedMLP experts module for the given local expert count."""
    dims = QWEN3_30B_A3B_MODEL_DIMS
    transformer_config = TransformerConfig(
        num_layers=1,
        hidden_size=dims["hidden_size"],
        num_attention_heads=dims["num_attention_heads"],
        num_query_groups=dims["num_query_groups"],
        kv_channels=dims["kv_channels"],
        ffn_hidden_size=dims["ffn_hidden_size"],
        moe_ffn_hidden_size=dims["moe_ffn_hidden_size"],
        num_moe_experts=num_local_experts,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=True,
        use_te_activation_func=False,
        fp4="e2m1" if enable_fp4_config else None,
        fp4_recipe="nvfp4" if enable_fp4_config else None,
    )
    layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=num_local_experts, moe_grouped_gemm=True
    )
    experts_spec = layer_spec.submodules.mlp.submodules.experts
    process_groups = ProcessGroupCollection.use_mpu_process_groups()
    experts_module = build_module(
        experts_spec, num_local_experts, transformer_config, pg_collection=process_groups
    )
    return experts_module.cuda().eval()


def set_grouped_linear_first_microbatch_flag(experts_module, is_first: bool) -> None:
    for linear_name in ("linear_fc1", "linear_fc2"):
        linear_module = getattr(experts_module, linear_name, None)
        if linear_module is not None and hasattr(linear_module, "is_first_microbatch"):
            linear_module.is_first_microbatch = is_first


def disable_minite_weight_requantization(experts_module) -> None:
    """Keep quantized weight cache steady after the priming microbatch."""
    try:
        from miniTransformer.module.group_linear import GroupedLinear as MiniGroupedLinear
    except ImportError:
        return
    for submodule in experts_module.modules():
        if isinstance(submodule, MiniGroupedLinear):
            quantize_op = submodule.qlinear_params.quantize_op
            if hasattr(quantize_op, "should_update_quantize_weights"):
                quantize_op.should_update_quantize_weights = False


def nvfp4_autocast_recipe_kind_for_backend(backend_name: str) -> Optional[str]:
    if backend_name == "te_nvfp4":
        return "block"
    if backend_name == "te_nvfp4_pertoken":
        return "pertoken"
    return None


def patch_tegrouped_linear_to_accept_input_row_amax(experts_module) -> None:
    """Forward ``input_row_amax`` through Megatron's TEGroupedLinear wrapper.

    Stock Megatron wrapper only calls ``forward(x, m_splits)``. Both TE and miniTE
    GroupedLinear bases already accept ``input_row_amax``.
    """
    import inspect

    for linear_name in ("linear_fc1", "linear_fc2"):
        linear_module = getattr(experts_module, linear_name, None)
        if linear_module is None or getattr(
            linear_module, "_accepts_input_row_amax_patched", False
        ):
            continue

        base_class_with_amax = None
        for candidate_class in type(linear_module).mro():
            if candidate_class is object:
                break
            forward_fn = getattr(candidate_class, "forward", None)
            if forward_fn is None:
                continue
            try:
                parameters = inspect.signature(forward_fn).parameters
            except (TypeError, ValueError):
                continue
            if "input_row_amax" in parameters:
                base_class_with_amax = candidate_class
                break
        if base_class_with_amax is None:
            raise RuntimeError(
                f"{type(linear_module).__name__} has no input_row_amax; "
                "cannot enable amax-fuse"
            )

        def make_forward_with_amax(module, base_class):
            try:
                base_forward_params = inspect.signature(base_class.forward).parameters
            except (TypeError, ValueError):
                base_forward_params = {}

            def forward_with_optional_row_amax(
                hidden_states, tokens_per_expert, input_row_amax=None
            ):
                forward_kwargs = {}
                if "input_row_amax" in base_forward_params:
                    forward_kwargs["input_row_amax"] = input_row_amax
                # Older TE GroupedLinear accepted is_first_microbatch as a
                # forward kwarg; newer builds keep it as module state only.
                if "is_first_microbatch" in base_forward_params:
                    forward_kwargs["is_first_microbatch"] = (
                        None
                        if getattr(module, "disable_parameter_transpose_cache", False)
                        else getattr(module, "is_first_microbatch", None)
                    )
                output = base_class.forward(
                    module,
                    hidden_states,
                    tokens_per_expert,
                    **forward_kwargs,
                )
                if hasattr(module, "is_first_microbatch"):
                    module.is_first_microbatch = False
                if getattr(module, "te_return_bias", False):
                    return output
                if isinstance(output, tuple):
                    return output
                # Megatron TEGroupedLinear always returns (output, bias_or_None).
                return output, None

            return forward_with_optional_row_amax

        linear_module.forward = make_forward_with_amax(
            linear_module, base_class_with_amax
        )
        linear_module._accepts_input_row_amax_patched = True


def apply_tegrouped_mlp_quantization_padding(
    experts_module,
    hidden_states: torch.Tensor,
    tokens_per_expert,
    routing_probs: torch.Tensor,
):
    """Match TEGroupedMLP padding so token counts / row_amax lengths stay aligned."""
    tokens_per_expert_list = (
        tokens_per_expert.tolist()
        if torch.is_tensor(tokens_per_expert)
        else list(tokens_per_expert)
    )
    if experts_module.config.fp8 or experts_module.config.fp4:
        tokens_per_expert_before_pad = tokens_per_expert_list
        hidden_states, tokens_per_expert_list = experts_module.quantization_padding(
            hidden_states, tokens_per_expert_list
        )
        routing_probs, _ = experts_module.quantization_padding(
            routing_probs.unsqueeze(-1), tokens_per_expert_before_pad
        )
    else:
        routing_probs = routing_probs.unsqueeze(-1)
    return hidden_states, tokens_per_expert_list, routing_probs


def apply_weighted_bias_swiglu(
    experts_module, fc1_output, bias_parallel, routing_probs
):
    from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl

    return weighted_bias_swiglu_impl(
        fc1_output,
        bias_parallel,
        routing_probs,
        experts_module.config.activation_func_fp8_input_store,
    )


def tegrouped_mlp_forward_with_prefused_row_amax(
    experts_module,
    hidden_states: torch.Tensor,
    tokens_per_expert,
    routing_probs: torch.Tensor,
    fc1_row_amax: torch.Tensor,
    fc2_row_amax: torch.Tensor,
):
    """TEGroupedMLP-equivalent forward using prefused ``input_row_amax`` on both linears."""
    hidden_states, tokens_per_expert_list, routing_probs = (
        apply_tegrouped_mlp_quantization_padding(
            experts_module, hidden_states, tokens_per_expert, routing_probs
        )
    )
    if fc1_row_amax.numel() != hidden_states.shape[0]:
        raise RuntimeError(
            f"fc1_row_amax length {fc1_row_amax.numel()} != "
            f"padded tokens {hidden_states.shape[0]}"
        )
    fc1_output, bias_parallel = experts_module.linear_fc1(
        hidden_states, tokens_per_expert_list, input_row_amax=fc1_row_amax
    )
    swiglu_activation = apply_weighted_bias_swiglu(
        experts_module, fc1_output, bias_parallel, routing_probs
    )
    if fc2_row_amax.numel() != swiglu_activation.shape[0]:
        raise RuntimeError(
            f"fc2_row_amax length {fc2_row_amax.numel()} != "
            f"activation tokens {swiglu_activation.shape[0]}"
        )
    mlp_output, _bias = experts_module.linear_fc2(
        swiglu_activation, tokens_per_expert_list, input_row_amax=fc2_row_amax
    )
    return mlp_output


def precompute_fc1_and_fc2_row_amax(
    experts_module,
    hidden_states,
    tokens_per_expert,
    routing_probs,
    recipe_kind: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simulate upstream sort/SwiGLU amax fuse: compute row_amax outside the timer."""
    with torch.no_grad(), nvfp4_autocast_context(recipe_kind):
        padded_hidden, tokens_per_expert_list, padded_probs = (
            apply_tegrouped_mlp_quantization_padding(
                experts_module, hidden_states, tokens_per_expert, routing_probs
            )
        )
        fc1_row_amax = (
            padded_hidden.detach().abs().amax(dim=-1).to(dtype=torch.float32)
        )
        fc1_output, bias_parallel = experts_module.linear_fc1(
            padded_hidden, tokens_per_expert_list, input_row_amax=fc1_row_amax
        )
        swiglu_activation = apply_weighted_bias_swiglu(
            experts_module, fc1_output, bias_parallel, padded_probs
        )
        fc2_row_amax = (
            swiglu_activation.detach().abs().amax(dim=-1).to(dtype=torch.float32)
        )
    return fc1_row_amax, fc2_row_amax


def expand_amax_fuse_cli_mode_to_bools(amax_fuse_mode: str) -> List[bool]:
    if amax_fuse_mode == "off":
        return [False]
    if amax_fuse_mode == "on":
        return [True]
    if amax_fuse_mode == "both":
        return [False, True]
    raise ValueError(f"unknown --amax-fuse mode: {amax_fuse_mode!r}")


def expand_bench_mode_cli_to_modes(bench_mode: str) -> List[str]:
    canonical = normalize_bench_mode_alias(bench_mode)
    if canonical == "both":
        return ["forward", "backward", "fwb"]
    return [canonical]


def unpack_tegrouped_mlp_output(mlp_output) -> torch.Tensor:
    """TEGroupedMLP returns ``(output, bias_or_None)``; custom fuse path returns tensor."""
    if isinstance(mlp_output, tuple):
        return mlp_output[0]
    return mlp_output


def zero_expert_parameter_gradients(experts_module) -> None:
    experts_module.zero_grad(set_to_none=True)


def benchmark_one_backend_sequence_config(
    backend_name: str,
    sequence_length: int,
    *,
    bench_mode: str,
    amax_fuse_enabled: bool,
    tensor_parallel_shard: int,
    warmup_iters: int,
    timed_iters: int,
    random_seed: int,
) -> BenchTimingResult:
    dims = QWEN3_30B_A3B_MODEL_DIMS
    num_local_experts = dims["num_experts_global"] // tensor_parallel_shard
    num_tokens = (sequence_length * dims["top_k"]) // tensor_parallel_shard
    if num_tokens % num_local_experts != 0:
        raise SystemExit(
            f"num_tokens={num_tokens} not divisible by "
            f"num_local_experts={num_local_experts}"
        )
    tokens_per_expert = num_tokens // num_local_experts
    if tokens_per_expert % 128 != 0:
        raise SystemExit(
            f"tokens_per_expert={tokens_per_expert} must be a multiple of 128"
        )

    configure_backend_environment(backend_name)
    reload_megatron_transformer_engine_extension()

    global get_gpt_layer_with_transformer_engine_spec, build_module
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec as get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.transformer.spec_utils import build_module as build_module

    recipe_kind = nvfp4_autocast_recipe_kind_for_backend(backend_name)
    enable_fp4_config = backend_name != "te_bf16"
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    experts_module = build_tegrouped_mlp(
        num_local_experts, enable_fp4_config=enable_fp4_config
    )
    if amax_fuse_enabled:
        patch_tegrouped_linear_to_accept_input_row_amax(experts_module)
    tokens_per_expert_tensor = torch.full(
        (num_local_experts,), tokens_per_expert, device="cuda", dtype=torch.int64
    )

    def make_fixed_inputs():
        hidden_states = torch.randn(
            num_tokens,
            dims["hidden_size"],
            device="cuda",
            dtype=torch.bfloat16,
        )
        routing_probs = torch.ones(num_tokens, device="cuda", dtype=torch.bfloat16)
        return hidden_states, routing_probs

    disable_minite_weight_requantization(experts_module)
    if bench_mode not in SUPPORTED_BENCH_MODES:
        raise ValueError(f"unsupported bench_mode={bench_mode!r}")
    needs_grad = bench_mode in ("backward", "fwb")
    if needs_grad:
        experts_module.train()
    else:
        experts_module.eval()

    prime_hidden, prime_probs = make_fixed_inputs()
    set_grouped_linear_first_microbatch_flag(experts_module, True)

    def run_mlp_once(
        hidden_states: torch.Tensor,
        routing_probs: torch.Tensor,
        *,
        fc1_amax: Optional[torch.Tensor],
        fc2_amax: Optional[torch.Tensor],
        enable_grad: bool,
    ) -> torch.Tensor:
        grad_ctx = contextlib.nullcontext() if enable_grad else torch.no_grad()
        with grad_ctx, nvfp4_autocast_context(recipe_kind):
            if amax_fuse_enabled:
                assert fc1_amax is not None and fc2_amax is not None
                mlp_output = tegrouped_mlp_forward_with_prefused_row_amax(
                    experts_module,
                    hidden_states,
                    tokens_per_expert_tensor,
                    routing_probs,
                    fc1_amax,
                    fc2_amax,
                )
            else:
                mlp_output = experts_module(
                    hidden_states, tokens_per_expert_tensor, routing_probs
                )
            return unpack_tegrouped_mlp_output(mlp_output)

    # Priming microbatch: populate weight quant caches (and one bwd if needed).
    if amax_fuse_enabled:
        prime_fc1_amax, prime_fc2_amax = precompute_fc1_and_fc2_row_amax(
            experts_module,
            prime_hidden,
            tokens_per_expert_tensor,
            prime_probs,
            recipe_kind,
        )
    else:
        prime_fc1_amax = prime_fc2_amax = None

    if needs_grad:
        zero_expert_parameter_gradients(experts_module)
        prime_hidden_leaf = prime_hidden.detach().requires_grad_(True)
        prime_output = run_mlp_once(
            prime_hidden_leaf,
            prime_probs,
            fc1_amax=prime_fc1_amax,
            fc2_amax=prime_fc2_amax,
            enable_grad=True,
        )
        prime_output.float().sum().backward()
    else:
        _ = run_mlp_once(
            prime_hidden,
            prime_probs,
            fc1_amax=prime_fc1_amax,
            fc2_amax=prime_fc2_amax,
            enable_grad=False,
        )
    set_grouped_linear_first_microbatch_flag(experts_module, False)
    synchronize_cuda()

    # Fixed activations outside the timer — measure fwd / bwd / fwb only.
    timed_hidden, timed_probs = make_fixed_inputs()
    fc1_row_amax = fc2_row_amax = None
    if amax_fuse_enabled:
        fc1_row_amax, fc2_row_amax = precompute_fc1_and_fc2_row_amax(
            experts_module,
            timed_hidden,
            tokens_per_expert_tensor,
            timed_probs,
            recipe_kind,
        )
        synchronize_cuda()

    def run_forward_for_backward() -> torch.Tensor:
        """Build a fresh autograd graph; caller times ``.backward()`` separately."""
        zero_expert_parameter_gradients(experts_module)
        hidden_leaf = timed_hidden.detach().requires_grad_(True)
        return run_mlp_once(
            hidden_leaf,
            timed_probs,
            fc1_amax=fc1_row_amax,
            fc2_amax=fc2_row_amax,
            enable_grad=True,
        )

    def run_one_timed_iteration() -> None:
        if bench_mode == "fwb":
            output = run_forward_for_backward()
            output.float().sum().backward()
        elif bench_mode == "forward":
            _ = run_mlp_once(
                timed_hidden,
                timed_probs,
                fc1_amax=fc1_row_amax,
                fc2_amax=fc2_row_amax,
                enable_grad=False,
            )
        else:
            raise RuntimeError(f"run_one_timed_iteration does not handle {bench_mode}")

    for _ in range(warmup_iters):
        if bench_mode == "backward":
            output = run_forward_for_backward()
            synchronize_cuda()
            output.float().sum().backward()
        else:
            run_one_timed_iteration()

    elapsed_ms_samples: list[float] = []
    for _ in range(timed_iters):
        if bench_mode == "backward":
            # Forward builds the graph outside the timer; only bwd is measured.
            output = run_forward_for_backward()
            synchronize_cuda()
            start_time = time.perf_counter()
            output.float().sum().backward()
            synchronize_cuda()
        else:
            synchronize_cuda()
            start_time = time.perf_counter()
            run_one_timed_iteration()
            synchronize_cuda()
        elapsed_ms_samples.append((time.perf_counter() - start_time) * 1e3)

    del experts_module
    gc.collect()
    torch.cuda.empty_cache()

    return BenchTimingResult(
        backend=backend_name,
        bench_mode=bench_mode,
        amax_fuse_enabled=amax_fuse_enabled,
        sequence_length=sequence_length,
        num_local_experts=num_local_experts,
        num_tokens=num_tokens,
        tokens_per_expert=tokens_per_expert,
        median_ms=statistics.median(elapsed_ms_samples),
        mean_ms=statistics.fmean(elapsed_ms_samples),
        min_ms=min(elapsed_ms_samples),
        max_ms=max(elapsed_ms_samples),
    )


def format_result_column_label(
    backend_name: str, amax_fuse_enabled: bool, bench_mode: str = "forward"
) -> str:
    label = f"{backend_name}+fuse" if amax_fuse_enabled else backend_name
    mode_tag = {
        "forward": "fwd",
        "backward": "bwd",
        "fwb": "fwb",
    }.get(bench_mode, bench_mode)
    if mode_tag != "fwd":
        label = f"{label}/{mode_tag}"
    return label


def print_timing_summary_tables(
    results: Sequence[BenchTimingResult], sequence_lengths: Sequence[int]
) -> None:
    column_keys: list[tuple[str, bool, str]] = []
    seen_keys = set()
    for result in results:
        key = (result.backend, result.amax_fuse_enabled, result.bench_mode)
        if key not in seen_keys:
            seen_keys.add(key)
            column_keys.append(key)

    column_labels = [
        format_result_column_label(backend, fuse, mode)
        for backend, fuse, mode in column_keys
    ]
    column_width = max(14, max((len(label) for label in column_labels), default=14))

    print("\n" + "=" * 96)
    print(
        f"{'seq':>7}",
        *[f"{label:>{column_width}}" for label in column_labels],
        f"{'fastest':>14}",
    )
    print("-" * 96)

    result_by_key = {
        (r.backend, r.amax_fuse_enabled, r.bench_mode, r.sequence_length): r
        for r in results
    }
    for sequence_length in sequence_lengths:
        row_cells = [f"{sequence_length:7d}"]
        median_by_label: dict[str, float] = {}
        for backend, fuse, mode in column_keys:
            hit = result_by_key.get((backend, fuse, mode, sequence_length))
            label = format_result_column_label(backend, fuse, mode)
            if hit is None:
                row_cells.append(f"{'—':>{column_width}}")
            else:
                row_cells.append(f"{hit.median_ms:{column_width}.3f}")
                median_by_label[label] = hit.median_ms
        fastest_label = (
            min(median_by_label, key=median_by_label.get) if median_by_label else "—"
        )
        row_cells.append(f"{fastest_label:>14}")
        print(*row_cells)

    # Speedup vs te_bf16 within the same bench_mode.
    for mode in sorted({r.bench_mode for r in results}):
        bf16_hits = [
            r
            for r in results
            if r.backend == "te_bf16"
            and not r.amax_fuse_enabled
            and r.bench_mode == mode
        ]
        if not bf16_hits:
            continue
        mode_columns = [
            (b, f, m) for b, f, m in column_keys if m == mode and not (b == "te_bf16" and not f)
        ]
        if not mode_columns:
            continue
        print(f"\nvs TE BF16 [{mode}] (median speedup, >1 = faster than BF16)")
        print(
            f"{'seq':>7}",
            *[
                f"{format_result_column_label(b, f, m):>{column_width}}"
                for b, f, m in mode_columns
            ],
        )
        for sequence_length in sequence_lengths:
            bf16_result = result_by_key.get(("te_bf16", False, mode, sequence_length))
            if bf16_result is None or bf16_result.median_ms <= 0:
                continue
            row_cells = [f"{sequence_length:7d}"]
            for backend, fuse, col_mode in mode_columns:
                hit = result_by_key.get((backend, fuse, col_mode, sequence_length))
                if hit is None:
                    row_cells.append(f"{'—':>{column_width}}")
                else:
                    speedup = bf16_result.median_ms / hit.median_ms
                    row_cells.append(f"{speedup:{column_width}.2f}x")
            print(*row_cells)
    print("=" * 96)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        default=IMPORT_TIME_BACKEND,
        choices=list(SUPPORTED_BACKENDS),
        help="Import-time backend (Megatron TE replace); usually match --backends",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(SUPPORTED_BACKENDS),
        choices=list(SUPPORTED_BACKENDS),
        help="Backends to measure in this process (prefer one per process)",
    )
    parser.add_argument(
        "--amax-fuse",
        choices=["off", "on", "both"],
        default="off",
        help="Amax-fuse simulation feature (default off)",
    )
    parser.add_argument(
        "--bench-mode",
        choices=["forward", "fwd", "backward", "bwd", "fwb", "both"],
        default="forward",
        help=(
            "forward|fwd: timed forward; "
            "backward|bwd: timed backward only; "
            "fwb: timed forward+backward; "
            "both: run all three (default forward)"
        ),
    )
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=[16384, 32768, 65536, 131072],
    )
    parser.add_argument("--tp-shard", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", type=str, default="")
    args = parser.parse_args()

    if QWEN3_30B_A3B_MODEL_DIMS["num_experts_global"] % args.tp_shard != 0:
        raise SystemExit("tp_shard must divide 128")

    try:
        Utils.destroy_model_parallel()
    except Exception:
        pass
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(42)

    amax_fuse_enabled_flags = expand_amax_fuse_cli_mode_to_bools(args.amax_fuse)
    bench_modes = expand_bench_mode_cli_to_modes(args.bench_mode)
    dims = QWEN3_30B_A3B_MODEL_DIMS

    print("=" * 96)
    print("TEGroupedMLP microbench (fc1+GLU+fc2, pre-permuted)")
    print(
        f"preset=qwen3-30b-a3b  h={dims['hidden_size']}  "
        f"ffn={dims['moe_ffn_hidden_size']}  G_global={dims['num_experts_global']}  "
        f"tp_shard={args.tp_shard}  "
        f"G_local={dims['num_experts_global'] // args.tp_shard}  "
        f"top_k={dims['top_k']}"
    )
    print(
        f"backends={args.backends}  bench_mode={args.bench_mode}  "
        f"amax_fuse={args.amax_fuse}  seq_lens={args.seq_lens}"
    )
    print(
        f"warmup={args.warmup}  iters={args.iters}  seed={args.seed}  "
        f"device={torch.cuda.get_device_name(0)}  torch={torch.__version__}  "
        f"te={get_transformer_engine_version()}"
    )
    print(
        f"MINITE_NVFP4_PATH={os.environ.get('MINITE_NVFP4_PATH', '')!r}  "
        f"NVTE_NVFP4_DISABLE_2D_QUANTIZATION="
        f"{os.environ.get('NVTE_NVFP4_DISABLE_2D_QUANTIZATION', '')!r}"
    )
    print("=" * 96)

    timing_results: list[BenchTimingResult] = []
    skipped_messages: list[str] = []

    for sequence_length in args.seq_lens:
        for backend_name in args.backends:
            for amax_fuse_enabled in amax_fuse_enabled_flags:
                if (
                    amax_fuse_enabled
                    and backend_name not in BACKENDS_SUPPORTING_AMAX_FUSE
                ):
                    warning = (
                        f"skip amax_fuse=on for backend={backend_name} "
                        f"(supported: {sorted(BACKENDS_SUPPORTING_AMAX_FUSE)})"
                    )
                    if warning not in skipped_messages:
                        skipped_messages.append(warning)
                        print(f"\nWARN  {warning}", flush=True)
                    continue

                for bench_mode in bench_modes:
                    column_label = format_result_column_label(
                        backend_name, amax_fuse_enabled, bench_mode
                    )
                    print(
                        f"\n>>> {column_label} seq={sequence_length} ...",
                        flush=True,
                    )
                    try:
                        timing_result = benchmark_one_backend_sequence_config(
                            backend_name,
                            sequence_length,
                            bench_mode=bench_mode,
                            amax_fuse_enabled=amax_fuse_enabled,
                            tensor_parallel_shard=args.tp_shard,
                            warmup_iters=args.warmup,
                            timed_iters=args.iters,
                            random_seed=args.seed,
                        )
                    except Exception as exc:
                        print(f"FAIL  {type(exc).__name__}: {exc}")
                        continue
                    timing_results.append(timing_result)
                    print(
                        f"OK  median={timing_result.median_ms:.3f} ms  "
                        f"mean={timing_result.mean_ms:.3f}  "
                        f"tokens={timing_result.num_tokens}  "
                        f"G={timing_result.num_local_experts}  "
                        f"m={timing_result.tokens_per_expert}  "
                        f"mode={timing_result.bench_mode}  "
                        f"amax_fuse={timing_result.amax_fuse_enabled}"
                    )

    print_timing_summary_tables(timing_results, args.seq_lens)

    payload = {
        "meta": {
            "preset": "qwen3-30b-a3b",
            "model": QWEN3_30B_A3B_MODEL_DIMS,
            "tp_shard": args.tp_shard,
            "backends": list(args.backends),
            "bench_mode": args.bench_mode,
            "amax_fuse": args.amax_fuse,
            "seq_lens": list(args.seq_lens),
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformer_engine": get_transformer_engine_version(),
            "env": {
                "MINITE_NVFP4_PATH": os.environ.get("MINITE_NVFP4_PATH"),
                "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": os.environ.get(
                    "NVTE_NVFP4_DISABLE_2D_QUANTIZATION"
                ),
                "SHOULD_REPLACE_TE_GROUPLINEAR": os.environ.get(
                    "SHOULD_REPLACE_TE_GROUPLINEAR"
                ),
                "QAT_PARAMS": os.environ.get("QAT_PARAMS"),
            },
            "skipped": skipped_messages,
            "note": (
                "TEGroupedMLP only; input pre-permuted; no router/dispatch; "
                "bench_mode=forward times forward; "
                "bench_mode=backward times backward only (forward outside timer); "
                "bench_mode=fwb times forward+backward (no optimizer); "
                "amax_fuse=on precomputes input_row_amax outside timer (skip-K1 sim)"
            ),
        },
        "results": [asdict(result) for result in timing_results],
    }
    output_json_path = (
        Path(args.out_json)
        if args.out_json
        else Path("profiles/tegrouped_mlp.json")
    )
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output_json_path}")


if __name__ == "__main__":
    main()
