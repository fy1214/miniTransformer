"""Megatron collector for logpdiff: capture one MoE layer's expert fc1/fc2
kernel-boundary tensors plus router/dispatch/swiglu boundary tensors, and
write them as schema TensorRecords into a logpdiff run directory.

Two capture paths are combined:

1. MiniTE logpdiff sink (provenance=actual_kernel). The target layer's minite
   GroupedLinear modules (experts.linear_fc1 / experts.linear_fc2) are wrapped
   in ``logpdiff_scope`` via pre/post forward hooks, so the sink records real
   quantize/GEMM kernel-boundary tensors under
   ``<run_dir>/megatron/<module_name>/*.pt``. After the run, the sink's raw
   index is translated into TensorRecords appended to ``megatron/index.jsonl``;
   payload files are referenced in place (never copied). The raw sink index is
   preserved as ``megatron/sink_index.jsonl``.

   sink phase -> tensor_point (role taken from linear_role; fc1 shown):
     input / tensor "x"        -> fc1_raw_input
     input / tensor "m_splits" -> (not a tensor_point; kept in the raw sink
                                  index only, counted as skipped-metadata)
     x_qdata, w_qdata          -> fc1_qdata   (payload tensors)
                                  fc1_scales  (names containing scale/offset,
                                               e.g. scale_flat, global_scale,
                                               row_offsets, scale_offsets)
     dequant_view              -> fc1_qdata / fc1_scales (same name rule)
     gemm_output               -> fc1_output

2. Module hooks (provenance=hook_boundary): layer_input, dispatch_input,
   router_topk_ids, router_topk_weights, swiglu_output, expert_combined_output.
   token_logprobs is computed from the actual output logits
   (provenance=derived_metric). router_logits is NOT emitted: Megatron's
   TopKRouter never exposes pre-softmax logits outside the module, and
   fabricating internal quantities is disallowed (mark replayed, or skip).

P0 scope: TP=EP=PP=1, single process. Launch with
``torchrun --nproc_per_node=1`` (WORLD_SIZE must be 1).

Usage:
    SHOULD_REPLACE_TE_GROUPLINEAR=1 QAT_PARAMS=8 \
    NVTE_NVFP4_DISABLE_RHT=0 NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0 \
    NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1 \
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
        dump_expert_fc1_and_fc2_megatron.py \
        --model-path /path/to/hf/Qwen3-30B-A3B \
        --ref-load /path/to/Qwen3-30B-A3B_torch_dist/ \
        --input-file test_inputs_long.jsonl \
        --output-dir runs/ --layer-idx 0

    # join an existing run (writes <run>/megatron/; inputs.jsonl must match
    # byte-for-byte; the run's config/manifest are merged, not overwritten):
    torchrun --nproc_per_node=1 dump_expert_fc1_and_fc2_megatron.py \
        --run-dir runs/<run-id> --model-path ... --ref-load ... --input-file ...

    # equivalent via config file (CLI flags override config values):
    torchrun --nproc_per_node=1 dump_expert_fc1_and_fc2_megatron.py \
        --config capture.megatron.json [--layer-idx 3]

Config JSON keys: model_path, ref_load, layer_idx, input_file, output_dir,
run_dir, run_id, ep_size, force_routing.

Old CLI flags (--input-file/--output-dir/--ep-size/--model-path/--layer-idx/
--force-routing) are unchanged; --ref-load-path remains as a deprecated alias
of --ref-load. The previously hardcoded MODEL_PATH/REF_LOAD are gone:
--model-path and --ref-load are now required (CLI or --config).
--output-dir is now the BASE directory under which ``<run-id>/`` is created
(old default wrote a bare .pt into the output dir; that format is gone);
--run-dir joins an existing run instead of creating one.

Heavy dependencies (torch, torch.distributed, megatron, minite) are imported
inside functions, so ``--help`` and the sink-index translation work without
them installed.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from logpdiff import schema
from logpdiff.writer import RunWriter, assert_internal_payload, create_run_dir

FRAMEWORK = "megatron"
SINK_RAW_INDEX_NAME = "sink_index.jsonl"

# Megatron/slime source checkouts on the capture machine (old behaviour).
MEGATRON_SRC = "/root/Megatron-LM"
MINITE_SRC = "/root/minite"

_CONFIG_KEYS = (
    "model_path",
    "ref_load",
    "layer_idx",
    "input_file",
    "output_dir",
    "run_dir",
    "run_id",
    "ep_size",
    "force_routing",
)

_DEFAULTS = {
    "model_path": None,
    "ref_load": None,
    "layer_idx": 0,
    "input_file": None,
    "output_dir": "runs",
    "run_dir": None,
    "run_id": None,
    "ep_size": 1,
    "force_routing": None,
}

# Hook captures saved per sample, in this fixed order.
# provenance defaults to hook_boundary; token_logprobs is computed from the
# actual logits payload and is therefore a derived metric.
_HOOK_CAPTURE_ORDER = (
    ("layer_input", schema.LAYER_INPUT),
    ("dispatch_input", schema.DISPATCH_INPUT),
    ("router_topk_ids", schema.ROUTER_TOPK_IDS),
    ("router_topk_weights", schema.ROUTER_TOPK_WEIGHTS),
    ("swiglu_output", schema.SWIGLU_OUTPUT),
    ("expert_combined_output", schema.EXPERT_COMBINED_OUTPUT),
    ("token_logprobs", schema.TOKEN_LOGPROBS),
)
_HOOK_DERIVED = frozenset({"token_logprobs"})


# ---------------------------------------------------------------------------
# CLI / config resolution (stdlib only)
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None,
                        help="JSON config file; CLI flags override its values")
    parser.add_argument("--model-path", default=None,
                        help="HF model path (required; was hardcoded before)")
    parser.add_argument("--ref-load", default=None,
                        help="Megatron torch_dist checkpoint dir (required; "
                             "was hardcoded as REF_LOAD before)")
    parser.add_argument("--ref-load-path", dest="ref_load", default=None,
                        help="deprecated alias of --ref-load")
    parser.add_argument("--layer-idx", type=int, default=None,
                        help="transformer layer index to capture (default 0)")
    parser.add_argument("--input-file", default=None,
                        help="input JSONL with {\"name\", \"input_ids\"} records (required)")
    parser.add_argument("--output-dir", default=None,
                        help="base directory; the run is created at <output-dir>/<run-id>/ "
                             "(default: runs)")
    parser.add_argument("--run-dir", default=None,
                        help="join an existing run directory instead of creating one "
                             "(writes its megatron/ subdir; mutually exclusive with --run-id)")
    parser.add_argument("--run-id", default=None,
                        help="run id (default: UTC timestamp + short uuid)")
    parser.add_argument("--ep-size", type=int, default=None,
                        help="expert parallel size; P0 supports only 1")
    parser.add_argument("--force-routing", default=None,
                        help="Path to sglang dump .pt to force router output "
                             "(topk_ids/topk_weights)")
    return parser


def resolve_config(args, parser) -> dict:
    """Merge --config JSON with CLI overrides; validate required values."""
    cfg = dict(_DEFAULTS)
    if args.config:
        try:
            with open(args.config, encoding="utf-8") as f:
                file_cfg = json.load(f)
        except OSError as e:
            parser.error(f"cannot read --config {args.config}: {e}")
        if not isinstance(file_cfg, dict):
            parser.error(f"--config {args.config} must contain a JSON object")
        unknown = sorted(set(file_cfg) - set(_CONFIG_KEYS))
        if unknown:
            parser.error(
                f"--config {args.config} has unknown keys {unknown}; "
                f"supported keys: {sorted(_CONFIG_KEYS)}"
            )
        cfg.update(file_cfg)
    for key in _CONFIG_KEYS:
        cli_value = getattr(args, key, None)
        if cli_value is not None:
            cfg[key] = cli_value

    for required in ("model_path", "ref_load", "input_file"):
        if not cfg.get(required):
            parser.error(
                f"--{required.replace('_', '-')} is required "
                f"(no hardcoded default anymore; pass it via CLI or --config)"
            )
    if cfg["run_dir"] and cfg["run_id"]:
        parser.error("--run-id only applies when creating a run (with --output-dir), "
                     "not when joining one via --run-dir")
    try:
        cfg["ep_size"] = int(cfg["ep_size"])
        cfg["layer_idx"] = int(cfg["layer_idx"])
    except (TypeError, ValueError):
        parser.error("ep_size and layer_idx must be integers")
    try:
        schema.check_single_device_parallel(tp=1, ep=cfg["ep_size"], pp=1)
    except ValueError as e:
        parser.error(str(e))
    return cfg


def load_inputs(path):
    """Load input JSONL records; every record needs ``name`` and ``input_ids``."""
    inputs = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if "name" not in rec or "input_ids" not in rec:
                raise ValueError(
                    f"{path}:{lineno}: input record must have 'name' and 'input_ids', "
                    f"got keys {sorted(rec)}"
                )
            inputs.append(rec)
    if not inputs:
        raise ValueError(f"{path}: no input records found")
    return inputs


def _get_global_rank() -> int:
    """Rank from torch.distributed when initialized, else 0."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# run directory setup / top-level files (torch-free)
# ---------------------------------------------------------------------------


def _setup_run(cfg):
    """Create a new run under --output-dir, or join an existing --run-dir.

    Joining reuses the run directory (this side writes its megatron/ subdir);
    an existing megatron index means a megatron capture is already present
    and is never mixed into. Returns (RunWriter, joined_existing).
    """
    common = dict(
        model_path=cfg["model_path"],
        seed=1234,  # megatron default; no --seed flag is passed in setup_megatron_argv
        backend="megatron",
        tp=1,
        ep=cfg["ep_size"],
        pp=1,
        force_routing=bool(cfg["force_routing"]),
    )
    if cfg["run_dir"]:
        run_dir = Path(cfg["run_dir"]).resolve()
        if not run_dir.is_dir():
            raise SystemExit(
                f"--run-dir {run_dir} does not exist; "
                "to create a new run use --output-dir instead"
            )
        (run_dir / FRAMEWORK / "tensors").mkdir(parents=True, exist_ok=True)
        for stale in ("index.jsonl", SINK_RAW_INDEX_NAME):
            if (run_dir / FRAMEWORK / stale).exists():
                raise SystemExit(
                    f"{run_dir / FRAMEWORK / stale} already exists; refusing to "
                    "mix a second megatron capture into the same run"
                )
        return RunWriter(run_dir=run_dir, run_id=run_dir.name, **common), True
    return create_run_dir(cfg["output_dir"], run_id=cfg["run_id"], **common), False


def _write_or_verify_inputs(run_writer, test_inputs):
    """Write inputs.jsonl in the canonical record schema (same shape the
    sglang collector writes). When joining a run that already has one, the
    existing content must be byte-identical (no silent input mismatch)."""
    records = [
        {
            "sample_id": str(inp["name"]),
            "input_ids": list(inp["input_ids"]),
            "num_tokens": len(inp["input_ids"]),
        }
        for inp in test_inputs
    ]
    path = run_writer.run_dir / "inputs.jsonl"
    if path.exists():
        existing = path.read_bytes()
        canonical = "".join(
            json.dumps(rec, ensure_ascii=False) + "\n" for rec in records
        ).encode("utf-8")
        if existing != canonical:
            raise SystemExit(
                f"{path} already exists and does not match --input-file; "
                "refusing to mix different inputs into one run"
            )
        print(f"inputs.jsonl already present and identical ({len(records)} samples)")
        return records
    run_writer.write_inputs(records)
    print(f"Wrote {len(records)} input samples to {path}")
    return records


def _merge_write_config(run_writer, megatron_section):
    """Write config.resolved.json under the "megatron" section, preserving
    other frameworks' sections when joining an existing run."""
    path = run_writer.run_dir / "config.resolved.json"
    merged = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            merged = json.load(f)
        if not isinstance(merged, dict):
            raise SystemExit(f"{path} exists but is not a JSON object; refusing to merge")
    merged.setdefault("run_id", run_writer.run_id)
    merged[FRAMEWORK] = megatron_section
    run_writer.write_config_resolved(merged)
    print(f"Wrote {path}")


def _finalize_manifest(run_writer):
    path = run_writer.run_dir / "manifest.json"
    if path.exists():
        print(f"{path} already exists (joined run); leaving it untouched")
        return
    run_writer.finalize_manifest()
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# sink raw index -> TensorRecord translation (torch-free, unit-testable)
# ---------------------------------------------------------------------------


def _quant_component_point(prefix: str, tensor_name: str) -> str:
    """Classify a quantized-payload component tensor into qdata vs scales.

    The CUTLASS NVFP4 path emits extra_info tensors named data / scale_flat /
    global_scale / row_offsets / scale_offsets; scale-side metadata (anything
    containing "scale" or "offset") goes to <prefix>_scales, the packed data
    itself to <prefix>_qdata. The same rule covers dequant_view tensors
    (x_*/w_* names).
    """
    name = tensor_name.lower()
    if "scale" in name or "offset" in name:
        return f"{prefix}_scales"
    return f"{prefix}_qdata"


def sink_record_to_tensor_record(
    sink_rec: dict,
    *,
    run_id: str,
    sample_id,
    global_rank: int,
    expected_layer_id=None,
) -> "schema.TensorRecord | None":
    """Translate one raw sink index record into a schema TensorRecord.

    Returns None for records that carry no tensor_point payload (currently
    only the "m_splits" grouping metadata of the input phase); those stay
    available in the preserved raw sink index. Raises ValueError on malformed
    or unexpected records -- nothing is silently dropped or truncated.
    """
    required = (
        "module_name", "layer_id", "linear_role", "phase", "seq",
        "tensor_name", "shape", "dtype", "path", "provenance",
    )
    missing = [k for k in required if k not in sink_rec]
    if missing:
        raise ValueError(f"sink record missing fields {missing}: {sink_rec!r}")

    role = sink_rec["linear_role"]
    if role not in ("fc1", "fc2"):
        raise ValueError(f"unknown linear_role {role!r} in sink record: {sink_rec!r}")
    layer_id = int(sink_rec["layer_id"])
    if expected_layer_id is not None and layer_id != expected_layer_id:
        raise ValueError(
            f"sink record layer_id {layer_id} != captured layer {expected_layer_id}: "
            f"{sink_rec!r}"
        )

    phase = sink_rec["phase"]
    tensor_name = sink_rec["tensor_name"]
    if phase == "input":
        if tensor_name == "x":
            tensor_point = f"{role}_raw_input"
        elif tensor_name == "m_splits":
            return None  # per-expert row grouping metadata; raw index keeps it
        else:
            raise ValueError(
                f"unknown input-phase tensor {tensor_name!r}: {sink_rec!r}"
            )
    elif phase in ("x_qdata", "w_qdata", "dequant_view"):
        tensor_point = _quant_component_point(role, tensor_name)
    elif phase == "gemm_output":
        tensor_point = f"{role}_output"
    else:
        raise ValueError(f"unknown sink phase {phase!r}: {sink_rec!r}")

    provenance = schema.validate_provenance(sink_rec["provenance"])
    rel = str(sink_rec["path"]).replace(os.sep, "/")
    return schema.TensorRecord(
        run_id=run_id,
        framework=FRAMEWORK,
        sample_id=sample_id,
        layer_id=layer_id,
        tensor_point=tensor_point,
        shape=tuple(int(d) for d in sink_rec["shape"]),
        dtype=str(sink_rec["dtype"]),
        provenance=provenance,
        payload_path=f"{FRAMEWORK}/{rel}",
        global_rank=global_rank,
    )


def translate_and_append_sink_records(
    run_writer, sink_records, *, sample_id, global_rank: int, expected_layer_id
):
    """Translate raw sink records and append them to megatron/index.jsonl.

    Payload files are referenced in place; every referenced file must exist
    inside the run directory (FileNotFoundError otherwise -- no silent drops).

    Returns (written, skipped_metadata) counts.
    """
    written = 0
    skipped = 0
    for rec in sink_records:
        tensor_record = sink_record_to_tensor_record(
            rec,
            run_id=run_writer.run_id,
            sample_id=sample_id,
            global_rank=global_rank,
            expected_layer_id=expected_layer_id,
        )
        if tensor_record is None:
            skipped += 1
            continue
        payload = assert_internal_payload(tensor_record.payload_path, run_writer.run_dir)
        if not payload.is_file():
            raise FileNotFoundError(
                f"sink payload {payload} referenced by record {rec!r} does not exist"
            )
        run_writer.append_record(FRAMEWORK, tensor_record)
        written += 1
    return written, skipped


def _count_lines(path: Path) -> int:
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


# ---------------------------------------------------------------------------
# Megatron bootstrap (heavy imports inside)
# ---------------------------------------------------------------------------


def setup_megatron_argv(cfg):
    ep = cfg["ep_size"]
    sys.argv = [
        "dump_expert_fc1_and_fc2_megatron.py",
        "--disable-bias-linear",
        "--qk-layernorm",
        "--group-query-attention",
        "--num-attention-heads", "32",
        "--num-query-groups", "4",
        "--kv-channels", "128",
        "--num-layers", "48",
        "--hidden-size", "2048",
        "--ffn-hidden-size", "6144",
        "--normalization", "RMSNorm",
        "--position-embedding-type", "rope",
        "--norm-epsilon", "1e-6",
        "--rotary-percent", "1.0",
        "--swiglu",
        "--untie-embeddings-and-output-weights",
        "--vocab-size", "151936",
        "--rotary-base", "1000000",
        "--moe-ffn-hidden-size", "768",
        "--moe-router-score-function", "softmax",
        "--moe-token-dispatcher-type", "alltoall",
        "--moe-router-topk", "8",
        "--num-experts", "128",
        "--moe-grouped-gemm",
        "--moe-token-drop-policy", "probs",
        "--moe-router-dtype", "fp32",
        "--moe-permute-fusion",
        "--moe-aux-loss-coeff", "0",
        "--moe-layer-freq", "[" + ",".join(["1"] * 48) + "]",
        "--tensor-model-parallel-size", "1",
        "--pipeline-model-parallel-size", "1",
        "--context-parallel-size", "1",
        "--expert-model-parallel-size", str(ep),
        "--expert-tensor-parallel-size", "1",
        "--sequence-parallel",
        "--transformer-impl", "transformer_engine",
        "--bf16",
        "--fp4-format", "e2m1",
        "--fp4-recipe", "nvfp4",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend", "flash",
        "--micro-batch-size", "1",
        "--global-batch-size", str(ep),
        "--seq-length", "4096",
        "--lr", "1e-6",
        "--lr-decay-style", "constant",
        "--weight-decay", "0.1",
        "--adam-beta1", "0.9",
        "--adam-beta2", "0.98",
        "--recompute-granularity", "full",
        "--recompute-method", "uniform",
        "--recompute-num-layers", "1",
        "--tokenizer-type", "NullTokenizer",
        "--hf-checkpoint", cfg["model_path"],
        "--ref-load", cfg["ref_load"],
        "--load", cfg["ref_load"],
        "--lr-warmup-iters", "0",
        "--lr-decay-iters", "10",
        "--train-iters", "10",
    ]


def _init_megatron(cfg, rank, local_rank):
    """Parse megatron args, init slime, load the model. Returns (raw_model, args)."""
    import torch  # noqa: F401
    import torch.distributed as dist

    setup_megatron_argv(cfg)

    sys.path.insert(0, MEGATRON_SRC)
    sys.path.insert(0, MINITE_SRC)

    from megatron.training.arguments import parse_args as megatron_parse_args
    from megatron.training.arguments import validate_args as megatron_validate_args

    args = megatron_parse_args(extra_args_provider=None, ignore_unknown_args=True)

    args.hf_checkpoint = cfg["model_path"]
    args.ref_load = cfg["ref_load"]
    args.load = cfg["ref_load"]

    from slime.backends.megatron_utils.arguments import set_default_megatron_args

    args = set_default_megatron_args(args)

    missing_defaults = {
        'virtual_pipeline_model_parallel_size': None,
        'pipeline_model_parallel_comm_backend': None,
        'hierarchical_context_parallel_sizes': None,
        'num_distributed_optimizer_instances': 1,
        'distributed_timeout_minutes': 10,
        'nccl_communicator_config_path': None,
        'use_tp_pp_dp_mapping': False,
        'enable_gloo_process_groups': False,
        'enable_experimental': False,
        'data_parallel_random_init': False,
        'te_rng_tracker': False,
        'inference_rng_tracker': False,
        'rampup_batch_size': None,
        'decrease_batch_size_if_needed': False,
        'deterministic_mode': False,
        'tp_comm_overlap': False,
        'custom_megatron_init_path': None,
        'seed': 1234,
        'data_parallel_size': 1,
        'dist_ckpt_save_pre_mcore_014': True,
        'megatron_to_hf_mode': None,
        'custom_model_provider_path': None,
        'moe_use_legacy_grouped_gemm': False,
        'multi_latent_attention': False,
        'fp8_param_gather': False,
        'only_train_params_name_list': None,
        'freeze_params_name_list': ['.*'],
        'mtp_num_layers': None,
        'moe_use_upcycling': False,
        'pretrained_checkpoint': None,
        'variable_seq_lengths': True,
        'decoder_first_pipeline_num_layers': None,
        'decoder_last_pipeline_num_layers': None,
        'num_rollout': 1,
        'rollout_batch_size': 1,
        'n_samples_per_prompt': 1,
    }
    for k, v in missing_defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    megatron_validate_args(args)

    if rank == 0:
        print(f"Args parsed. TP={args.tensor_model_parallel_size}, "
              f"EP={args.expert_model_parallel_size}, PP={args.pipeline_model_parallel_size}")

    from slime.backends.megatron_utils.initialize import init

    init(args)

    if rank == 0:
        print("Megatron initialized. Loading model...")

    from slime.backends.megatron_utils.model import initialize_model_and_optimizer
    from megatron.training import get_args

    t0 = time.time()
    model_list, optimizer, scheduler, step = initialize_model_and_optimizer(get_args(), "actor")
    model = model_list[0]
    model.eval()

    if rank == 0:
        print(f"Model loaded in {time.time() - t0:.1f}s")

    raw_model = model
    while hasattr(raw_model, 'module'):
        raw_model = raw_model.module
    return raw_model, args


# ---------------------------------------------------------------------------
# capture driver (heavy imports inside)
# ---------------------------------------------------------------------------


def run_capture(cfg, run_writer, test_inputs):
    """Run the capture. On rank 0 returns a summary dict; other ranks return None."""
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size != 1:
        raise RuntimeError(
            f"P0 supports single-process capture only (WORLD_SIZE={world_size}); "
            "launch with `torchrun --nproc_per_node=1`. "
            "Multi-rank capture is planned for P1."
        )

    if rank == 0:
        print(f"World size: {world_size}, EP: {cfg['ep_size']}, rank: {rank}")
        print(f"SHOULD_REPLACE_TE_GROUPLINEAR={os.environ.get('SHOULD_REPLACE_TE_GROUPLINEAR', '')}")
        print(f"QAT_PARAMS={os.environ.get('QAT_PARAMS', '')}")

    torch.cuda.set_device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    raw_model, megatron_args = _init_megatron(cfg, rank, local_rank)

    layer_idx = cfg["layer_idx"]
    target_layer = raw_model.decoder.layers[layer_idx]
    moe_layer = target_layer.mlp
    experts_module = moe_layer.experts if hasattr(moe_layer, 'experts') else None

    if experts_module is None:
        dist.destroy_process_group()
        raise RuntimeError(f"Cannot find experts in layer {layer_idx}!")

    linear_fc1 = experts_module.linear_fc1
    linear_fc2 = experts_module.linear_fc2

    if rank == 0:
        print(f"\nLayer {layer_idx} structure:")
        for name, child in target_layer.named_children():
            print(f"  {name}: {type(child).__name__}")
        print(f"Experts module: {type(experts_module).__name__}")
        print(f"linear_fc1: {type(linear_fc1).__name__}")
        if hasattr(linear_fc1, 'num_gemms'):
            print(f"  num_gemms={linear_fc1.num_gemms}")
        print(f"linear_fc2: {type(linear_fc2).__name__}")

    # -- minite logpdiff sink scopes ----------------------------------------
    # The sink reads MINITE_LOGPDIFF_ENABLE / MINITE_LOGPDIFF_OUTPUT_DIR at
    # import time; main() sets both before this function imports minite.
    from minite.module._logpdiff_sink import logpdiff_scope

    fc1_module_name = f"decoder.layers.{layer_idx}.mlp.experts.linear_fc1"
    fc2_module_name = f"decoder.layers.{layer_idx}.mlp.experts.linear_fc2"

    captured = {}

    def _enter_scope(module, module_name, role):
        ctx = logpdiff_scope(module_name=module_name, layer_id=layer_idx, linear_role=role)
        ctx.__enter__()
        module._logpdiff_scope_ctx = ctx

    def _exit_scope(module):
        ctx = getattr(module, "_logpdiff_scope_ctx", None)
        if ctx is not None:
            module._logpdiff_scope_ctx = None
            ctx.__exit__(None, None, None)

    def fc1_pre_hook(module, args_tuple):
        _enter_scope(module, fc1_module_name, "fc1")

    def fc1_post_hook(module, args_tuple, output):
        _exit_scope(module)

    def fc2_pre_hook(module, args_tuple):
        _enter_scope(module, fc2_module_name, "fc2")
        inp = args_tuple[0]
        if isinstance(inp, torch.Tensor):
            # post-SwiGLU (probs-weighted) input to linear_fc2, before QDQ
            captured["swiglu_output"] = inp.detach().clone()

    def fc2_post_hook(module, args_tuple, output):
        _exit_scope(module)

    def moe_pre_hook(module, args_tuple):
        x = args_tuple[0]
        if isinstance(x, (list, tuple)):
            x = x[0]
        if isinstance(x, torch.Tensor):
            captured["layer_input"] = x.detach().clone()

    def moe_hook(module, args_tuple, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            captured["expert_combined_output"] = out.detach().clone()

    def dispatcher_pre_hook(module, args_tuple):
        x = args_tuple[0]
        if isinstance(x, torch.Tensor):
            captured["dispatch_input"] = x.detach().clone()

    # Load forced routing from sglang dump if specified
    forced_routing = {}
    if cfg["force_routing"]:
        if rank == 0:
            print(f"\nLoading forced routing from: {cfg['force_routing']}")
        fr_data = torch.load(cfg["force_routing"], map_location="cpu", weights_only=False)
        for sample_name, sample_data in fr_data.items():
            fr_topk_ids = sample_data.get("topk_ids")
            fr_topk_weights = sample_data.get("topk_weights")
            if fr_topk_ids is not None and fr_topk_weights is not None:
                forced_routing[sample_name] = {
                    "topk_ids": fr_topk_ids,
                    "topk_weights": fr_topk_weights,
                }
        if rank == 0:
            print(f"  Loaded routing for {len(forced_routing)} samples: "
                  f"{list(forced_routing.keys())}")

    current_sample_name = [None]

    def router_hook(module, args_tuple, output):
        # Megatron router returns (probs [T,E] dense float, routing_map [T,E] bool).
        # Pre-softmax router logits never leave the TopKRouter module, so
        # router_logits is intentionally not captured (no fabricated data).
        if isinstance(output, tuple) and len(output) >= 2:
            probs_dense, routing_map = output[0], output[1]
            if routing_map.dtype == torch.bool:
                k = routing_map.sum(dim=-1).max().item()
                topk_w, topk_i = torch.topk(probs_dense, k=int(k), dim=-1)
                captured["router_topk_weights"] = topk_w.detach().cpu().clone()
                captured["router_topk_ids"] = topk_i.detach().cpu().clone()
            else:
                captured["router_topk_weights"] = probs_dense.detach().cpu().clone()
                captured["router_topk_ids"] = routing_map.detach().cpu().clone()

        sample_key = current_sample_name[0]
        if sample_key and sample_key in forced_routing:
            fr = forced_routing[sample_key]
            device = output[0].device if isinstance(output, tuple) else output.device
            dtype = output[0].dtype if isinstance(output, tuple) else output.dtype
            num_experts = output[0].shape[1] if isinstance(output, tuple) else 128

            topk_ids = fr["topk_ids"].to(device=device).long()
            topk_weights = fr["topk_weights"].to(device=device, dtype=dtype)
            num_tokens = topk_ids.shape[0]

            forced_probs = torch.zeros(num_tokens, num_experts, device=device, dtype=dtype)
            forced_probs.scatter_(1, topk_ids, topk_weights)
            forced_routing_map = torch.zeros(num_tokens, num_experts, device=device, dtype=torch.bool)
            forced_routing_map.scatter_(1, topk_ids, True)

            captured["router_topk_weights"] = topk_weights.detach().cpu().clone()
            captured["router_topk_ids"] = topk_ids.detach().cpu().clone()
            captured["routing_forced"] = True

            if isinstance(output, tuple):
                new_output = (forced_probs, forced_routing_map) + output[2:]
            else:
                new_output = (forced_probs, forced_routing_map)
            return new_output
        return output

    router_module = None
    for attr_name in ['router', 'gate', 'topk_router', 'moe_router']:
        if hasattr(moe_layer, attr_name):
            router_module = getattr(moe_layer, attr_name)
            if rank == 0:
                print(f"  Found router as moe_layer.{attr_name}: {type(router_module).__name__}")
            break
    if router_module is None and rank == 0:
        print(f"  WARNING: No router found. moe_layer children: "
              f"{[n for n, _ in moe_layer.named_children()]}")

    dispatcher_module = None
    for attr_name in ['token_dispatcher', 'dispatcher']:
        if hasattr(moe_layer, attr_name):
            dispatcher_module = getattr(moe_layer, attr_name)
            if rank == 0:
                print(f"  Found dispatcher as moe_layer.{attr_name}: "
                      f"{type(dispatcher_module).__name__}")
            break
    if dispatcher_module is None and rank == 0:
        print("  WARNING: No token dispatcher found; dispatch_input will not be captured.")

    handles = [
        linear_fc1.register_forward_pre_hook(fc1_pre_hook),
        linear_fc1.register_forward_hook(fc1_post_hook),
        linear_fc2.register_forward_pre_hook(fc2_pre_hook),
        linear_fc2.register_forward_hook(fc2_post_hook),
        moe_layer.register_forward_pre_hook(moe_pre_hook),
        moe_layer.register_forward_hook(moe_hook),
    ]
    if router_module is not None:
        handles.append(router_module.register_forward_hook(router_hook))
    if dispatcher_module is not None:
        handles.append(dispatcher_module.register_forward_pre_hook(dispatcher_pre_hook))

    if rank == 0:
        print(f"\nHooks registered: fc1/fc2 logpdiff scopes + swiglu capture, "
              f"moe layer pre/post, router={'YES' if router_module else 'NO'}, "
              f"dispatcher={'YES' if dispatcher_module else 'NO'}")

    # -- per-sample forward loop ---------------------------------------------
    sink_index_path = run_writer.run_dir / FRAMEWORK / "index.jsonl"
    hook_records = []   # list of (sample_name, [TensorRecord, ...]); payloads already saved
    sink_spans = []     # list of (sample_name, sink_index_line_before, after)
    sample_names = set()

    def _save_hook_records(sample_name):
        records = []
        for key, tensor_point in _HOOK_CAPTURE_ORDER:
            if key not in captured:
                print(f"  WARNING: {key} not captured for sample {sample_name}!")
                continue
            tensor = captured[key]
            provenance = (
                schema.DERIVED_METRIC if key in _HOOK_DERIVED else schema.HOOK_BOUNDARY
            )
            record = schema.TensorRecord(
                run_id=run_writer.run_id,
                framework=FRAMEWORK,
                sample_id=sample_name,
                layer_id=layer_idx,
                tensor_point=tensor_point,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype),
                provenance=provenance,
                payload_path="",  # backfilled by _save_tensor_file below
                global_rank=_get_global_rank(),
            )
            # payloads are written immediately (records stay in memory; the
            # index file is the sink's raw index until finalization)
            record.payload_path = run_writer._save_tensor_file(FRAMEWORK, tensor)
            records.append(record)
        return records

    for inp in test_inputs:
        name = inp["name"]
        input_ids_list = inp["input_ids"]
        seq_len = len(input_ids_list)
        if name in sample_names and rank == 0:
            print(f"  WARNING: duplicate sample name {name!r}; both captures are kept")
        sample_names.add(name)
        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=f"cuda:{local_rank}")

        current_sample_name[0] = name
        if rank == 0:
            print(f"\nProcessing: {name} ({seq_len} tokens)")
            if name in forced_routing:
                print(f"  [FORCED ROUTING from sglang]")

        captured.clear()
        sink_before = _count_lines(sink_index_path)

        try:
            with torch.no_grad():
                tokens = input_ids.unsqueeze(0)
                position_ids = torch.arange(seq_len, dtype=torch.long, device=tokens.device).unsqueeze(0)
                output = raw_model(tokens, position_ids, attention_mask=None)
        finally:
            # if a forward raised between pre/post hooks, unwind dangling scopes
            _exit_scope(linear_fc1)
            _exit_scope(linear_fc2)

        logits = None
        if isinstance(output, torch.Tensor):
            logits = output
        elif isinstance(output, tuple):
            logits = output[0]
        elif hasattr(output, 'logits'):
            logits = output.logits
        elif hasattr(output, 'contiguous'):
            logits = output

        if logits is not None and isinstance(logits, torch.Tensor):
            logits_f = logits.squeeze(0).float()
            log_probs = F.log_softmax(logits_f[:-1], dim=-1)
            target = input_ids[1:]
            token_logprobs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
            captured['token_logprobs'] = token_logprobs
            if rank == 0:
                print(f"  token_logprobs: {len(token_logprobs)} values, "
                      f"range=[{token_logprobs.min():.4f}, {token_logprobs.max():.4f}], "
                      f"mean={token_logprobs.mean():.4f}")
        elif rank == 0:
            print(f"  WARNING: logits not available (output type {type(output).__name__})")

        sink_after = _count_lines(sink_index_path)
        sink_spans.append((name, sink_before, sink_after))
        if sink_after == sink_before and rank == 0:
            print(f"  WARNING: minite logpdiff sink produced no records for {name} "
                  f"(is SHOULD_REPLACE_TE_GROUPLINEAR=1 set and GroupedLinear patched?)")

        if rank == 0:
            records = _save_hook_records(name)
            hook_records.append((name, records))
            for key, _ in _HOOK_CAPTURE_ORDER:
                if key in captured and isinstance(captured[key], torch.Tensor):
                    print(f"  {key}: shape={tuple(captured[key].shape)}, "
                          f"dtype={captured[key].dtype}")

    for h in handles:
        h.remove()

    dist.barrier()

    if rank != 0:
        dist.destroy_process_group()
        return None

    num_experts = getattr(megatron_args, "num_experts", None)
    if num_experts is None and hasattr(linear_fc1, "num_gemms"):
        num_experts = linear_fc1.num_gemms
    summary = {
        "hook_records": hook_records,
        "sink_spans": sink_spans,
        "sink_record_total": _count_lines(sink_index_path),
        "num_experts": num_experts,
        "router_topk": getattr(megatron_args, "moe_router_topk", None),
        "routing_forced_samples": sorted(
            n for n in sample_names if n in forced_routing
        ),
    }
    dist.destroy_process_group()
    return summary


# ---------------------------------------------------------------------------
# finalization (torch-free)
# ---------------------------------------------------------------------------


def finalize_run(cfg, run_writer, test_inputs, capture) -> int:
    """Translate the sink raw index, write all run files. Returns exit code."""
    sink_index_path = run_writer.run_dir / FRAMEWORK / "index.jsonl"
    raw_sink_records = []
    if sink_index_path.is_file():
        with open(sink_index_path, encoding="utf-8") as f:
            raw_sink_records = [json.loads(line) for line in f if line.strip()]

    # integrity: per-sample spans must tile the raw sink index exactly --
    # no unattributed records, no overlap, no gaps.
    expected = 0
    for name, before, after in capture["sink_spans"]:
        if before != expected:
            raise RuntimeError(
                f"sink index attribution gap before sample {name!r}: expected line "
                f"{expected}, got {before}; refusing to guess sample attribution"
            )
        expected = after
    if expected != len(raw_sink_records):
        raise RuntimeError(
            f"sink index has {len(raw_sink_records)} records but per-sample spans "
            f"cover {expected}; refusing to guess sample attribution"
        )

    # preserve the raw sink index next to the translated TensorRecord index
    if sink_index_path.is_file():
        os.replace(sink_index_path, sink_index_path.with_name(SINK_RAW_INDEX_NAME))

    hook_records = dict()
    for name, records in capture["hook_records"]:
        hook_records.setdefault(name, []).extend(records)

    n_hook = 0
    n_sink_written = 0
    n_sink_skipped = 0
    for name, before, after in capture["sink_spans"]:
        for record in hook_records.get(name, []):
            run_writer.append_record(FRAMEWORK, record)
            n_hook += 1
        written, skipped = translate_and_append_sink_records(
            run_writer,
            raw_sink_records[before:after],
            sample_id=name,
            global_rank=_get_global_rank(),
            expected_layer_id=cfg["layer_idx"],
        )
        n_sink_written += written
        n_sink_skipped += skipped

    samples_summary = [
        {
            "name": inp["name"],
            "num_tokens": len(inp["input_ids"]),
            "input_ids": inp["input_ids"],
        }
        for inp in test_inputs
    ]
    megatron_section = {
        "framework": FRAMEWORK,
        "layer_idx": cfg["layer_idx"],
        "model_path": cfg["model_path"],
        "ref_load": cfg["ref_load"],
        "input_file": cfg["input_file"],
        "parallel": {"tp": 1, "ep": cfg["ep_size"], "pp": 1},
        "force_routing": cfg["force_routing"],
        "num_samples": len(samples_summary),
        "total_tokens": sum(s["num_tokens"] for s in samples_summary),
        "samples": samples_summary,
        "num_experts": capture["num_experts"],
        "router_topk": capture["router_topk"],
        "routing_forced_samples": capture["routing_forced_samples"],
        "capture": {
            "hook_records": n_hook,
            "sink_records_translated": n_sink_written,
            "sink_records_metadata_skipped": n_sink_skipped,
            "sink_raw_index": f"{FRAMEWORK}/{SINK_RAW_INDEX_NAME}",
            "router_logits": "not captured (Megatron TopKRouter does not expose "
                             "pre-softmax logits; not fabricated)",
        },
    }
    _merge_write_config(run_writer, megatron_section)
    _finalize_manifest(run_writer)

    print(f"\nRun directory: {run_writer.run_dir}")
    print(f"  hook-boundary records written: {n_hook}")
    print(f"  sink records translated:       {n_sink_written} "
          f"(+{n_sink_skipped} metadata-only, kept in {FRAMEWORK}/{SINK_RAW_INDEX_NAME})")
    print(f"  config: {run_writer.run_dir / 'config.resolved.json'}")
    print(f"  index:  {run_writer.run_dir / FRAMEWORK / 'index.jsonl'}")

    if not raw_sink_records:
        print(
            "\nERROR: the minite logpdiff sink produced no records -- the actual "
            "kernel payloads (fc1/fc2 qdata/scales/output) are missing. Check that "
            "SHOULD_REPLACE_TE_GROUPLINEAR=1 is set and the Megatron TEGroupedLinear "
            "patch is applied. Hook-boundary records were still written.",
            file=sys.stderr,
        )
        return 1
    return 0


def main():
    parser = build_parser()
    our_args, unknown = parser.parse_known_args()
    if unknown:
        print(f"WARNING: ignoring unrecognized args: {unknown}", file=sys.stderr)
    cfg = resolve_config(our_args, parser)

    run_writer, reused = _setup_run(cfg)
    print(f"Run directory: {run_writer.run_dir} "
          f"(run_id={run_writer.run_id}, joined_existing={reused})")

    test_inputs = load_inputs(cfg["input_file"])
    _write_or_verify_inputs(run_writer, test_inputs)

    # enable the minite sink BEFORE minite is imported anywhere (the sink
    # reads these at import time); payloads land in <run_dir>/megatron/
    # (the reused run's megatron/ when joining via --run-dir)
    os.environ["MINITE_LOGPDIFF_ENABLE"] = "1"
    os.environ["MINITE_LOGPDIFF_OUTPUT_DIR"] = str(run_writer.run_dir / FRAMEWORK)

    capture = run_capture(cfg, run_writer, test_inputs)
    if capture is None:  # non-zero rank
        return
    sys.exit(finalize_run(cfg, run_writer, test_inputs, capture))


if __name__ == "__main__":
    main()
