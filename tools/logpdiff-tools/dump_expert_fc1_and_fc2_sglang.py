"""SGLang collector for logpdiff: capture one MoE layer's internals into a run dir.

Loads the model via sglang's ModelRunner (same path as sgl.Engine) so hidden
states come from sglang's real attention/norm kernels, then records the
router/expert pipeline of one MoE layer as logpdiff TensorRecords under
``<run>/sglang/`` (see ``logpdiff/schema.py`` next to this script).

Recorded tensor points (per sample, per layer) and their provenance:

  tensor_point             provenance           how it is obtained
  -----------------------  -------------------  ----------------------------------
  layer_input              actual_kernel        MoE module pre-hook (untouched)
  router_logits            actual_kernel        gate output hook (pre-forcing)
  router_topk_ids          replayed_reference   softmax->topk->renorm replayed in
  router_topk_weights      replayed_reference   Python from captured logits, or
                                                forced via --force-routing
  dispatch_input           hook_boundary        per-expert gather of layer_input
  fc1_raw_input            hook_boundary        same payload as dispatch_input
                                                (dispatch is a pure gather here)
  fc1_qdata                replayed_reference   NVFP4 per-token re-quantization
  fc1_scales               replayed_reference   (minite; packed uint8 data,
                                                row-major fp8 group scales)
  fc1_output               replayed_reference   dequant(x) @ w13.T in Python
  swiglu_output            replayed_reference   silu(gate) * up of replayed fc1
  fc2_raw_input            replayed_reference   swiglu_output * router weight
  fc2_qdata                replayed_reference   NVFP4 per-token re-quantization
  fc2_scales               replayed_reference
  fc2_output               replayed_reference   dequant(x) @ w2.T in Python
  expert_combined_output   actual_kernel        MoE module output hook (untouched)
  token_logprobs           replayed_reference   lm_head re-run on captured final
                                                hidden states

SGLang fuses dispatch/quant/GEMM into a single kernel, so the packed fc1/fc2
qdata+scales are never materialized in Python-visible memory. They are
reproduced by re-running minite's NVFP4 quantization on the captured inputs
and are therefore marked replayed_reference, never actual_kernel. Per-expert
payloads hold tokens in ascending token_position order; alignment with the
other framework goes through router_topk_ids (always saved), no silent
truncation anywhere.

Usage (new):
    python dump_expert_fc1_and_fc2_sglang.py \
        --model-path /path/to/Qwen3-30B-A3B \
        --input-file inputs.jsonl --output-dir runs/ --layer-idx 0

    # join an existing run (writes <run>/sglang/):
    python dump_expert_fc1_and_fc2_sglang.py --run-dir runs/<run-id> ...

    # config file + CLI overrides (CLI wins):
    python dump_expert_fc1_and_fc2_sglang.py --config capture.json --layer-idx 3

Legacy invocation keeps working -- all old flags keep their meaning; note that
--model-path is now REQUIRED (the hardcoded default was removed) and
--output-dir is now a run ROOT (the run lands in <output-dir>/<run-id>/):
    python dump_expert_fc1_and_fc2_sglang.py \
        --model-path ... --input-file ... --output-dir ... --layer-idx 0

Recommended env (unchanged from the legacy script):
    SHOULD_REPLACE_TE_GROUPLINEAR=1 SGLANG_NVFP4_PERTOKEN_SCALE=1 QAT_PARAMS=8 \
    NVTE_NVFP4_DISABLE_RHT=0 NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0 \
    NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1 \
    CUDA_VISIBLE_DEVICES=0 python dump_expert_fc1_and_fc2_sglang.py ...

P0 scope: TP=EP=PP=1 only. Heavy deps (torch, sglang, minite) are imported
inside functions so that --help works on machines without them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from logpdiff import schema  # noqa: E402
from logpdiff.writer import RunWriter, create_run_dir  # noqa: E402

FRAMEWORK = "sglang"

# config keys accepted both on the CLI and in --config JSON files
_DEFAULTS = {
    "model_path": None,
    "tokenizer_path": None,
    "input_file": None,
    "output_dir": "runs",
    "run_dir": None,
    "run_id": None,
    "layer_idx": 0,
    "mem_fraction": 0.8,
    "moe_backend": "triton",
    "hf_weights": None,
    "force_routing": None,
    "seed": None,
}


# ---------------------------------------------------------------------------
# CLI / config resolution
# ---------------------------------------------------------------------------


def _parse_cli(argv=None):
    parser = argparse.ArgumentParser(
        prog="dump_expert_fc1_and_fc2_sglang.py",
        description="SGLang MoE-layer collector writing logpdiff run directories.",
        epilog="tensor_point / provenance mapping: see this script's module docstring.",
    )
    parser.add_argument("--config", default=None,
                        help="JSON config file; keys match CLI dest names "
                             "(model_path, input_file, layer_idx, ...); CLI flags override it")
    parser.add_argument("--model-path", default=None,
                        help="HF model path (REQUIRED; the legacy hardcoded default was removed)")
    parser.add_argument("--tokenizer-path", default=None,
                        help="tokenizer path (default: same as --model-path)")
    parser.add_argument("--input-file", default=None,
                        help="JSONL inputs, one {\"name\", \"input_ids\"} object per line (REQUIRED)")
    parser.add_argument("--output-dir", default=None,
                        help="run ROOT directory; the run is created at <output-dir>/<run-id>/ "
                             "(default: ./runs)")
    parser.add_argument("--run-dir", default=None,
                        help="join an existing run directory instead of creating one "
                             "(writes its sglang/ subdir; mutually exclusive with --run-id)")
    parser.add_argument("--run-id", default=None,
                        help="explicit run id when creating a run under --output-dir")
    parser.add_argument("--layer-idx", type=int, default=None,
                        help="MoE layer index to capture (default: 0)")
    parser.add_argument("--mem-fraction", type=float, default=None,
                        help="sglang mem_fraction_static (default: 0.8)")
    parser.add_argument("--moe-backend", default=None,
                        help="sglang moe_runner_backend: triton (standard 3D weights) or "
                             "flashinfer (4D tiled) (default: triton)")
    parser.add_argument("--hf-weights", default=None,
                        help="HF raw w13 weights .pt (from compare_expert_fc1_and_fc2.py "
                             "--mode dump-hf); required for the fc1/fc2 replay when "
                             "w13_weight is 4D (flashinfer backend)")
    parser.add_argument("--force-routing", default=None, metavar="PT",
                        help="legacy dump .pt mapping sample name -> {topk_ids, topk_weights}; "
                             "forces the gate output so the model uses that routing "
                             "(records manifest force_routing=True)")
    parser.add_argument("--seed", type=int, default=None,
                        help="recorded in the manifest; also passed to torch.manual_seed "
                             "and sglang random_seed")
    return parser, parser.parse_args(argv)


def _load_config_file(path, parser):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        parser.error(f"cannot read --config {path}: {e}")
    except json.JSONDecodeError as e:
        parser.error(f"--config {path} is not valid JSON: {e}")
    if not isinstance(data, dict):
        parser.error(f"--config {path} must contain a JSON object")
    unknown = set(data) - set(_DEFAULTS)
    if unknown:
        parser.error(
            f"unknown keys in --config {path}: {sorted(unknown)}; "
            f"valid keys: {sorted(_DEFAULTS)}"
        )
    return data


def _resolve_config(args, parser):
    """Merge defaults < --config file < CLI flags; validate required fields."""
    cfg = dict(_DEFAULTS)
    if args.config is not None:
        cfg.update(_load_config_file(args.config, parser))
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            cfg[key] = value

    if cfg["run_dir"] and cfg["run_id"]:
        parser.error("--run-id only applies when creating a run (with --output-dir), "
                     "not when joining one via --run-dir")
    if not cfg["model_path"]:
        parser.error("--model-path is required (or set \"model_path\" in --config); "
                     "the legacy hardcoded default was removed")
    if not cfg["input_file"]:
        parser.error("--input-file is required (or set \"input_file\" in --config)")
    if cfg["layer_idx"] < 0:
        parser.error("--layer-idx must be >= 0")
    if not os.path.isfile(cfg["input_file"]):
        parser.error(f"--input-file not found: {cfg['input_file']}")
    if cfg["force_routing"] is not None and not os.path.isfile(cfg["force_routing"]):
        parser.error(f"--force-routing file not found: {cfg['force_routing']}")
    if cfg["hf_weights"] is not None and not os.path.isfile(cfg["hf_weights"]):
        parser.error(f"--hf-weights file not found: {cfg['hf_weights']}")
    if not cfg["tokenizer_path"]:
        cfg["tokenizer_path"] = cfg["model_path"]
    return cfg


# ---------------------------------------------------------------------------
# run directory setup / top-level files
# ---------------------------------------------------------------------------


def _setup_run(cfg):
    """Create a new run under --output-dir, or join an existing --run-dir."""
    force = cfg["force_routing"] is not None
    common = dict(
        model_path=cfg["model_path"],
        tokenizer_path=cfg["tokenizer_path"],
        seed=cfg["seed"],
        backend=cfg["moe_backend"],
        force_routing=force,
    )
    if cfg["run_dir"]:
        run_dir = Path(cfg["run_dir"]).resolve()
        if not run_dir.is_dir():
            raise SystemExit(
                f"--run-dir {run_dir} does not exist; "
                "to create a new run use --output-dir instead"
            )
        (run_dir / FRAMEWORK / "tensors").mkdir(parents=True, exist_ok=True)
        if (run_dir / FRAMEWORK / "index.jsonl").exists():
            raise SystemExit(
                f"{run_dir / FRAMEWORK / 'index.jsonl'} already exists; refusing to "
                "mix a second sglang capture into the same run"
            )
        return RunWriter(run_dir=run_dir, run_id=run_dir.name, **common), True
    return create_run_dir(cfg["output_dir"], run_id=cfg["run_id"], **common), False


def _write_or_verify_inputs(writer, test_inputs):
    """Write inputs.jsonl; when joining a run that already has one, the
    existing content must be identical (no silent input mismatch)."""
    records = [
        {
            "sample_id": str(inp["name"]),
            "input_ids": list(inp["input_ids"]),
            "num_tokens": len(inp["input_ids"]),
        }
        for inp in test_inputs
    ]
    path = writer.run_dir / "inputs.jsonl"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            existing = [json.loads(line) for line in f if line.strip()]
        if existing != records:
            raise SystemExit(
                f"{path} already exists and does not match --input-file; "
                "refusing to mix different inputs into one run"
            )
        print(f"inputs.jsonl already present and identical ({len(records)} samples)")
        return records
    writer.write_inputs(records)
    print(f"Wrote {len(records)} input samples to {path}")
    return records


def _merge_write_config(writer, sglang_section):
    """Write config.resolved.json, preserving other frameworks' sections."""
    path = writer.run_dir / "config.resolved.json"
    merged = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            merged = json.load(f)
        if not isinstance(merged, dict):
            raise SystemExit(f"{path} exists but is not a JSON object; refusing to merge")
    merged.setdefault("run_id", writer.run_id)
    merged[FRAMEWORK] = sglang_section
    writer.write_config_resolved(merged)
    print(f"Wrote {path}")


def _finalize_manifest(writer):
    path = writer.run_dir / "manifest.json"
    if path.exists():
        print(f"{path} already exists (joined run); leaving it untouched")
        if writer.force_routing:
            print("  NOTE: this sglang capture used --force-routing but the existing "
                  "manifest was not rewritten; check its force_routing flag yourself")
        return
    writer.finalize_manifest()
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# record saving
# ---------------------------------------------------------------------------


def _save(writer, sample_id, layer_id, tensor_point, provenance, tensor, expert_id=None):
    """Save one tensor payload and append its TensorRecord to the sglang index."""
    record = schema.TensorRecord(
        run_id=writer.run_id,
        framework=FRAMEWORK,
        sample_id=sample_id,
        layer_id=layer_id,
        tensor_point=tensor_point,
        provenance=provenance,
        global_rank=0,
        expert_id=expert_id,
    )
    return writer.save_tensor(record, tensor)


def _alias_record(writer, sample_id, layer_id, tensor_point, provenance,
                  tensor, payload_path, expert_id=None):
    """Append a second record pointing at an already-saved payload (used for
    fc1_raw_input, which is byte-identical to dispatch_input on sglang)."""
    record = schema.TensorRecord(
        run_id=writer.run_id,
        framework=FRAMEWORK,
        sample_id=sample_id,
        layer_id=layer_id,
        tensor_point=tensor_point,
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        provenance=provenance,
        payload_path=payload_path,
        global_rank=0,
        expert_id=expert_id,
    )
    writer.append_record(FRAMEWORK, record)


# ---------------------------------------------------------------------------
# NVFP4 replay (reference re-quantization of captured activations)
# ---------------------------------------------------------------------------


def _nvfp4_quant_replay(x):
    """Re-run NVFP4 per-token quantization on a captured activation.

    Returns ``(qdata, scales, dequant)``: packed uint8 FP4 data [m, n//2],
    row-major fp8 group scales [m, n//16], and the BF16 dequantized tensor.
    This replays what sglang's fused MoE kernel does internally; the packed
    internals are never visible to Python, hence replayed_reference.
    """
    import torch
    from minite.ops.nvfp4_quantize import (
        dequantize_nvfp4_pertoken,
        quantize_nvfp4_pertoken,
    )

    k = x.shape[-1]
    pad_k = ((k + 127) // 128) * 128
    if pad_k > k:
        x_pad = torch.zeros(*x.shape[:-1], pad_k, dtype=x.dtype, device=x.device)
        x_pad[..., :k] = x
    else:
        x_pad = x
    qr = quantize_nvfp4_pertoken(x_pad)
    dq = dequantize_nvfp4_pertoken(qr).to(torch.bfloat16)[..., :k]
    return qr.data, qr.scale, dq


# ---------------------------------------------------------------------------
# model loading and hooks (sglang imports live here, never at module level)
# ---------------------------------------------------------------------------


def _load_model(cfg):
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.utils import suppress_other_loggers

    suppress_other_loggers()

    server_kwargs = {}
    if cfg["seed"] is not None:
        server_kwargs["random_seed"] = cfg["seed"]
    server_args = ServerArgs(
        model_path=cfg["model_path"],
        tokenizer_path=cfg["tokenizer_path"],
        host="127.0.0.1",
        port=30000,
        disable_cuda_graph=True,
        tp_size=1,
        mem_fraction_static=cfg["mem_fraction"],
        trust_remote_code=True,
        moe_runner_backend=cfg["moe_backend"],
        **server_kwargs,
    )
    _set_envs_and_config(server_args)

    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)

    port_args = PortArgs.init_new(server_args)
    model_config = ModelConfig.from_server_args(server_args)

    print(f"Loading model via sglang ModelRunner from {cfg['model_path']} ...")
    t0 = time.time()
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=0,
        tp_rank=0,
        tp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    return model_runner, model_config


def _register_hooks(model, layer_idx, captured, current_forced):
    """Register capture hooks on layer ``layer_idx`` of the sglang model.

    ``current_forced`` is a single-element list holding None or the forced
    routing dict {"topk_ids", "topk_weights"} for the sample in flight; when
    set, the gate hook rewrites its output logits so that softmax->topk->renorm
    reproduces the forced routing exactly (records stay honest: router_logits
    keeps the actual kernel output, and the forcing is visible via
    router_topk_ids/weights + manifest force_routing=True).
    """
    import torch

    layer = model.model.layers[layer_idx]
    moe = layer.mlp

    def moe_pre_hook(module, args_tuple):
        x = args_tuple[0]
        if isinstance(x, (list, tuple)):
            x = x[0]
        captured["layer_input"] = x.detach().clone()

    def gate_post_hook(module, args_tuple, output):
        logits = output[0] if isinstance(output, tuple) else output
        captured["router_logits"] = logits.detach().clone()
        forced = current_forced[0]
        if forced is None:
            return None
        flat = logits.reshape(-1, logits.shape[-1])
        ids = forced["topk_ids"].to(device=flat.device)
        weights = forced["topk_weights"].to(device=flat.device, dtype=torch.float32)
        if ids.shape[0] != flat.shape[0]:
            raise RuntimeError(
                f"--force-routing token count mismatch: forced topk_ids has "
                f"{ids.shape[0]} tokens but the gate produced {flat.shape[0]}; "
                "refusing to silently truncate"
            )
        if ids.numel() and int(ids.max()) >= flat.shape[-1]:
            raise RuntimeError(
                f"--force-routing expert id {int(ids.max())} out of range for "
                f"{flat.shape[-1]} experts"
            )
        # log-space construction: softmax(forced_logits) reproduces the forced
        # weights (already normalized per token) at the forced ids
        forced_logits = torch.full(flat.shape, -1e4, dtype=torch.float32,
                                   device=flat.device)
        forced_logits.scatter_(1, ids, weights.clamp_min(1e-30).log())
        forced_logits = forced_logits.view_as(logits)
        if isinstance(output, tuple):
            return (forced_logits,) + tuple(output[1:])
        return forced_logits

    def moe_post_hook(module, args_tuple, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            captured["expert_combined_output"] = out.detach().clone()

    def logits_pre_hook(module, args_tuple):
        captured["final_hidden_states"] = args_tuple[1].detach().clone()

    handles = [
        moe.register_forward_pre_hook(moe_pre_hook),
        moe.gate.register_forward_hook(gate_post_hook),
        moe.register_forward_hook(moe_post_hook),
        model.logits_processor.register_forward_pre_hook(logits_pre_hook),
    ]
    print(f"Hooks registered on layer {layer_idx} MoE + gate + logits_processor")
    return moe, handles


def _resolve_top_k(model_runner, moe):
    """Router top-k for this model; falls back to 8 (Qwen3-30B-A3B) with a warning."""
    hf_config = getattr(model_runner.model_config, "hf_config", None)
    for holder in (hf_config, moe, getattr(moe, "experts", None)):
        if holder is None:
            continue
        for attr in ("num_experts_per_tok", "top_k"):
            value = getattr(holder, attr, None)
            if isinstance(value, int) and value > 0:
                print(f"router top_k={value} (from {type(holder).__name__}.{attr})")
                return value
    print("WARNING: could not resolve router top_k from the model; falling back to 8")
    return 8


# ---------------------------------------------------------------------------
# inputs / forced routing
# ---------------------------------------------------------------------------


def load_inputs(path):
    inputs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                inputs.append(json.loads(line))
    return inputs


def _load_forced_routing(path):
    """Load a legacy dump .pt mapping sample name -> {topk_ids, topk_weights}.

    Weights are normalized per token so that what the model uses downstream
    (softmax->topk->renorm) matches the saved router_topk_weights records
    exactly. Legacy sglang dumps already store normalized weights, so this is
    a no-op for them.
    """
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise SystemExit(f"--force-routing {path}: expected a dict, got {type(data).__name__}")
    forced = {}
    for name, sample in data.items():
        ids = sample.get("topk_ids")
        weights = sample.get("topk_weights")
        if ids is None or weights is None:
            raise SystemExit(
                f"--force-routing {path}: sample {name!r} lacks topk_ids/topk_weights"
            )
        weights = weights.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        forced[str(name)] = {"topk_ids": ids.long(), "topk_weights": weights}
    print(f"Loaded forced routing for {len(forced)} samples: {sorted(forced)}")
    return forced


# ---------------------------------------------------------------------------
# per-sample capture
# ---------------------------------------------------------------------------


def _save_routing_and_layer_records(writer, sample_id, layer_idx, captured, moe,
                                    mlp_input, top_k, forced):
    """Save layer_input / router_logits / router_topk_ids / router_topk_weights
    and return (topk_ids, topk_weights, router_logits) on the model device."""
    import torch
    import torch.nn.functional as F

    if "router_logits" in captured:
        router_logits = captured["router_logits"]
    else:
        print("  WARNING: router_logits not captured by hook; recomputing via gate")
        router_logits, _ = moe.gate(mlp_input.to(moe.gate.weight.dtype))
    if router_logits.ndim == 3:
        router_logits = router_logits.squeeze(0)
    _save(writer, sample_id, layer_idx, schema.ROUTER_LOGITS,
          schema.ACTUAL_KERNEL, router_logits)

    if forced is not None:
        topk_ids = forced["topk_ids"].to(mlp_input.device)
        topk_weights = forced["topk_weights"].to(mlp_input.device)
    else:
        routing_weights = F.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(routing_weights, k=top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    _save(writer, sample_id, layer_idx, schema.ROUTER_TOPK_IDS,
          schema.REPLAYED_REFERENCE, topk_ids)
    _save(writer, sample_id, layer_idx, schema.ROUTER_TOPK_WEIGHTS,
          schema.REPLAYED_REFERENCE, topk_weights)
    _save(writer, sample_id, layer_idx, schema.LAYER_INPUT,
          schema.ACTUAL_KERNEL, mlp_input)

    if "expert_combined_output" in captured:
        combined = captured["expert_combined_output"]
        if combined.ndim == 3:
            combined = combined.squeeze(0)
        _save(writer, sample_id, layer_idx, schema.EXPERT_COMBINED_OUTPUT,
              schema.ACTUAL_KERNEL, combined)
    else:
        print("  WARNING: expert_combined_output not captured!")

    return topk_ids, topk_weights, router_logits


def _replay_experts(writer, sample_id, layer_idx, mlp_input, topk_ids, topk_weights,
                    experts, hf_weights, is_4d):
    """Per-expert NVFP4 replay of fc1/swiglu/fc2; saves all per-expert records.

    Returns (active_experts, skipped_experts): skipped experts occur only when
    w13_weight is 4D-tiled (flashinfer backend) and no HF raw weight is
    available for that expert -- their dispatch/fc1 quant records are still
    saved, only the GEMM replay is skipped.
    """
    import torch
    import torch.nn.functional as F

    active_experts = topk_ids.unique().tolist()
    skipped = []
    w13_weight = experts.w13_weight

    for eid in sorted(active_experts):
        mask = (topk_ids == eid).any(dim=-1)
        tidx = mask.nonzero(as_tuple=True)[0]
        if tidx.numel() == 0:
            continue

        x_raw = mlp_input[tidx]  # ascending token order; pre-quant, post-dispatch
        payload = _save(writer, sample_id, layer_idx, schema.DISPATCH_INPUT,
                        schema.HOOK_BOUNDARY, x_raw, expert_id=eid)
        _alias_record(writer, sample_id, layer_idx, schema.FC1_RAW_INPUT,
                      schema.HOOK_BOUNDARY, x_raw, payload, expert_id=eid)

        qdata, scales, fc1_dq = _nvfp4_quant_replay(x_raw)
        _save(writer, sample_id, layer_idx, schema.FC1_QDATA,
              schema.REPLAYED_REFERENCE, qdata, expert_id=eid)
        _save(writer, sample_id, layer_idx, schema.FC1_SCALES,
              schema.REPLAYED_REFERENCE, scales, expert_id=eid)

        if hf_weights is not None and eid in hf_weights:
            w13_e = hf_weights[eid].to(device=x_raw.device)
        elif not is_4d:
            w13_e = w13_weight[eid]
        else:
            skipped.append(eid)
            continue

        fc1_out = (fc1_dq.float() @ w13_e.float().T).to(fc1_dq.dtype)
        _save(writer, sample_id, layer_idx, schema.FC1_OUTPUT,
              schema.REPLAYED_REFERENCE, fc1_out, expert_id=eid)

        moe_ffn = w13_e.shape[0] // 2
        swiglu = F.silu(fc1_out[:, :moe_ffn]) * fc1_out[:, moe_ffn:]
        _save(writer, sample_id, layer_idx, schema.SWIGLU_OUTPUT,
              schema.REPLAYED_REFERENCE, swiglu, expert_id=eid)

        expert_mask = (topk_ids[tidx] == eid)  # [n_tok, top_k]
        prob_per_tok = (topk_weights[tidx] * expert_mask.float()).sum(dim=-1)
        fc2_raw = swiglu * prob_per_tok.view(-1, 1).to(swiglu.dtype)
        _save(writer, sample_id, layer_idx, schema.FC2_RAW_INPUT,
              schema.REPLAYED_REFERENCE, fc2_raw, expert_id=eid)

        qdata2, scales2, fc2_dq = _nvfp4_quant_replay(fc2_raw)
        _save(writer, sample_id, layer_idx, schema.FC2_QDATA,
              schema.REPLAYED_REFERENCE, qdata2, expert_id=eid)
        _save(writer, sample_id, layer_idx, schema.FC2_SCALES,
              schema.REPLAYED_REFERENCE, scales2, expert_id=eid)

        w2_e = experts.w2_weight[eid]
        fc2_out = (fc2_dq.float() @ w2_e.float().T).to(fc2_dq.dtype)
        _save(writer, sample_id, layer_idx, schema.FC2_OUTPUT,
              schema.REPLAYED_REFERENCE, fc2_out, expert_id=eid)

    return sorted(active_experts), skipped


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None):
    parser, args = _parse_cli(argv)
    cfg = _resolve_config(args, parser)
    layer_idx = cfg["layer_idx"]

    print(f"SHOULD_REPLACE_TE_GROUPLINEAR={os.environ.get('SHOULD_REPLACE_TE_GROUPLINEAR', '')}")
    print(f"QAT_PARAMS={os.environ.get('QAT_PARAMS', '')}")
    # Ensure SGLANG_NVFP4_PERTOKEN_SCALE is set
    if not os.environ.get("SGLANG_NVFP4_PERTOKEN_SCALE"):
        os.environ["SGLANG_NVFP4_PERTOKEN_SCALE"] = "1"
    print(f"SGLANG_NVFP4_PERTOKEN_SCALE={os.environ.get('SGLANG_NVFP4_PERTOKEN_SCALE', '')}")

    test_inputs = load_inputs(cfg["input_file"])
    if not test_inputs:
        raise SystemExit(f"--input-file {cfg['input_file']} contains no samples")

    writer, reused = _setup_run(cfg)
    print(f"Run directory: {writer.run_dir} (run_id={writer.run_id}, joined_existing={reused})")
    input_records = _write_or_verify_inputs(writer, test_inputs)

    # heavy deps only past this point
    import torch
    import torch.nn.functional as F

    if cfg["seed"] is not None:
        torch.manual_seed(cfg["seed"])

    # minite must come from the environment (pip install or PYTHONPATH); as a
    # convenience, fall back to the enclosing repo checkout if present
    try:
        import minite  # noqa: F401
    except ImportError:
        repo_root = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
        if os.path.isdir(os.path.join(repo_root, "minite")):
            sys.path.insert(0, repo_root)
        try:
            import minite  # noqa: F401
        except ImportError:
            raise SystemExit(
                "could not import 'minite' (needed for the NVFP4 replay); "
                "pip install it or set PYTHONPATH to the repo root"
            )

    forced_routing = None
    if cfg["force_routing"] is not None:
        forced_routing = _load_forced_routing(cfg["force_routing"])

    model_runner, model_config = _load_model(cfg)
    model = model_runner.model
    layer = model.model.layers[layer_idx]
    moe = layer.mlp
    experts = moe.experts

    print(f"\nLayer {layer_idx}: {type(moe).__name__}")
    print(f"  experts: {type(experts).__name__}")
    if hasattr(experts, "w13_weight"):
        print(f"  w13_weight shape: {experts.w13_weight.shape}, dtype: {experts.w13_weight.dtype}")
    if hasattr(experts, "w2_weight"):
        print(f"  w2_weight shape: {experts.w2_weight.shape}, dtype: {experts.w2_weight.dtype}")

    top_k = _resolve_top_k(model_runner, moe)
    if forced_routing is not None:
        widths = {v["topk_ids"].shape[1] for v in forced_routing.values()}
        if len(widths) != 1:
            raise SystemExit(f"--force-routing file has inconsistent top-k widths: {sorted(widths)}")
        if widths.pop() != top_k:
            raise SystemExit(f"--force-routing top-k width does not match model top_k={top_k}")

    hf_weights = None
    if cfg["hf_weights"]:
        print(f"\nLoading HF raw weights from {cfg['hf_weights']}...")
        hf_weights = torch.load(cfg["hf_weights"], map_location="cpu", weights_only=False)
        print(f"  Loaded {len(hf_weights)} expert w13 weights")

    is_4d = experts.w13_weight.ndim == 4
    if is_4d:
        print(f"  w13_weight is 4D (flashinfer tiled): {experts.w13_weight.shape}")
        if hf_weights is None:
            print("  WARNING: 4D tiled weights cannot be untiled correctly; "
                  "per-expert GEMM replay will be skipped (raw/quant records still saved). "
                  "Use --hf-weights to provide raw HF weights for the fc1/fc2 replay.")
        else:
            print("  Using HF raw weights for the fc1 replay (bypassing 4D tiling)")

    captured = {}
    current_forced = [None]
    moe, handles = _register_hooks(model, layer_idx, captured, current_forced)

    from sglang.bench_one_batch import TreeCacheNamespace
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    num_experts = None
    sample_summaries = []

    for inp_idx, inp in enumerate(test_inputs):
        name = str(inp["name"])
        input_ids_list = list(inp["input_ids"])
        seq_len = len(input_ids_list)
        print(f"\nProcessing: {name} ({seq_len} tokens)")

        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool_allocator.clear()
        captured.clear()

        if forced_routing is not None:
            current_forced[0] = forced_routing.get(name)
            if current_forced[0] is None:
                print("  WARNING: no forced routing entry for this sample; "
                      "running with the model's own routing")
            else:
                n_forced = current_forced[0]["topk_ids"].shape[0]
                if n_forced != seq_len:
                    raise SystemExit(
                        f"--force-routing entry for {name!r} has {n_forced} tokens, "
                        f"input has {seq_len}; refusing to silently truncate"
                    )
                print("  [FORCED ROUTING]")

        sampling_params = SamplingParams(temperature=0, max_new_tokens=0)
        req = Req(
            rid=inp_idx,
            origin_input_text="",
            origin_input_ids=input_ids_list,
            sampling_params=sampling_params,
        )
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))

        dummy_tree_cache = TreeCacheNamespace(
            page_size=model_runner.server_args.page_size,
            device=model_runner.device,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        )
        batch = ScheduleBatch.init_new(
            reqs=[req],
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
            tree_cache=dummy_tree_cache,
            model_config=model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)

        with torch.no_grad():
            model_runner.forward(forward_batch)

        # --- token logprobs (lm_head re-run on captured hidden states) ---
        if "final_hidden_states" in captured:
            hs = captured["final_hidden_states"].float()
            lm_weight = model.lm_head.weight.float()
            logits = hs @ lm_weight.T
            log_probs = F.log_softmax(logits[:-1], dim=-1)
            input_ids_tensor = torch.tensor(input_ids_list, device=logits.device)
            target = input_ids_tensor[1:]
            token_logprobs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
            _save(writer, name, layer_idx, schema.TOKEN_LOGPROBS,
                  schema.REPLAYED_REFERENCE, token_logprobs)
            print(f"  token_logprobs: {len(token_logprobs)} values, "
                  f"range=[{token_logprobs.min():.4f}, {token_logprobs.max():.4f}], "
                  f"mean={token_logprobs.mean():.4f}")
        else:
            print("  WARNING: final_hidden_states not captured!")

        ids_sha = hashlib.sha256(
            json.dumps(input_ids_list).encode("utf-8")
        ).hexdigest()
        summary = {
            "sample_id": name,
            "num_tokens": seq_len,
            "input_ids": input_ids_list,
            "input_ids_sha256": ids_sha,
            "routing_forced": current_forced[0] is not None,
        }

        if "layer_input" not in captured:
            print("  WARNING: layer_input not captured; skipping MoE records "
                  "for this sample!")
            sample_summaries.append(summary)
            continue

        mlp_input = captured["layer_input"]
        if mlp_input.ndim == 3:
            mlp_input = mlp_input.squeeze(0)

        topk_ids, topk_weights, router_logits = _save_routing_and_layer_records(
            writer, name, layer_idx, captured, moe, mlp_input, top_k,
            current_forced[0],
        )
        if num_experts is None:
            num_experts = router_logits.shape[-1]

        active_experts, skipped = _replay_experts(
            writer, name, layer_idx, mlp_input, topk_ids, topk_weights,
            experts, hf_weights, is_4d,
        )
        summary["active_experts"] = len(active_experts)
        if skipped:
            summary["replay_skipped_experts"] = skipped
            print(f"  WARNING: GEMM replay skipped for {len(skipped)} experts "
                  "(4D tiled weights without HF raw weights)")
        print(f"  active experts: {len(active_experts)}")
        sample_summaries.append(summary)

    for h in handles:
        h.remove()

    # --- top-level run files -------------------------------------------------
    index_path = writer.run_dir / FRAMEWORK / "index.jsonl"
    num_records = 0
    captured_points = set()
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    num_records += 1
                    captured_points.add(json.loads(line)["tensor_point"])

    sglang_section = {
        "collector": os.path.basename(__file__),
        "framework": FRAMEWORK,
        "model_path": cfg["model_path"],
        "tokenizer_path": cfg["tokenizer_path"],
        "input_file": os.path.abspath(cfg["input_file"]),
        "layer_idx": layer_idx,
        "num_samples": len(sample_summaries),
        "samples": sample_summaries,
        "total_tokens": sum(s["num_tokens"] for s in sample_summaries),
        "num_experts": num_experts,
        "top_k": top_k,
        "moe_backend": cfg["moe_backend"],
        "mem_fraction": cfg["mem_fraction"],
        "force_routing": cfg["force_routing"] is not None,
        "force_routing_path": cfg["force_routing"],
        "hf_weights": cfg["hf_weights"],
        "seed": cfg["seed"],
        "expert_token_order": "ascending_token_position",
        "captured_tensor_points": sorted(captured_points),
        "num_records": num_records,
    }
    _merge_write_config(writer, sglang_section)
    _finalize_manifest(writer)

    print(f"\nDone. {num_records} sglang records in {index_path}")
    print(f"Run directory: {writer.run_dir}")


if __name__ == "__main__":
    main()
