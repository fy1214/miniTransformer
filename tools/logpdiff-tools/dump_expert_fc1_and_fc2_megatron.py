"""Dump expert fc1+fc2 activations and token logprobs from Megatron model.

Captures at Layer 0 expert fc1:
  - fc1_input: padded dispatched tokens entering linear_fc1 (raw, before QDQ)
  - fc1_gemm_input: activation after NVFP4 pertoken QDQ (what enters GEMM)
  - fc1_output: fc1 GEMM result
  - tokens_per_expert: padded and unpadded token counts

Captures at Layer 0 expert fc2:
  - fc2_input: post-SwiGLU, post-probs input to linear_fc2 (raw, before QDQ)
  - fc2_gemm_input: activation after NVFP4 pertoken QDQ
  - fc2_output: fc2 GEMM result

Also captures token_logprobs from the model's final output logits
(merged from dump_megatron_logprob.py).

Usage:
    SHOULD_REPLACE_TE_GROUPLINEAR=1 QAT_PARAMS=8 \
    NVTE_NVFP4_DISABLE_RHT=0 NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0 NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1 \
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 dump_expert_fc1_and_fc2_megatron.py \
        --input-file test_inputs_long.jsonl \
        --output-dir dump_expert_fc1_megatron/ \
        --ep-size 1 \
        --layer-idx 0
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F


MODEL_PATH = "/home/admin/mingfa/model/hf/Qwen3-30B-A3B/Qwen3-30B-A3B"
REF_LOAD = "/home/admin/mingfa/model/torch_dist/Qwen3-30B-A3B_torch_dist/"


def parse_our_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input-file", default="/root/compare_logprob/test_inputs_long.jsonl")
    parser.add_argument("--output-dir", default="/root/compare_logprob/dump_expert_fc1_megatron/")
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--ref-load-path", default=REF_LOAD)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--force-routing", default=None,
                        help="Path to sglang dump .pt to force router output (topk_ids/topk_weights)")
    our_args, _ = parser.parse_known_args()
    return our_args


def setup_megatron_argv(our_args):
    ep = our_args.ep_size
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
        "--hf-checkpoint", our_args.model_path,
        "--ref-load", our_args.ref_load_path,
        "--load", our_args.ref_load_path,
        "--lr-warmup-iters", "0",
        "--lr-decay-iters", "10",
        "--train-iters", "10",
    ]


def load_inputs(path):
    inputs = []
    with open(path) as f:
        for line in f:
            inputs.append(json.loads(line))
    return inputs


def main():
    our_args = parse_our_args()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank == 0:
        os.makedirs(our_args.output_dir, exist_ok=True)
        print(f"World size: {world_size}, EP: {our_args.ep_size}, rank: {rank}")
        print(f"SHOULD_REPLACE_TE_GROUPLINEAR={os.environ.get('SHOULD_REPLACE_TE_GROUPLINEAR', '')}")
        print(f"QAT_PARAMS={os.environ.get('QAT_PARAMS', '')}")

    torch.cuda.set_device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    setup_megatron_argv(our_args)

    sys.path.insert(0, '/root/Megatron-LM')
    sys.path.insert(0, '/root/miniTransformer')

    from megatron.training.arguments import parse_args as megatron_parse_args
    from megatron.training.arguments import validate_args as megatron_validate_args
    args = megatron_parse_args(extra_args_provider=None, ignore_unknown_args=True)

    args.hf_checkpoint = our_args.model_path
    args.ref_load = our_args.ref_load_path
    args.load = our_args.ref_load_path

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

    layer_idx = our_args.layer_idx
    target_layer = raw_model.decoder.layers[layer_idx]
    moe_layer = target_layer.mlp
    experts_module = moe_layer.experts if hasattr(moe_layer, 'experts') else None

    if experts_module is None:
        if rank == 0:
            print(f"ERROR: Cannot find experts in layer {layer_idx}!")
        dist.barrier()
        dist.destroy_process_group()
        return

    linear_fc1 = experts_module.linear_fc1
    linear_fc2 = experts_module.linear_fc2

    if rank == 0:
        print(f"\nLayer {layer_idx} structure:")
        for name, child in target_layer.named_children():
            print(f"  {name}: {type(child).__name__}")
        print(f"\nExperts module: {type(experts_module).__name__}")
        print(f"linear_fc1: {type(linear_fc1).__name__}")
        if hasattr(linear_fc1, 'num_gemms'):
            print(f"  num_gemms={linear_fc1.num_gemms}")
        if hasattr(linear_fc1, 'weight0'):
            print(f"  weight0 shape={linear_fc1.weight0.shape}")
        print(f"linear_fc2: {type(linear_fc2).__name__}")
        if hasattr(linear_fc2, 'weight0'):
            print(f"  weight0 shape={linear_fc2.weight0.shape}")

    # Setup hooks
    captured = {}

    def fc1_pre_hook(module, args_tuple):
        inp = args_tuple[0]
        m_splits = args_tuple[1]
        captured["fc1_input"] = inp.detach().cpu().clone()
        if isinstance(m_splits, torch.Tensor):
            captured["tokens_per_expert_padded"] = m_splits.detach().cpu().clone()
        else:
            captured["tokens_per_expert_padded"] = list(m_splits)

    def fc1_hook(module, args_tuple, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        captured["fc1_output"] = out.detach().cpu().clone()

    def fc2_pre_hook(module, args_tuple):
        inp = args_tuple[0]
        captured["fc2_input"] = inp.detach().cpu().clone()

    def fc2_hook(module, args_tuple, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        captured["fc2_output"] = out.detach().cpu().clone()

    def experts_pre_hook(module, args_tuple):
        inp = args_tuple[0]
        tokens_per_expert = args_tuple[1]
        captured["experts_input_raw"] = inp.detach().cpu().clone()
        if isinstance(tokens_per_expert, torch.Tensor):
            captured["tokens_per_expert_unpadded"] = tokens_per_expert.detach().cpu().clone()
        else:
            captured["tokens_per_expert_unpadded"] = list(tokens_per_expert)

    def moe_pre_hook(module, args_tuple):
        x = args_tuple[0]
        if isinstance(x, (list, tuple)):
            x = x[0]
        captured["mlp_input_original"] = x.detach().cpu().clone()

    # Load forced routing from sglang dump if specified
    forced_routing = {}
    if our_args.force_routing:
        if rank == 0:
            print(f"\nLoading forced routing from: {our_args.force_routing}")
        fr_data = torch.load(our_args.force_routing, map_location="cpu", weights_only=False)
        for sample_name, sample_data in fr_data.items():
            fr_topk_ids = sample_data.get("topk_ids")
            fr_topk_weights = sample_data.get("topk_weights")
            if fr_topk_ids is not None and fr_topk_weights is not None:
                forced_routing[sample_name] = {
                    "topk_ids": fr_topk_ids,
                    "topk_weights": fr_topk_weights,
                }
        if rank == 0:
            print(f"  Loaded routing for {len(forced_routing)} samples: {list(forced_routing.keys())}")

    current_sample_name = [None]

    def router_hook(module, args_tuple, output):
        # Megatron router returns (probs [T,E] dense float, routing_map [T,E] bool)
        if isinstance(output, tuple) and len(output) >= 2:
            probs_dense, routing_map = output[0], output[1]
            if routing_map.dtype == torch.bool:
                k = routing_map.sum(dim=-1).max().item()
                topk_w, topk_i = torch.topk(probs_dense, k=int(k), dim=-1)
                captured["router_scores"] = topk_w.detach().cpu().clone()
                captured["router_indices"] = topk_i.detach().cpu().clone()
            else:
                captured["router_scores"] = probs_dense.detach().cpu().clone()
                captured["router_indices"] = routing_map.detach().cpu().clone()

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

            captured["router_scores"] = topk_weights.detach().cpu().clone()
            captured["router_indices"] = topk_ids.detach().cpu().clone()
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
        print(f"  moe_layer attrs: {[a for a in dir(moe_layer) if not a.startswith('_')]}")

    handles = [
        linear_fc1.register_forward_pre_hook(fc1_pre_hook),
        linear_fc1.register_forward_hook(fc1_hook),
        linear_fc2.register_forward_pre_hook(fc2_pre_hook),
        linear_fc2.register_forward_hook(fc2_hook),
        experts_module.register_forward_pre_hook(experts_pre_hook),
        moe_layer.register_forward_pre_hook(moe_pre_hook),
    ]
    if router_module is not None:
        handles.append(router_module.register_forward_hook(router_hook))

    if rank == 0:
        print(f"\nHooks registered: experts pre_hook + linear_fc1 (pre/post) + linear_fc2 (pre/post)"
              f" + router={'YES' if router_module else 'NO'}")

    # Run inference
    test_inputs = load_inputs(our_args.input_file)
    results = {}

    for inp in test_inputs:
        name = inp["name"]
        input_ids_list = inp["input_ids"]
        seq_len = len(input_ids_list)
        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=f"cuda:{local_rank}")

        current_sample_name[0] = name
        if rank == 0:
            print(f"\nProcessing: {name} ({seq_len} tokens)")
            if name in forced_routing:
                print(f"  [FORCED ROUTING from sglang]")

        captured.clear()

        with torch.no_grad():
            tokens = input_ids.unsqueeze(0)
            position_ids = torch.arange(seq_len, dtype=torch.long, device=tokens.device).unsqueeze(0)
            output = raw_model(tokens, position_ids, attention_mask=None)

            logits = None
            if isinstance(output, torch.Tensor):
                logits = output
            elif isinstance(output, tuple):
                logits = output[0]
            elif hasattr(output, 'logits'):
                logits = output.logits
            elif hasattr(output, 'contiguous'):
                logits = output

            if rank == 0:
                print(f"  output type: {type(output).__name__}, logits type: "
                      f"{type(logits).__name__ if logits is not None else 'None'}")
                if logits is not None and isinstance(logits, torch.Tensor):
                    print(f"  logits shape: {logits.shape}")

            if logits is not None and isinstance(logits, torch.Tensor):
                logits_f = logits.squeeze(0).float()
                log_probs = F.log_softmax(logits_f[:-1], dim=-1)
                target = input_ids[1:]
                token_logprobs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
                captured['token_logprobs'] = token_logprobs.cpu()
                if rank == 0:
                    print(f"  token_logprobs: {len(token_logprobs)} values, "
                          f"range=[{token_logprobs.min():.4f}, {token_logprobs.max():.4f}], "
                          f"mean={token_logprobs.mean():.4f}")
            else:
                if rank == 0:
                    print(f"  WARNING: logits is None or not tensor")
                    if output is not None:
                        print(f"  output attrs: {[a for a in dir(output) if not a.startswith('_')][:20]}")

            captured['final_output'] = output

        if rank == 0:
            result = {
                "input_ids": input_ids_list,
                "layer_idx": layer_idx,
            }

            if "token_logprobs" in captured:
                result["token_logprobs"] = captured["token_logprobs"]

            if "router_scores" in captured:
                result["topk_weights"] = captured["router_scores"]
                result["topk_ids"] = captured["router_indices"]
                print(f"  topk_ids: shape={captured['router_indices'].shape}, "
                      f"topk_weights: shape={captured['router_scores'].shape}")
                n_active = captured["router_indices"].unique().numel()
                print(f"  active experts: {n_active}/128")
            else:
                print(f"  WARNING: router_scores not captured! router_module={router_module}")

            for key in ["fc1_input", "fc1_output", "fc2_input", "fc2_output",
                         "experts_input_raw",
                         "tokens_per_expert_padded", "tokens_per_expert_unpadded"]:
                if key in captured:
                    val = captured[key]
                    result[key] = val
                    if isinstance(val, torch.Tensor):
                        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
                        if val.ndim >= 2:
                            print(f"    [0,:5] = {val[0, :5].float().tolist()}")
                    else:
                        print(f"  {key}: {val}")
                else:
                    print(f"  WARNING: {key} not captured!")

            # Compute fc1_gemm_input via dequantize of the activation QDQ
            if "fc1_input" in captured:
                fc1_input_gpu = captured["fc1_input"].to(f"cuda:{local_rank}")
                tpe = captured.get("tokens_per_expert_padded")
                if isinstance(tpe, torch.Tensor):
                    m_splits = tpe.tolist()
                else:
                    m_splits = list(tpe)
                m_splits_int = [int(x) for x in m_splits]

                from miniTransformer.ops.nvfp4_quantize import (
                    group_quantize_nvfp4_pertoken_swizzled,
                    dequantize_group_nvfp4_pertoken_swizzled,
                )

                gq = group_quantize_nvfp4_pertoken_swizzled(fc1_input_gpu, m_splits_int)
                fc1_gemm_input = dequantize_group_nvfp4_pertoken_swizzled(gq)
                result["fc1_gemm_input"] = fc1_gemm_input.detach().cpu().clone()
                print(f"  fc1_gemm_input (after QDQ): shape={fc1_gemm_input.shape}")

                orig = captured["fc1_input"].float()
                qdq_rel2 = (fc1_gemm_input.float().cpu() - orig).norm(dim=-1) / orig.norm(dim=-1).clamp(min=1e-8)
                print(f"  activation QDQ rel2: mean={qdq_rel2.mean():.6f}, max={qdq_rel2.max():.6f}")

            # Compute fc2_gemm_input via dequantize of the activation QDQ
            if "fc2_input" in captured:
                fc2_input_gpu = captured["fc2_input"].to(f"cuda:{local_rank}")
                tpe = captured.get("tokens_per_expert_padded")
                if isinstance(tpe, torch.Tensor):
                    m_splits2 = tpe.tolist()
                else:
                    m_splits2 = list(tpe)
                m_splits2_int = [int(x) for x in m_splits2]

                from miniTransformer.ops.nvfp4_quantize import (
                    group_quantize_nvfp4_pertoken_swizzled,
                    dequantize_group_nvfp4_pertoken_swizzled,
                )

                gq2 = group_quantize_nvfp4_pertoken_swizzled(fc2_input_gpu, m_splits2_int)
                fc2_gemm_input = dequantize_group_nvfp4_pertoken_swizzled(gq2)
                result["fc2_gemm_input"] = fc2_gemm_input.detach().cpu().clone()
                print(f"  fc2_gemm_input (after QDQ): shape={fc2_gemm_input.shape}")

                orig2 = captured["fc2_input"].float()
                qdq_rel2_fc2 = (fc2_gemm_input.float().cpu() - orig2).norm(dim=-1) / orig2.norm(dim=-1).clamp(min=1e-8)
                print(f"  fc2 activation QDQ rel2: mean={qdq_rel2_fc2.mean():.6f}, max={qdq_rel2_fc2.max():.6f}")

            # Compute per-expert original token indices via value matching
            if "mlp_input_original" in captured and "experts_input_raw" in captured:
                print("  [DEBUG] Starting value matching...", flush=True)
                device = f"cuda:{local_rank}"
                torch.cuda.synchronize(device)
                print(f"  [DEBUG] CUDA synced. mlp_input_original type={type(captured['mlp_input_original'])}, "
                      f"shape={captured['mlp_input_original'].shape}, device={captured['mlp_input_original'].device}", flush=True)

                mlp_orig_cpu = captured["mlp_input_original"].float()
                print(f"  [DEBUG] mlp_orig_cpu .float() done, shape={mlp_orig_cpu.shape}", flush=True)
                while mlp_orig_cpu.ndim > 2:
                    mlp_orig_cpu = mlp_orig_cpu.squeeze()
                print(f"  [DEBUG] mlp_orig_cpu squeezed: shape={mlp_orig_cpu.shape}", flush=True)

                experts_raw_cpu = captured["experts_input_raw"].float()
                print(f"  [DEBUG] experts_raw_cpu .float() done, shape={experts_raw_cpu.shape}", flush=True)
                while experts_raw_cpu.ndim > 2:
                    experts_raw_cpu = experts_raw_cpu.squeeze()

                tpe_u_list = captured.get("tokens_per_expert_unpadded")
                if isinstance(tpe_u_list, torch.Tensor):
                    tpe_u_list = tpe_u_list.tolist()
                else:
                    tpe_u_list = list(tpe_u_list)

                mlp_norm = mlp_orig_cpu / mlp_orig_cpu.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                print(f"  [DEBUG] mlp_norm ready (CPU), starting expert loop ({len(tpe_u_list)} experts)...", flush=True)

                per_expert_original_indices = {}
                offset = 0
                n_poor_match = 0
                for eid in range(len(tpe_u_list)):
                    n = int(tpe_u_list[eid])
                    if n == 0:
                        continue
                    expert_tokens = experts_raw_cpu[offset:offset + n]
                    expert_norm = expert_tokens / expert_tokens.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    cos_matrix = expert_norm @ mlp_norm.T
                    original_indices = cos_matrix.argmax(dim=-1)
                    max_cos = cos_matrix.max(dim=-1).values
                    if max_cos.min() < 0.999:
                        n_poor_match += 1
                    per_expert_original_indices[eid] = original_indices.int()
                    offset += n
                    if eid % 16 == 0:
                        print(f"  [DEBUG] expert loop: eid={eid}, offset={offset}, n={n}", flush=True)

                print(f"  [DEBUG] Expert loop done.", flush=True)
                result["per_expert_original_indices"] = per_expert_original_indices
                result["mlp_input_original"] = captured["mlp_input_original"]
                print(f"  per_expert_original_indices: {len(per_expert_original_indices)} experts"
                      f" ({n_poor_match} with poor match)")
            else:
                print(f"  [DEBUG] Value matching SKIPPED: mlp_input_original={'mlp_input_original' in captured}, experts_input_raw={'experts_input_raw' in captured}")

            results[name] = result

    for h in handles:
        h.remove()

    if rank == 0:
        output_path = os.path.join(our_args.output_dir, "megatron_expert_fc1.pt")
        torch.save(results, output_path)
        print(f"\nSaved to {output_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
