"""Dump FC1+FC2 GEMM internals from sglang model (using actual sglang ModelRunner).

Loads the model via sglang's ModelRunner (same path as sgl.Engine) to get
accurate hidden_states from sglang's attention/norm kernels. Then captures
MoE layer input and does per-expert NVFP4 replay for FC1 and FC2 analysis.

Per-expert captures:
  - fc1_gemm_in: activation after NVFP4 pertoken QDQ
  - fc1_out: BF16 matmul result (fc1_gemm_in @ w13.T)
  - swiglu_out: SwiGLU output * router probs (fc2 raw input)
  - fc2_gemm_in: swiglu_out after NVFP4 pertoken QDQ
  - fc2_out: BF16 matmul result (fc2_gemm_in @ w2.T)

Also captures token_logprobs from the model's final output logits.

Usage:
    SHOULD_REPLACE_TE_GROUPLINEAR=1 SGLANG_NVFP4_PERTOKEN_SCALE=1 QAT_PARAMS=8 \
    NVTE_NVFP4_DISABLE_RHT=0 NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0 \
    NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1 \
    CUDA_VISIBLE_DEVICES=0 python dump_expert_fc1_and_fc2_sglang.py \
        --input-file test_inputs_long.jsonl \
        --output-dir dump_fc1_gemm_sglang_v2/ --layer-idx 0 \
        --moe-backend triton
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

MODEL_PATH = "/home/admin/mingfa/model/hf/Qwen3-30B-A3B/Qwen3-30B-A3B"

captured = {}


def nvfp4_qdq_pertoken(x):
    from miniTransformer.ops.nvfp4_quantize import (
        quantize_nvfp4_pertoken, dequantize_nvfp4_pertoken, _get_fp4_grid,
    )
    _get_fp4_grid(x.device)
    k = x.shape[-1]
    pad_k = ((k + 127) // 128) * 128
    if pad_k > k:
        x_pad = torch.zeros(*x.shape[:-1], pad_k, dtype=x.dtype, device=x.device)
        x_pad[..., :k] = x
    else:
        x_pad = x
    qr = quantize_nvfp4_pertoken(x_pad)
    return dequantize_nvfp4_pertoken(qr).to(torch.bfloat16)[..., :k]


def load_inputs(path):
    inputs = []
    with open(path) as f:
        for line in f:
            inputs.append(json.loads(line))
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default="/root/compare_logprob/test_inputs_long.jsonl")
    parser.add_argument("--output-dir", default="/root/compare_logprob/dump_fc1_gemm_sglang_v2/")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--mem-fraction", type=float, default=0.8)
    parser.add_argument("--moe-backend", default="triton",
                        help="MoE runner backend: triton (standard 3D weight) or flashinfer (4D tiled)")
    parser.add_argument("--hf-weights", default=None,
                        help="HF raw w13 weights .pt (from compare_expert_fc1_and_fc2.py --mode dump-hf). "
                             "Required when w13_weight is 4D (flashinfer_trtllm backend).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sys.path.insert(0, '/root/miniTransformer')

    print(f"SHOULD_REPLACE_TE_GROUPLINEAR={os.environ.get('SHOULD_REPLACE_TE_GROUPLINEAR', '')}")
    print(f"QAT_PARAMS={os.environ.get('QAT_PARAMS', '')}")
    # Ensure SGLANG_NVFP4_PERTOKEN_SCALE is set
    if not os.environ.get("SGLANG_NVFP4_PERTOKEN_SCALE"):
        os.environ["SGLANG_NVFP4_PERTOKEN_SCALE"] = "1"
    print(f"SGLANG_NVFP4_PERTOKEN_SCALE={os.environ.get('SGLANG_NVFP4_PERTOKEN_SCALE', '')}")

    # --- Initialize sglang ModelRunner ---
    from sglang.srt.server_args import ServerArgs, PortArgs
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.utils import suppress_other_loggers
    from sglang.bench_one_batch import TreeCacheNamespace

    suppress_other_loggers()

    server_args = ServerArgs(
        model_path=args.model_path,
        tokenizer_path=args.model_path,
        host="127.0.0.1",
        port=30000,
        disable_cuda_graph=True,
        tp_size=1,
        mem_fraction_static=args.mem_fraction,
        trust_remote_code=True,
        moe_runner_backend=args.moe_backend,
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

    print(f"Loading model via sglang ModelRunner from {args.model_path} ...")
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

    model = model_runner.model
    layer_idx = args.layer_idx
    layer = model.model.layers[layer_idx]
    moe = layer.mlp
    experts = moe.experts

    print(f"\nLayer {layer_idx}: {type(moe).__name__}")
    print(f"  experts: {type(experts).__name__}")
    if hasattr(experts, 'w13_weight'):
        print(f"  w13_weight shape: {experts.w13_weight.shape}, dtype: {experts.w13_weight.dtype}")
    if hasattr(experts, 'w2_weight'):
        print(f"  w2_weight shape: {experts.w2_weight.shape}, dtype: {experts.w2_weight.dtype}")

    # --- Register hooks ---
    def moe_pre_hook(module, args_tuple):
        x = args_tuple[0]
        captured["mlp_input"] = x.detach().clone()

    def gate_post_hook(module, args_tuple, output):
        router_logits = output[0] if isinstance(output, tuple) else output
        captured["router_logits"] = router_logits.detach().clone()

    def logits_pre_hook(module, args_tuple):
        captured["final_hidden_states"] = args_tuple[1].detach().clone()

    handles = [
        moe.register_forward_pre_hook(moe_pre_hook),
        moe.gate.register_forward_hook(gate_post_hook),
        model.logits_processor.register_forward_pre_hook(logits_pre_hook),
    ]
    print(f"Hooks registered on layer {layer_idx} MoE + logits_processor")

    hf_weights = None
    if args.hf_weights:
        print(f"\nLoading HF raw weights from {args.hf_weights}...")
        hf_weights = torch.load(args.hf_weights, map_location="cpu", weights_only=False)
        print(f"  Loaded {len(hf_weights)} expert w13 weights")

    # --- Load inputs and run ---
    test_inputs = load_inputs(args.input_file)
    results = {}

    for inp_idx, inp in enumerate(test_inputs):
        name = inp["name"]
        input_ids_list = inp["input_ids"]
        seq_len = len(input_ids_list)
        print(f"\nProcessing: {name} ({seq_len} tokens)")

        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool_allocator.clear()
        captured.clear()

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

        # --- Compute token logprobs from captured hidden states ---
        if "final_hidden_states" in captured:
            hs = captured["final_hidden_states"].float()
            lm_weight = model.lm_head.weight.float()
            logits = hs @ lm_weight.T
            log_probs = F.log_softmax(logits[:-1], dim=-1)
            input_ids_tensor = torch.tensor(input_ids_list, device=logits.device)
            target = input_ids_tensor[1:]
            token_logprobs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
            captured["token_logprobs"] = token_logprobs.cpu()
            print(f"  token_logprobs: {len(token_logprobs)} values, "
                  f"range=[{token_logprobs.min():.4f}, {token_logprobs.max():.4f}], "
                  f"mean={token_logprobs.mean():.4f}")
        else:
            print("  WARNING: final_hidden_states not captured!")

        # --- Per-expert FC1 replay ---
        if "mlp_input" not in captured:
            print("  WARNING: mlp_input not captured!")
            results[name] = {"input_ids": input_ids_list, "layer_idx": layer_idx}
            continue

        mlp_input = captured["mlp_input"]
        if mlp_input.ndim == 3:
            mlp_input = mlp_input.squeeze(0)

        if "router_logits" in captured:
            routing_weights = F.softmax(captured["router_logits"], dim=-1)
        else:
            router_logits, _ = moe.gate(mlp_input.to(moe.gate.weight.dtype))
            routing_weights = F.softmax(router_logits, dim=-1)

        topk_weights, topk_ids = torch.topk(routing_weights, k=8, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        active_experts = topk_ids.unique().tolist()
        print(f"  active experts: {len(active_experts)}")

        fc1_gemm_input_all = nvfp4_qdq_pertoken(mlp_input)

        w13_weight = experts.w13_weight
        is_4d = w13_weight.ndim == 4
        if inp_idx == 0:
            if is_4d:
                print(f"  w13_weight is 4D (FlashInfer tiled): {w13_weight.shape}")
                if hf_weights is None:
                    print(f"  WARNING: 4D tiled weights cannot be untiled correctly. "
                          f"Use --hf-weights to provide raw HF weights for fc1 replay.")
                else:
                    print(f"  Using HF raw weights for fc1 replay (bypassing 4D tiling)")
            else:
                print(f"  w13_weight shape: {w13_weight.shape} (standard 3D)")

        per_expert = {}
        for eid in sorted(active_experts):
            mask = (topk_ids == eid).any(dim=-1)
            tidx = mask.nonzero(as_tuple=True)[0]
            if len(tidx) == 0:
                continue

            x_tok = fc1_gemm_input_all[tidx]

            if hf_weights is not None and eid in hf_weights:
                w13_e = hf_weights[eid].to(device=x_tok.device)
            elif not is_4d:
                w13_e = w13_weight[eid]
            else:
                per_expert[eid] = {
                    "token_indices": tidx.cpu(),
                    "fc1_gemm_in": x_tok.detach().cpu(),
                }
                continue

            fc1_out = (x_tok.float() @ w13_e.float().T).to(x_tok.dtype)

            # FC2 replay: SwiGLU → ×probs → QDQ → matmul
            moe_ffn = w13_e.shape[0] // 2  # 768
            swiglu = F.silu(fc1_out[:, :moe_ffn]) * fc1_out[:, moe_ffn:]

            expert_mask = (topk_ids[tidx] == eid)  # [n_tok, topk_k]
            prob_per_tok = (topk_weights[tidx] * expert_mask.float()).sum(dim=-1)
            swiglu_probs = swiglu * prob_per_tok.view(-1, 1).to(swiglu.dtype)

            fc2_gemm_in = nvfp4_qdq_pertoken(swiglu_probs)
            w2_e = experts.w2_weight[eid]  # [2048, 768]
            fc2_out = (fc2_gemm_in.float() @ w2_e.float().T).to(fc2_gemm_in.dtype)

            per_expert[eid] = {
                "token_indices": tidx.cpu(),
                "fc1_gemm_in": x_tok.detach().cpu(),
                "fc1_out": fc1_out.detach().cpu(),
                "swiglu_out": swiglu_probs.detach().cpu(),
                "fc2_gemm_in": fc2_gemm_in.detach().cpu(),
                "fc2_out": fc2_out.detach().cpu(),
            }

        print(f"  {len(per_expert)} experts computed")

        results[name] = {
            "input_ids": input_ids_list,
            "layer_idx": layer_idx,
            "mlp_input": mlp_input.cpu(),
            "topk_ids": topk_ids.cpu(),
            "topk_weights": topk_weights.cpu(),
            "router_logits": captured.get("router_logits", routing_weights).cpu(),
            "per_expert": per_expert,
            "active_experts": active_experts,
            "token_logprobs": captured.get("token_logprobs"),
        }

    for h in handles:
        h.remove()

    output_path = os.path.join(args.output_dir, "fc1_gemm_sglang_v2.pt")
    torch.save(results, output_path)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
