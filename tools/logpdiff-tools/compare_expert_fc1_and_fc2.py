"""Compare expert fc1 activations and weights between sglang and Megatron.

Modes:
  activations (default) - Compare fc1 activation dumps (gemm_input, output, etc.)
  dump-hf   - Dump per-expert w13 weights from HF safetensors (no GPU)
  dump-mg   - Dump per-expert fc1 weights from Megatron model (torchrun)
  weights   - Compare two weight dump .pt files

Usage:
    # Compare activations (original behavior)
    python compare_expert_fc1_and_fc2.py --mode activations \
        --sglang-dump dump_fc1_gemm_sglang_v2/fc1_gemm_sglang_v2.pt \
        --megatron-dump dump_expert_fc1_megatron/megatron_expert_fc1.pt

    # Compare activations with raw-matmul reference (recommended)
    python compare_expert_fc1_and_fc2.py --mode activations \
        --sglang-dump dump_fc1_gemm_sglang_v2/fc1_gemm_sglang_v2.pt \
        --megatron-dump dump_expert_fc1_megatron/megatron_expert_fc1.pt \
        --hf-weights dump_weights/hf_w13.pt

    # Dump HF weights (no GPU needed)
    python compare_expert_fc1_and_fc2.py --mode dump-hf --layer-idx 0

    # Dump Megatron weights (needs torchrun + env vars)
    SHOULD_REPLACE_TE_GROUPLINEAR=1 QAT_PARAMS=8 \
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    compare_expert_fc1_and_fc2.py --mode dump-mg --layer-idx 0 --ep-size 1

    # Compare weight dumps
    python compare_expert_fc1_and_fc2.py --mode weights \
        --hf-dump dump_weights/hf_w13.pt --mg-dump dump_weights/mg_fc1.pt
"""

import argparse
import glob
import gc
import math
import os
import sys
import time

import torch
import torch.nn.functional as F


MODEL_PATH = "/home/admin/mingfa/model/hf/Qwen3-30B-A3B/Qwen3-30B-A3B"
REF_LOAD = "/home/admin/mingfa/model/torch_dist/Qwen3-30B-A3B_torch_dist/"


def cos_sim(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


def cos_sim_flat(a, b):
    return F.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0),
    ).item()


def stats_line(name, a, b):
    a_f, b_f = a.float(), b.float()
    diff = (a_f - b_f).abs()
    cos = cos_sim(a, b)
    rel = diff / b_f.abs().clamp(min=1e-8)
    print(f"  {name}:")
    print(f"    shape: sg={list(a.shape)}, mg={list(b.shape)}")
    print(f"    abs_diff: mean={diff.mean():.6f}, max={diff.max():.6f}")
    print(f"    cos_sim:  mean={cos.mean():.6f}, min={cos.min():.6f}")
    print(f"    rel_err:  mean={rel.mean():.6f}, max={rel.max():.6f}")
    return cos.mean().item(), cos.min().item()


# ============================================================
# Weight dump/compare functions
# ============================================================

def load_hf_weights(layer_idx):
    from safetensors import safe_open

    st_files = sorted(glob.glob(os.path.join(MODEL_PATH, "*.safetensors")))
    if not st_files:
        raise FileNotFoundError(f"No safetensors in {MODEL_PATH}")
    print(f"Found {len(st_files)} safetensors files")

    prefix = f"model.layers.{layer_idx}.mlp.experts."
    gate_w, up_w = {}, {}

    for st_file in st_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if not key.startswith(prefix):
                    continue
                rest = key[len(prefix):]
                parts = rest.split(".")
                eid = int(parts[0])
                proj = parts[1]
                if proj == "gate_proj":
                    gate_w[eid] = f.get_tensor(key)
                elif proj == "up_proj":
                    up_w[eid] = f.get_tensor(key)

    result = {}
    for eid in sorted(gate_w.keys()):
        if eid in up_w:
            w13 = torch.cat([gate_w[eid], up_w[eid]], dim=0)
            result[eid] = w13

    print(f"Loaded {len(result)} expert w13 from HF safetensors")
    if result:
        eid0 = min(result.keys())
        print(f"  expert {eid0}: shape={list(result[eid0].shape)}, dtype={result[eid0].dtype}")
    return result


def setup_megatron(our_args):
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    ep = our_args.ep_size
    sys.argv = [
        "compare_expert_fc1_and_fc2.py",
        "--disable-bias-linear", "--qk-layernorm",
        "--group-query-attention",
        "--num-attention-heads", "32", "--num-query-groups", "4",
        "--kv-channels", "128", "--num-layers", "48",
        "--hidden-size", "2048", "--ffn-hidden-size", "6144",
        "--normalization", "RMSNorm",
        "--position-embedding-type", "rope",
        "--norm-epsilon", "1e-6", "--rotary-percent", "1.0",
        "--swiglu", "--untie-embeddings-and-output-weights",
        "--vocab-size", "151936", "--rotary-base", "1000000",
        "--moe-ffn-hidden-size", "768",
        "--moe-router-score-function", "softmax",
        "--moe-token-dispatcher-type", "alltoall",
        "--moe-router-topk", "8", "--num-experts", "128",
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
        "--fp4-format", "e2m1", "--fp4-recipe", "nvfp4",
        "--attention-dropout", "0.0", "--hidden-dropout", "0.0",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend", "flash",
        "--micro-batch-size", "1",
        "--global-batch-size", str(ep),
        "--seq-length", "4096",
        "--lr", "1e-6", "--lr-decay-style", "constant",
        "--weight-decay", "0.1",
        "--adam-beta1", "0.9", "--adam-beta2", "0.98",
        "--recompute-granularity", "full",
        "--recompute-method", "uniform",
        "--recompute-num-layers", "1",
        "--tokenizer-type", "NullTokenizer",
        "--hf-checkpoint", our_args.model_path,
        "--ref-load", our_args.ref_load_path,
        "--load", our_args.ref_load_path,
        "--lr-warmup-iters", "0",
        "--lr-decay-iters", "10", "--train-iters", "10",
    ]

    sys.path.insert(0, '/root/Megatron-LM')
    sys.path.insert(0, '/root/miniTransformer')

    from megatron.training.arguments import parse_args as mg_parse_args
    from megatron.training.arguments import validate_args as mg_validate_args
    args = mg_parse_args(extra_args_provider=None, ignore_unknown_args=True)
    args.hf_checkpoint = our_args.model_path
    args.ref_load = our_args.ref_load_path
    args.load = our_args.ref_load_path

    from slime.backends.megatron_utils.arguments import set_default_megatron_args
    args = set_default_megatron_args(args)

    defaults = {
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
        'te_rng_tracker': False, 'inference_rng_tracker': False,
        'rampup_batch_size': None,
        'decrease_batch_size_if_needed': False,
        'deterministic_mode': False, 'tp_comm_overlap': False,
        'custom_megatron_init_path': None, 'seed': 1234,
        'data_parallel_size': 1,
        'dist_ckpt_save_pre_mcore_014': True,
        'megatron_to_hf_mode': None,
        'custom_model_provider_path': None,
        'moe_use_legacy_grouped_gemm': False,
        'multi_latent_attention': False, 'fp8_param_gather': False,
        'only_train_params_name_list': None,
        'freeze_params_name_list': ['.*'],
        'mtp_num_layers': None, 'moe_use_upcycling': False,
        'pretrained_checkpoint': None,
        'variable_seq_lengths': True,
        'decoder_first_pipeline_num_layers': None,
        'decoder_last_pipeline_num_layers': None,
        'num_rollout': 1, 'rollout_batch_size': 1,
        'n_samples_per_prompt': 1,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    mg_validate_args(args)

    from slime.backends.megatron_utils.initialize import init
    init(args)

    if rank == 0:
        print("Megatron initialized. Loading model...")

    from slime.backends.megatron_utils.model import initialize_model_and_optimizer
    from megatron.training import get_args

    t0 = time.time()
    model_list, _, _, _ = initialize_model_and_optimizer(get_args(), "actor")
    model = model_list[0]
    model.eval()

    if rank == 0:
        print(f"Model loaded in {time.time() - t0:.1f}s")

    raw_model = model
    while hasattr(raw_model, 'module'):
        raw_model = raw_model.module

    return raw_model, rank, local_rank


def extract_mg_weights(raw_model, layer_idx, rank):
    layer = raw_model.decoder.layers[layer_idx]
    experts = layer.mlp.experts if hasattr(layer.mlp, 'experts') else None
    if experts is None:
        print(f"ERROR: No experts in layer {layer_idx}")
        return None

    linear_fc1 = experts.linear_fc1
    if rank == 0:
        print(f"\nlinear_fc1: {type(linear_fc1).__name__}")
        print(f"  Named parameters:")
        for pname, p in linear_fc1.named_parameters():
            print(f"    {pname}: shape={list(p.shape)}, dtype={p.dtype}")

    num_experts = 128
    if hasattr(linear_fc1, 'num_gemms'):
        num_experts = linear_fc1.num_gemms

    result = {}
    for eid in range(num_experts):
        attr = f"weight{eid}"
        if hasattr(linear_fc1, attr):
            w = getattr(linear_fc1, attr)
            if isinstance(w, (torch.Tensor, torch.nn.Parameter)):
                w_cpu = w.detach().cpu()
                if w_cpu.dtype not in (torch.bfloat16, torch.float32, torch.float16):
                    if hasattr(w, 'dequantize'):
                        w_cpu = w.dequantize().detach().cpu().to(torch.bfloat16)
                result[eid] = w_cpu.clone()

    if result:
        if rank == 0:
            eid0 = min(result.keys())
            print(f"\n  Extracted {len(result)} expert weights from weight{{i}} params")
            print(f"  expert {eid0}: shape={list(result[eid0].shape)}, dtype={result[eid0].dtype}")
        return result

    for attr in ['weight', 'weight_master']:
        if hasattr(linear_fc1, attr):
            w = getattr(linear_fc1, attr)
            if isinstance(w, (torch.Tensor, torch.nn.Parameter)):
                if rank == 0:
                    print(f"\n  Found combined weight '{attr}': shape={list(w.shape)}")
                w_cpu = w.detach().cpu().float()
                if w_cpu.ndim == 3:
                    for eid in range(w_cpu.shape[0]):
                        result[eid] = w_cpu[eid].to(torch.bfloat16)
                elif w_cpu.ndim == 2:
                    es = w_cpu.shape[0] // num_experts
                    for eid in range(num_experts):
                        result[eid] = w_cpu[eid * es:(eid + 1) * es].to(torch.bfloat16)
                if result:
                    return result

    if rank == 0:
        print(f"\n  ERROR: Could not extract weights from linear_fc1")
    return result if result else None


def compare_weights(hf_weights, mg_weights):
    print(f"\n{'='*70}")
    print(f"  Weight Comparison: HF vs Megatron")
    print(f"{'='*70}")

    hf_eids = set(hf_weights.keys())
    mg_eids = set(mg_weights.keys())
    common = sorted(hf_eids & mg_eids)
    print(f"  HF experts: {len(hf_eids)}, MG experts: {len(mg_eids)}, Common: {len(common)}")

    if not common:
        print("  ERROR: No common expert IDs!")
        return

    eid0 = common[0]
    hf_s, mg_s = list(hf_weights[eid0].shape), list(mg_weights[eid0].shape)
    print(f"  HF shape: {hf_s}, dtype: {hf_weights[eid0].dtype}")
    print(f"  MG shape: {mg_s}, dtype: {mg_weights[eid0].dtype}")

    need_transpose = False
    if hf_s != mg_s and hf_s == mg_s[::-1]:
        need_transpose = True
        print(f"  NOTE: shapes are transposed, will transpose MG weights for comparison")

    exact = (hf_weights[eid0].float() - mg_weights[eid0].float()).abs().max().item()
    if exact == 0.0:
        print(f"\n  >>> EXACT MATCH on expert {eid0}! Weights are bit-identical.")

    print(f"\n  {'eid':>4} {'cos_w13':>10} {'cos_gate':>10} {'cos_up':>10} "
          f"{'max_abs':>10} {'exact':>6}")
    print(f"  {'----':>4} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10} "
          f"{'-'*10:>10} {'------':>6}")

    all_cos = []
    n_exact = 0
    for eid in common:
        hf_w = hf_weights[eid].float()
        mg_w = mg_weights[eid].float()
        if need_transpose:
            mg_w = mg_w.T

        if hf_w.shape != mg_w.shape:
            print(f"  {eid:>4} SHAPE MISMATCH: hf={list(hf_w.shape)}, mg={list(mg_w.shape)}")
            continue

        is_exact = torch.equal(hf_w, mg_w)
        if is_exact:
            n_exact += 1

        cos_w13 = cos_sim_flat(hf_w, mg_w)
        half = hf_w.shape[0] // 2
        cos_gate = cos_sim_flat(hf_w[:half], mg_w[:half])
        cos_up = cos_sim_flat(hf_w[half:], mg_w[half:])

        diff = (hf_w - mg_w).abs()
        max_abs = diff.max().item()
        all_cos.append(cos_w13)

        if len(common) <= 20 or eid < 10 or eid >= common[-3]:
            print(f"  {eid:>4} {cos_w13:>10.6f} {cos_gate:>10.6f} {cos_up:>10.6f} "
                  f"{max_abs:>10.6f} {'YES' if is_exact else 'no':>6}")

    if len(common) > 23:
        print(f"  ... ({len(common) - 13} experts omitted)")

    if all_cos:
        print(f"\n  Aggregate ({len(all_cos)} experts):")
        print(f"    cos_w13: mean={sum(all_cos)/len(all_cos):.6f}, min={min(all_cos):.6f}")
        print(f"    exact matches: {n_exact}/{len(all_cos)}")

    if all_cos and min(all_cos) < 0.5:
        print(f"\n  >>> LOW COSINE - cross-expert matching (first 5 HF experts)...")
        for hf_eid in common[:5]:
            best_cos, best_eid = -1, -1
            hf_w = hf_weights[hf_eid].float()
            for mg_eid in common:
                mg_w = mg_weights[mg_eid].float()
                if need_transpose:
                    mg_w = mg_w.T
                if hf_w.shape != mg_w.shape:
                    continue
                c = cos_sim_flat(hf_w, mg_w)
                if c > best_cos:
                    best_cos = c
                    best_eid = mg_eid
            print(f"    HF[{hf_eid}] best match -> MG[{best_eid}] cos={best_cos:.6f}")


# ============================================================
# Activation comparison functions
# ============================================================

def compare_topk(sg_topk, mg_topk, sg_tw, mg_tw):
    print(f"\n  --- Top-K Routing ---")
    print(f"  shapes: sg={list(sg_topk.shape)}, mg={list(mg_topk.shape)}")

    n_tokens = min(sg_topk.shape[0], mg_topk.shape[0])
    sg_t = sg_topk[:n_tokens]
    mg_t = mg_topk[:n_tokens]

    exact = (sg_t == mg_t).all(dim=-1).float().mean()
    print(f"  exact match (position-wise): {exact:.4f}")

    sg_s = sg_t.sort(dim=-1).values
    mg_s = mg_t.sort(dim=-1).values
    set_match = (sg_s == mg_s).all(dim=-1).float().mean()
    print(f"  set match (ignoring order):  {set_match:.4f}")

    overlaps = []
    for i in range(n_tokens):
        overlaps.append(len(set(sg_t[i].tolist()) & set(mg_t[i].tolist())))
    ov = torch.tensor(overlaps, dtype=torch.float)
    print(f"  per-token overlap: mean={ov.mean():.2f}/8, min={ov.min().int().item()}/8")

    sg_active = set(sg_t.view(-1).tolist())
    mg_active = set(mg_t.view(-1).tolist())
    print(f"  active experts: sg={len(sg_active)}, mg={len(mg_active)}, "
          f"intersection={len(sg_active & mg_active)}")

    if sg_tw is not None and mg_tw is not None:
        sg_w = sg_tw[:n_tokens].float()
        mg_w = mg_tw[:n_tokens].float()
        match_mask = (sg_t == mg_t)
        if match_mask.any():
            w_diff = (sg_w[match_mask] - mg_w[match_mask]).abs()
            print(f"  topk_weights (where experts match): "
                  f"mean_diff={w_diff.mean():.6f}, max_diff={w_diff.max():.6f}")

    return set_match.item()


def compare_token_logprobs(sg_lp, mg_lp):
    print(f"\n  --- Token Logprobs ---")
    n = min(len(sg_lp), len(mg_lp))
    sg_lp = sg_lp[:n].float()
    mg_lp = mg_lp[:n].float()
    diff = (sg_lp - mg_lp).abs()
    print(f"  num_tokens: sg={len(sg_lp)}, mg={len(mg_lp)}")
    print(f"  abs_diff: mean={diff.mean():.6f}, max={diff.max():.6f}, "
          f"p95={torch.quantile(diff, 0.95):.6f}")
    print(f"  sg  range: [{sg_lp.min():.4f}, {sg_lp.max():.4f}], mean={sg_lp.mean():.4f}")
    print(f"  mg  range: [{mg_lp.min():.4f}, {mg_lp.max():.4f}], mean={mg_lp.mean():.4f}")

    large_diff_mask = diff > 0.1
    n_large = large_diff_mask.sum().item()
    print(f"  tokens with |diff|>0.1: {n_large}/{n} ({n_large/n*100:.2f}%)")
    if n_large > 0 and n_large <= 20:
        idx = large_diff_mask.nonzero(as_tuple=True)[0]
        for i in idx[:10]:
            print(f"    token[{i.item()}]: sg={sg_lp[i]:.4f}, mg={mg_lp[i]:.4f}, "
                  f"diff={diff[i]:.4f}")
    return diff.mean().item()


def compare_mlp_input(sg_mlp, mg_mlp):
    print(f"\n  --- MLP Input (MoE layer input) ---")
    while sg_mlp.ndim > 2:
        sg_mlp = sg_mlp.squeeze()
    while mg_mlp.ndim > 2:
        mg_mlp = mg_mlp.squeeze()

    n = min(sg_mlp.shape[0], mg_mlp.shape[0])
    sg_mlp = sg_mlp[:n]
    mg_mlp = mg_mlp[:n]
    return stats_line("mlp_input", sg_mlp, mg_mlp)


def compare_per_expert_fc1(sg_per_expert, mg_data, hf_weights=None):
    print(f"\n  --- Per-Expert FC1 (token-aligned) ---")

    mg_fc1_in = mg_data.get("fc1_input")
    mg_fc1_gemm = mg_data.get("fc1_gemm_input")
    mg_fc1_out = mg_data.get("fc1_output")
    mg_indices = mg_data.get("per_expert_original_indices", {})
    tpe_padded = mg_data.get("tokens_per_expert_padded")
    tpe_unpadded = mg_data.get("tokens_per_expert_unpadded")

    if mg_fc1_out is None or tpe_padded is None:
        print("  WARNING: megatron fc1_output or tokens_per_expert_padded missing!")
        return None, None

    if isinstance(tpe_padded, torch.Tensor):
        tpe_p = tpe_padded.tolist()
    else:
        tpe_p = list(tpe_padded)
    if isinstance(tpe_unpadded, torch.Tensor):
        tpe_u = tpe_unpadded.tolist()
    else:
        tpe_u = list(tpe_unpadded) if tpe_unpadded is not None else tpe_p
    tpe_p = [int(x) for x in tpe_p]
    tpe_u = [int(x) for x in tpe_u]

    if mg_fc1_out.ndim == 3:
        mg_fc1_out = mg_fc1_out.squeeze(0)
    if mg_fc1_gemm is not None and mg_fc1_gemm.ndim == 3:
        mg_fc1_gemm = mg_fc1_gemm.squeeze(0)

    swap_cos_all = []
    has_hf = hf_weights is not None
    raw_cos_all = []

    hdr_extra = f" {'raw_cos':>11}" if has_hf else ""
    sep_extra = f" {'-----------':>11}" if has_hf else ""
    print(f"  {'eid':>4} {'sg_n':>5} {'mg_n':>5} {'common':>6} "
          f"{'gemm_in_cos':>11} {'fc1_out_cos':>11} {'fc1_out_cos_min':>15}"
          f" {'swap_cos':>11}{hdr_extra}")
    print(f"  {'----':>4} {'-----':>5} {'-----':>5} {'------':>6} "
          f"{'-----------':>11} {'-----------':>11} {'---------------':>15}"
          f" {'-----------':>11}{sep_extra}")

    gemm_in_cos_all = []
    fc1_out_cos_all = []

    for eid in sorted(sg_per_expert.keys()):
        sg_e = sg_per_expert[eid]
        sg_indices = sg_e["token_indices"]
        sg_gemm_in = sg_e["fc1_gemm_in"]
        sg_fc1_out_e = sg_e.get("fc1_out")
        n_sg = sg_gemm_in.shape[0]

        n_unpadded = int(tpe_u[eid])
        if n_unpadded == 0:
            continue
        expert_offset = sum(tpe_p[:eid])

        mg_out_e = mg_fc1_out[expert_offset:expert_offset + n_unpadded]
        mg_gemm_e = (mg_fc1_gemm[expert_offset:expert_offset + n_unpadded]
                     if mg_fc1_gemm is not None else None)

        mg_idx = mg_indices.get(eid) if mg_indices else None
        if mg_idx is not None:
            sg_idx_map = {int(v): i for i, v in enumerate(sg_indices)}
            mg_idx_map = {int(v): i for i, v in enumerate(mg_idx)}
            common = sorted(set(sg_idx_map.keys()) & set(mg_idx_map.keys()))
            n_common = len(common)
            if n_common == 0:
                print(f"  {eid:>4} {n_sg:>5} {n_unpadded:>5} {'0':>6} "
                      f"{'N/A':>11} {'N/A':>11} {'N/A':>15}")
                continue
            sg_rows = [sg_idx_map[t] for t in common]
            mg_rows = [mg_idx_map[t] for t in common]
        else:
            n_common = min(n_sg, n_unpadded)
            sg_rows = list(range(n_common))
            mg_rows = list(range(n_common))

        gemm_cos_str = "N/A"
        if mg_gemm_e is not None:
            sg_gi = sg_gemm_in[sg_rows]
            mg_gi = mg_gemm_e[mg_rows]
            gi_cos = cos_sim(sg_gi, mg_gi)
            gemm_cos_str = f"{gi_cos.mean():.6f}"
            gemm_in_cos_all.append(gi_cos.mean().item())

        mg_fo = mg_out_e[mg_rows]

        fc1_cos_str = "N/A"
        fc1_min_str = f"{'N/A':>15}"
        swap_cos_val = float('nan')
        if sg_fc1_out_e is not None:
            sg_fo = sg_fc1_out_e[sg_rows]
            fo_cos = cos_sim(sg_fo, mg_fo)
            fc1_out_cos_all.append(fo_cos.mean().item())
            fc1_cos_str = f"{fo_cos.mean():.6f}"
            fc1_min_str = f"{fo_cos.min():>15.6f}"

            half = sg_fo.shape[-1] // 2
            if half > 0 and sg_fo.shape[-1] == mg_fo.shape[-1]:
                swap_cos = cos_sim(
                    torch.cat([sg_fo[:, half:], sg_fo[:, :half]], dim=-1),
                    mg_fo,
                )
                swap_cos_val = swap_cos.mean().item()
        swap_cos_all.append(swap_cos_val)

        raw_cos_str = ""
        if has_hf and eid in hf_weights:
            hf_w = hf_weights[eid].float()
            sg_act = sg_gemm_in[sg_rows].float()
            raw_out = (sg_act @ hf_w.T).to(sg_gemm_in.dtype)
            raw_cos_v = cos_sim(raw_out.float(), mg_fo.float())
            raw_cos_val = raw_cos_v.mean().item()
            raw_cos_all.append(raw_cos_val)
            raw_cos_str = f" {raw_cos_val:>11.6f}"

        print(f"  {eid:>4} {n_sg:>5} {n_unpadded:>5} {n_common:>6} "
              f"{gemm_cos_str:>11} {fc1_cos_str:>11} {fc1_min_str}"
              f" {swap_cos_val:>11.6f}{raw_cos_str}")

    has_results = fc1_out_cos_all or raw_cos_all
    if has_results:
        n_agg = max(len(fc1_out_cos_all), len(raw_cos_all))
        print(f"\n  Aggregate ({n_agg} experts):")
        if gemm_in_cos_all:
            print(f"    fc1_gemm_input cos: mean={sum(gemm_in_cos_all)/len(gemm_in_cos_all):.6f}")
        if fc1_out_cos_all:
            print(f"    fc1_output cos:     mean={sum(fc1_out_cos_all)/len(fc1_out_cos_all):.6f}, "
                  f"min={min(fc1_out_cos_all):.6f}")
        if raw_cos_all:
            print(f"    raw_matmul cos:     mean={sum(raw_cos_all)/len(raw_cos_all):.6f}, "
                  f"min={min(raw_cos_all):.6f}")
        valid_swap = [v for v in swap_cos_all if not math.isnan(v)]
        if valid_swap:
            avg_swap = sum(valid_swap) / len(valid_swap)
            print(f"    fc1_out SWAPPED cos: mean={avg_swap:.6f}, "
                  f"min={min(valid_swap):.6f}")
            if fc1_out_cos_all and avg_swap > sum(fc1_out_cos_all)/len(fc1_out_cos_all) + 0.1:
                print(f"    >>> GATE/UP SWAP DETECTED: swapped cos ({avg_swap:.4f}) >> "
                      f"direct cos ({sum(fc1_out_cos_all)/len(fc1_out_cos_all):.4f})")

    avg_gemm = sum(gemm_in_cos_all)/len(gemm_in_cos_all) if gemm_in_cos_all else None
    avg_out = sum(fc1_out_cos_all)/len(fc1_out_cos_all) if fc1_out_cos_all else None
    return avg_gemm, avg_out


def compare_per_expert_fc2(sg_per_expert, mg_data):
    """Compare FC2 intermediates between sglang and Megatron per-expert."""
    print(f"\n  --- Per-Expert FC2 (token-aligned) ---")

    mg_fc2_in = mg_data.get("fc2_input")
    mg_fc2_gemm = mg_data.get("fc2_gemm_input")
    mg_fc2_out = mg_data.get("fc2_output")
    mg_indices = mg_data.get("per_expert_original_indices", {})
    tpe_padded = mg_data.get("tokens_per_expert_padded")
    tpe_unpadded = mg_data.get("tokens_per_expert_unpadded")

    if mg_fc2_out is None or tpe_padded is None:
        print("  WARNING: megatron fc2_output or tokens_per_expert_padded missing!")
        return None, None

    if isinstance(tpe_padded, torch.Tensor):
        tpe_p = tpe_padded.tolist()
    else:
        tpe_p = list(tpe_padded)
    if isinstance(tpe_unpadded, torch.Tensor):
        tpe_u = tpe_unpadded.tolist()
    else:
        tpe_u = list(tpe_unpadded) if tpe_unpadded is not None else tpe_p
    tpe_p = [int(x) for x in tpe_p]
    tpe_u = [int(x) for x in tpe_u]

    if mg_fc2_out.ndim == 3:
        mg_fc2_out = mg_fc2_out.squeeze(0)
    if mg_fc2_gemm is not None and mg_fc2_gemm.ndim == 3:
        mg_fc2_gemm = mg_fc2_gemm.squeeze(0)
    if mg_fc2_in is not None and mg_fc2_in.ndim == 3:
        mg_fc2_in = mg_fc2_in.squeeze(0)

    print(f"  {'eid':>4} {'sg_n':>5} {'mg_n':>5} {'common':>6} "
          f"{'fc2_in_cos':>11} {'fc2_gemm_cos':>12} {'fc2_out_cos':>11} {'fc2_out_min':>11}")
    print(f"  {'----':>4} {'-----':>5} {'-----':>5} {'------':>6} "
          f"{'-----------':>11} {'------------':>12} {'-----------':>11} {'-----------':>11}")

    fc2_in_cos_all = []
    fc2_gemm_cos_all = []
    fc2_out_cos_all = []

    for eid in sorted(sg_per_expert.keys()):
        sg_e = sg_per_expert[eid]
        sg_indices = sg_e["token_indices"]
        sg_fc2_in = sg_e.get("swiglu_out")
        sg_fc2_gemm = sg_e.get("fc2_gemm_in")
        sg_fc2_out_e = sg_e.get("fc2_out")

        if sg_fc2_out_e is None:
            continue

        n_sg = sg_fc2_out_e.shape[0]
        n_unpadded = int(tpe_u[eid])
        if n_unpadded == 0:
            continue
        expert_offset = sum(tpe_p[:eid])

        mg_out_e = mg_fc2_out[expert_offset:expert_offset + n_unpadded]
        mg_gemm_e = (mg_fc2_gemm[expert_offset:expert_offset + n_unpadded]
                     if mg_fc2_gemm is not None else None)
        mg_in_e = (mg_fc2_in[expert_offset:expert_offset + n_unpadded]
                   if mg_fc2_in is not None else None)

        mg_idx = mg_indices.get(eid) if mg_indices else None
        if mg_idx is not None:
            sg_idx_map = {int(v): i for i, v in enumerate(sg_indices)}
            mg_idx_map = {int(v): i for i, v in enumerate(mg_idx)}
            common = sorted(set(sg_idx_map.keys()) & set(mg_idx_map.keys()))
            n_common = len(common)
            if n_common == 0:
                continue
            sg_rows = [sg_idx_map[t] for t in common]
            mg_rows = [mg_idx_map[t] for t in common]
        else:
            n_common = min(n_sg, n_unpadded)
            sg_rows = list(range(n_common))
            mg_rows = list(range(n_common))

        fc2_in_str = "N/A"
        if sg_fc2_in is not None and mg_in_e is not None:
            in_cos = cos_sim(sg_fc2_in[sg_rows], mg_in_e[mg_rows])
            fc2_in_str = f"{in_cos.mean():.6f}"
            fc2_in_cos_all.append(in_cos.mean().item())

        gemm_cos_str = "N/A"
        if sg_fc2_gemm is not None and mg_gemm_e is not None:
            gi_cos = cos_sim(sg_fc2_gemm[sg_rows], mg_gemm_e[mg_rows])
            gemm_cos_str = f"{gi_cos.mean():.6f}"
            fc2_gemm_cos_all.append(gi_cos.mean().item())

        sg_fo = sg_fc2_out_e[sg_rows]
        mg_fo = mg_out_e[mg_rows]
        fo_cos = cos_sim(sg_fo, mg_fo)
        fc2_out_cos_all.append(fo_cos.mean().item())
        fc2_cos_str = f"{fo_cos.mean():.6f}"
        fc2_min_str = f"{fo_cos.min():.6f}"

        print(f"  {eid:>4} {n_sg:>5} {n_unpadded:>5} {n_common:>6} "
              f"{fc2_in_str:>11} {gemm_cos_str:>12} {fc2_cos_str:>11} {fc2_min_str:>11}")

    if fc2_out_cos_all:
        print(f"\n  Aggregate ({len(fc2_out_cos_all)} experts):")
        if fc2_in_cos_all:
            print(f"    fc2_input cos:      mean={sum(fc2_in_cos_all)/len(fc2_in_cos_all):.6f}")
        if fc2_gemm_cos_all:
            print(f"    fc2_gemm_input cos: mean={sum(fc2_gemm_cos_all)/len(fc2_gemm_cos_all):.6f}")
        print(f"    fc2_output cos:     mean={sum(fc2_out_cos_all)/len(fc2_out_cos_all):.6f}, "
              f"min={min(fc2_out_cos_all):.6f}")

    avg_gemm = sum(fc2_gemm_cos_all)/len(fc2_gemm_cos_all) if fc2_gemm_cos_all else None
    avg_out = sum(fc2_out_cos_all)/len(fc2_out_cos_all) if fc2_out_cos_all else None
    return avg_gemm, avg_out


# ============================================================
# Main with mode dispatch
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="activations",
                        choices=["activations", "dump-hf", "dump-mg", "weights"])
    parser.add_argument("--sglang-dump", help="sglang activation dump .pt")
    parser.add_argument("--megatron-dump", help="Megatron activation dump .pt")
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--output-dir", default="dump_weights/")
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--ref-load-path", default=REF_LOAD)
    parser.add_argument("--hf-dump", help="HF weight dump .pt for weights mode")
    parser.add_argument("--mg-dump", help="Megatron weight dump .pt for weights mode")
    parser.add_argument("--hf-weights", default=None,
                        help="HF raw w13 weights .pt for raw-matmul reference in activations mode")
    args, _ = parser.parse_known_args()

    if args.mode == "dump-hf":
        weights = load_hf_weights(args.layer_idx)
        os.makedirs(args.output_dir, exist_ok=True)
        out = os.path.join(args.output_dir, "hf_w13.pt")
        torch.save(weights, out)
        print(f"\nSaved {len(weights)} expert weights to {out}")

    elif args.mode == "dump-mg":
        raw_model, rank, local_rank = setup_megatron(args)
        weights = extract_mg_weights(raw_model, args.layer_idx, rank)
        os.makedirs(args.output_dir, exist_ok=True)
        if rank == 0 and weights:
            out = os.path.join(args.output_dir, "mg_fc1.pt")
            torch.save(weights, out)
            print(f"\nSaved {len(weights)} expert weights to {out}")
        elif rank == 0:
            print("\nERROR: Could not extract weights!")

        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()

    elif args.mode == "weights":
        if not args.hf_dump or not args.mg_dump:
            print("Need --hf-dump and --mg-dump for weights mode")
            return
        print(f"Loading HF dump: {args.hf_dump}")
        hf_w = torch.load(args.hf_dump, map_location="cpu", weights_only=False)
        print(f"Loading MG dump: {args.mg_dump}")
        mg_w = torch.load(args.mg_dump, map_location="cpu", weights_only=False)
        compare_weights(hf_w, mg_w)

    else:  # activations
        if not args.sglang_dump or not args.megatron_dump:
            print("Need --sglang-dump and --megatron-dump for activations mode")
            return
        print("Loading sglang dump...")
        sglang_data = torch.load(args.sglang_dump, map_location="cpu", weights_only=False)
        print("Loading megatron dump...")
        megatron_data = torch.load(args.megatron_dump, map_location="cpu", weights_only=False)

        hf_weights = None
        if args.hf_weights:
            print(f"Loading HF raw weights from {args.hf_weights}...")
            hf_weights = torch.load(args.hf_weights, map_location="cpu", weights_only=False)
            print(f"  Loaded {len(hf_weights)} expert w13 for raw-matmul reference")

        for sample_name in sglang_data:
            if sample_name not in megatron_data:
                print(f"WARNING: {sample_name} not in megatron dump, skipping")
                continue

            sg = sglang_data[sample_name]
            mg = megatron_data[sample_name]

            print(f"\n{'#'*70}")
            print(f"  Sample: {sample_name}")
            print(f"{'#'*70}")

            sg_lp = sg.get("token_logprobs")
            mg_lp = mg.get("token_logprobs")
            if sg_lp is not None and mg_lp is not None:
                compare_token_logprobs(sg_lp, mg_lp)

            sg_mlp = sg.get("mlp_input")
            mg_mlp = mg.get("mlp_input_original")
            if sg_mlp is not None and mg_mlp is not None:
                compare_mlp_input(sg_mlp, mg_mlp)

            sg_topk = sg.get("topk_ids")
            mg_topk = mg.get("topk_ids")
            if sg_topk is not None and mg_topk is not None:
                compare_topk(sg_topk, mg_topk,
                             sg.get("topk_weights"), mg.get("topk_weights"))

            sg_per_expert = sg.get("per_expert")
            if sg_per_expert is not None:
                compare_per_expert_fc1(sg_per_expert, mg, hf_weights=hf_weights)

                has_fc2 = any(sg_per_expert[e].get("fc2_out") is not None
                             for e in sg_per_expert)
                if has_fc2 and mg.get("fc2_output") is not None:
                    compare_per_expert_fc2(sg_per_expert, mg)


if __name__ == "__main__":
    main()
