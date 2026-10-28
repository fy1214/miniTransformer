"""Auto-applied when this directory is on PYTHONPATH.

If ``TE_FP4_PERTOKEN=1``, monkey-patch Megatron ``fp4_utils`` so native TE
``NVFP4PerTokenBlockScaling`` is used under ``--fp4-format e2m1`` (same hook as
``tests/python/megatron/test_lowbit_param.py`` / fair MoE bench).

If ``MOE_DUMP_DISPATCH=1``, dump per-expert token counts for selected steps
(``MOE_DUMP_STEPS``, default ``8``) into ``$LOG_DIR/dispatch_dump`` (or
``MOE_DUMP_DIR``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _maybe_patch_te_fp4_pertoken() -> None:
    if os.environ.get("TE_FP4_PERTOKEN") != "1":
        return

    from contextlib import nullcontext

    import megatron.core.fp4_utils as megatron_fp4_utils
    import transformer_engine.common.recipe as te_recipe

    def _te_native_pertoken_fp4_recipe(config):
        del config
        return te_recipe.NVFP4PerTokenBlockScaling()

    def _te_native_pertoken_fp4_context(config, layer_no: int = -1, is_init: bool = False):
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
            or (
                layer_no >= 0
                and config.first_last_layers_bf16
                and (is_first_layer or is_last_layer)
            )
        ):
            return nullcontext()

        fp4_recipe = _te_native_pertoken_fp4_recipe(config)
        import inspect

        import transformer_engine.pytorch as te

        if not is_init:
            return te.fp8_autocast(enabled=True, fp8_recipe=fp4_recipe, fp8_group=None)

        context_args = {"enabled": True}
        if "recipe" in inspect.signature(te.fp8_model_init).parameters:
            context_args["recipe"] = fp4_recipe
        return te.fp8_model_init(**context_args)

    megatron_fp4_utils.get_fp4_recipe = _te_native_pertoken_fp4_recipe
    megatron_fp4_utils.get_fp4_context = _te_native_pertoken_fp4_context


def _maybe_patch_moe_dispatch_dump() -> None:
    if os.environ.get("MOE_DUMP_DISPATCH") != "1":
        return

    import torch
    import torch.distributed as dist

    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher

    dump_steps = {
        int(x)
        for x in os.environ.get("MOE_DUMP_STEPS", "8").split(",")
        if x.strip()
    }
    max_mbs = int(os.environ.get("MOE_DUMP_MAX_MBS", "2"))
    dump_dir = Path(
        os.environ.get(
            "MOE_DUMP_DIR",
            os.path.join(os.environ.get("LOG_DIR", "/tmp"), "dispatch_dump"),
        )
    )
    dump_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "mb_by_step": {},
        "last_layer_by_step": {},
        "written_meta": False,
        "files": {},
    }

    def _tolist(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return [int(v) for v in x.detach().cpu().flatten().tolist()]
        # CPU splits may already be numpy arrays after dtoh sync.
        if hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
            flat = x.tolist()
            if isinstance(flat, list):
                return [int(v) for v in flat]
            return [int(flat)]
        if isinstance(x, (list, tuple)):
            return [int(v) for v in x]
        return int(x)

    def _stats(vals):
        if not vals:
            return {}
        s = sorted(vals)
        n = len(s)
        mean = sum(s) / n
        var = sum((v - mean) ** 2 for v in s) / n
        med = s[n // 2]
        return {
            "min": s[0],
            "max": s[-1],
            "sum": int(sum(s)),
            "mean": mean,
            "median": med,
            "cv": (var**0.5) / mean if mean > 0 else 0.0,
            "max_over_median": (s[-1] / med) if med > 0 else None,
            "nonzero": sum(1 for v in s if v > 0),
            "zero": sum(1 for v in s if v == 0),
        }

    def _should_dump(step: int, mb: int) -> bool:
        return step in dump_steps and 0 <= mb < max_mbs

    def _curr_step() -> int:
        try:
            from megatron.training import get_args

            return int(get_args().curr_iteration)
        except Exception:
            return -1

    def _world_meta(dispatcher) -> dict:
        rank = dist.get_rank() if dist.is_initialized() else 0
        return {
            "world_rank": rank,
            "ep_size": int(getattr(dispatcher, "ep_size", -1)),
            "tp_size": int(getattr(dispatcher, "tp_size", -1)),
            "num_experts": int(getattr(dispatcher, "num_experts", -1)),
            "num_local_experts": int(getattr(dispatcher, "num_local_experts", -1)),
            "local_expert_indices": _tolist(
                getattr(dispatcher, "local_expert_indices", None)
            ),
        }

    def _ensure_meta(dispatcher):
        if state["written_meta"]:
            return
        meta = {
            "num_experts_config": int(getattr(dispatcher, "num_experts", -1)),
            "note": "Qwen3-30B-A3B MoE e2e uses num_experts=128 (not 256).",
            "dump_steps": sorted(dump_steps),
            "max_mbs": max_mbs,
            **_world_meta(dispatcher),
        }
        (dump_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        state["written_meta"] = True

    def _fh(rank: int, step: int, mb: int):
        key = (rank, step, mb)
        if key not in state["files"]:
            path = dump_dir / f"rank{rank}_step{step}_mb{mb}.jsonl"
            state["files"][key] = path.open("a", encoding="utf-8")
        return state["files"][key]

    def routed_experts_compute(self, hidden_states, probs, residual):
        step = _curr_step()
        layer = int(getattr(self, "layer_number", -1) or -1)
        last = state["last_layer_by_step"].get(step, 10**9)
        if layer <= last:
            state["mb_by_step"][step] = state["mb_by_step"].get(step, -1) + 1
        state["last_layer_by_step"][step] = layer
        mb = int(state["mb_by_step"].get(step, 0))

        td = self.token_dispatcher
        do_dump = _should_dump(step, mb) and isinstance(td, MoEAlltoAllTokenDispatcher)

        pre_local = None
        ep_global = None
        if do_dump and getattr(td, "routing_map", None) is not None:
            pre_local_t = td.routing_map.sum(dim=0).long()
            pre_local = pre_local_t.detach().cpu()
            # EP-group global tokens per expert (= sum of each rank's local routing).
            ep_global_t = pre_local_t.clone()
            if dist.is_initialized() and getattr(td, "ep_group", None) is not None:
                dist.all_reduce(ep_global_t, group=td.ep_group, op=dist.ReduceOp.SUM)
            ep_global = ep_global_t.detach().cpu()

        dispatched_input, tokens_per_expert, permuted_probs, row_amax = (
            td.dispatch_postprocess(hidden_states, probs)
        )

        if do_dump:
            _ensure_meta(td)
            post_list = _tolist(tokens_per_expert)
            pre_list = _tolist(pre_local) if pre_local is not None else None
            ep_list = _tolist(ep_global) if ep_global is not None else None
            rec = {
                "step": step,
                "microbatch": mb,
                "layer": layer,
                **_world_meta(td),
                "input_tokens_this_rank": int(
                    td.routing_map.size(0)
                    if getattr(td, "routing_map", None) is not None
                    else -1
                ),
                "topk": int(getattr(td.config, "moe_router_topk", -1)),
                "routed_tokens_this_rank": (
                    int(pre_local.sum().item()) if pre_local is not None else None
                ),
                "tokens_per_expert_pre_ep_local": pre_list,
                "tokens_per_expert_ep_global": ep_list,
                "tokens_per_local_expert_post_ep": post_list,
                "input_splits": _tolist(getattr(td, "input_splits", None)),
                "output_splits": _tolist(getattr(td, "output_splits", None)),
                "post_ep_total_M": int(sum(post_list)) if post_list else None,
                "stats_ep_global": _stats(ep_list) if ep_list else {},
                "stats_post_ep_local": _stats(post_list) if post_list else {},
            }
            rank = dist.get_rank() if dist.is_initialized() else 0
            fh = _fh(rank, step, mb)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

        expert_output, mlp_bias = self.experts(
            dispatched_input, tokens_per_expert, permuted_probs, row_amax=row_amax
        )
        assert mlp_bias is None, f"mlp_bias is not supported for {type(td)}"
        output = td.combine_preprocess(expert_output)
        return output, mlp_bias

    MoELayer.routed_experts_compute = routed_experts_compute
    print(
        f"[moe_dump_dispatch] enabled dump_steps={sorted(dump_steps)} "
        f"max_mbs={max_mbs} dir={dump_dir}",
        flush=True,
    )


_maybe_patch_te_fp4_pertoken()
_maybe_patch_moe_dispatch_dump()
