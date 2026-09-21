"""Shared fixtures for the logpdiff P0 test suite.

CPU-only / synthetic data: no GPU, sglang or megatron is required. The
``tools/logpdiff-tools`` directory is added to ``sys.path`` so ``import
logpdiff`` works no matter where pytest is invoked from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from logpdiff import schema  # noqa: E402
from logpdiff.writer import create_run_dir  # noqa: E402

# synthetic model dimensions
HIDDEN = 16
N_TOKENS = 8
# token positions routed to each expert (top_k = 1)
EXPERT_TOKENS = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
FC1_DIM = 8


def require_torch():
    """Import torch or skip the calling test (CPU-only usage)."""
    return pytest.importorskip("torch", reason="test needs torch (CPU is fine)")


def make_record(run_id, framework, tensor_point, **kwargs):
    """A TensorRecord with sane defaults; override anything via kwargs."""
    fields = dict(
        run_id=run_id,
        framework=framework,
        sample_id="sample-0",
        layer_id=0,
        tensor_point=tensor_point,
        shape=(4,),
        dtype="torch.float32",
        provenance=schema.ACTUAL_KERNEL,
        payload_path=f"{framework}/tensors/synthetic.pt",
    )
    fields.update(kwargs)
    return schema.TensorRecord(**fields)


def save_payload(run, framework, tensor_point, tensor, *, sample_id="sample-0",
                 layer_id=0, expert_id=None, token_position=None,
                 route_slot=None, provenance=schema.ACTUAL_KERNEL):
    """Save ``tensor`` into ``run`` and return the appended record."""
    rec = schema.TensorRecord(
        run_id=run.run_id,
        framework=framework,
        sample_id=sample_id,
        layer_id=layer_id,
        tensor_point=tensor_point,
        expert_id=expert_id,
        token_position=token_position,
        route_slot=route_slot,
        provenance=provenance,
    )
    run.save_tensor(rec, tensor)
    return rec


def make_tensors(torch, seed=42):
    """Deterministic synthetic payloads for one MoE layer capture."""
    g = torch.Generator().manual_seed(seed)
    return {
        "layer_input": torch.randn(N_TOKENS, HIDDEN, generator=g),
        "topk_ids": torch.tensor(
            [[tok // 4] for tok in range(N_TOKENS)], dtype=torch.long
        ),
        # top-k=1 routing weights are normalized per token.
        "topk_weights": torch.ones(N_TOKENS, 1),
        "fc1_by_expert": {
            0: torch.randn(len(EXPERT_TOKENS[0]), FC1_DIM, generator=g),
            1: torch.randn(len(EXPERT_TOKENS[1]), FC1_DIM, generator=g),
        },
        "combined": torch.randn(N_TOKENS, HIDDEN, generator=g),
        "logprobs": -torch.rand(N_TOKENS, generator=g) * 5.0,
    }


def populate_run(run, tensors, *, megatron_delta=0.0):
    """Write matching records into both frameworks of ``run``.

    ``megatron_delta != 0`` shifts the megatron-side fc1/combined/logprob
    payloads so downstream comparisons FAIL (used for strict-exit tests).
    """
    for fw in ("sglang", "megatron"):
        delta = megatron_delta if fw == "megatron" else 0.0
        save_payload(run, fw, schema.LAYER_INPUT, tensors["layer_input"])
        save_payload(run, fw, schema.ROUTER_TOPK_IDS, tensors["topk_ids"],
                     provenance=schema.REPLAYED_REFERENCE)
        save_payload(run, fw, schema.ROUTER_TOPK_WEIGHTS,
                     tensors["topk_weights"],
                     provenance=schema.REPLAYED_REFERENCE)
        for eid, toks in EXPERT_TOKENS.items():
            mat = tensors["fc1_by_expert"][eid]
            for row, tok in enumerate(toks):
                save_payload(
                    run, fw, schema.FC1_OUTPUT, mat[row] + delta,
                    expert_id=eid, token_position=tok, route_slot=0,
                    provenance=schema.REPLAYED_REFERENCE,
                )
        save_payload(run, fw, schema.EXPERT_COMBINED_OUTPUT,
                     tensors["combined"] + delta)
        save_payload(run, fw, schema.TOKEN_LOGPROBS,
                     tensors["logprobs"] - delta,
                     provenance=schema.DERIVED_METRIC)
    run.write_config_resolved(
        {
            "model_path": run.model_path,
            "layer_idx": 0,
            "seed": run.seed,
            "tp": run.tp,
            "ep": run.ep,
            "pp": run.pp,
            "force_routing": run.force_routing,
        }
    )
    run.write_inputs([{"sample_id": "sample-0", "text": "synthetic input"}])
    run.finalize_manifest()
    return run


def make_run(tmp_path, torch, run_id="synthetic-run", *, megatron_delta=0.0,
             seed=42):
    """Create a fully populated synthetic run directory under tmp_path."""
    run = create_run_dir(
        tmp_path / "runs", run_id=run_id,
        model_path="/models/synthetic", seed=1234,
    )
    populate_run(run, make_tensors(torch, seed=seed),
                 megatron_delta=megatron_delta)
    return run


@pytest.fixture
def synth_run(tmp_path):
    """A synthetic run whose two frameworks agree numerically (PASS path)."""
    torch = require_torch()
    return make_run(tmp_path, torch)


@pytest.fixture
def synth_run_fail(tmp_path):
    """A synthetic run whose megatron side diverges (FAIL path)."""
    torch = require_torch()
    return make_run(tmp_path, torch, run_id="synthetic-fail",
                    megatron_delta=0.5)
