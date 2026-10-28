"""Schema definitions for logpdiff train/inference consistency diagnostics.

Pure-Python module: importing it must NOT require torch, so that metadata
handling (index files, manifests, reports) works in environments without a
GPU / torch installation.

A single capture run records tensors from both frameworks (sglang inference
side and megatron training side) for one model layer, plus derived metrics.
Every tensor is described by a :class:`TensorRecord` row in the per-framework
``index.jsonl`` files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# provenance: where the payload bytes actually came from
# ---------------------------------------------------------------------------

ACTUAL_KERNEL = "actual_kernel"
"""Captured from the real forward pass of the framework under test."""

REPLAYED_REFERENCE = "replayed_reference"
"""Produced by re-running a reference implementation on captured inputs."""

HOOK_BOUNDARY = "hook_boundary"
"""Captured at a module hook boundary (may include hook-side transforms)."""

DERIVED_METRIC = "derived_metric"
"""Computed from other payloads (e.g. per-token logprob diffs); not raw data."""

PROVENANCE_ALL = frozenset(
    {ACTUAL_KERNEL, REPLAYED_REFERENCE, HOOK_BOUNDARY, DERIVED_METRIC}
)


def validate_provenance(provenance: str) -> str:
    """Return ``provenance`` if it is a known value, else raise ValueError."""
    if provenance not in PROVENANCE_ALL:
        raise ValueError(
            f"unknown provenance {provenance!r}; "
            f"must be one of {sorted(PROVENANCE_ALL)}"
        )
    return provenance


# ---------------------------------------------------------------------------
# tensor_point: semantic location inside one MoE layer (plus model-level)
# ---------------------------------------------------------------------------

LAYER_INPUT = "layer_input"
ROUTER_LOGITS = "router_logits"
ROUTER_TOPK_IDS = "router_topk_ids"
ROUTER_TOPK_WEIGHTS = "router_topk_weights"
DISPATCH_INPUT = "dispatch_input"
FC1_RAW_INPUT = "fc1_raw_input"
FC1_QDATA = "fc1_qdata"
FC1_SCALES = "fc1_scales"
FC1_OUTPUT = "fc1_output"
SWIGLU_OUTPUT = "swiglu_output"
FC2_RAW_INPUT = "fc2_raw_input"
FC2_QDATA = "fc2_qdata"
FC2_SCALES = "fc2_scales"
FC2_OUTPUT = "fc2_output"
EXPERT_COMBINED_OUTPUT = "expert_combined_output"
TOKEN_LOGPROBS = "token_logprobs"

TENSOR_POINT_ALL = frozenset(
    {
        LAYER_INPUT,
        ROUTER_LOGITS,
        ROUTER_TOPK_IDS,
        ROUTER_TOPK_WEIGHTS,
        DISPATCH_INPUT,
        FC1_RAW_INPUT,
        FC1_QDATA,
        FC1_SCALES,
        FC1_OUTPUT,
        SWIGLU_OUTPUT,
        FC2_RAW_INPUT,
        FC2_QDATA,
        FC2_SCALES,
        FC2_OUTPUT,
        EXPERT_COMBINED_OUTPUT,
        TOKEN_LOGPROBS,
    }
)


def validate_tensor_point(tensor_point: str) -> str:
    """Return ``tensor_point`` if it is a known value, else raise ValueError."""
    if tensor_point not in TENSOR_POINT_ALL:
        raise ValueError(
            f"unknown tensor_point {tensor_point!r}; "
            f"must be one of {sorted(TENSOR_POINT_ALL)}"
        )
    return tensor_point


# ---------------------------------------------------------------------------
# pipeline stage order (comparison/reporting walks stages in this order)
# ---------------------------------------------------------------------------

PIPELINE_STAGES = [
    "layer_input",
    "routing",
    "dispatch",
    "fc1_quant",
    "fc1_gemm",
    "swiglu",
    "fc2_quant",
    "fc2_gemm",
    "combined_output",
    "logprob",
]


# ---------------------------------------------------------------------------
# parallelism guard: P0 supports single-device only
# ---------------------------------------------------------------------------


def check_single_device_parallel(tp: int = 1, ep: int = 1, pp: int = 1) -> None:
    """Raise ValueError unless TP=EP=PP=1.

    P0 only supports single-device runs (TP=EP=PP=1); multi-device capture
    (sharded tensors, cross-rank gather) is planned for P1.
    """
    if tp != 1 or ep != 1 or pp != 1:
        raise ValueError(
            f"P0 only supports TP=EP=PP=1, got tp={tp}, ep={ep}, pp={pp}. "
            "Multi-parallel capture will be supported in P1."
        )


# ---------------------------------------------------------------------------
# TensorRecord
# ---------------------------------------------------------------------------


def _check_optional_non_negative_int(value: Optional[int], name: str) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative int or None, got {value!r}")


@dataclass
class TensorRecord:
    """One row of a per-framework ``index.jsonl`` file.

    ``expert_id`` / ``token_position`` / ``route_slot`` / ``global_rank`` are
    optional: whether they are meaningful depends on ``tensor_point`` (e.g.
    ``router_logits`` has no ``expert_id``; P0 single-device runs leave
    ``global_rank`` as None or 0).
    """

    run_id: str
    framework: str
    sample_id: Any
    layer_id: int
    tensor_point: str
    shape: tuple = field(default_factory=tuple)
    dtype: str = ""
    provenance: str = ""
    payload_path: str = ""
    global_rank: Optional[int] = None
    expert_id: Optional[int] = None
    token_position: Optional[int] = None
    route_slot: Optional[int] = None

    def __post_init__(self) -> None:
        # normalize shape so from_dict(list) and direct construction agree
        self.shape = tuple(self.shape)

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "framework": self.framework,
            "sample_id": self.sample_id,
            "layer_id": self.layer_id,
            "tensor_point": self.tensor_point,
            "global_rank": self.global_rank,
            "expert_id": self.expert_id,
            "token_position": self.token_position,
            "route_slot": self.route_slot,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "provenance": self.provenance,
            "payload_path": self.payload_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TensorRecord":
        known = {
            "run_id",
            "framework",
            "sample_id",
            "layer_id",
            "tensor_point",
            "global_rank",
            "expert_id",
            "token_position",
            "route_slot",
            "shape",
            "dtype",
            "provenance",
            "payload_path",
        }
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown TensorRecord fields: {sorted(unknown)}")
        return cls(
            run_id=d["run_id"],
            framework=d["framework"],
            sample_id=d["sample_id"],
            layer_id=d["layer_id"],
            tensor_point=d["tensor_point"],
            global_rank=d.get("global_rank"),
            expert_id=d.get("expert_id"),
            token_position=d.get("token_position"),
            route_slot=d.get("route_slot"),
            shape=tuple(d["shape"]),
            dtype=d["dtype"],
            provenance=d["provenance"],
            payload_path=d["payload_path"],
        )

    # -- validation ---------------------------------------------------------

    def validate(self) -> None:
        """Raise ValueError on any invalid field; returns None when valid."""
        for name in ("run_id", "framework", "dtype", "payload_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string, got {value!r}")
        if self.sample_id is None or (isinstance(self.sample_id, str) and not self.sample_id):
            raise ValueError(f"sample_id must be set, got {self.sample_id!r}")
        if (
            not isinstance(self.layer_id, int)
            or isinstance(self.layer_id, bool)
            or self.layer_id < 0
        ):
            raise ValueError(f"layer_id must be a non-negative int, got {self.layer_id!r}")
        validate_provenance(self.provenance)
        validate_tensor_point(self.tensor_point)
        if not isinstance(self.shape, tuple):
            raise ValueError(f"shape must be a tuple, got {type(self.shape).__name__}")
        for dim in self.shape:
            if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
                raise ValueError(f"shape dims must be non-negative ints, got {self.shape!r}")
        _check_optional_non_negative_int(self.global_rank, "global_rank")
        _check_optional_non_negative_int(self.expert_id, "expert_id")
        _check_optional_non_negative_int(self.token_position, "token_position")
        _check_optional_non_negative_int(self.route_slot, "route_slot")
