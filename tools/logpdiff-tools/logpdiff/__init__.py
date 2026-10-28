"""logpdiff: train/inference consistency diagnostics toolkit.

P0 scope: single-device (TP=EP=PP=1), single-layer capture with structured
tensor records and a run directory layout shared by the sglang (inference)
and megatron (training) collectors.

The package imports without torch; torch is required only when actually
saving/loading tensor payloads (see :mod:`logpdiff.writer`).
"""

from .schema import (
    ACTUAL_KERNEL,
    DERIVED_METRIC,
    DISPATCH_INPUT,
    EXPERT_COMBINED_OUTPUT,
    FC1_OUTPUT,
    FC1_QDATA,
    FC1_RAW_INPUT,
    FC1_SCALES,
    FC2_OUTPUT,
    FC2_QDATA,
    FC2_RAW_INPUT,
    FC2_SCALES,
    HOOK_BOUNDARY,
    LAYER_INPUT,
    PIPELINE_STAGES,
    PROVENANCE_ALL,
    REPLAYED_REFERENCE,
    ROUTER_LOGITS,
    ROUTER_TOPK_IDS,
    ROUTER_TOPK_WEIGHTS,
    SWIGLU_OUTPUT,
    TENSOR_POINT_ALL,
    TOKEN_LOGPROBS,
    TensorRecord,
    check_single_device_parallel,
    validate_provenance,
    validate_tensor_point,
)
from .aligner import (
    ALIGNMENT_INVALID,
    COSINE_FALLBACK_THRESHOLD,
    METHOD_COSINE_FALLBACK,
    METHOD_DISPATCHER_INDEX,
    METHOD_HASH_EXACT,
    AlignedPair,
    AlignmentResult,
    align_records,
)
from .compare import (
    COMPARISON_ACTUAL_VS_REPLAY,
    REFERENCE_FRAMEWORK,
    TARGET_FRAMEWORK,
    TENSOR_POINT_TO_STAGE,
    run_compare,
)
from .metrics import (
    DEFAULT_THRESHOLDS,
    equal_summary,
    evaluate_stages,
    find_first_failure,
    fp4_underflow_ratio,
    logprob_metrics,
    routing_metrics,
    tensor_metrics,
    weighted_summary,
    worst_entries,
)
from .reporter import render_report
from .writer import (
    FRAMEWORKS,
    RunWriter,
    assert_internal_payload,
    create_run_dir,
)

__all__ = [
    # provenance
    "ACTUAL_KERNEL",
    "REPLAYED_REFERENCE",
    "HOOK_BOUNDARY",
    "DERIVED_METRIC",
    "PROVENANCE_ALL",
    "validate_provenance",
    # tensor points
    "LAYER_INPUT",
    "ROUTER_LOGITS",
    "ROUTER_TOPK_IDS",
    "ROUTER_TOPK_WEIGHTS",
    "DISPATCH_INPUT",
    "FC1_RAW_INPUT",
    "FC1_QDATA",
    "FC1_SCALES",
    "FC1_OUTPUT",
    "SWIGLU_OUTPUT",
    "FC2_RAW_INPUT",
    "FC2_QDATA",
    "FC2_SCALES",
    "FC2_OUTPUT",
    "EXPERT_COMBINED_OUTPUT",
    "TOKEN_LOGPROBS",
    "TENSOR_POINT_ALL",
    "validate_tensor_point",
    # pipeline / parallelism
    "PIPELINE_STAGES",
    "check_single_device_parallel",
    # record
    "TensorRecord",
    # writer
    "FRAMEWORKS",
    "RunWriter",
    "create_run_dir",
    "assert_internal_payload",
    # aligner
    "ALIGNMENT_INVALID",
    "COSINE_FALLBACK_THRESHOLD",
    "METHOD_DISPATCHER_INDEX",
    "METHOD_HASH_EXACT",
    "METHOD_COSINE_FALLBACK",
    "AlignedPair",
    "AlignmentResult",
    "align_records",
    # metrics
    "DEFAULT_THRESHOLDS",
    "tensor_metrics",
    "fp4_underflow_ratio",
    "routing_metrics",
    "logprob_metrics",
    "weighted_summary",
    "equal_summary",
    "worst_entries",
    "evaluate_stages",
    "find_first_failure",
    # compare
    "run_compare",
    "TENSOR_POINT_TO_STAGE",
    "COMPARISON_ACTUAL_VS_REPLAY",
    "REFERENCE_FRAMEWORK",
    "TARGET_FRAMEWORK",
    # reporter
    "render_report",
]
