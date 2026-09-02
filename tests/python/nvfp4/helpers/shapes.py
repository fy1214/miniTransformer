"""Shared tensor shapes for NVFP4 pytest parametrization and eval scripts."""
from __future__ import annotations

from typing import List, Sequence, Tuple

__all__ = [
    "FC1_DUMP_PATH",
    "FC1_REAL_INPUT_SHAPE",
    "FC1_WEIGHT_SHAPE",
    "MOE_FC1_M_SPLITS_8",
    "GROUPED_GEMM_V2_CASES",
    "GEMM_WORKLOAD_MKN",
    "DEEPGEMM_MKN",
    "PERTOKEN_COL_FUSE_LIMIT",
]

# Real MoE fc1 dump (see README — collect via LINEAR_STATS_* env vars).
FC1_DUMP_PATH = "/tmp/dump/fc1_step0000_inp.pt"
FC1_REAL_INPUT_SHAPE = (148_480, 2048)
FC1_WEIGHT_SHAPE = (1536, 2048)  # per-expert weight rows in fc1 dump

# Equal 8-way split of fc1 activation (eval-only convention, not an op constraint).
MOE_FC1_M_SPLITS_8: List[int] = [18_560] * 8

# Fused pertoken quant switches to two-pass above this K (see nvfp4_quantize.py).
PERTOKEN_COL_FUSE_LIMIT = 8192

# grouped_cutlass_gemm_v2 pytest cases: (m_splits, n, k)
GROUPED_GEMM_V2_CASES: List[Tuple[List[int], int, int]] = [
    ([128], 128, 128),
    ([128, 256], 256, 256),
    ([128, 128, 128], 512, 256),
    ([256, 512], 256, 512),
    ([128], 7168, 4096),
    ([128] * 8, 4096, 7168),
]

# Dense NVFP4 GEMM workload shapes (M, K, N) — legacy backend comparison tests.
GEMM_WORKLOAD_MKN: Sequence[Tuple[int, int, int]] = (
    (2048, 5120, 10240),
    (8192, 2048, 1280),
    (2048, 5120, 51200),
    (8192, 6400, 1280),
)

DEEPGEMM_MKN: Sequence[Tuple[int, int, int]] = (
    (128, 16384, 7168),
    (128, 7168, 4096),
    (4096, 1536, 24576),
    (4096, 512, 32768),
    (4096, 16384, 7168),
    (4096, 7168, 4096),
    (4096, 2048, 7168),
    (4096, 7168, 256),
    (128, 7168, 129280),
    (4096, 7168, 129280),
)
