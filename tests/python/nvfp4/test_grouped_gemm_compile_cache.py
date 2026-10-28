"""Unit tests for grouped CuTeDSL compile cache helpers (no CUDA required).

What: ``infer_alpha_modes``, ``build_grouped_compile_cache_key``, clear cache.
Where: ``grouped_gemm_compile_cache`` used by CuTeDSL grouped GEMM (QAT 6–7).
Compare: pure Python key stability / alpha mode inference.

Markers: ``production``.
"""

import pytest
import torch

from miniTransformer.ops.cuteDSL.nvfp4.blockwise_group_gemm_cute_alpha_epilogue import (
    GroupedBlockScaledProfile,
)
from miniTransformer.ops.cuteDSL.nvfp4.grouped_gemm_compile_cache import (
    build_grouped_compile_cache_key,
    clear_grouped_gemm_compile_cache,
    infer_alpha_modes,
    profile_cfg_cache_tuple,
)

pytestmark = pytest.mark.production


def test_infer_alpha_modes_scalar_and_row():
    """Scalar alpha → mode 0; per-row tensor alpha → mode 1."""
    ps = [(64, 512, 1024, 1), (64, 512, 1024, 1)]
    scalars = [0.5, 1.0]
    assert infer_alpha_modes(scalars, ps) == (0, 0)

    row = torch.ones(64)
    assert infer_alpha_modes([row, scalars[1]], ps) == (1, 0)


def test_compile_cache_key_stable():
    """Identical inputs produce identical cache keys."""
    profile = GroupedBlockScaledProfile.NVFP4
    cfg = profile.config
    ps = [(64, 512, 1024, 1), (64, 512, 1024, 1)]
    key_a = build_grouped_compile_cache_key(
        variant="alpha_epilogue",
        device=torch.device("cuda:0"),
        profile_name=profile.value,
        profile_cfg=cfg,
        num_groups=2,
        problem_sizes=ps,
        out_dtype=torch.bfloat16,
        alpha_modes=(1, 0),
    )
    key_b = build_grouped_compile_cache_key(
        variant="alpha_epilogue",
        device=torch.device("cuda:0"),
        profile_name=profile.value,
        profile_cfg=cfg,
        num_groups=2,
        problem_sizes=ps,
        out_dtype=torch.bfloat16,
        alpha_modes=(1, 0),
    )
    assert key_a == key_b
    assert profile_cfg_cache_tuple(cfg) == profile_cfg_cache_tuple(cfg)


def test_clear_cache():
    """``clear_grouped_gemm_compile_cache`` is callable (smoke)."""
    clear_grouped_gemm_compile_cache()
