"""
In-memory and optional on-disk cache for CuTeDSL grouped blockscaled GEMM compilation.

Enable persistent (disk) cache:
  export NVFP4_CUTE_PERSISTENT_CACHE=1
  export NVFP4_CUTE_CACHE_DIR=/path/to/cache   # optional

Grouped GEMM defaults persistent cache to enabled when the env var is unset
(``MINITRANSFORMER_GROUPED_GEMM_PERSISTENT_CACHE_DEFAULT=0`` disables that).
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

import cutlass.cute as cute

from miniTransformer.ops.cuteDSL.nvfp4.kernel_utils import (
    get_device_arch_from_torch_device,
    try_export_compiled_to_disk,
    try_load_compiled_from_disk,
)

GroupedGemmAlpha = Union[float, int, torch.Tensor]

_CACHE_VERSION = "grouped_blockscaled_v1"
_FUNCTION_NAME = "grouped_blockscaled_gemm"
_IN_MEMORY_CACHE: Dict[Tuple[Any, ...], Any] = {}
_COMPILE_LOCK = threading.Lock()
_CACHE_HITS = 0
_CACHE_MISSES = 0


def _persistent_cache_enabled() -> bool:
    explicit = os.getenv("NVFP4_CUTE_PERSISTENT_CACHE")
    if explicit is not None:
        return explicit.strip().lower() in {"1", "true", "yes", "on"}
    default_off = os.getenv(
        "MINITRANSFORMER_GROUPED_GEMM_PERSISTENT_CACHE_DEFAULT", "1"
    ).strip().lower() in {"0", "false", "no", "off"}
    return not default_off


def profile_cfg_cache_tuple(profile_cfg: Any) -> Tuple[Any, ...]:
    return (
        str(profile_cfg.ab_dtype),
        str(profile_cfg.sf_dtype),
        int(profile_cfg.sf_vec_size),
        tuple(profile_cfg.mma_tiler_mn),
        tuple(profile_cfg.cluster_shape_mn),
        profile_cfg.a_major,
        profile_cfg.b_major,
        profile_cfg.c_major,
    )


def infer_alpha_modes(
    alphas: Optional[List[GroupedGemmAlpha]],
    problem_sizes: List[Tuple[int, int, int, int]],
) -> Tuple[int, ...]:
    """Per-group alpha epilogue mode: 0=scalar, 1=row [M]."""
    modes: List[int] = []
    for i, (m, _n, _k, _l) in enumerate(problem_sizes):
        if alphas is None:
            modes.append(0)
            continue
        alpha = alphas[i]
        if isinstance(alpha, (float, int)):
            modes.append(0)
            continue
        if isinstance(alpha, torch.Tensor) and (alpha.ndim == 0 or alpha.numel() == 1):
            modes.append(0)
            continue
        modes.append(1)
    return tuple(modes)


def build_grouped_compile_cache_key(
    *,
    variant: str,
    device: torch.device,
    profile_name: str,
    profile_cfg: Any,
    num_groups: int,
    problem_sizes: List[Tuple[int, int, int, int]],
    out_dtype: torch.dtype,
    alpha_modes: Optional[Tuple[int, ...]] = None,
    host_problem_shape_available: bool = True,
) -> Tuple[Any, ...]:
    return (
        _CACHE_VERSION,
        variant,
        get_device_arch_from_torch_device(device),
        profile_name,
        profile_cfg_cache_tuple(profile_cfg),
        int(num_groups),
        tuple(tuple(ps) for ps in problem_sizes),
        str(out_dtype),
        alpha_modes if alpha_modes is not None else ("no_alpha",),
        bool(host_problem_shape_available),
    )


def get_or_compile_grouped_gemm(
    cache_key: Tuple[Any, ...],
    compile_fn: Callable[[], Any],
) -> Any:
    global _CACHE_HITS, _CACHE_MISSES

    cached = _IN_MEMORY_CACHE.get(cache_key)
    if cached is not None:
        _CACHE_HITS += 1
        return cached

    with _COMPILE_LOCK:
        cached = _IN_MEMORY_CACHE.get(cache_key)
        if cached is not None:
            _CACHE_HITS += 1
            return cached

        _CACHE_MISSES += 1
        compiled = try_load_compiled_from_disk(
            cache_key,
            persistent_cache_enabled=_persistent_cache_enabled(),
            function_name=_FUNCTION_NAME,
        )
        if compiled is None:
            compiled = compile_fn()
            try_export_compiled_to_disk(
                cache_key,
                compiled,
                persistent_cache_enabled=_persistent_cache_enabled(),
                function_name=_FUNCTION_NAME,
            )
        _IN_MEMORY_CACHE[cache_key] = compiled
        return compiled


def grouped_gemm_cache_stats() -> Dict[str, Any]:
    return {
        "in_memory_entries": len(_IN_MEMORY_CACHE),
        "hits": _CACHE_HITS,
        "misses": _CACHE_MISSES,
        "persistent_cache_enabled": _persistent_cache_enabled(),
    }


def clear_grouped_gemm_compile_cache() -> None:
    global _CACHE_HITS, _CACHE_MISSES
    _IN_MEMORY_CACHE.clear()
    _CACHE_HITS = 0
    _CACHE_MISSES = 0
