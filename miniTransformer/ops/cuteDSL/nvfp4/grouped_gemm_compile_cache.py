"""
In-memory and optional on-disk cache for CuTeDSL grouped blockscaled GEMM compilation.

In-process cache is always enabled (compile once per shape signature per process).

Disk cache is opt-in only (grouped ``export_to_c`` must match loader symbol ``func``):
  export NVFP4_CUTE_PERSISTENT_CACHE=1
  export NVFP4_CUTE_CACHE_DIR=/path/to/cache   # optional

Stale or corrupt ``.o`` entries are removed automatically on load failure.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

import cutlass.cute as cute

from miniTransformer.ops.cuteDSL.nvfp4.kernel_utils import (
    FileLock,
    get_device_arch_from_torch_device,
    get_persistent_cache_dir,
    persistent_cache_hash,
    try_export_compiled_to_disk,
)

logger = logging.getLogger(__name__)

# Bump when compile cache key semantics change (invalidates old on-disk blobs).
_CACHE_VERSION = "grouped_blockscaled_v2"
# Must match ``export_to_c(..., function_name=...)`` / TVM FFI symbol (same as single GEMM).
_DISK_FUNCTION_NAME = "func"
_DISK_FUNCTION_FALLBACKS = ("func",)
_IN_MEMORY_CACHE: Dict[Tuple[Any, ...], Any] = {}
_COMPILE_LOCK = threading.Lock()
_CACHE_HITS = 0
_CACHE_MISSES = 0
_DISK_LOAD_FAILURES = 0


def _persistent_cache_enabled() -> bool:
    """Grouped GEMM disk cache is opt-in; in-memory cache is always on."""
    return os.getenv("NVFP4_CUTE_PERSISTENT_CACHE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _try_load_grouped_compiled_from_disk(
    cache_key: Tuple[Any, ...],
) -> Any | None:
    global _DISK_LOAD_FAILURES

    if not _persistent_cache_enabled():
        return None

    cache_dir = get_persistent_cache_dir()
    key_hash = persistent_cache_hash(cache_key)
    obj_path = cache_dir / f"{key_hash}.o"
    lock_path = cache_dir / f"{key_hash}.lock"

    if not obj_path.exists():
        return None

    with FileLock(lock_path, exclusive=False):
        if not obj_path.exists():
            return None
        try:
            # Use enable_tvm_ffi=False: the fallback export_to_c path writes CuTeDSL
            # metadata symbols (func_args_spec, func_version, etc.), not TVM FFI symbols
            # (__tvm_ffi_func). enable_tvm_ffi=False reads metadata and finds the
            # _mlir_func__mlir_ciface_... callable, which matches the fallback export format.
            module = cute.runtime.load_module(str(obj_path), enable_tvm_ffi=False)
        except Exception as exc:
            _DISK_LOAD_FAILURES += 1
            logger.warning(
                "Grouped GEMM disk cache: failed to load %s (%s); recompiling.",
                obj_path,
                exc,
            )
            _remove_disk_cache_entry(obj_path, lock_path)
            return None

        for fn_name in _DISK_FUNCTION_FALLBACKS:
            try:
                return getattr(module, fn_name)
            except Exception:
                continue

        # _DISK_FUNCTION_FALLBACKS didn't match — the export_to_c fallback path
        # may have written the symbol under a prefix-derived name (e.g. "func_0").
        # Discover it by scanning all public callables in the module.
        discovered_name: Optional[str] = None
        discovered_fn = None
        try:
            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue
                try:
                    attr = getattr(module, attr_name)
                    if callable(attr):
                        discovered_name = attr_name
                        discovered_fn = attr
                        break
                except Exception:
                    continue
        except Exception:
            pass

        if discovered_fn is not None:
            logger.warning(
                "Grouped GEMM disk cache: found callable '%s' in %s "
                "(not in expected %s; consider updating _DISK_FUNCTION_FALLBACKS).",
                discovered_name, obj_path, _DISK_FUNCTION_FALLBACKS,
            )
            return discovered_fn

        _DISK_LOAD_FAILURES += 1
        logger.warning(
            "Grouped GEMM disk cache: no callable in %s (expected %s); recompiling.",
            obj_path,
            _DISK_FUNCTION_FALLBACKS,
        )
        _remove_disk_cache_entry(obj_path, lock_path)
        return None


def _remove_disk_cache_entry(obj_path: Path, lock_path: Path) -> None:
    try:
        obj_path.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


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


def build_grouped_compile_cache_key(
    *,
    variant: str,
    device: torch.device,
    profile_name: str,
    profile_cfg: Any,
    num_groups: int,
    problem_sizes: List[Tuple[int, int, int, int]],
    out_dtype: torch.dtype,
    host_problem_shape_available: bool = True,
) -> Tuple[Any, ...]:
    if host_problem_shape_available:
        # Exact M baked into kernel via total_num_clusters; each distinct M set compiles separately.
        shape_key: tuple = tuple(tuple(ps) for ps in problem_sizes)
    else:
        # Persistent kernel: total_num_clusters = max_active_clusters.
        # M, N, K are all runtime values (problem_shape_mnkl tensor); no shape affects compilation.
        shape_key = ()
    return (
        _CACHE_VERSION,
        variant,
        get_device_arch_from_torch_device(device),
        profile_name,
        profile_cfg_cache_tuple(profile_cfg),
        int(num_groups),
        shape_key,
        str(out_dtype),
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
        compiled = _try_load_grouped_compiled_from_disk(cache_key)
        if compiled is None:
            compiled = compile_fn()
            if _persistent_cache_enabled():
                try:
                    try_export_compiled_to_disk(
                        cache_key,
                        compiled,
                        persistent_cache_enabled=True,
                        function_name=_DISK_FUNCTION_NAME,
                    )
                except Exception as exc:
                    logger.warning(
                        "Grouped GEMM disk cache: export skipped (%s).",
                        exc,
                    )
        _IN_MEMORY_CACHE[cache_key] = compiled
        return compiled


def grouped_gemm_cache_stats() -> Dict[str, Any]:
    return {
        "in_memory_entries": len(_IN_MEMORY_CACHE),
        "hits": _CACHE_HITS,
        "misses": _CACHE_MISSES,
        "disk_load_failures": _DISK_LOAD_FAILURES,
        "persistent_cache_enabled": _persistent_cache_enabled(),
    }


def clear_grouped_gemm_compile_cache() -> None:
    global _CACHE_HITS, _CACHE_MISSES
    _IN_MEMORY_CACHE.clear()
    _CACHE_HITS = 0
    _CACHE_MISSES = 0
