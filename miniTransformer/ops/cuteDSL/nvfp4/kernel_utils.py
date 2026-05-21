import fcntl
import hashlib
import os
import pickle
import tempfile
from dataclasses import dataclass
from getpass import getuser
from itertools import product
from pathlib import Path
from typing import Any, List, Tuple

import cutlass
import cutlass.cute as cute
import torch

_PERSISTENT_CACHE_VERSION = "v1"


def validate_group_shapes_basic(
    m_list: List[int], n_list: List[int], k_list: List[int]
) -> None:
    if not (len(m_list) == len(n_list) == len(k_list)):
        raise ValueError("m_list, n_list and k_list must have identical lengths")
    if len(m_list) == 0:
        raise ValueError("grouped problem cannot be empty")
    for idx, (m, n, k) in enumerate(zip(m_list, n_list, k_list)):
        if m <= 0 or n <= 0 or k <= 0:
            raise ValueError(f"group {idx}: non-positive shape ({m}, {n}, {k})")


DEFAULT_MMA_TILER_MN = (256, 256)
DEFAULT_CLUSTER_SHAPE_MNK = (2, 1, 1)
DEFAULT_NUM_AB_STAGE = 5
DEFAULT_NUM_ACC_STAGE = 1


@dataclass(frozen=True)
class GemmTuneConfig:
    mma_tiler_mn: Tuple[int, int]
    cluster_shape_mnk: Tuple[int, int, int]
    use_2cta_instrs: bool
    num_ab_stage: int
    num_acc_stage: int


def build_default_tune_search_space() -> List[GemmTuneConfig]:
    search_space: List[GemmTuneConfig] = []
    for _mma_tiler_mn, _cluster_shape_mnk, _use_2cta_instrs, _num_ab_stage in product(
        [
            (128, 128),
            (128, 256),
            (256, 128),
            (256, 256),
            (64, 64),
            (64, 128),
            (128, 64),
            (32, 32),
        ],
        [(1, 1, 1), (2, 1, 1)],
        [False, True],
        [1, 2, 3, 4, 5],
    ):
        if _mma_tiler_mn[0] == 128 and _use_2cta_instrs:
            continue
        search_space.append(
            GemmTuneConfig(
                mma_tiler_mn=_mma_tiler_mn,
                cluster_shape_mnk=_cluster_shape_mnk,
                use_2cta_instrs=_use_2cta_instrs,
                num_ab_stage=_num_ab_stage,
                num_acc_stage=1,
            )
        )
    return search_space


def get_persistent_cache_dir() -> Path:
    cache_dir = os.getenv("NVFP4_CUTE_CACHE_DIR")
    if cache_dir:
        path = Path(cache_dir)
    else:
        path = Path(tempfile.gettempdir()) / getuser() / "minitransformer_cute_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def persistent_cache_hash(config_key: Tuple[Any, ...]) -> str:
    payload = (_PERSISTENT_CACHE_VERSION, config_key)
    return hashlib.sha256(pickle.dumps(payload)).hexdigest()


class FileLock:
    def __init__(self, lock_path: Path, exclusive: bool):
        self.lock_path = lock_path
        self.exclusive = exclusive
        self._fd = None

    def __enter__(self):
        self._fd = os.open(
            str(self.lock_path),
            os.O_CREAT | (os.O_WRONLY if self.exclusive else os.O_RDONLY),
        )
        mode = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
        fcntl.flock(self._fd, mode)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


def get_device_arch_from_torch_device(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


def try_load_compiled_from_disk(
    config_key: Tuple[Any, ...],
    *,
    persistent_cache_enabled: bool,
    function_name: str,
):
    if not persistent_cache_enabled:
        return None
    cache_dir = get_persistent_cache_dir()
    key_hash = persistent_cache_hash(config_key)
    obj_path = cache_dir / f"{key_hash}.o"
    lock_path = cache_dir / f"{key_hash}.lock"
    with FileLock(lock_path, exclusive=False):
        if not obj_path.exists():
            return None
        module = cute.runtime.load_module(str(obj_path), enable_tvm_ffi=True)
        return getattr(module, function_name)


def try_export_compiled_to_disk(
    config_key: Tuple[Any, ...],
    compiled,
    *,
    persistent_cache_enabled: bool,
    function_name: str,
) -> None:
    if not persistent_cache_enabled:
        return
    cache_dir = get_persistent_cache_dir()
    key_hash = persistent_cache_hash(config_key)
    obj_path = cache_dir / f"{key_hash}.o"
    lock_path = cache_dir / f"{key_hash}.lock"
    with FileLock(lock_path, exclusive=True):
        if obj_path.exists():
            return
        try:
            compiled.export_to_c(
                object_file_path=str(obj_path),
                function_name=function_name,
            )
        except TypeError:
            # Primary signature not supported by this CuTeDSL version.
            # Remove any partial file it may have written before retrying.
            obj_path.unlink(missing_ok=True)
            try:
                compiled.export_to_c(
                    str(cache_dir),
                    key_hash,
                    function_prefix=function_name,
                )
            except TypeError:
                return
            # The alternate form may write to cache_dir/key_hash (no .o suffix).
            # Rename to the canonical obj_path so the loader can find it.
            alt_path = cache_dir / key_hash
            if alt_path.exists() and not obj_path.exists():
                alt_path.rename(obj_path)

def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_ptr: cute.Pointer,
    sf_mma_ptr: cute.Pointer,
    mn: int,
    sf_k: int,
    l: int,
    mma_shape: tuple,
):
    mma_permute_order = (3, 4, 1, 5, 2, 0)
    permuted_shape = tuple(mma_shape[i] for i in mma_permute_order)
    cute_layout = cute.make_ordered_layout(permuted_shape, order=(2, 1, 4, 0, 3, 5))

    sf_ref_tensor = cute.make_tensor(
        sf_ref_ptr, cute.make_layout((mn, sf_k, l), stride=(sf_k, 1, mn * sf_k))
    )
    sf_mma_tensor = cute.make_tensor(sf_mma_ptr, cute_layout)

    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]
    pass


_CVT_SF_COMPILED: dict = {}  # mma_shape → compiled callable


def get_or_compile_cvt_sf(
    ref_ptr: cute.Pointer,
    mma_ptr: cute.Pointer,
    mn: int,
    sf_k: int,
    l: int,
    mma_shape: tuple,
):
    """Compile cvt_sf_MKL_to_M32x4xrm_K4xrk_L once per mma_shape, cache for reuse.

    @cute.jit recompiles on every call with no cross-call caching.  This wrapper
    calls cute.compile() explicitly — ~91 ms one-time cost per unique mma_shape —
    and re-uses the compiled callable for all subsequent calls with the same shape.
    """
    compiled = _CVT_SF_COMPILED.get(mma_shape)
    if compiled is None:
        import logging
        logging.getLogger(__name__).info(
            "[CuTeDSL] cvt_sf: compiling for mma_shape=%s", mma_shape
        )
        compiled = cute.compile(
            cvt_sf_MKL_to_M32x4xrm_K4xrk_L,
            ref_ptr, mma_ptr, mn, sf_k, l, mma_shape,
        )
        _CVT_SF_COMPILED[mma_shape] = compiled
    compiled(ref_ptr, mma_ptr)