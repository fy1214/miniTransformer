import fcntl
import hashlib
import os
import pickle
import tempfile
from enum import Enum
from getpass import getuser
from pathlib import Path
from typing import Any, Tuple

import cutlass
import cutlass.cute as cute

_PERSISTENT_CACHE_VERSION = "v1"


class Nvfp4KernelVariant(str, Enum):
    OPTS0 = "opts0"
    OPTS1 = "opts1"
    OPTS2 = "opts2"
    OPTS3 = "opts3"
    OPTS4 = "opts4"
    OPTS5 = "opts5"
    OPTS6 = "opts6"


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

@cute.jit
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