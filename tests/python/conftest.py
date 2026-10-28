"""
Pytest conftest: ensure project root is on sys.path so "miniTransformer" is importable
no matter where pytest is run from (workspace root or miniTransformer).

NVFP4 tests may request ``nvfp4_cuda_device`` and pass ``--nvfp4-cuda-device cuda:7``
or set ``NVFP4_TEST_CUDA_DEVICE`` (CLI overrides env).
"""
import os
import sys
from pathlib import Path

import pytest
import torch


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _find_project_root() -> Path:
    """Walk up from this file until we find the directory that contains the miniTransformer package."""
    path = Path(__file__).resolve().parent
    for _ in range(10):  # avoid infinite loop
        if (path / "miniTransformer" / "__init__.py").exists():
            return path
        parent = path.parent
        if parent == path:
            break
        path = parent
    return Path(__file__).resolve().parent.parent.parent


def _normalize_cuda_device_string(dev_str: str) -> str:
    """Return a valid torch device string; treat malformed ``cuda:`` as ``cuda``."""
    s = (dev_str or "").strip()
    if not s:
        return "cuda"
    if s.lower() == "cuda":
        return "cuda"
    prefix, sep, rest = s.partition(":")
    if sep and prefix.lower() == "cuda":
        rest = rest.strip()
        if rest == "":
            return "cuda"
        if not rest.isdigit():
            raise ValueError(
                f"Invalid NVFP4 CUDA device {dev_str!r}: expected 'cuda' or 'cuda:<int>', "
                f"got empty or non-integer index after ':'"
            )
    return s


def pytest_addoption(parser):
    parser.addoption(
        "--nvfp4-cuda-device",
        action="store",
        default=None,
        help=(
            "CUDA device for NVFP4 tests (e.g. cuda, cuda:7). "
            "Overrides NVFP4_TEST_CUDA_DEVICE when set."
        ),
    )


@pytest.fixture(scope="session")
def nvfp4_cuda_device(request) -> str:
    opt = request.config.getoption("--nvfp4-cuda-device", default=None)
    if opt is not None and str(opt).strip() != "":
        dev_str = _normalize_cuda_device_string(str(opt))
    else:
        dev_str = _normalize_cuda_device_string(
            os.environ.get("NVFP4_TEST_CUDA_DEVICE", "cuda").strip() or "cuda"
        )

    if torch.cuda.is_available():
        dev = torch.device(dev_str)
        idx = dev.index if dev.index is not None else 0
        if idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"nvfp4_cuda_device={dev_str!r} invalid: only "
                f"{torch.cuda.device_count()} CUDA device(s) visible"
            )
    return dev_str


def pytest_configure(config):
    root = _find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )


def pytest_ignore_collect(collection_path, config):
    path_str = str(collection_path)
    if path_str.endswith("test_group_linear_cutedsl_forward.py"):
        return not _env_enabled("MINITRANSFORMER_ENABLE_GROUPED_GEMM_CUTEDSL")
    return False


def pytest_collection_modifyitems(config, items):
    if _env_enabled("MINITRANSFORMER_ENABLE_GROUPED_GEMM_CUTEDSL"):
        return

    skip_reason = (
        "requires MINITRANSFORMER_ENABLE_GROUPED_GEMM_CUTEDSL=1 "
        "to run grouped cuteDSL forward tests"
    )
    skip_marker = pytest.mark.skip(reason=skip_reason)
    for item in items:
        if "test_group_linear_cutedsl_forward.py" in str(item.fspath):
            item.add_marker(skip_marker)
