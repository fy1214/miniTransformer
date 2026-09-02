"""NVFP4 test suite fixtures, markers, and runtime guards."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_NVFP4_ROOT = Path(__file__).resolve().parent
if str(_NVFP4_ROOT) not in sys.path:
    sys.path.insert(0, str(_NVFP4_ROOT))

try:
    import fp4_gemm  # noqa: F401
except ImportError:
    fp4_gemm = None


def fp4_gemm_skip_reason() -> str | None:
    if fp4_gemm is None:
        return "fp4_gemm extension not built"
    if not torch.cuda.is_available():
        return "CUDA is not available"
    major, _minor = torch.cuda.get_device_capability()
    if major < 10:
        return f"NVFP4 CUDA kernels require sm_100+ (got sm_{major}{_minor})"
    return None


def pytest_configure(config):
    for name, desc in (
        ("production", "MoE GroupedLinear / QAT production path"),
        ("legacy", "dense GEMM, TE, quartet, or exploratory backend tests"),
        ("eval", "requires real fc1 dump under /tmp/dump"),
        ("bench", "performance benchmark, not correctness"),
        ("slow", "long-running test"),
    ):
        config.addinivalue_line("markers", f"{name}: {desc}")


@pytest.fixture(scope="session")
def nvfp4_test_root() -> Path:
    return _NVFP4_ROOT


@pytest.fixture(scope="session")
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


@pytest.fixture(scope="session")
def require_sm100(require_cuda):
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip(f"Requires SM100 (Blackwell); got SM{major}{minor}")


@pytest.fixture(scope="session")
def require_fp4_gemm():
    reason = fp4_gemm_skip_reason()
    if reason:
        pytest.skip(reason)


@pytest.fixture(scope="session")
def fp4_gemm_module(require_fp4_gemm):
    return fp4_gemm
