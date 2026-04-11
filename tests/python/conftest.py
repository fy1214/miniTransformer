"""
Pytest conftest: ensure project root is on sys.path so "miniTransformer" is importable
no matter where pytest is run from (workspace root or miniTransformer).
"""
import sys
from pathlib import Path


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


def pytest_configure(config):
    root = _find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
