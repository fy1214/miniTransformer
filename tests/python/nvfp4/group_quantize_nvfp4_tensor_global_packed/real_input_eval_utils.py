"""Backward-compatible re-export — prefer ``from helpers.metrics import ...``."""
import sys
from pathlib import Path

_NVFP4_ROOT = Path(__file__).resolve().parents[1]
if str(_NVFP4_ROOT) not in sys.path:
    sys.path.insert(0, str(_NVFP4_ROOT))

from helpers.metrics import *  # noqa: F403, F401
