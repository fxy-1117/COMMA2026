"""Runtime setup for the extracted logic engine.

The engine imports neural models at module import time. This loader applies the
offline/cache configuration first, then imports the engine exactly once through
Python's normal module system.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

from .runtime_utils import patch_torch_hub


def load_logic_engine() -> ModuleType:
    """Load the experiment logic engine after applying local runtime patches."""
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    patch_torch_hub()
    return importlib.import_module("comma_core.logic_engine")
