"""Runtime loading for the extracted the original experiment code core."""

from __future__ import annotations

import importlib
import os
from types import ModuleType

from .utils import patch_torch_hub


def load_method_module() -> ModuleType:
    """Load the experiment core after applying offline/local patches."""
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    patch_torch_hub()
    return importlib.import_module("comma_core.method")
