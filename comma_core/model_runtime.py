"""Runtime setup for the extracted logic engine.

The engine imports neural models at module import time. This loader applies the
offline/cache configuration first, then imports the engine exactly once through
Python's normal module system.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import os
from types import ModuleType

from .runtime_utils import patch_torch_hub


class _SilentStream:
    """No-op stream used while third-party model loaders initialize."""

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def load_logic_engine(quiet: bool = True) -> ModuleType:
    """Load the experiment logic engine after applying local runtime patches."""
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    patch_torch_hub()

    logging.getLogger("fairseq").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    if not quiet or os.environ.get("COMMA_VERBOSE_MODEL_LOADING") == "1":
        return importlib.import_module("comma_core.logic_engine")

    silent = _SilentStream()
    with contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
        return importlib.import_module("comma_core.logic_engine")
