"""Shared runtime, seeding, and cache I/O helpers."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Iterable


def stable_key(parts: Iterable[Any]) -> str:
    """Create a deterministic cache key for text pairs or sentence lists."""
    payload = json.dumps(list(parts), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def set_seed(seed: int) -> None:
    """Set the same seed knobs used during the notebook experiments."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def load_json_cache(path: Path) -> Dict[str, Any]:
    """Load a JSON dictionary cache, returning an empty cache if absent."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_cache(path: Path, data: Dict[str, Any]) -> None:
    """Atomically write a JSON dictionary cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_pickle_cache(path: Path) -> Dict[str, Any]:
    """Load a pickle dictionary cache, returning an empty cache if absent."""
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_pickle_cache(path: Path, data: Dict[str, Any]) -> None:
    """Atomically write a pickle dictionary cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(data, handle)
    tmp.replace(path)


def patch_torch_hub() -> None:
    """Use the locally cached fairseq hub repository when available."""
    import torch

    original_load = torch.hub.load
    local_repo = Path.home() / ".cache" / "torch" / "hub" / "pytorch_fairseq_main"

    def patched_load(repo_or_dir: str, model: str, *args: Any, **kwargs: Any) -> Any:
        if repo_or_dir == "pytorch/fairseq" and local_repo.exists():
            kwargs.pop("source", None)
            return original_load(str(local_repo), model, *args, source="local", **kwargs)
        return original_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = patched_load
