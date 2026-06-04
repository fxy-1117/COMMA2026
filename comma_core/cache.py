"""Disk caches for neural model calls."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

from .utils import load_json_cache, save_json_cache, stable_key


def install_neural_caches(method: ModuleType, cache_dir: Path):
    """Wrap the notebook NLI and similarity functions with JSON caches."""
    nli_path = cache_dir / "nli_cache.json"
    score_path = cache_dir / "score_cache.json"
    nli_cache = load_json_cache(nli_path)
    score_cache = load_json_cache(score_path)
    dirty = {"nli": False, "score": False}

    original_nli = method.NLI
    original_score = method.score

    def cached_nli(x: str, y: str, tokenizer: Any = None, model: Any = None):
        key = stable_key([x, y])
        if key not in nli_cache:
            label, confidence = original_nli(x, y, method.nli_tokenizer, method.model_nli)
            nli_cache[key] = [label, confidence]
            dirty["nli"] = True
        label, confidence = nli_cache[key]
        return label, confidence

    def cached_score(s1: str, s2: str) -> float:
        key = stable_key([s1, s2])
        if key not in score_cache:
            value = original_score(s1, s2)
            if hasattr(value, "item"):
                value = value.item()
            score_cache[key] = float(value)
            dirty["score"] = True
        return float(score_cache[key])

    def flush() -> None:
        if dirty["nli"]:
            save_json_cache(nli_path, nli_cache)
            dirty["nli"] = False
        if dirty["score"]:
            save_json_cache(score_path, score_cache)
            dirty["score"] = False

    method.NLI = cached_nli
    method.score = cached_score
    return flush
