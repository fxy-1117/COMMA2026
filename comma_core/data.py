"""Data loading that mirrors the notebook's sample construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


def canonical_label(label: Any) -> str:
    value = str(label)
    if value in {"entailment", "ent"}:
        return "ent"
    if value in {"contradiction", "con"}:
        return "con"
    if value in {"neutral", "neu"}:
        return "neu"
    raise ValueError(f"unknown label: {label!r}")


def split_helpful_like_notebook(value: Any) -> List[str]:
    """Split Helpful exactly as the notebook does.

    Non-string values deliberately raise. In the original notebook this happens
    inside a broad try/except, so those rows are skipped before evaluation.
    """
    if not isinstance(value, str):
        raise AttributeError("Helpful is not a string")
    return [part.strip() for part in value.split(" \u2192 ")[:10] if part.strip()]


def build_sd_sent(source: Path, seed: int, neutral_source: Path) -> Tuple[List[List[str]], Dict[str, int]]:
    """Build the notebook-style sd_sent list from one source CSV and neu2.csv."""
    import pandas as pd

    sd = pd.read_csv(source)[:]
    sd_sent: List[List[str]] = []
    stats = {"source_rows": len(sd), "source_kept": 0, "source_skipped": 0, "neutral_added": 0}

    for _, group in sd.groupby("label"):
        for index in range(5000):
            try:
                row = group.iloc[index]
                steps = split_helpful_like_notebook(row["Helpful"])
                label = canonical_label(row["label"])
                sd_sent.append([row["Premise"], row["Claim"]] + steps + [label])
                stats["source_kept"] += 1
            except Exception:
                stats["source_skipped"] += 1
                continue

    neutral = pd.read_csv(neutral_source).sample(frac=1, random_state=seed)
    for index in range(250):
        sd_sent.append([neutral.iloc[index]["0"], neutral.iloc[index]["1"], "neu"])
        stats["neutral_added"] += 1

    return sd_sent, stats
