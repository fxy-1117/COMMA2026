"""Build evaluation items from the checked-in CSV data.

The original notebook assembled evaluation examples by scanning rows per class,
skipping rows whose ``Helpful`` field could not be split, and then appending a
shuffled neutral set. This module keeps that behavior explicit so the experiment
runner can reproduce the reported sampling order.
"""

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


def split_reasoning_steps(value: Any) -> List[str]:
    """Split the ``Helpful`` reasoning chain exactly as the notebook does.

    Non-string values deliberately raise. In the original notebook this happens
    inside a broad try/except, so those rows are skipped before evaluation.
    """
    if not isinstance(value, str):
        raise AttributeError("Helpful is not a string")
    return [part.strip() for part in value.split(" \u2192 ")[:10] if part.strip()]


def build_evaluation_items(
    source: Path,
    seed: int,
    neutral_source: Path,
) -> Tuple[List[List[str]], Dict[str, int]]:
    """Build the evaluation list from one experiment source CSV and neutral pairs."""
    import pandas as pd

    source_rows = pd.read_csv(source)[:]
    evaluation_items: List[List[str]] = []
    stats = {
        "source_rows": len(source_rows),
        "source_kept": 0,
        "source_skipped": 0,
        "neutral_added": 0,
    }

    # The notebook scanned a fixed upper bound per class and skipped failures.
    for _, group in source_rows.groupby("label"):
        for index in range(5000):
            try:
                row = group.iloc[index]
                steps = split_reasoning_steps(row["Helpful"])
                label = canonical_label(row["label"])
                evaluation_items.append([row["Premise"], row["Claim"]] + steps + [label])
                stats["source_kept"] += 1
            except Exception:
                stats["source_skipped"] += 1
                continue

    # Neutral rows are appended after a deterministic shuffle with the run seed.
    neutral = pd.read_csv(neutral_source).sample(frac=1, random_state=seed)
    for index in range(250):
        evaluation_items.append([neutral.iloc[index]["0"], neutral.iloc[index]["1"], "neu"])
        stats["neutral_added"] += 1

    return evaluation_items, stats
