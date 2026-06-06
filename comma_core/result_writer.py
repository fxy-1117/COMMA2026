"""Write experiment outputs.

Each run directory receives the full JSON results and a compact CSV summary.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from .experiment_config import LABELS


def result_to_summary_row(result: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Flatten one result dictionary into a CSV-friendly row."""
    row: Dict[str, Any] = {
        "experiment": result["experiment"],
        "tau_m": result["tau_m"],
        "tau_c": result["tau_c"],
        "step": "" if result["step"] is None else result["step"],
        "source": source,
        "n": result["n"],
        "accuracy": result["accuracy"],
        "skipped": result["skipped"],
    }
    for label in LABELS:
        metrics = result["report"][label]
        row[f"{label}_precision"] = metrics["precision"]
        row[f"{label}_recall"] = metrics["recall"]
        row[f"{label}_f1"] = metrics["f1-score"]
        row[f"{label}_support"] = metrics["support"]
    return row


def write_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a CSV table when there is at least one row."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir: Path, results: List[Dict[str, Any]], source_by_experiment: Dict[str, str]) -> None:
    """Write all output artifacts for one independent run directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = [
        result_to_summary_row(result, source_by_experiment[result["experiment"]])
        for result in results
    ]
    write_table(output_dir / "experiment_summary.csv", summary)
