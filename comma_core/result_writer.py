"""Write experiment outputs and paper-reference comparisons.

Each run directory receives the full JSON results, a compact CSV summary, and
an optional comparison table against the values reported in the paper figures.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from .paper_reference import EXPECTED_EXP3_METRICS, LABELS


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
        "expected_accuracy": result["expected_accuracy"],
        "accuracy_delta": result["accuracy_delta"],
        "skipped": result["skipped"],
    }
    for label in LABELS:
        metrics = result["report"][label]
        row[f"{label}_precision"] = metrics["precision"]
        row[f"{label}_recall"] = metrics["recall"]
        row[f"{label}_f1"] = metrics["f1-score"]
        row[f"{label}_support"] = metrics["support"]
    return row


def comparison_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build rows comparing actual metrics with paper-reported references."""
    rows: List[Dict[str, Any]] = []
    for result in results:
        key = (result["experiment"], round(result["tau_m"], 2), result["tau_c"], result["step"])
        if result["expected_accuracy"] is not None:
            rows.append(
                {
                    "experiment": result["experiment"],
                    "tau_m": result["tau_m"],
                    "tau_c": result["tau_c"],
                    "step": "" if result["step"] is None else result["step"],
                    "class": "overall",
                    "metric": "accuracy",
                    "actual": result["accuracy"],
                    "expected": result["expected_accuracy"],
                    "delta": result["accuracy"] - result["expected_accuracy"],
                }
            )
        if result["experiment"] == "exp3" and key[3] in EXPECTED_EXP3_METRICS:
            expected_metrics = EXPECTED_EXP3_METRICS[key[3]]
            for label in LABELS:
                for metric in ["precision", "recall"]:
                    actual = round(result["report"][label][metric], 2)
                    expected = expected_metrics[label][metric]
                    rows.append(
                        {
                            "experiment": result["experiment"],
                            "tau_m": result["tau_m"],
                            "tau_c": result["tau_c"],
                            "step": result["step"],
                            "class": label,
                            "metric": metric,
                            "actual": actual,
                            "expected": expected,
                            "delta": actual - expected,
                        }
                    )
    return rows


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
    write_table(output_dir / "experiment_comparison.csv", comparison_rows(results))
