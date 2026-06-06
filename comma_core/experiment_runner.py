"""Evaluation loop for the three COMMA experiment families.

The runner is intentionally close to the notebook control flow: it scans the
prepared items in order, skips failed proof attempts, and stops each class only
after 140 successful predictions have been appended.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional

from .experiment_config import LABELS
from .runtime_utils import load_pickle_cache, save_pickle_cache, stable_key


LOGIC_CACHE_VERSION = "batch-item-v1"


def item_logic_key(sentences: List[str]) -> str:
    """Cache key for item-level AMR logic generated with notebook batching."""
    return stable_key([LOGIC_CACHE_VERSION, *sentences])


class ExperimentRunner:
    """Run settings with the same control flow as the original experiment code.

    The important control-flow detail is that class counters are incremented only
    after a prediction is appended. Failed examples and "both" results are
    skipped, then scanning continues until each class has up to 140 appended
    examples.
    """

    def __init__(
        self,
        logic_engine: ModuleType,
        evaluation_items: List[List[str]],
        cache_dir: Path,
        preload_logic: bool,
    ) -> None:
        self.logic_engine = logic_engine
        self.evaluation_items = evaluation_items
        self.cache_dir = cache_dir
        self.item_cache = load_pickle_cache(cache_dir / "logic_cache.pkl")
        self.logic_dirty = False

        self.logic_engine.LOGIC = {}
        self.logic_engine.not_work = []
        self.logic_engine.both = []

        if preload_logic:
            self.preload_stats = self.preload_logic()
        else:
            self.preload_stats = {
                "items": len(evaluation_items),
                "preloaded": 0,
                "missing": len(evaluation_items),
                "duplicates": 0,
            }

    def preload_logic(self) -> Dict[str, int]:
        """Fill ``logic_engine.LOGIC`` from item-level disk caches.

        The original notebook generated all sentences in one example with a
        single ``generate_logic(i[:-1])`` call. The cache therefore stores the
        full item result, not independently generated sentence formulas.
        """
        stats = {
            "items": 0,
            "from_item_cache": 0,
            "missing": 0,
            "duplicates": 0,
        }
        logic = self.logic_engine.LOGIC
        for item in self.evaluation_items:
            stats["items"] += 1
            notebook_key = item[0] + item[1]
            if notebook_key in logic:
                stats["duplicates"] += 1
                continue
            sentences = item[:-1]
            item_key = item_logic_key(sentences)
            if item_key in self.item_cache and len(self.item_cache[item_key]) == len(sentences):
                logic[notebook_key] = self.item_cache[item_key]
                stats["from_item_cache"] += 1
            else:
                stats["missing"] += 1
        return stats

    def flush_logic_caches(self) -> None:
        if not self.logic_dirty:
            return
        save_pickle_cache(self.cache_dir / "logic_cache.pkl", self.item_cache)
        self.logic_dirty = False

    def get_logic_forms(self, item: List[str]) -> List[Any]:
        """Get or generate logic forms for one notebook item.

        Missing logic is generated with the same batch call as the notebook:
        premise, claim, and any implicit-premise steps are parsed together.
        """
        notebook_key = item[0] + item[1]
        logic = self.logic_engine.LOGIC
        if notebook_key not in logic:
            sentences = item[:-1]
            item_key = item_logic_key(sentences)
            if item_key in self.item_cache and len(self.item_cache[item_key]) == len(sentences):
                forms = self.item_cache[item_key]
            else:
                print(f"generating batch logic for item: {item[0][:80]!r} -> {item[1][:80]!r}")
                raw_forms = self.logic_engine.generate_logic(sentences)[-2]
                if len(raw_forms) != len(sentences):
                    raise ValueError(
                        f"expected {len(sentences)} logic forms, got {len(raw_forms)}"
                    )
                forms = [self.logic_engine.transform_logic(formula) for formula in raw_forms]
                self.item_cache[item_key] = forms
                self.logic_dirty = True
                self.flush_logic_caches()
            logic[notebook_key] = forms
        return logic[notebook_key]

    @staticmethod
    def proof_input(experiment: str, pre_data: List[Any], step: Optional[int]) -> List[Any]:
        """Select premise, claim, and implicit-premise steps for one run."""
        if experiment == "exp1" or step == 0:
            return [pre_data[0], pre_data[1]]
        if experiment == "exp2":
            return [pre_data[0], pre_data[1]] + pre_data[2:3]
        if experiment == "exp3":
            if step is None:
                raise ValueError("exp3 requires step")
            return [pre_data[0], pre_data[1]] + pre_data[2 : 2 + step]
        raise ValueError(f"unknown experiment: {experiment}")

    def evaluate(
        self,
        experiment: str,
        tau_m: float,
        tau_c: int,
        step: Optional[int],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        from sklearn.metrics import accuracy_score, classification_report

        y_true: List[str] = []
        y_pred: List[str] = []
        counters = {"ent": 0, "con": 0, "neu": 0}
        skipped = 0
        # The notebook used ``> 139``, which yields 140 successful predictions
        # per class after skipped and ambiguous examples are ignored.
        fix_number = 139

        def emit_progress(scanned: int) -> None:
            if progress_callback is None:
                return
            correct = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == predicted)
            evaluated = len(y_true)
            progress_callback(
                {
                    "scanned": scanned,
                    "evaluated": evaluated,
                    "skipped": skipped,
                    "counters": counters.copy(),
                    "accuracy": correct / evaluated if evaluated else 0.0,
                }
            )

        for scanned, item in enumerate(self.evaluation_items[:], start=1):
            label = item[-1]
            if label == "ent" and counters["ent"] > fix_number:
                emit_progress(scanned)
                continue
            if label == "con" and counters["con"] > fix_number:
                emit_progress(scanned)
                continue
            if label == "neu" and counters["neu"] > fix_number:
                emit_progress(scanned)
                continue

            try:
                pre_data = self.get_logic_forms(item)
                proof_data = self.proof_input(experiment, pre_data, step)
                result = self.logic_engine.prove(proof_data, tau_m, tau_c)
                if result and result[0] != "both":
                    predicted = result[0]
                else:
                    skipped += 1
                    emit_progress(scanned)
                    continue

                y_true.append(label)
                y_pred.append(predicted)
                counters[label] += 1
                emit_progress(scanned)
            except Exception:
                skipped += 1
                emit_progress(scanned)
                continue

        report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred) if y_true else 0.0
        return {
            "experiment": experiment,
            "tau_m": tau_m,
            "tau_c": tau_c,
            "step": step,
            "n": len(y_true),
            "class_counts": counters,
            "skipped": skipped,
            "accuracy": accuracy,
            "report": report,
            "preload_stats": self.preload_stats,
        }
