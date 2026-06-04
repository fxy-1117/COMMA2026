"""Evaluation loop for Exp1/Exp2/Exp3."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional

from .expected import EXPECTED_ACCURACY, LABELS
from .utils import load_pickle_cache, save_pickle_cache, stable_key


class ExperimentEvaluator:
    """Run settings with the same control flow as the original experiment code.

    The important control-flow detail is that class counters are incremented only
    after a prediction is appended. Failed examples and "both" results are
    skipped, then scanning continues until each class has up to 140 appended
    examples.
    """

    def __init__(
        self,
        method: ModuleType,
        sd_sent: List[List[str]],
        cache_dir: Path,
        preload_logic: bool,
    ) -> None:
        self.method = method
        self.sd_sent = sd_sent
        self.cache_dir = cache_dir
        self.sentence_cache = load_pickle_cache(cache_dir / "sentence_logic_cache.pkl")
        self.item_cache = load_pickle_cache(cache_dir / "logic_cache.pkl")
        self.logic_dirty = False

        self.method.LOGIC = {}
        self.method.not_work = []
        self.method.both = []

        if preload_logic:
            self.preload_stats = self.preload_logic()
        else:
            self.preload_stats = {"items": len(sd_sent), "preloaded": 0, "missing": len(sd_sent), "duplicates": 0}

    def preload_logic(self) -> Dict[str, int]:
        """Fill method.LOGIC from disk caches using the notebook key format."""
        stats = {"items": 0, "from_sentence_cache": 0, "from_item_cache": 0, "missing": 0, "duplicates": 0}
        logic = self.method.LOGIC
        for item in self.sd_sent:
            stats["items"] += 1
            notebook_key = item[0] + item[1]
            if notebook_key in logic:
                stats["duplicates"] += 1
                continue
            sentences = item[:-1]
            item_key = stable_key(sentences)
            if item_key in self.item_cache and len(self.item_cache[item_key]) == len(sentences):
                logic[notebook_key] = self.item_cache[item_key]
                stats["from_item_cache"] += 1
            elif all(sentence in self.sentence_cache for sentence in sentences):
                logic[notebook_key] = [self.sentence_cache[sentence] for sentence in sentences]
                stats["from_sentence_cache"] += 1
            else:
                stats["missing"] += 1
        return stats

    def flush_logic_caches(self) -> None:
        if not self.logic_dirty:
            return
        save_pickle_cache(self.cache_dir / "sentence_logic_cache.pkl", self.sentence_cache)
        save_pickle_cache(self.cache_dir / "logic_cache.pkl", self.item_cache)
        self.logic_dirty = False

    def get_logic_forms(self, item: List[str]) -> List[Any]:
        """Get or generate logic forms for one notebook item.

        Missing logic is generated sentence-by-sentence. This preserves the
        final formulas while avoiding fragile large AMR batches.
        """
        notebook_key = item[0] + item[1]
        logic = self.method.LOGIC
        if notebook_key not in logic:
            sentences = item[:-1]
            forms = []
            for sentence in sentences:
                if sentence not in self.sentence_cache:
                    print(f"generating logic for sentence: {sentence[:120]!r}")
                    formula = self.method.generate_logic([sentence])[-2][0]
                    self.sentence_cache[sentence] = self.method.transform_logic(formula)
                    self.logic_dirty = True
                    self.flush_logic_caches()
                forms.append(self.sentence_cache[sentence])
            logic[notebook_key] = forms
            self.item_cache[stable_key(sentences)] = forms
            self.logic_dirty = True
            self.flush_logic_caches()
        return logic[notebook_key]

    @staticmethod
    def proof_input(experiment: str, pre_data: List[Any], step: Optional[int]) -> List[Any]:
        if experiment == "exp1" or step == 0:
            return [pre_data[0], pre_data[1]]
        if experiment == "exp2":
            return [pre_data[0], pre_data[1]] + pre_data[2:3]
        if experiment == "exp3":
            if step is None:
                raise ValueError("exp3 requires step")
            return [pre_data[0], pre_data[1]] + pre_data[2 : 2 + step]
        raise ValueError(f"unknown experiment: {experiment}")

    def evaluate(self, experiment: str, tau_m: float, tau_c: int, step: Optional[int]) -> Dict[str, Any]:
        from sklearn.metrics import accuracy_score, classification_report

        y_true: List[str] = []
        y_pred: List[str] = []
        counters = {"ent": 0, "con": 0, "neu": 0}
        skipped = 0
        fix_number = 139

        for item in self.sd_sent[:]:
            label = item[-1]
            if label == "ent" and counters["ent"] > fix_number:
                continue
            if label == "con" and counters["con"] > fix_number:
                continue
            if label == "neu" and counters["neu"] > fix_number:
                continue

            try:
                pre_data = self.get_logic_forms(item)
                proof_data = self.proof_input(experiment, pre_data, step)
                result = self.method.prove(proof_data, tau_m, tau_c)
                if result and result[0] != "both" and result[0] != "neu":
                    predicted = result[0]
                elif result and result[0] != "both":
                    predicted = result[0]
                else:
                    skipped += 1
                    continue

                y_true.append(label)
                y_pred.append(predicted)
                counters[label] += 1
            except Exception:
                skipped += 1
                continue

        report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred) if y_true else 0.0
        expected = EXPECTED_ACCURACY.get((experiment, round(tau_m, 2), tau_c, step))
        return {
            "experiment": experiment,
            "tau_m": tau_m,
            "tau_c": tau_c,
            "step": step,
            "n": len(y_true),
            "class_counts": counters,
            "skipped": skipped,
            "accuracy": accuracy,
            "expected_accuracy": expected,
            "accuracy_delta": None if expected is None else accuracy - expected,
            "report": report,
            "preload_stats": self.preload_stats,
        }
