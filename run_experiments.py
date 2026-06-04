"""Run the COMMA experiments from the original Method.ipynb runtime.

The notebook contains several fragile behaviours that affect the reported
numbers. This script preserves those behaviours intentionally:

* examples with non-string Helpful values are skipped while building sd_sent;
* LOGIC is keyed by premise + claim, as in the notebook;
* the notebook's original pysat_formula implementation is used.

The script adds operational conveniences: local/offline model loading,
disk caches for NLI/similarity calls, optional logic-cache preloading, JSON/CSV
outputs, and automatic comparison to the reported values.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "Method.ipynb"
DEFAULT_CACHE_DIR = ROOT / ".experiment_cache"
DEFAULT_OUTPUT_DIR = ROOT / "experiment_outputs"

LABELS = ["con", "ent", "neu"]
TAU_M_VALUES = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
TAU_C_VALUES = [80, 90, 100]
EXP3_STEPS = [0, 1, 2, 3, 4, 5]

EXPECTED_ACCURACY = {
    ("exp1", 0.50, 80, None): 0.410072,
    ("exp1", 0.50, 90, None): 0.390476,
    ("exp1", 0.50, 100, None): 0.373810,
    ("exp1", 0.55, 80, None): 0.426190,
    ("exp1", 0.55, 90, None): 0.411905,
    ("exp1", 0.55, 100, None): 0.390476,
    ("exp1", 0.60, 80, None): 0.421429,
    ("exp1", 0.60, 90, None): 0.416667,
    ("exp1", 0.60, 100, None): 0.409524,
    ("exp1", 0.65, 80, None): 0.371429,
    ("exp1", 0.65, 90, None): 0.366667,
    ("exp1", 0.65, 100, None): 0.361905,
    ("exp1", 0.70, 80, None): 0.378571,
    ("exp1", 0.70, 90, None): 0.376190,
    ("exp1", 0.70, 100, None): 0.352381,
    ("exp1", 0.75, 80, None): 0.354762,
    ("exp1", 0.75, 90, None): 0.352381,
    ("exp1", 0.75, 100, None): 0.338095,
    ("exp1", 0.80, 80, None): 0.347619,
    ("exp1", 0.80, 90, None): 0.342857,
    ("exp1", 0.80, 100, None): 0.340476,
    ("exp2", 0.50, 80, None): 0.433333,
    ("exp2", 0.50, 90, None): 0.433333,
    ("exp2", 0.50, 100, None): 0.400000,
    ("exp2", 0.55, 80, None): 0.464286,
    ("exp2", 0.55, 90, None): 0.461905,
    ("exp2", 0.55, 100, None): 0.430952,
    ("exp2", 0.60, 80, None): 0.507143,
    ("exp2", 0.60, 90, None): 0.511905,
    ("exp2", 0.60, 100, None): 0.485714,
    ("exp2", 0.65, 80, None): 0.476190,
    ("exp2", 0.65, 90, None): 0.480952,
    ("exp2", 0.65, 100, None): 0.450000,
    ("exp2", 0.70, 80, None): 0.438095,
    ("exp2", 0.70, 90, None): 0.433333,
    ("exp2", 0.70, 100, None): 0.388095,
    ("exp2", 0.75, 80, None): 0.402381,
    ("exp2", 0.75, 90, None): 0.397619,
    ("exp2", 0.75, 100, None): 0.364286,
    ("exp2", 0.80, 80, None): 0.373810,
    ("exp2", 0.80, 90, None): 0.366667,
    ("exp2", 0.80, 100, None): 0.354762,
    ("exp3", 0.60, 80, 0): 0.426,
    ("exp3", 0.60, 80, 1): 0.507,
    ("exp3", 0.60, 80, 2): 0.531,
    ("exp3", 0.60, 80, 3): 0.540,
    ("exp3", 0.60, 80, 4): 0.555,
    ("exp3", 0.60, 80, 5): 0.552,
}

EXPECTED_EXP3_METRICS = {
    0: {"con": {"precision": 0.49, "recall": 0.31}, "ent": {"precision": 0.44, "recall": 0.34}, "neu": {"precision": 0.39, "recall": 0.63}},
    1: {"con": {"precision": 0.56, "recall": 0.49}, "ent": {"precision": 0.47, "recall": 0.40}, "neu": {"precision": 0.49, "recall": 0.63}},
    2: {"con": {"precision": 0.57, "recall": 0.54}, "ent": {"precision": 0.47, "recall": 0.43}, "neu": {"precision": 0.55, "recall": 0.63}},
    3: {"con": {"precision": 0.53, "recall": 0.52}, "ent": {"precision": 0.49, "recall": 0.47}, "neu": {"precision": 0.60, "recall": 0.63}},
    4: {"con": {"precision": 0.51, "recall": 0.53}, "ent": {"precision": 0.50, "recall": 0.51}, "neu": {"precision": 0.66, "recall": 0.63}},
    5: {"con": {"precision": 0.52, "recall": 0.52}, "ent": {"precision": 0.51, "recall": 0.51}, "neu": {"precision": 0.63, "recall": 0.63}},
}


def stable_key(parts: Iterable[Any]) -> str:
    payload = json.dumps(list(parts), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def set_seed(seed: int) -> None:
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
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_cache(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_pickle_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_pickle_cache(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(data, handle)
    tmp.replace(path)


def patch_torch_hub() -> None:
    import torch

    original_load = torch.hub.load
    local_repo = Path.home() / ".cache" / "torch" / "hub" / "pytorch_fairseq_main"

    def patched_load(repo_or_dir: str, model: str, *args: Any, **kwargs: Any) -> Any:
        if repo_or_dir == "pytorch/fairseq" and local_repo.exists():
            kwargs.pop("source", None)
            return original_load(str(local_repo), model, *args, source="local", **kwargs)
        return original_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = patched_load


def exec_notebook_cell(ns: Dict[str, Any], notebook: Dict[str, Any], index: int) -> None:
    cell = notebook["cells"][index]
    if cell.get("cell_type") != "code":
        return
    source = "".join(cell.get("source", []))
    source = source.replace(
        "from pattern.en import conjugate, lemma, lexeme, PRESENT, SG\n",
        "",
    )
    if source.strip():
        exec(compile(source, f"{NOTEBOOK.name}:cell{index}", "exec"), ns)


def load_method_runtime() -> Dict[str, Any]:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    patch_torch_hub()

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8", errors="ignore"))
    ns: Dict[str, Any] = {"__name__": "__comma_experiments__"}
    for index in range(0, 17):
        exec_notebook_cell(ns, notebook, index)
    return ns


def install_neural_caches(ns: Dict[str, Any], cache_dir: Path):
    nli_path = cache_dir / "nli_cache.json"
    score_path = cache_dir / "score_cache.json"
    nli_cache = load_json_cache(nli_path)
    score_cache = load_json_cache(score_path)
    dirty = {"nli": False, "score": False}

    original_nli = ns["NLI"]
    original_score = ns["score"]

    def cached_nli(x: str, y: str, tokenizer: Any = None, model: Any = None):
        key = stable_key([x, y])
        if key not in nli_cache:
            label, confidence = original_nli(x, y, ns["nli_tokenizer"], ns["model_nli"])
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

    ns["NLI"] = cached_nli
    ns["score"] = cached_score
    return flush


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
    if not isinstance(value, str):
        raise AttributeError("Helpful is not a string")
    return [part.strip() for part in value.split(" \u2192 ")[:10] if part.strip()]


def build_sd_sent(source: Path, seed: int, neutral_source: Path) -> Tuple[List[List[str]], Dict[str, int]]:
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


class NotebookStyleEvaluator:
    def __init__(
        self,
        ns: Dict[str, Any],
        sd_sent: List[List[str]],
        cache_dir: Path,
        preload_logic: bool,
        detail_stream: Any,
    ) -> None:
        self.ns = ns
        self.sd_sent = sd_sent
        self.cache_dir = cache_dir
        self.detail_stream = detail_stream
        self.sentence_cache = load_pickle_cache(cache_dir / "sentence_logic_cache.pkl")
        self.item_cache = load_pickle_cache(cache_dir / "logic_cache.pkl")
        self.logic_dirty = False
        self.ns["LOGIC"] = {}
        self.ns["not_work"] = []
        self.ns["both"] = []
        if preload_logic:
            self.preload_stats = self.preload_logic()
        else:
            self.preload_stats = {"items": len(sd_sent), "preloaded": 0, "missing": len(sd_sent), "duplicates": 0}

    def preload_logic(self) -> Dict[str, int]:
        stats = {"items": 0, "from_sentence_cache": 0, "from_item_cache": 0, "missing": 0, "duplicates": 0}
        logic = self.ns["LOGIC"]
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

    def get_logic_forms(self, item: List[str]) -> List[Any]:
        notebook_key = item[0] + item[1]
        logic = self.ns["LOGIC"]
        if notebook_key not in logic:
            sentences = item[:-1]
            forms = []
            for sentence in sentences:
                if sentence not in self.sentence_cache:
                    print(f"generating logic for sentence: {sentence[:120]!r}")
                    formula = self.ns["generate_logic"]([sentence])[-2][0]
                    self.sentence_cache[sentence] = self.ns["transform_logic"](formula)
                    self.logic_dirty = True
                    self.flush_logic_caches()
                forms.append(self.sentence_cache[sentence])
            logic[notebook_key] = forms
            item_key = stable_key(sentences)
            self.item_cache[item_key] = forms
            self.logic_dirty = True
            self.flush_logic_caches()
        return logic[notebook_key]

    def flush_logic_caches(self) -> None:
        if not self.logic_dirty:
            return
        save_pickle_cache(self.cache_dir / "sentence_logic_cache.pkl", self.sentence_cache)
        save_pickle_cache(self.cache_dir / "logic_cache.pkl", self.item_cache)
        self.logic_dirty = False

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
                result = self.ns["prove"](proof_data, tau_m, tau_c)
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

        report = classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )
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


def settings_for(
    experiments: List[str],
    exp3_steps: Optional[List[int]] = None,
) -> List[Tuple[str, float, int, Optional[int]]]:
    settings: List[Tuple[str, float, int, Optional[int]]] = []
    for experiment in experiments:
        if experiment in {"exp1", "exp2"}:
            for tau_m in TAU_M_VALUES:
                for tau_c in TAU_C_VALUES:
                    settings.append((experiment, tau_m, tau_c, None))
        elif experiment == "exp3":
            for step in (exp3_steps or EXP3_STEPS):
                settings.append(("exp3", 0.6, 80, step))
        else:
            raise ValueError(f"unknown experiment: {experiment}")
    return settings


def result_to_summary_row(result: Dict[str, Any], source: str) -> Dict[str, Any]:
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
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir: Path, results: List[Dict[str, Any]], source_by_experiment: Dict[str, str]) -> None:
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


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["exp1", "exp2", "exp3"],
        default=["exp1", "exp2", "exp3"],
    )
    parser.add_argument("--exp12-source", default="exp1&exp2.csv")
    parser.add_argument("--exp3-source", default="exp3.csv")
    parser.add_argument("--neutral-source", default="neu2.csv")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=1129)
    parser.add_argument("--no-preload-logic", action="store_true")
    parser.add_argument(
        "--exp3-steps",
        nargs="+",
        type=int,
        choices=EXP3_STEPS,
        help="Subset of exp3 steps to run.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    print("loading Method.ipynb runtime")
    sys.stdout.flush()
    ns = load_method_runtime()
    flush_neural_caches = install_neural_caches(ns, args.cache_dir)

    source_by_experiment = {
        "exp1": args.exp12_source,
        "exp2": args.exp12_source,
        "exp3": args.exp3_source,
    }
    settings = settings_for(args.experiments, exp3_steps=args.exp3_steps)
    results: List[Dict[str, Any]] = []
    detail_path = args.output_dir / "experiment_detail.log"
    detail_path.parent.mkdir(parents=True, exist_ok=True)

    with detail_path.open("w", encoding="utf-8") as detail:
        for source_name in [args.exp12_source, args.exp3_source]:
            group_settings = [
                setting for setting in settings if source_by_experiment[setting[0]] == source_name
            ]
            if not group_settings:
                continue

            print(f"building sd_sent from {source_name}")
            sys.stdout.flush()
            sd_sent, data_stats = build_sd_sent(
                ROOT / source_name,
                args.seed,
                ROOT / args.neutral_source,
            )
            evaluator = NotebookStyleEvaluator(
                ns,
                sd_sent,
                args.cache_dir,
                preload_logic=not args.no_preload_logic,
                detail_stream=detail,
            )
            print(f"data stats: {data_stats}")
            print(f"logic preload: {evaluator.preload_stats}")
            sys.stdout.flush()

            for experiment, tau_m, tau_c, step in group_settings:
                label = f"{experiment} tau_m={tau_m} tau_c={tau_c}"
                if step is not None:
                    label += f" step={step}"
                print(f"running {label}")
                sys.stdout.flush()
                with contextlib.redirect_stdout(detail), contextlib.redirect_stderr(detail):
                    result = evaluator.evaluate(experiment, tau_m, tau_c, step)
                result["source"] = source_name
                result["data_stats"] = data_stats
                results.append(result)
                evaluator.flush_logic_caches()
                flush_neural_caches()
                write_outputs(args.output_dir, results, source_by_experiment)
                expected = result["expected_accuracy"]
                delta = result["accuracy_delta"]
                expected_text = "n/a" if expected is None else f"{expected:.6f}"
                delta_text = "n/a" if delta is None else f"{delta:+.6f}"
                print(
                    f"done {label}: accuracy={result['accuracy']:.6f}, "
                    f"expected={expected_text}, delta={delta_text}, n={result['n']}"
                )
                sys.stdout.flush()

    flush_neural_caches()
    write_outputs(args.output_dir, results, source_by_experiment)
    print(f"wrote outputs under {args.output_dir}")


if __name__ == "__main__":
    run()
