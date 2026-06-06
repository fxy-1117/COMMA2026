"""Experiment-list entry point for the COMMA experiment pipeline.

Select experiment groups by editing ``RUN_EXPERIMENTS`` below. The rest of
the script keeps the notebook's experiment semantics and writes reproducible,
review-friendly artifacts into an independent output directory.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from comma_core.dataset_builder import build_evaluation_items
from comma_core.experiment_config import EXP3_STEPS, TAU_C_VALUES, TAU_M_VALUES
from comma_core.experiment_runner import ExperimentRunner
from comma_core.model_runtime import load_logic_engine
from comma_core.neural_cache import install_neural_caches
from comma_core.result_writer import write_outputs
from comma_core.runtime_utils import set_seed

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = ROOT / ".experiment_cache"
DEFAULT_OUTPUT_ROOT = ROOT / "experiment_outputs"


# Edit this list to choose what to run.
#
# Examples:
# - Keep all three lines to run all paper experiments.
# - Comment out a line to skip that experiment.
# - Leave only "exp3" uncommented to run only the step analysis.
RUN_EXPERIMENTS: List[str] = [
    "exp1",
    "exp2",
    "exp3",
]


def selected_settings(experiments: List[str]) -> List[Tuple[str, float, int, Optional[int]]]:
    """Expand selected experiment groups into concrete parameter settings."""
    if not experiments:
        raise ValueError("RUN_EXPERIMENTS is empty; uncomment at least one experiment.")

    settings: List[Tuple[str, float, int, Optional[int]]] = []
    for experiment in experiments:
        if experiment in {"exp1", "exp2"}:
            for tau_m in TAU_M_VALUES:
                for tau_c in TAU_C_VALUES:
                    settings.append((experiment, tau_m, tau_c, None))
        elif experiment == "exp3":
            for step in EXP3_STEPS:
                settings.append(("exp3", 0.6, 80, step))
        else:
            raise ValueError(f"unknown experiment: {experiment}")
    return settings


def format_setting(setting: Tuple[str, float, int, Optional[int]]) -> str:
    """Render one task for logs or ``--list-tasks`` output."""
    experiment, tau_m, tau_c, step = setting
    label = f"{experiment} tau_m={tau_m} tau_c={tau_c}"
    if step is not None:
        label += f" step={step}"
    return label


def progress_bar(items: List[Tuple[str, float, int, Optional[int]]], source_name: str):
    """Return a tqdm progress bar when tqdm is installed."""
    if tqdm is None:
        return items
    return tqdm(items, desc=f"settings from {source_name}", unit="setting", position=0)


def item_progress_bar(total: int, label: str) -> Optional[Any]:
    """Return a per-item progress bar for one concrete setting."""
    if tqdm is None:
        return None
    return tqdm(total=total, desc=f"items: {label}", unit="item", leave=False, position=1)


def make_item_progress_callback(
    item_progress: Optional[Any],
) -> Optional[Callable[[Dict[str, Any]], None]]:
    """Convert runner progress dictionaries into compact tqdm updates."""
    if item_progress is None:
        return None

    last_scanned = {"value": 0}

    def callback(state: Dict[str, Any]) -> None:
        scanned = int(state["scanned"])
        delta = scanned - last_scanned["value"]

        counters = state["counters"]
        item_progress.set_postfix_str(
            "eval={evaluated} skip={skipped} con={con}/140 ent={ent}/140 "
            "neu={neu}/140 acc={accuracy:.3f}".format(
                evaluated=state["evaluated"],
                skipped=state["skipped"],
                con=counters["con"],
                ent=counters["ent"],
                neu=counters["neu"],
                accuracy=state["accuracy"],
            ),
            refresh=False,
        )
        if delta > 0:
            item_progress.update(delta)
            last_scanned["value"] = scanned

    return callback


def progress_write(message: str) -> None:
    """Print without breaking the tqdm display."""
    if tqdm is None:
        print(message)
    else:
        tqdm.write(message)


def resolve_output_dir(output_root: Path, output_dir: Optional[Path], run_id: Optional[str]) -> Path:
    """Return an independent output directory for this run."""
    if output_dir is not None:
        return output_dir
    actual_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / actual_run_id


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp12-source", default="data/rte_pairs_exp1_exp2.csv")
    parser.add_argument("--exp3-source", default="data/rte_pairs_exp3.csv")
    parser.add_argument("--neutral-source", default="data/neutral_pairs.csv")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=1129)
    parser.add_argument("--no-preload-logic", action="store_true")
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the settings expanded from RUN_EXPERIMENTS and exit.",
    )
    args = parser.parse_args()

    settings = selected_settings(RUN_EXPERIMENTS)
    if args.list_tasks:
        for index, setting in enumerate(settings, start=1):
            print(f"{index:02d}. {format_setting(setting)}")
        return

    output_dir = resolve_output_dir(args.output_root, args.output_dir, args.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    print("loading experiment runtime")
    sys.stdout.flush()
    logic_engine = load_logic_engine()
    flush_neural_caches = install_neural_caches(logic_engine, args.cache_dir)

    source_by_experiment = {
        "exp1": args.exp12_source,
        "exp2": args.exp12_source,
        "exp3": args.exp3_source,
    }
    results = []
    detail_path = output_dir / "experiment_detail.log"

    with detail_path.open("w", encoding="utf-8") as detail:
        # Exp1/Exp2 and Exp3 use different source CSVs but share the same
        # neutral examples and evaluation runner.
        for source_name in [args.exp12_source, args.exp3_source]:
            group_settings = [
                setting for setting in settings if source_by_experiment[setting[0]] == source_name
            ]
            if not group_settings:
                continue

            print(f"building evaluation items from {source_name}")
            sys.stdout.flush()
            evaluation_items, data_stats = build_evaluation_items(
                ROOT / source_name,
                args.seed,
                ROOT / args.neutral_source,
            )
            evaluator = ExperimentRunner(
                logic_engine,
                evaluation_items,
                args.cache_dir,
                preload_logic=not args.no_preload_logic,
            )
            print(f"data stats: {data_stats}")
            print(f"logic preload: {evaluator.preload_stats}")
            sys.stdout.flush()

            progress = progress_bar(group_settings, source_name)
            for experiment, tau_m, tau_c, step in progress:
                label = format_setting((experiment, tau_m, tau_c, step))
                if hasattr(progress, "set_postfix_str"):
                    progress.set_postfix_str(label)
                else:
                    print(f"running {label}")
                sys.stdout.flush()

                # The original proof code is very chatty; keep detailed traces
                # in a per-run log instead of flooding stdout.
                item_progress = item_progress_bar(len(evaluation_items), label)
                progress_callback = make_item_progress_callback(item_progress)
                try:
                    with contextlib.redirect_stdout(detail), contextlib.redirect_stderr(detail):
                        result = evaluator.evaluate(
                            experiment,
                            tau_m,
                            tau_c,
                            step,
                            progress_callback=progress_callback,
                        )
                finally:
                    if item_progress is not None:
                        item_progress.close()
                result["source"] = source_name
                result["data_stats"] = data_stats
                results.append(result)

                evaluator.flush_logic_caches()
                flush_neural_caches()
                write_outputs(output_dir, results, source_by_experiment)

                counts = result["class_counts"]
                progress_write(
                    f"done {label}: accuracy={result['accuracy']:.6f}, "
                    f"n={result['n']}, con={counts['con']}, "
                    f"ent={counts['ent']}, neu={counts['neu']}, skipped={result['skipped']}"
                )
                sys.stdout.flush()

    flush_neural_caches()
    write_outputs(output_dir, results, source_by_experiment)
    print(f"wrote outputs under {output_dir}")


if __name__ == "__main__":
    run()
