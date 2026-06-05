"""Task-list entry point for the COMMA experiment pipeline.

Select experiment settings by editing ``EXPERIMENT_TASKS`` below. The rest of
the script keeps the notebook's experiment semantics and writes reproducible,
review-friendly artifacts into an independent output directory.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from comma_core.dataset_builder import build_evaluation_items
from comma_core.experiment_runner import ExperimentRunner
from comma_core.model_runtime import load_logic_engine
from comma_core.neural_cache import install_neural_caches
from comma_core.paper_reference import EXP3_STEPS
from comma_core.result_writer import write_outputs
from comma_core.runtime_utils import set_seed


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = ROOT / ".experiment_cache"
DEFAULT_OUTPUT_ROOT = ROOT / "experiment_outputs"


@dataclass(frozen=True)
class ExperimentTask:
    """One concrete experiment setting.

    Comment out task lines in ``EXPERIMENT_TASKS`` to skip them. Uncomment or
    add a line to run that setting.
    """

    experiment: str
    tau_m: float
    tau_c: int
    step: Optional[int] = None

    def as_setting(self) -> Tuple[str, float, int, Optional[int]]:
        return (self.experiment, self.tau_m, self.tau_c, self.step)


# Edit this list to choose what to run.
#
# Examples:
# - Comment out a line to skip that setting.
# - Leave only one line uncommented for a quick single-setting run.
# - Exp3 uses ``step``; Exp1/Exp2 leave ``step`` empty.
EXPERIMENT_TASKS: List[ExperimentTask] = [
    # Experiment 1: no implicit premise.
    ExperimentTask("exp1", 0.50, 80),
    ExperimentTask("exp1", 0.50, 90),
    ExperimentTask("exp1", 0.50, 100),
    ExperimentTask("exp1", 0.55, 80),
    ExperimentTask("exp1", 0.55, 90),
    ExperimentTask("exp1", 0.55, 100),
    ExperimentTask("exp1", 0.60, 80),
    ExperimentTask("exp1", 0.60, 90),
    ExperimentTask("exp1", 0.60, 100),
    ExperimentTask("exp1", 0.65, 80),
    ExperimentTask("exp1", 0.65, 90),
    ExperimentTask("exp1", 0.65, 100),
    ExperimentTask("exp1", 0.70, 80),
    ExperimentTask("exp1", 0.70, 90),
    ExperimentTask("exp1", 0.70, 100),
    ExperimentTask("exp1", 0.75, 80),
    ExperimentTask("exp1", 0.75, 90),
    ExperimentTask("exp1", 0.75, 100),
    ExperimentTask("exp1", 0.80, 80),
    ExperimentTask("exp1", 0.80, 90),
    ExperimentTask("exp1", 0.80, 100),

    # Experiment 2: single implicit premise.
    ExperimentTask("exp2", 0.50, 80),
    ExperimentTask("exp2", 0.50, 90),
    ExperimentTask("exp2", 0.50, 100),
    ExperimentTask("exp2", 0.55, 80),
    ExperimentTask("exp2", 0.55, 90),
    ExperimentTask("exp2", 0.55, 100),
    ExperimentTask("exp2", 0.60, 80),
    ExperimentTask("exp2", 0.60, 90),
    ExperimentTask("exp2", 0.60, 100),
    ExperimentTask("exp2", 0.65, 80),
    ExperimentTask("exp2", 0.65, 90),
    ExperimentTask("exp2", 0.65, 100),
    ExperimentTask("exp2", 0.70, 80),
    ExperimentTask("exp2", 0.70, 90),
    ExperimentTask("exp2", 0.70, 100),
    ExperimentTask("exp2", 0.75, 80),
    ExperimentTask("exp2", 0.75, 90),
    ExperimentTask("exp2", 0.75, 100),
    ExperimentTask("exp2", 0.80, 80),
    ExperimentTask("exp2", 0.80, 90),
    ExperimentTask("exp2", 0.80, 100),

    # Experiment 3: step analysis at tau_m=0.60, tau_c=80.
    ExperimentTask("exp3", 0.60, 80, step=0),
    ExperimentTask("exp3", 0.60, 80, step=1),
    ExperimentTask("exp3", 0.60, 80, step=2),
    ExperimentTask("exp3", 0.60, 80, step=3),
    ExperimentTask("exp3", 0.60, 80, step=4),
    ExperimentTask("exp3", 0.60, 80, step=5),
]


def selected_settings(tasks: List[ExperimentTask]) -> List[Tuple[str, float, int, Optional[int]]]:
    """Validate configured tasks and convert them to runner settings."""
    if not tasks:
        raise ValueError("EXPERIMENT_TASKS is empty; uncomment at least one task.")

    settings: List[Tuple[str, float, int, Optional[int]]] = []
    for task in tasks:
        if task.experiment not in {"exp1", "exp2", "exp3"}:
            raise ValueError(f"unknown experiment: {task.experiment}")
        if task.experiment == "exp3" and task.step not in EXP3_STEPS:
            raise ValueError(f"exp3 requires step in {EXP3_STEPS}: {task}")
        if task.experiment in {"exp1", "exp2"} and task.step is not None:
            raise ValueError(f"{task.experiment} does not use step: {task}")
        settings.append(task.as_setting())
    return settings


def format_setting(setting: Tuple[str, float, int, Optional[int]]) -> str:
    """Render one task for logs or ``--list-tasks`` output."""
    experiment, tau_m, tau_c, step = setting
    label = f"{experiment} tau_m={tau_m} tau_c={tau_c}"
    if step is not None:
        label += f" step={step}"
    return label


def resolve_output_dir(output_root: Path, output_dir: Optional[Path], run_id: Optional[str]) -> Path:
    """Return an independent output directory for this run."""
    if output_dir is not None:
        return output_dir
    actual_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / actual_run_id


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp12-source", default="data/exp1&exp2.csv")
    parser.add_argument("--exp3-source", default="data/exp3.csv")
    parser.add_argument("--neutral-source", default="data/neu2.csv")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=1129)
    parser.add_argument("--no-preload-logic", action="store_true")
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the tasks configured in EXPERIMENT_TASKS and exit.",
    )
    args = parser.parse_args()

    settings = selected_settings(EXPERIMENT_TASKS)
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

            for experiment, tau_m, tau_c, step in group_settings:
                label = format_setting((experiment, tau_m, tau_c, step))
                print(f"running {label}")
                sys.stdout.flush()

                # The original proof code is very chatty; keep detailed traces
                # in a per-run log instead of flooding stdout.
                with contextlib.redirect_stdout(detail), contextlib.redirect_stderr(detail):
                    result = evaluator.evaluate(experiment, tau_m, tau_c, step)
                result["source"] = source_name
                result["data_stats"] = data_stats
                results.append(result)

                evaluator.flush_logic_caches()
                flush_neural_caches()
                write_outputs(output_dir, results, source_by_experiment)

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
    write_outputs(output_dir, results, source_by_experiment)
    print(f"wrote outputs under {output_dir}")


if __name__ == "__main__":
    run()
