"""Command-line entry point for the COMMA experiment pipeline.

The CLI keeps the notebook's experiment semantics but writes reproducible,
review-friendly artifacts into an independent output directory.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from comma_core.dataset_builder import build_evaluation_items
from comma_core.experiment_runner import ExperimentRunner
from comma_core.model_runtime import load_logic_engine
from comma_core.neural_cache import install_neural_caches
from comma_core.paper_reference import EXP3_STEPS, TAU_C_VALUES, TAU_M_VALUES
from comma_core.result_writer import write_outputs
from comma_core.runtime_utils import set_seed


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = ROOT / ".experiment_cache"
DEFAULT_OUTPUT_ROOT = ROOT / "experiment_outputs"


def settings_for(
    experiments: List[str],
    exp3_steps: Optional[List[int]] = None,
) -> List[Tuple[str, float, int, Optional[int]]]:
    """Expand experiment names into concrete parameter settings."""
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


def resolve_output_dir(output_root: Path, output_dir: Optional[Path], run_id: Optional[str]) -> Path:
    """Return an independent output directory for this run."""
    if output_dir is not None:
        return output_dir
    actual_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / actual_run_id


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["exp1", "exp2", "exp3"],
        default=["exp1", "exp2", "exp3"],
    )
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
        "--exp3-steps",
        nargs="+",
        type=int,
        choices=EXP3_STEPS,
        help="Subset of exp3 steps to run.",
    )
    args = parser.parse_args()

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
    settings = settings_for(args.experiments, exp3_steps=args.exp3_steps)
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
                label = f"{experiment} tau_m={tau_m} tau_c={tau_c}"
                if step is not None:
                    label += f" step={step}"
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
