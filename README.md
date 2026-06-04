# COMMA Experiments

This repository contains the data and cleaned Python code for the COMMA
experiments.

## Code Layout

- `comma_core/method.py`: core logic from the original experiment code.
- `comma_core/data.py`: data construction matching the original experiment.
- `comma_core/cache.py`: disk caches for NLI and similarity calls.
- `comma_core/evaluator.py`: Exp1/Exp2/Exp3 evaluation loop.
- `comma_core/outputs.py`: JSON/CSV output writing and comparison tables.
- `run_experiments.py`: thin command-line entry point.

## Main Entry Point

Use `run_experiments.py` to run the cleaned experiment pipeline. Run it from
the repository root with the same Python environment used for the original
experiments:

```powershell
$env:PYTHONHASHSEED='1129'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python run_experiments.py
```

The script writes:

- `experiment_outputs/<run_id>/experiment_results.json`
- `experiment_outputs/<run_id>/experiment_summary.csv`
- `experiment_outputs/<run_id>/experiment_comparison.csv`
- `experiment_outputs/<run_id>/experiment_detail.log`

By default `<run_id>` is a timestamp. You can set it explicitly:

```powershell
python run_experiments.py --run-id full_run_1129
```

## Experiment Details

The script intentionally preserves the original behaviours that affect the
reported numbers:

- examples with non-string `Helpful` values are skipped while building
  `sd_sent`;
- `LOGIC` is keyed by `premise + claim`, matching the original experiment code;
- the original `pysat_formula` string parser is used;
- neutral examples are shuffled with `random_state=1129`;
- each class is scanned until up to 140 successful evaluated examples are
  appended.

Disk caches are stored under `.experiment_cache/` for AMR logic, NLI predictions,
and sentence-similarity scores. These files are ignored by Git.

To run only the Experiment 3 step analysis:

```powershell
python run_experiments.py --experiments exp3
```

To run a subset of Experiment 3 steps:

```powershell
python run_experiments.py --experiments exp3 --exp3-steps 2 3 4 5
```
