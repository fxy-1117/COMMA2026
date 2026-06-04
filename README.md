# COMMA Experiments

This repository contains the data, notebooks, and a cleaned Python runner for
the COMMA experiments.

## Main Entry Point

Use `run_experiments.py` to run the experiment pipeline from the original
`Method.ipynb` runtime. Run it from the repository root with the same Python
environment used by the notebook:

```powershell
$env:PYTHONHASHSEED='1129'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python run_experiments.py
```

The script writes:

- `experiment_outputs/experiment_results.json`
- `experiment_outputs/experiment_summary.csv`
- `experiment_outputs/experiment_comparison.csv`
- `experiment_outputs/experiment_detail.log`

## Notebook Details

The script intentionally preserves the notebook behaviours that affect the
reported numbers:

- examples with non-string `Helpful` values are skipped while building
  `sd_sent`;
- `LOGIC` is keyed by `premise + claim`, matching `Method.ipynb`;
- the original notebook `pysat_formula` string parser is used;
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
