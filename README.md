# COMMA Reproduction

This repository contains the data and notebook code used to reproduce the
reported COMMA experiments.

## Main Entry Point

Use `reproduce_paper.py` to reproduce the paper figures from the original
`Method.ipynb` runtime. Run it from the repository root with the same Python
environment used by the notebook:

```powershell
$env:PYTHONHASHSEED='1129'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python reproduce_paper.py
```

The script writes:

- `reproduction_outputs/paper_reproduction_results.json`
- `reproduction_outputs/paper_reproduction_summary.csv`
- `reproduction_outputs/paper_reproduction_comparison.csv`
- `reproduction_outputs/paper_reproduction_detail.log`

## Reproduction Details

The script intentionally preserves the notebook behaviours that affect the
reported numbers:

- examples with non-string `Helpful` values are skipped while building
  `sd_sent`;
- `LOGIC` is keyed by `premise + claim`, matching `Method.ipynb`;
- the original notebook `pysat_formula` string parser is used;
- neutral examples are shuffled with `random_state=1129`;
- each class is scanned until up to 140 successful evaluated examples are
  appended.

Disk caches are stored under `.repro_cache/` for AMR logic, NLI predictions,
and sentence-similarity scores. These files are ignored by Git.

To rerun only the Experiment 3 step analysis:

```powershell
python reproduce_paper.py --experiments exp3
```

To run a subset of Experiment 3 steps:

```powershell
python reproduce_paper.py --experiments exp3 --exp3-steps 2 3 4 5
```
