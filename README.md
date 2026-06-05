# COMMA Experiments

This repository contains the data and cleaned Python code for the COMMA
experiments.

## Code Layout

- `comma_core/logic_engine.py`: neurosymbolic proof, AMR logic, NLI, and similarity engine.
- `comma_core/dataset_builder.py`: evaluation-item construction matching the original experiment.
- `comma_core/experiment_runner.py`: Exp1/Exp2/Exp3 evaluation loop.
- `comma_core/neural_cache.py`: disk caches for NLI and similarity calls.
- `comma_core/model_runtime.py`: offline/local runtime setup for loading the logic engine.
- `comma_core/result_writer.py`: JSON/CSV output writing and comparison tables.
- `comma_core/paper_reference.py`: parameter grids and paper-reported values used for comparison.
- `comma_core/prompting.py`: prompt templates, DeepSeek wrapper, and output parser.
- `comma_core/runtime_utils.py`: shared seeding, cache I/O, and local model-loading helpers.
- `data/`: checked-in CSV files and ArgGraph XML corpus.
- `prompts/reasoning_chain.md`: human-readable prompt template.
- `scripts/generate_data.py`: XML-to-CSV and neutral-pair data generation.
- `run_experiments.py`: command-line entry point for experiment runs.

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

- examples with non-string `Helpful` values are skipped while building the
  evaluation item list;
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

## Prompt Generation

The prompt template used for implicit-premise generation is documented in
`prompts/reasoning_chain.md` and implemented in `comma_core.prompting`.

```python
from comma_core.prompting import build_reasoning_chain_prompt, parse_chain_output

prompt = build_reasoning_chain_prompt(
    premise="...",
    claim="...",
    num_statements="six",
    topic="...",
    label="entailment",
)
```

API keys are not stored in the repository. The optional DeepSeek helper reads
`DEEPSEEK_API_KEY` from the environment.

## Data Generation

The checked-in CSV files are already enough to run the experiments. To rebuild
the ArgGraph-derived CSV from XML files:

```powershell
python scripts/generate_data.py arggraph
```

To regenerate neutral pairs with the NLI model:

```powershell
python scripts/generate_data.py neutral
```
