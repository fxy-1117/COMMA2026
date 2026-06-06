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
- `prompts/single_implicit_premise.md`: prompt used for the single implicit premise in Experiment 2.
- `prompts/reasoning_chain.md`: prompt used for the reasoning chain in Experiment 3.
- `scripts/generate_data.py`: XML-to-CSV and neutral-pair data generation.
- `run_comma_experiments.py`: command-line entry point for experiment runs.

## Main Entry Point

Use `run_comma_experiments.py` to run the cleaned experiment pipeline. The
script reads the `RUN_EXPERIMENTS` list near the top of the file; comment or
uncomment `"exp1"`, `"exp2"`, or `"exp3"` there to choose which experiment
groups to run.

Run it from the repository root with the same Python environment used for the
original experiments:

```powershell
$env:PYTHONHASHSEED='1129'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python run_comma_experiments.py
```

To check the currently enabled task list without loading the neural models:

```powershell
python run_comma_experiments.py --list-tasks
```

The script writes:

- `experiment_outputs/<run_id>/experiment_results.json`
- `experiment_outputs/<run_id>/experiment_summary.csv`
- `experiment_outputs/<run_id>/experiment_comparison.csv`
- `experiment_outputs/<run_id>/experiment_detail.log`

By default `<run_id>` is a timestamp. You can set it explicitly:

```powershell
python run_comma_experiments.py --run-id full_run_1129
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

For example, to run only Experiment 3, edit `run_comma_experiments.py` like this:

```python
RUN_EXPERIMENTS = [
    # "exp1",
    # "exp2",
    "exp3",
]
```

Disk caches are stored under `.experiment_cache/` for AMR logic, NLI predictions,
and sentence-similarity scores. These files are ignored by Git.

## Prompt Generation

The prompt templates used for implicit-premise generation are documented in
`prompts/single_implicit_premise.md` and `prompts/reasoning_chain.md`, and are
implemented in `comma_core.prompting`.

```python
from comma_core.prompting import (
    build_reasoning_chain_prompt,
    build_single_implicit_prompt,
    parse_chain_output,
)

single_prompt = build_single_implicit_prompt(
    premise="...",
    claim="...",
    topic="...",
    label="entailment",
)

chain_prompt = build_reasoning_chain_prompt(
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
