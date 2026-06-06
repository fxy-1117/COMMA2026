# Implicit Premises for Argument Graph Reconstruction

This repository contains the data and cleaned Python implementation for the
experiments in "Identifying Implicit Premises for Logical Reconstruction of
Argument Graphs." The pipeline generates implicit premises for premise-claim
pairs from argument graphs, translates the resulting statements into logical
formulae, and evaluates whether the reconstructed relation is entailment,
contradiction, or neutral.

## Code Layout

- `comma_core/logic_engine.py`: neuro-symbolic proof, AMR logic, entailment classification, and similarity engine.
- `comma_core/dataset_builder.py`: evaluation-item construction matching the original experiment.
- `comma_core/experiment_config.py`: experiment labels and parameter grids.
- `comma_core/experiment_runner.py`: Exp1/Exp2/Exp3 evaluation loop.
- `comma_core/neural_cache.py`: disk caches for entailment-label and similarity calls.
- `comma_core/model_runtime.py`: offline/local runtime setup for loading the logic engine.
- `comma_core/result_writer.py`: JSON/CSV output writing.
- `comma_core/prompting.py`: prompt templates, DeepSeek wrapper, and output parser.
- `comma_core/runtime_utils.py`: shared seeding, cache I/O, and local model-loading helpers.
- `data/arggraph_xml/`: original ArgGraph XML files.
- `data/arggraph_relations.csv`: pairwise support, rebuttal, and undercut relations extracted from ArgGraph XML.
- `data/premise_claim_relations_exp1_exp2.csv`: premise-claim examples for Experiment 1 and Experiment 2.
- `data/premise_claim_chains_exp3.csv`: premise-claim examples with multi-step implicit-premise chains for Experiment 3.
- `data/neutral_pairs.csv`: neutral premise-claim examples appended during evaluation.
- `prompts/single_implicit_premise.md`: prompt used for the single implicit premise in Experiment 2.
- `prompts/reasoning_chain.md`: prompt used for the reasoning chain in Experiment 3.
- `results/exp1_exp2_accuracy.csv`: accuracy values shown in the Exp1/Exp2 figure.
- `results/exp3_step_metrics.csv`: precision, recall, and accuracy values shown in the Exp3 figure.
- `scripts/generate_data.py`: XML-to-CSV and neutral-pair data generation.
- `run_comma_experiments.py`: command-line entry point for experiment runs.

## Main Entry Point

Use `run_comma_experiments.py` to run the cleaned experiment pipeline. The
script reads the `RUN_EXPERIMENTS` list near the top of the file; comment or
uncomment `"exp1"`, `"exp2"`, or `"exp3"` there to choose which experiment
groups to run.

Run it from the repository root with the Python environment used for the
neuro-symbolic argument-graph experiments:

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
    "exp3",
]
```

Disk caches are stored under `.experiment_cache/` for AMR logic, entailment-label
predictions, and sentence-similarity scores. These files are ignored by Git.

## Reference Results

The checked-in files under `results/` contain only the metrics plotted in the
paper figures. They are derived from a full verified run with seed `1129`.
Runtime cache files, detailed logs, per-example outputs, and support counts are
intentionally not included.

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
`data/arggraph_relations.csv` from XML files:

```powershell
python scripts/generate_data.py arggraph
```

To regenerate `data/neutral_pairs.csv` with the entailment classifier:

```powershell
python scripts/generate_data.py neutral
```
