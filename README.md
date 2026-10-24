# DR-BOLT

Offline dual-reference optimization for adaptive vision-language reasoning.

## Install

Use Python 3.11 or 3.12 on Linux with NVIDIA CUDA. The default configuration
uses four GPUs, bf16, and LoRA.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install .
```

## Run

Preview the ScienceQA training and evaluation pipeline:

```sh
dualref run --suite eval_scienceqa_dr_bolt --dry-run
```

Run the pipeline:

```sh
dualref run --suite eval_scienceqa_dr_bolt
```

The pipeline prepares ScienceQA, collects candidates from the Qwen3-VL-8B
Instruct and Thinking references, trains DR-BOLT, and evaluates the adapter.
Models and datasets are downloaded when needed.

Outputs are saved under `outputs/experiments/runs/`. Predictions and metrics
are in its `eval_scienceqa_dr_bolt/` directory.

Use `--config path/to/config.yaml` and repeatable `--set key=value` options
to override the defaults in `configs/`. Run `dualref --help` for commands.
