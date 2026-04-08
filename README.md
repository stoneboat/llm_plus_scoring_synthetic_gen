# LLM + Scoring Synthetic Generation

Reproduction of **"Private prediction for large-scale synthetic text generation"**
(Amin, Bie, Kong, Kurakin, Ponomareva, Syed, Terzis, Vassilvitskii — Google, EMNLP 2024 Findings).

Paper: https://arxiv.org/abs/2407.12108

## Overview

This project implements the private prediction algorithm for generating differentially
private synthetic text using LLMs. The core insight is that standard softmax sampling
from an LLM is distributionally equivalent to the exponential mechanism, so once logit
clipping and aggregation bound score sensitivity, token selection can be analyzed
without adding explicit noise to the logits themselves.

Key components:
- **Logit clipping with recentering** — bounds sensitivity for the exponential mechanism
- **Sparse vector technique (SVT)** — skips privacy cost for tokens predictable from public data
- **zCDP privacy accounting** — via Google's `dp-accounting` library
- **Parallel composition** — fixed, hash-based disjoint batches instead of re-sampling

## Setup

### Amazon EC2

```bash
bash scripts/local_scripts/install_amazon.sh
```

This creates a Python venv in `/tmp/python-venv/llm_scoring_venv`, installs
all dependencies, downloads the Gemma 2B model, and registers a Jupyter kernel.

#### HuggingFace authentication (required for Gemma)

Gemma is a gated model. Before running the install script you must:

1. Accept the license at <https://huggingface.co/google/gemma-2-2b-it>
2. Create a token with **Read** access at <https://huggingface.co/settings/tokens>
3. Pass your token to the install script via the `HF_TOKEN` environment variable:

```bash
export HF_TOKEN=hf_YOUR_TOKEN
bash scripts/local_scripts/install_amazon.sh
```

The script will authenticate automatically before downloading the model.

### Running an experiment

```bash
source /tmp/python-venv/llm_scoring_venv/bin/activate
python scripts/run_experiment.py \
  --dataset agnews \
  --epsilon 3.0 \
  --batch_size 255 \
  --clip_bound 10.0 \
  --temperature 2.0 \
  --top_k_vocab 1024 \
  --public_temperature 1.5 \
  --svt_threshold 0.5 \
  --svt_noise 0.2
```

This example is aligned with the hyperparameter settings explored in Amin et al. (2024) for AG News.
It uses the full training split by default and lets the script compute `max_private_tokens` from
`epsilon`, `delta = 1/n`, and the chosen mechanism parameters.

Runs are checkpointed batch-by-batch to the output JSONL as they execute. If a long run is interrupted,
resume it with the same arguments plus `--resume`. By default, the runner will pick the latest matching
checkpoint in `data/outputs`; use `--output_path /path/to/file.jsonl` if you want to resume a specific file.

Resume command for the existing AG News run artifact:

```bash
python scripts/run_experiment.py \
  --dataset agnews \
  --epsilon 3.0 \
  --delta 8.333333333333334e-06 \
  --batch_size 255 \
  --clip_bound 10.0 \
  --temperature 2.0 \
  --public_temperature 1.5 \
  --svt_threshold 0.5 \
  --svt_noise 0.2 \
  --top_k_vocab 1024 \
  --max_private_tokens 177 \
  --max_total_tokens 256 \
  --seed 42 \
  --micro_batch_size 32 \
  --output_path data/outputs/agnews_eps3.0_s255_20260404_194803.jsonl \
  --resume
```

`scripts/sweep_hyperparams.py` is still available as an optional research helper,
but it is not the canonical workflow and it does not replace the checkpoint/resume
path in `scripts/run_experiment.py`.

## Project Structure

```text
src/
  datasets/             Dataset adapters and registry
  prompts/              Prompt templates and builders
  batching/             Fixed hash-based batch assignment
  backends/             Model backend abstraction + HuggingFace implementation
  mechanisms/           PrivatePredictionMechanism and helpers
  privacy/              Bounds, events, analyses, planning, reporting
  runtime/              Generation orchestration and generation stats
  artifacts/            Checkpoint JSONL writing, metadata, resume, simple JSONL I/O
  evaluation/           Downstream evaluation package
  generate.py           Backward-compatible generation facade
  evaluate.py           Backward-compatible evaluation facade
  privacy_accounting.py Backward-compatible privacy facade
scripts/
  run_experiment.py     Canonical generation command with checkpoint/resume
  evaluate_downstream.py
                        Canonical downstream evaluation command
  sweep_hyperparams.py  Optional research helper
  local_scripts/        Machine-specific install scripts
data/                   (gitignored) Models, datasets, outputs
```

## Reference

```bibtex
@inproceedings{amin2024private,
  title={Private prediction for large-scale synthetic text generation},
  author={Amin, Kareem and Bie, Alex and Kong, Weiwei and Kurakin, Alexey
          and Ponomareva, Natalia and Syed, Umar and Terzis, Andreas
          and Vassilvitskii, Sergei},
  booktitle={Findings of EMNLP},
  year={2024}
}
```
