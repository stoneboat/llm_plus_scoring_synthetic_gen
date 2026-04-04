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
`epsilon`, `delta = 1/n`, and the chosen mechanism parameters. If you want to reproduce the paper's
selection procedure more closely, run `scripts/sweep_hyperparams.py` and choose the best AG News
configuration on a held-out validation split.

Runs are checkpointed batch-by-batch to the output JSONL as they execute. If a long run is interrupted,
resume it with the same arguments plus `--resume`. By default, the runner will pick the latest matching
checkpoint in `data/outputs`; use `--output_path /path/to/file.jsonl` if you want to resume a specific file.

## Project Structure

```
src/                    Source code
  config.py             Hyperparameters and privacy budget configuration
  clip_utils.py         Logit clipping with recentering (Eq. 1)
  sparse_vector.py      Sparse vector technique for public/private switching
  privacy_accounting.py Thin wrapper around dp-accounting for Theorem 1
  generate.py           Core Algorithm 1 — private synthetic text generation
  evaluate.py           Downstream evaluation (ICL, fine-tuning)
scripts/
  run_experiment.py     End-to-end experiment runner
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
