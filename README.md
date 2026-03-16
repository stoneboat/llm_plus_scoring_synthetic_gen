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
python scripts/run_experiment.py --dataset agnews --epsilon 1.0
```

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
