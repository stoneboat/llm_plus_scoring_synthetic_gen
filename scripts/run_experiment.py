#!/usr/bin/env python3
"""
End-to-end experiment runner for private prediction synthetic text generation.

Usage:
    python scripts/run_experiment.py --dataset agnews --epsilon 1.0
    python scripts/run_experiment.py --dataset agnews --epsilon 1.0 --batch_size 127 --svt_threshold 0.5 --svt_noise 0.2
    python scripts/run_experiment.py --dataset agnews --epsilon 1.0 --num_examples 1000 --max_total_tokens 128

Loads a dataset, constructs prompts, runs Algorithm 1, saves synthetic data,
and prints generation statistics.
"""

import argparse
import json
import os
import sys
import time

# Reduce CUDA memory fragmentation, especially with large-vocabulary models
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Add project root to path so src/ is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    PrivacyConfig,
    GenerationConfig,
    ModelConfig,
    DatasetConfig,
    PROMPT_TEMPLATES,
    compute_max_private_tokens,
)
from src.privacy_accounting import privacy_report
from src.generate import generate_synthetic_dataset
from src.evaluate import save_synthetic_data, compute_generation_stats


DATASET_HF_MAP = {
    "agnews": ("fancyzhx/ag_news", "train", "text", "label"),
    "dbpedia": ("fancyzhx/dbpedia_14", "train", "content", "label"),
    "imdb": ("stanfordnlp/imdb", "train", "text", "label"),
    "yelp": ("fancyzhx/yelp_polarity", "train", "text", "label"),
    "trec": ("CogComp/trec", "train", "text", "coarse_label"),
}


def load_dataset_examples(
    dataset_name: str,
    num_examples: int = None,
    cache_dir: str = "data/datasets",
) -> list:
    """Load dataset from HuggingFace and convert to list of dicts."""
    from datasets import load_dataset

    if dataset_name not in DATASET_HF_MAP:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_HF_MAP.keys())}"
        )

    hf_name, split, text_col, label_col = DATASET_HF_MAP[dataset_name]

    print(f"Loading {hf_name} ({split} split)...")
    ds = load_dataset(hf_name, split=split, cache_dir=cache_dir)

    if num_examples is not None and num_examples < len(ds):
        ds = ds.shuffle(seed=42).select(range(num_examples))

    examples = []
    for row in ds:
        examples.append({
            "text": row[text_col],
            "label": row[label_col],
        })

    labels = set(e["label"] for e in examples)
    print(f"  Loaded {len(examples)} examples with {len(labels)} labels")
    return examples


def load_model(model_config: ModelConfig):
    """Load the LLM and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_path = model_config.model_name_or_path
    if not os.path.isdir(model_path):
        model_path = model_config.hf_model_id
        print(f"Local model not found at {model_config.model_name_or_path}, "
              f"using HF hub: {model_path}")

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_config.torch_dtype,
        device_map=model_config.device,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    print(f"  Model loaded on {device}, dtype={next(model.parameters()).dtype}")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Private prediction synthetic text generation"
    )

    parser.add_argument(
        "--dataset", type=str, default="agnews",
        choices=list(DATASET_HF_MAP.keys()),
        help="Dataset to generate synthetic data for",
    )
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Subsample N examples from the dataset (default: all)")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="Target (epsilon, delta)-DP epsilon")
    parser.add_argument("--delta", type=float, default=None,
                        help="Target delta (default: 1/n)")
    parser.add_argument("--batch_size", type=int, default=255,
                        help="Expected batch size s")
    parser.add_argument("--clip_bound", type=float, default=10.0,
                        help="Logit clipping bound c")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Sampling temperature tau for private tokens")
    parser.add_argument("--public_temperature", type=float, default=1.5,
                        help="Sampling temperature for public tokens")
    parser.add_argument("--svt_threshold", type=float, default=float("-inf"),
                        help="SVT threshold theta (-inf to disable)")
    parser.add_argument("--svt_noise", type=float, default=0.2,
                        help="SVT Laplace noise sigma")
    parser.add_argument("--max_private_tokens", type=int, default=None,
                        help="Max private tokens r (default: computed from budget)")
    parser.add_argument("--max_total_tokens", type=int, default=256,
                        help="Absolute max output length")
    parser.add_argument("--model_path", type=str, default="data/models/gemma-2-2b-it",
                        help="Path to local model or HF model ID")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference")
    parser.add_argument("--output_dir", type=str, default="data/outputs",
                        help="Directory for output files")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--micro_batch_size", type=int, default=32,
                        help="Prompts per GPU forward pass (reduce if OOM, default 32)")
    parser.add_argument("--top_k_vocab", type=int, default=0,
                        help="Restrict sampling to top-k of public logits (0=off, "
                             "paper uses 1024 at high temperature)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run BERT fine-tuning evaluation after generation")
    parser.add_argument("--bert_epochs", type=int, default=5,
                        help="BERT fine-tuning epochs (used with --evaluate)")
    parser.add_argument("--max_test", type=int, default=None,
                        help="Cap test examples for eval (default: all)")

    args = parser.parse_args()

    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load data
    examples = load_dataset_examples(
        args.dataset,
        num_examples=args.num_examples,
        cache_dir="data/datasets",
    )

    # Configure privacy
    delta = args.delta if args.delta else 1.0 / len(examples)

    svt_noise = args.svt_noise if args.svt_threshold != float("-inf") else None

    if args.max_private_tokens is not None:
        max_priv = args.max_private_tokens
    else:
        max_priv = compute_max_private_tokens(
            args.epsilon, delta, args.batch_size,
            args.clip_bound, args.temperature, svt_noise,
        )
        print(f"Computed max_private_tokens = {max_priv} for epsilon={args.epsilon}")

    privacy_config = PrivacyConfig(
        target_epsilon=args.epsilon,
        delta=delta,
        clip_bound=args.clip_bound,
        temperature=args.temperature,
        public_temperature=args.public_temperature,
        svt_threshold=args.svt_threshold,
        svt_noise=args.svt_noise,
    )

    gen_config = GenerationConfig(
        batch_size=args.batch_size,
        max_private_tokens=max_priv,
        max_total_tokens=args.max_total_tokens,
        top_k_vocab=args.top_k_vocab,
    )

    # Print privacy report
    report = privacy_report(
        max_priv, args.clip_bound, args.batch_size,
        args.temperature, delta, svt_noise,
    )
    print(f"\n{'='*60}")
    print(f"Privacy configuration:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    # Load model
    model_config = ModelConfig(
        model_name_or_path=args.model_path,
        device=args.device,
    )
    model, tokenizer = load_model(model_config)

    # Generate
    print(f"\nStarting generation...")
    t0 = time.time()

    synthetic_data = generate_synthetic_dataset(
        model, tokenizer, examples,
        dataset_name=args.dataset,
        privacy_config=privacy_config,
        gen_config=gen_config,
        text_column="text",
        label_column="label",
        device=args.device,
        verbose=True,
        micro_batch_size=args.micro_batch_size,
    )

    elapsed = time.time() - t0
    print(f"\nGeneration took {elapsed:.1f}s")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        args.output_dir,
        f"{args.dataset}_eps{args.epsilon}_s{args.batch_size}_{timestamp}.jsonl",
    )
    metadata = {
        "dataset": args.dataset,
        "epsilon": report["epsilon"],
        "delta": delta,
        "batch_size": args.batch_size,
        "clip_bound": args.clip_bound,
        "temperature": args.temperature,
        "svt_threshold": args.svt_threshold,
        "svt_noise": args.svt_noise,
        "max_private_tokens": max_priv,
        "num_source_examples": len(examples),
        "generation_time_seconds": elapsed,
        "seed": args.seed,
    }
    save_synthetic_data(synthetic_data, output_path, metadata=metadata)
    print(f"Saved {len(synthetic_data)} examples to {output_path}")

    # Stats
    stats = compute_generation_stats(synthetic_data)
    print(f"\nGeneration statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Optional downstream evaluation
    if args.evaluate:
        from src.evaluate import finetune_and_evaluate

        DATASET_NUM_LABELS = {
            "agnews": 4, "dbpedia": 14, "imdb": 2, "yelp": 2, "trec": 6,
        }
        DATASET_TEST_SPLIT = {
            "agnews": ("fancyzhx/ag_news", "test", "text", "label"),
            "dbpedia": ("fancyzhx/dbpedia_14", "test", "content", "label"),
            "imdb": ("stanfordnlp/imdb", "test", "text", "label"),
            "yelp": ("fancyzhx/yelp_polarity", "test", "text", "label"),
            "trec": ("CogComp/trec", "test", "text", "coarse_label"),
        }

        hf_name, split, text_col, label_col = DATASET_TEST_SPLIT[args.dataset]
        from datasets import load_dataset
        print(f"\nLoading test set: {hf_name} ({split})...")
        test_ds = load_dataset(hf_name, split=split, cache_dir="data/datasets")
        if args.max_test and args.max_test < len(test_ds):
            test_ds = test_ds.shuffle(seed=42).select(range(args.max_test))
        test_texts = [row[text_col] for row in test_ds]
        test_labels = [row[label_col] for row in test_ds]
        print(f"  {len(test_texts)} test examples")

        synth_texts = [e.text for e in synthetic_data]
        synth_labels = [e.label for e in synthetic_data]

        eval_results = finetune_and_evaluate(
            synth_texts, synth_labels,
            test_texts, test_labels,
            num_labels=DATASET_NUM_LABELS[args.dataset],
            epochs=args.bert_epochs,
            device=args.device,
        )
        print(f"\nDownstream accuracy: {eval_results['accuracy']:.4f}")
        print(f"Macro F1: {eval_results['macro_f1']:.4f}")

        eval_path = output_path.replace(".jsonl", "_eval.json")
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"Eval results saved to {eval_path}")


if __name__ == "__main__":
    main()
