#!/usr/bin/env python3
"""
Hyperparameter sweep for private synthetic text generation.

Runs Algorithm 1 with several (temperature, SVT, top_k) configurations,
evaluates each with BERT fine-tuning on the test set, and writes a
comparison table.

The sweep grid is derived from Table 7 of Amin et al. (2024):
  - temperature: [1.5, 2.0]
  - SVT settings: [off, (0.3,0.1), (0.5,0.2)]
  - top_k_vocab: [512, 1024]

Usage:
    python scripts/sweep_hyperparams.py --dataset agnews --epsilon 3.0

    # Quick smoke test (100 source examples, 100 test examples)
    python scripts/sweep_hyperparams.py --dataset agnews --epsilon 3.0 \\
        --num_examples 100 --max_test 100

    # Use full dataset
    python scripts/sweep_hyperparams.py --dataset agnews --epsilon 3.0 \\
        --num_examples 120000

This script is an optional research helper. The canonical generation workflow
remains ``scripts/run_experiment.py`` with checkpoint/resume support.
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    PrivacyConfig, GenerationConfig,
)
from src.privacy import privacy_report, compute_max_private_tokens
from src.generate import generate_synthetic_dataset
from src.artifacts import save_synthetic_data
from src.datasets.registry import DATASET_CHOICES, get_adapter
from src.runtime import compute_generation_stats
from src.evaluation import finetune_bert

# Grid: each entry is (temperature, svt_threshold, svt_noise, top_k_vocab)
DEFAULT_GRID = [
    (1.5, float("-inf"), None, 0),
    (1.5, 0.3, 0.1, 0),
    (1.5, 0.5, 0.2, 0),
    (2.0, float("-inf"), None, 1024),
    (2.0, 0.3, 0.1, 1024),
    (2.0, 0.5, 0.2, 1024),
    (2.0, 0.5, 0.2, 512),
]


def load_data(dataset_name, split, num_examples, cache_dir="data/datasets"):
    adapter = get_adapter(dataset_name)
    return adapter.load(split, num_examples=num_examples, cache_dir=cache_dir)


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep")
    parser.add_argument("--dataset", default="agnews",
                        choices=DATASET_CHOICES)
    parser.add_argument("--epsilon", type=float, default=3.0)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=255)
    parser.add_argument("--clip_bound", type=float, default=10.0)
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Source examples (default: all)")
    parser.add_argument("--max_total_tokens", type=int, default=256)
    parser.add_argument("--model_path", default="data/models/gemma-2-2b-it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--micro_batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="data/sweep_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_test", type=int, default=None,
                        help="Cap test examples for faster eval (default: all)")
    parser.add_argument("--bert_epochs", type=int, default=5)
    parser.add_argument("--skip_eval", action="store_true",
                        help="Only generate, skip BERT eval")
    args = parser.parse_args()

    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    num_labels = get_adapter(args.dataset).num_labels

    # Load source data
    print("Loading source data...")
    train_examples = load_data(args.dataset, "train", args.num_examples)
    delta = args.delta or 1.0 / len(train_examples)
    print(f"  {len(train_examples)} source examples, delta={delta:.2e}")

    # Load test data
    test_texts, test_labels = None, None
    if not args.skip_eval:
        print("Loading test data...")
        test_rows = load_data(args.dataset, "test", args.max_test)
        test_texts = [r["text"] for r in test_rows]
        test_labels = [r["label"] for r in test_rows]
        print(f"  {len(test_texts)} test examples")

    # Load model once
    print("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_path = args.model_path
    if not os.path.isdir(model_path):
        model_path = "google/gemma-2-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map=args.device,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for idx, (tau, svt_thresh, svt_noise, top_k) in enumerate(DEFAULT_GRID):
        label = (f"tau={tau}_svt={svt_thresh}_noise={svt_noise}_topk={top_k}")
        print(f"\n{'='*70}")
        print(f"Config {idx+1}/{len(DEFAULT_GRID)}: {label}")
        print(f"{'='*70}")

        svt_noise_for_budget = svt_noise if svt_thresh != float("-inf") else None
        max_priv = compute_max_private_tokens(
            args.epsilon, delta, args.clip_bound, args.batch_size,
            tau, svt_noise_for_budget,
        )
        print(f"  max_private_tokens = {max_priv}")

        report = privacy_report(
            max_priv, args.clip_bound, args.batch_size,
            tau, delta, svt_noise_for_budget,
        )
        print(f"  epsilon = {report['epsilon']:.4f}")

        priv_cfg = PrivacyConfig(
            target_epsilon=args.epsilon,
            delta=delta,
            clip_bound=args.clip_bound,
            temperature=tau,
            public_temperature=1.5,
            svt_threshold=svt_thresh,
            svt_noise=svt_noise if svt_noise is not None else 0.2,
        )
        gen_cfg = GenerationConfig(
            batch_size=args.batch_size,
            max_private_tokens=max_priv,
            max_total_tokens=args.max_total_tokens,
            top_k_vocab=top_k,
        )

        torch.manual_seed(args.seed)
        t0 = time.time()

        synthetic_data = generate_synthetic_dataset(
            model, tokenizer, train_examples,
            dataset_name=args.dataset,
            privacy_config=priv_cfg,
            gen_config=gen_cfg,
            text_column="text",
            label_column="label",
            device=args.device,
            verbose=True,
            micro_batch_size=args.micro_batch_size,
        )

        gen_time = time.time() - t0
        gen_stats = compute_generation_stats(synthetic_data)

        out_path = os.path.join(args.output_dir, f"{args.dataset}_{label}.jsonl")
        save_synthetic_data(synthetic_data, out_path, metadata={
            "config": label,
            "epsilon": report["epsilon"],
            "delta": delta,
            **gen_stats,
        })

        print(f"\n  Generated {len(synthetic_data)} examples in {gen_time:.1f}s")
        print(f"  Saved to {out_path}")

        # Show sample texts
        for i, ex in enumerate(synthetic_data[:3]):
            snippet = ex.text[:120].replace("\n", " ")
            print(f"  Sample [{ex.label_name}]: {snippet}...")

        entry = {
            "config": label,
            "temperature": tau,
            "svt_threshold": svt_thresh,
            "svt_noise": svt_noise,
            "top_k_vocab": top_k,
            "max_private_tokens": max_priv,
            "epsilon": report["epsilon"],
            "num_synthetic": len(synthetic_data),
            "gen_time_s": gen_time,
        }
        entry.update(gen_stats)

        if not args.skip_eval and test_texts:
            print("\n  Running BERT evaluation...")
            synth_texts = [e.text for e in synthetic_data]
            synth_labels = [e.label for e in synthetic_data]
            eval_results = finetune_bert(
                synth_texts, synth_labels,
                test_texts, test_labels,
                num_labels=num_labels,
                epochs=args.bert_epochs,
                device=args.device,
            )
            entry["accuracy"] = eval_results["accuracy"]
            entry["macro_f1"] = eval_results["macro_f1"]
            print(f"  accuracy={eval_results['accuracy']:.4f}  "
                  f"macro_f1={eval_results['macro_f1']:.4f}")

        results.append(entry)

    # Summary table
    print(f"\n\n{'='*90}")
    print("SWEEP RESULTS SUMMARY")
    print(f"{'='*90}")
    header = f"{'Config':<50} {'eps':>6} {'#syn':>5} {'acc':>7} {'F1':>7} {'time':>7}"
    print(header)
    print("-" * 90)
    for r in results:
        acc_str = f"{r.get('accuracy', 0):.4f}" if "accuracy" in r else "  N/A"
        f1_str = f"{r.get('macro_f1', 0):.4f}" if "macro_f1" in r else "  N/A"
        print(f"{r['config']:<50} {r['epsilon']:>6.2f} "
              f"{r['num_synthetic']:>5} {acc_str:>7} {f1_str:>7} "
              f"{r['gen_time_s']:>6.0f}s")

    summary_path = os.path.join(args.output_dir, f"{args.dataset}_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {summary_path}")


if __name__ == "__main__":
    main()
