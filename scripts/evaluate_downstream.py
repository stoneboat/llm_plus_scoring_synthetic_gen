#!/usr/bin/env python3
"""
Downstream evaluation of private synthetic text.

Implements the two evaluation modes from Sections 5.1–5.2 of Amin et al. (2024):

  1. **BERT fine-tuning** (default): Train BERT-base on synthetic data, evaluate
     on the real test set.
  2. **ICL (in-context learning)**: Use synthetic examples as few-shot exemplars
     with a local causal LM and evaluate on the real test set.

Usage:
    # Fine-tune BERT on synthetic data and evaluate
    python scripts/evaluate_downstream.py \\
        --synthetic_path data/outputs/agnews_eps3.0_s255_*.jsonl \\
        --dataset agnews --mode finetune

    # ICL evaluation with local model
    python scripts/evaluate_downstream.py \\
        --synthetic_path data/outputs/agnews_eps3.0_s255_*.jsonl \\
        --dataset agnews --mode icl --model_path data/models/gemma-2-2b-it

    # Baseline: fine-tune BERT on real training data (upper bound)
    python scripts/evaluate_downstream.py \\
        --dataset agnews --mode finetune --use_real_train

Phase 5: evaluation logic (BERT fine-tuning, ICL, data loading, result saving)
delegated to src/evaluation/; this script is now a thin argparse wrapper.
"""

import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.datasets.registry import DATASET_CHOICES, get_adapter
from src.evaluation import (
    load_test_set,
    load_real_train,
    load_synthetic_for_eval,
    finetune_bert,
    icl_evaluate,
    save_eval_results,
)


def main():
    parser = argparse.ArgumentParser(
        description="Downstream evaluation of private synthetic text"
    )
    parser.add_argument("--synthetic_path", type=str, default=None,
                        help="Path to synthetic JSONL file")
    parser.add_argument("--dataset", type=str, default="agnews",
                        choices=DATASET_CHOICES)
    parser.add_argument("--mode", type=str, default="finetune",
                        choices=["finetune", "icl"],
                        help="Evaluation mode: finetune (BERT) or icl")
    parser.add_argument("--use_real_train", action="store_true",
                        help="Use real training data instead of synthetic (baseline)")
    parser.add_argument("--max_real_train", type=int, default=None,
                        help="Cap on real training examples for baseline")
    parser.add_argument("--model_path", type=str, default="data/models/gemma-2-2b-it",
                        help="Model path for ICL mode")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=5,
                        help="BERT fine-tuning epochs")
    parser.add_argument("--bert_batch_size", type=int, default=32)
    parser.add_argument("--bert_lr", type=float, default=2e-5)
    parser.add_argument("--bert_max_length", type=int, default=128)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument("--num_shots", type=int, default=4,
                        help="Number of ICL shots")
    parser.add_argument("--max_test", type=int, default=None,
                        help="Cap on test examples (default: all)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Save evaluation results as JSON")

    args = parser.parse_args()

    if not args.synthetic_path and not args.use_real_train:
        parser.error("Provide --synthetic_path or --use_real_train")

    num_labels = get_adapter(args.dataset).num_labels

    # Load test set
    test_texts, test_labels = load_test_set(
        args.dataset, max_examples=args.max_test,
    )

    # Load training data
    if args.use_real_train:
        train_texts, train_labels = load_real_train(
            args.dataset, max_examples=args.max_real_train,
        )
        train_label_names = None
        source = "real_train"
    else:
        train_texts, train_labels, train_label_names, metadata = load_synthetic_for_eval(
            args.synthetic_path
        )
        if not train_texts:
            print("ERROR: No synthetic examples found in", args.synthetic_path)
            sys.exit(1)
        source = args.synthetic_path
        print(f"Loaded {len(train_texts)} synthetic examples from {args.synthetic_path}")
        if metadata:
            print(f"  Metadata: eps={metadata.get('epsilon', '?')}, "
                  f"batch_size={metadata.get('batch_size', '?')}, "
                  f"tau={metadata.get('temperature', '?')}")

    print(f"\nTraining on {len(train_texts)} examples ({source})")
    print(f"Testing on {len(test_texts)} examples")

    # Run evaluation
    t0 = time.time()

    if args.mode == "finetune":
        results = finetune_bert(
            train_texts, train_labels,
            test_texts, test_labels,
            num_labels=num_labels,
            bert_model=args.bert_model,
            epochs=args.epochs,
            batch_size=args.bert_batch_size,
            lr=args.bert_lr,
            max_length=args.bert_max_length,
            device=args.device,
        )
    elif args.mode == "icl":
        results = icl_evaluate(
            train_texts, train_labels, train_label_names,
            test_texts, test_labels,
            dataset_name=args.dataset,
            model_path=args.model_path,
            num_shots=args.num_shots,
            max_test=args.max_test or len(test_texts),
            device=args.device,
        )

    elapsed = time.time() - t0
    results["eval_time_seconds"] = elapsed
    results["mode"] = args.mode
    results["source"] = source
    results["dataset"] = args.dataset
    results["num_train"] = len(train_texts)
    results["num_test"] = len(test_texts)

    print(f"\nEvaluation completed in {elapsed:.1f}s")
    print(f"  Accuracy: {results['accuracy']:.4f}")

    if args.output_path:
        save_eval_results(results, args.output_path)

    return results


if __name__ == "__main__":
    main()
