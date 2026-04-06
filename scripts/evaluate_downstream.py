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
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.evaluate import load_synthetic_data
from src.prompts import PROMPT_TEMPLATES
from src.datasets.registry import DATASET_CHOICES, get_adapter


def load_test_set(dataset_name: str, cache_dir: str = "data/datasets",
                  max_examples: Optional[int] = None) -> Tuple[List[str], List[int]]:
    """Load the real test set from HuggingFace."""
    adapter = get_adapter(dataset_name)
    print(f"Loading test set: {adapter.hf_name} (test)...")
    examples = adapter.load("test", num_examples=max_examples, cache_dir=cache_dir)
    texts = [e["text"] for e in examples]
    labels = [e["label"] for e in examples]
    print(f"  {len(texts)} test examples, {len(set(labels))} classes")
    return texts, labels


def load_real_train(dataset_name: str, cache_dir: str = "data/datasets",
                    max_examples: Optional[int] = None) -> Tuple[List[str], List[int]]:
    """Load the real training set (for baseline comparison)."""
    adapter = get_adapter(dataset_name)
    print(f"Loading real training set: {adapter.hf_name} (train)...")
    examples = adapter.load("train", num_examples=max_examples, cache_dir=cache_dir)
    texts = [e["text"] for e in examples]
    labels = [e["label"] for e in examples]
    print(f"  {len(texts)} training examples")
    return texts, labels


# ── BERT Fine-tuning ────────────────────────────────────────────────────────

def finetune_bert(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    num_labels: int,
    bert_model: str = "bert-base-uncased",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_length: int = 128,
    device: str = "cuda",
) -> Dict[str, float]:
    """Fine-tune BERT-base on training texts and evaluate on test set."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import accuracy_score, classification_report

    print(f"\nFine-tuning {bert_model} ({epochs} epochs, lr={lr}, bs={batch_size})...")
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        bert_model, num_labels=num_labels,
    ).to(device)

    def encode(texts, labels):
        enc = tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        return TensorDataset(
            enc["input_ids"], enc["attention_mask"],
            torch.tensor(labels, dtype=torch.long),
        )

    train_ds = encode(train_texts, train_labels)
    test_ds = encode(test_texts, test_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            ids, mask, labs = [t.to(device) for t in batch]
            out = model(input_ids=ids, attention_mask=mask, labels=labs)
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids, mask, labs = [t.to(device) for t in batch]
            logits = model(input_ids=ids, attention_mask=mask).logits
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labs.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, digits=4, output_dict=True,
    )
    print(f"\n  Test accuracy: {accuracy:.4f}")
    print(classification_report(all_labels, all_preds, digits=4))

    return {
        "accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {
            str(k): v for k, v in report.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
    }


# ── ICL Evaluation ──────────────────────────────────────────────────────────

def icl_evaluate(
    synthetic_texts: List[str],
    synthetic_labels: List[int],
    synthetic_label_names: List[str],
    test_texts: List[str],
    test_labels: List[int],
    dataset_name: str,
    model_path: str,
    num_shots: int = 4,
    max_test: int = 500,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluate using in-context learning with a local causal LM.

    Presents num_shots synthetic examples as few-shot demonstrations,
    then asks the model to classify each test example.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.metrics import accuracy_score, classification_report

    templates = PROMPT_TEMPLATES[dataset_name]
    label_names = templates["labels"]
    name_to_id = {v.lower(): k for k, v in label_names.items()}

    print(f"\nICL evaluation: {num_shots}-shot, model={model_path}")
    print(f"  Evaluating on {min(max_test, len(test_texts))} test examples...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map=device,
    )
    model.eval()

    shots_by_label: Dict[int, List[str]] = {}
    for text, label in zip(synthetic_texts, synthetic_labels):
        shots_by_label.setdefault(label, []).append(text)

    def build_icl_prompt(test_text: str) -> str:
        lines = ["Classify the following text into one of these categories: "
                 + ", ".join(label_names.values()) + ".\n"]
        shots_used = 0
        for lab in sorted(shots_by_label.keys()):
            available = shots_by_label[lab]
            per_label = max(1, num_shots // len(label_names))
            for t in available[:per_label]:
                lines.append(f"Text: {t}")
                lines.append(f"Category: {label_names[lab]}\n")
                shots_used += 1
                if shots_used >= num_shots:
                    break
            if shots_used >= num_shots:
                break
        lines.append(f"Text: {test_text}")
        lines.append("Category:")
        return "\n".join(lines)

    all_preds, all_true = [], []
    correct = 0
    total = min(max_test, len(test_texts))

    for i in range(total):
        prompt = build_icl_prompt(test_texts[i])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=10, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()
        first_word = gen.split("\n")[0].strip().rstrip(".,;:!?").lower()

        pred_label = -1
        for name, lid in name_to_id.items():
            if name in first_word or first_word in name:
                pred_label = lid
                break

        all_preds.append(pred_label)
        all_true.append(test_labels[i])
        if pred_label == test_labels[i]:
            correct += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total}] running accuracy: {correct/(i+1):.4f}")

    accuracy = accuracy_score(all_true, all_preds)
    report = classification_report(
        all_true, all_preds, digits=4, output_dict=True, zero_division=0,
    )
    print(f"\n  ICL accuracy: {accuracy:.4f}")
    print(classification_report(all_true, all_preds, digits=4, zero_division=0))

    return {
        "accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "num_shots": num_shots,
        "num_test": total,
    }


# ── Main ────────────────────────────────────────────────────────────────────

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
        examples, metadata = load_synthetic_data(args.synthetic_path)
        if not examples:
            print("ERROR: No synthetic examples found in", args.synthetic_path)
            sys.exit(1)
        train_texts = [e.text for e in examples]
        train_labels = [e.label for e in examples]
        train_label_names = [e.label_name for e in examples]
        source = args.synthetic_path
        print(f"Loaded {len(examples)} synthetic examples from {args.synthetic_path}")
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
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_path}")

    return results


if __name__ == "__main__":
    main()
