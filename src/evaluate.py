"""
Evaluation utilities for synthetic data quality.

Supports two evaluation modes from the paper:
1. In-context learning (ICL): use synthetic examples as few-shot exemplars
2. Fine-tuning: train a classifier on synthetic data, evaluate on real test data

Reference: Sections 5.1 and 5.2 of Amin et al. (2024).
"""

import json
import os
from typing import List, Dict, Optional, Tuple

import torch
from sklearn.metrics import accuracy_score, classification_report

from src.generate import SyntheticExample
from src.config import PROMPT_TEMPLATES


def format_icl_prompt(
    synthetic_examples: List[SyntheticExample],
    test_text: str,
    dataset_name: str,
    num_shots: int = 4,
) -> str:
    """Format synthetic examples as an in-context learning prompt.

    Follows the evaluation format from Figure 2 in the paper.

    Args:
        synthetic_examples: generated synthetic data.
        test_text: the real test example to classify.
        dataset_name: name of the dataset for label formatting.
        num_shots: number of synthetic examples to include.

    Returns:
        Formatted ICL prompt string.
    """
    templates = PROMPT_TEMPLATES[dataset_name]

    lines = ["Classify the following examples:"]
    for ex in synthetic_examples[:num_shots]:
        lines.append(f"Text: {ex.text}")
        lines.append(f"Answer: {ex.label_name}")

    lines.append("")
    lines.append(f"Text: {test_text}")
    lines.append("Answer:")

    return "\n".join(lines)


def save_synthetic_data(
    synthetic_examples: List[SyntheticExample],
    output_path: str,
    metadata: Optional[dict] = None,
) -> None:
    """Save synthetic examples to a JSONL file.

    Args:
        synthetic_examples: list of SyntheticExample.
        output_path: path to write the file.
        metadata: optional dict of experiment metadata to include in header.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        if metadata:
            f.write(json.dumps({"_metadata": metadata}) + "\n")
        for ex in synthetic_examples:
            record = {
                "text": ex.text,
                "label": ex.label,
                "label_name": ex.label_name,
                "num_private_tokens": ex.num_private_tokens,
                "num_public_tokens": ex.num_public_tokens,
                "num_total_tokens": ex.num_total_tokens,
            }
            f.write(json.dumps(record) + "\n")


def load_synthetic_data(input_path: str) -> Tuple[List[SyntheticExample], Optional[dict]]:
    """Load synthetic examples from a JSONL file.

    Returns:
        (list of SyntheticExample, metadata dict or None)
    """
    examples = []
    metadata = None

    with open(input_path) as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Allow interrupted checkpoint files with a truncated tail line.
                print(f"Warning: ignoring unreadable JSONL tail at line {line_no} in {input_path}")
                break
            if "_metadata" in record:
                metadata = record["_metadata"]
                continue
            if "_batch_complete" in record:
                continue
            examples.append(SyntheticExample(
                text=record["text"],
                label=record["label"],
                label_name=record["label_name"],
                num_private_tokens=record["num_private_tokens"],
                num_public_tokens=record["num_public_tokens"],
                num_total_tokens=record["num_total_tokens"],
            ))

    return examples, metadata


def compute_generation_stats(examples: List[SyntheticExample]) -> dict:
    """Compute summary statistics over generated examples."""
    if not examples:
        return {}

    total_tokens = [e.num_total_tokens for e in examples]
    priv_tokens = [e.num_private_tokens for e in examples]
    pub_tokens = [e.num_public_tokens for e in examples]

    total_priv = sum(priv_tokens)
    total_pub = sum(pub_tokens)
    total_all = total_priv + total_pub

    label_counts: Dict[int, int] = {}
    for e in examples:
        label_counts[e.label] = label_counts.get(e.label, 0) + 1

    return {
        "num_examples": len(examples),
        "label_distribution": label_counts,
        "total_tokens": sum(total_tokens),
        "mean_tokens_per_example": sum(total_tokens) / len(examples),
        "total_private_tokens": total_priv,
        "total_public_tokens": total_pub,
        "public_token_fraction": total_pub / max(1, total_all),
        "max_private_tokens_in_example": max(priv_tokens),
    }


# ── BERT fine-tuning evaluation ─────────────────────────────────────────────

def finetune_and_evaluate(
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
    verbose: bool = True,
) -> Dict:
    """Fine-tune BERT on training data and evaluate on a test set.

    This is the paper's primary evaluation metric (Section 5.2):
    train a BERT-base classifier on synthetic examples and measure
    test-set accuracy against the real AG News / IMDB / etc. labels.

    Returns dict with accuracy, macro_f1, weighted_f1, and per_class report.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import DataLoader, TensorDataset

    if verbose:
        print(f"Fine-tuning {bert_model}  epochs={epochs} lr={lr} bs={batch_size}")

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

    train_loader = DataLoader(encode(train_texts, train_labels),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(encode(test_texts, test_labels),
                             batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            ids, mask, labs = [t.to(device) for t in batch]
            loss = model(input_ids=ids, attention_mask=mask, labels=labs).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        if verbose:
            print(f"  epoch {epoch+1}/{epochs}  loss={total_loss/len(train_loader):.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids, mask, labs = [t.to(device) for t in batch]
            preds = model(input_ids=ids, attention_mask=mask).logits.argmax(-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labs.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4, output_dict=True)

    if verbose:
        print(f"  accuracy: {acc:.4f}")
        print(classification_report(all_labels, all_preds, digits=4))

    return {
        "accuracy": acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {
            str(k): v for k, v in report.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
    }
