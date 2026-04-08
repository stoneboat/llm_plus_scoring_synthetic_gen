"""
BERT fine-tuning evaluator.

Phase 5: consolidated from two near-identical implementations:
  - src/evaluate.py::finetune_and_evaluate
  - scripts/evaluate_downstream.py::finetune_bert

Both implementations had the same training loop, same return schema, and the
same hyper-parameter defaults.  The only differences were formatting of print
output and a ``verbose`` parameter present in the src/evaluate.py version but
absent in the script version (where prints were unconditional).

The consolidated ``finetune_bert`` function:
  - adds ``verbose=True`` to gate all print output,
  - uses the script's clearer print format (``Epoch N/M  loss=X.XXXX``,
    ``Test accuracy: X.XXXX``),
  - preserves the exact return dict schema:
    ``{accuracy, macro_f1, weighted_f1, per_class}``.

``src/evaluate.py`` re-exports this function as ``finetune_and_evaluate``
(with identical signature and ``verbose=True`` default) so that all existing
callers remain unaffected.

Reference: Section 5.2 of Amin et al. (2024).
"""

from typing import Dict, List

import torch
from sklearn.metrics import accuracy_score, classification_report


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
    verbose: bool = True,
) -> Dict:
    """Fine-tune BERT-base on training texts and evaluate on a test set.

    This is the paper's primary evaluation metric (Section 5.2 of Amin et al.
    2024): train a BERT-base classifier on synthetic examples and measure
    test-set accuracy against the real held-out labels.

    Args:
        train_texts: training texts (synthetic or real).
        train_labels: integer label IDs (parallel to train_texts).
        test_texts: held-out test texts.
        test_labels: integer label IDs (parallel to test_texts).
        num_labels: number of classes.
        bert_model: HuggingFace model ID or local path (default: bert-base-uncased).
        epochs: number of fine-tuning epochs.
        batch_size: training and evaluation batch size.
        lr: AdamW learning rate.
        max_length: maximum token length for tokenization.
        device: torch device (``"cuda"`` or ``"cpu"``).
        verbose: print per-epoch loss and final evaluation report.

    Returns:
        Dict with keys:
        - ``accuracy``: float, overall accuracy.
        - ``macro_f1``: float, macro-averaged F1.
        - ``weighted_f1``: float, weighted-averaged F1.
        - ``per_class``: dict mapping class label string to sklearn metrics dict.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import DataLoader, TensorDataset

    if verbose:
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
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

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
        print(f"\n  Test accuracy: {acc:.4f}")
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
