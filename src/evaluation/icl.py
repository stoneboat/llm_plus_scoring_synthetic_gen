"""
In-context learning (ICL) evaluation.

Phase 5: extracted from scripts/evaluate_downstream.py.

Provides:
    build_icl_prompt  — construct a balanced few-shot ICL prompt from synthetic
                        examples and a test text, grouping shots by label.
    icl_evaluate      — run a full ICL evaluation loop using a local causal LM.

The prompt format used here ("Classify the following text... Category:") is the
evaluation-specific format from the paper's Section 5.1 experiments.  It is
distinct from the simpler ``format_icl_prompt`` in ``src/evaluate.py`` (which
uses "Classify the following examples... Answer:") and both are preserved for
backward compatibility.

Reference: Section 5.1 of Amin et al. (2024).
"""

from typing import Dict, List, Optional

import torch

from src.prompts import PROMPT_TEMPLATES


def build_icl_prompt(
    synthetic_texts: List[str],
    synthetic_labels: List[int],
    dataset_name: str,
    test_text: str,
    num_shots: int = 4,
) -> str:
    """Build a balanced few-shot ICL prompt.

    Selects up to ``num_shots`` examples from *synthetic_texts*, balanced
    across label classes (``num_shots // num_classes`` per class where
    possible), then appends the query text.

    Prompt format::

        Classify the following text into one of these categories: A, B, ...

        Text: <synthetic shot>
        Category: <label name>

        ...

        Text: <test_text>
        Category:

    Args:
        synthetic_texts: generated text strings.
        synthetic_labels: integer label IDs (parallel to synthetic_texts).
        dataset_name: key into PROMPT_TEMPLATES (for label-name lookup).
        test_text: the real test example to classify.
        num_shots: total number of few-shot demonstrations.

    Returns:
        Formatted prompt string.
    """
    templates = PROMPT_TEMPLATES[dataset_name]
    label_names = templates["labels"]

    shots_by_label: Dict[int, List[str]] = {}
    for text, label in zip(synthetic_texts, synthetic_labels):
        shots_by_label.setdefault(label, []).append(text)

    lines = [
        "Classify the following text into one of these categories: "
        + ", ".join(label_names.values()) + ".\n"
    ]

    shots_used = 0
    per_label = max(1, num_shots // len(label_names))
    for lab in sorted(shots_by_label.keys()):
        available = shots_by_label[lab]
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


def icl_evaluate(
    synthetic_texts: List[str],
    synthetic_labels: List[int],
    synthetic_label_names: Optional[List[str]],
    test_texts: List[str],
    test_labels: List[int],
    dataset_name: str,
    model_path: str,
    num_shots: int = 4,
    max_test: int = 500,
    device: str = "cuda",
) -> Dict:
    """Evaluate using in-context learning with a local causal LM.

    Presents ``num_shots`` synthetic examples as few-shot demonstrations
    (balanced by label via ``build_icl_prompt``), then asks the model to
    classify each test example by generating a short completion.  Predictions
    are extracted by substring-matching the first generated word against the
    known label names.

    Args:
        synthetic_texts: generated text strings used as shots.
        synthetic_labels: integer label IDs (parallel to synthetic_texts).
        synthetic_label_names: label name strings (parallel to synthetic_texts;
            currently unused but kept for API symmetry with the script).
        test_texts: real test texts to classify.
        test_labels: true integer label IDs (parallel to test_texts).
        dataset_name: key into PROMPT_TEMPLATES (for label-name lookup).
        model_path: local path or HuggingFace model ID for the LM.
        num_shots: number of few-shot demonstrations per query.
        max_test: maximum number of test examples to evaluate.
        device: torch device.

    Returns:
        Dict with keys:
        - ``accuracy``: float, overall accuracy.
        - ``macro_f1``: float, macro-averaged F1.
        - ``weighted_f1``: float, weighted-averaged F1.
        - ``num_shots``: int, shots used.
        - ``num_test``: int, test examples evaluated.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.metrics import accuracy_score, classification_report

    templates = PROMPT_TEMPLATES[dataset_name]
    label_names = templates["labels"]
    name_to_id = {v.lower(): k for k, v in label_names.items()}

    print(f"\nICL evaluation: {num_shots}-shot, model={model_path}")
    total = min(max_test, len(test_texts))
    print(f"  Evaluating on {total} test examples...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map=device,
    )
    model.eval()

    all_preds, all_true = [], []
    correct = 0

    for i in range(total):
        prompt = build_icl_prompt(
            synthetic_texts, synthetic_labels, dataset_name,
            test_texts[i], num_shots=num_shots,
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=10, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        ).strip()
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
