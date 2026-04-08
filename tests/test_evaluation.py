"""
Tests for src/evaluation/ — data loading helpers, ICL prompt building,
result saving, and finetune_bert import smoke.

All tests are mocked / in-memory; no GPU, no model weights, no HuggingFace
downloads, no BERT fine-tuning.
"""

import json
import os
import tempfile
from typing import List
from unittest.mock import patch, MagicMock

import pytest

from src.runtime.generation import SyntheticExample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_example(text="synthetic text", label=0, label_name="World",
                  n_priv=3, n_pub=2) -> SyntheticExample:
    return SyntheticExample(
        text=text,
        label=label,
        label_name=label_name,
        num_private_tokens=n_priv,
        num_public_tokens=n_pub,
        num_total_tokens=n_priv + n_pub,
    )


def _write_jsonl(path, examples, metadata=None):
    with open(path, "w") as f:
        if metadata:
            f.write(json.dumps({"_metadata": metadata}) + "\n")
        for ex in examples:
            f.write(json.dumps({
                "text": ex.text,
                "label": ex.label,
                "label_name": ex.label_name,
                "num_private_tokens": ex.num_private_tokens,
                "num_public_tokens": ex.num_public_tokens,
                "num_total_tokens": ex.num_total_tokens,
            }) + "\n")


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------

class TestEvaluationImports:
    def test_import_evaluation_package(self):
        from src.evaluation import (
            load_test_set,
            load_real_train,
            load_synthetic_for_eval,
            finetune_bert,
            build_icl_prompt,
            icl_evaluate,
            save_eval_results,
        )

    def test_finetune_and_evaluate_still_importable_from_evaluate(self):
        """Backward compat: finetune_and_evaluate must remain importable from src.evaluate."""
        from src.evaluate import finetune_and_evaluate
        assert callable(finetune_and_evaluate)

    def test_finetune_bert_is_same_as_finetune_and_evaluate(self):
        """finetune_and_evaluate should be the same function as finetune_bert."""
        from src.evaluate import finetune_and_evaluate
        from src.evaluation.finetune import finetune_bert
        assert finetune_and_evaluate is finetune_bert


# ---------------------------------------------------------------------------
# data.py — load_synthetic_for_eval
# ---------------------------------------------------------------------------

class TestLoadSyntheticForEval:
    def test_returns_four_tuple(self, tmp_path):
        from src.evaluation.data import load_synthetic_for_eval
        p = str(tmp_path / "synth.jsonl")
        _write_jsonl(p, [_make_example("hello", label=0, label_name="World")])
        texts, labels, label_names, meta = load_synthetic_for_eval(p)
        assert isinstance(texts, list)
        assert isinstance(labels, list)
        assert isinstance(label_names, list)
        assert meta is None

    def test_texts_parallel_to_labels(self, tmp_path):
        from src.evaluation.data import load_synthetic_for_eval
        p = str(tmp_path / "synth.jsonl")
        examples = [
            _make_example("text A", label=0, label_name="World"),
            _make_example("text B", label=1, label_name="Sports"),
        ]
        _write_jsonl(p, examples)
        texts, labels, label_names, _ = load_synthetic_for_eval(p)
        assert texts == ["text A", "text B"]
        assert labels == [0, 1]
        assert label_names == ["World", "Sports"]

    def test_metadata_returned(self, tmp_path):
        from src.evaluation.data import load_synthetic_for_eval
        p = str(tmp_path / "synth.jsonl")
        _write_jsonl(p, [_make_example()], metadata={"epsilon": 1.5, "dataset": "trec"})
        _, _, _, meta = load_synthetic_for_eval(p)
        assert meta == {"epsilon": 1.5, "dataset": "trec"}

    def test_empty_file_returns_empty_lists(self, tmp_path):
        from src.evaluation.data import load_synthetic_for_eval
        p = str(tmp_path / "empty.jsonl")
        with open(p, "w") as f:
            f.write("")
        texts, labels, label_names, meta = load_synthetic_for_eval(p)
        assert texts == []
        assert labels == []
        assert label_names == []


# ---------------------------------------------------------------------------
# icl.py — build_icl_prompt
# ---------------------------------------------------------------------------

class TestBuildIclPrompt:
    """Tests for the balanced ICL prompt builder."""

    def test_prompt_contains_test_text(self):
        from src.evaluation.icl import build_icl_prompt
        prompt = build_icl_prompt(
            synthetic_texts=["World politics article"],
            synthetic_labels=[0],
            dataset_name="agnews",
            test_text="Test sentence here",
            num_shots=1,
        )
        assert "Test sentence here" in prompt

    def test_prompt_ends_with_category_marker(self):
        from src.evaluation.icl import build_icl_prompt
        prompt = build_icl_prompt(
            synthetic_texts=["World article"],
            synthetic_labels=[0],
            dataset_name="agnews",
            test_text="Query",
            num_shots=1,
        )
        assert prompt.strip().endswith("Category:")

    def test_prompt_mentions_label_names(self):
        from src.evaluation.icl import build_icl_prompt
        from src.prompts import PROMPT_TEMPLATES
        prompt = build_icl_prompt(
            synthetic_texts=["World article", "Sports game"],
            synthetic_labels=[0, 1],
            dataset_name="agnews",
            test_text="Query",
            num_shots=2,
        )
        # Should mention at least one label name from the template
        label_names = PROMPT_TEMPLATES["agnews"]["labels"]
        assert any(name in prompt for name in label_names.values())

    def test_prompt_includes_shots(self):
        from src.evaluation.icl import build_icl_prompt
        prompt = build_icl_prompt(
            synthetic_texts=["shot text one"],
            synthetic_labels=[0],
            dataset_name="agnews",
            test_text="test",
            num_shots=1,
        )
        assert "shot text one" in prompt

    def test_zero_shots_still_produces_prompt(self):
        from src.evaluation.icl import build_icl_prompt
        prompt = build_icl_prompt(
            synthetic_texts=[],
            synthetic_labels=[],
            dataset_name="agnews",
            test_text="test text",
            num_shots=0,
        )
        assert "test text" in prompt
        assert prompt.strip().endswith("Category:")


# ---------------------------------------------------------------------------
# Prompt format parity: format_icl_prompt vs build_icl_prompt
# ---------------------------------------------------------------------------

class TestIclPromptFormatDistinction:
    """Verifies the two ICL formatters are different (both preserved)."""

    def test_format_icl_prompt_uses_answer_marker(self):
        from src.evaluate import format_icl_prompt
        ex = _make_example("example text", label=0, label_name="World")
        prompt = format_icl_prompt([ex], "query text", "agnews", num_shots=1)
        assert "Answer:" in prompt
        assert "Category:" not in prompt

    def test_build_icl_prompt_uses_category_marker(self):
        from src.evaluation.icl import build_icl_prompt
        prompt = build_icl_prompt(
            ["example text"], [0], "agnews", "query text", num_shots=1
        )
        assert "Category:" in prompt
        # Does not use "Answer:" as the response marker
        assert not prompt.strip().endswith("Answer:")


# ---------------------------------------------------------------------------
# results.py — save_eval_results
# ---------------------------------------------------------------------------

class TestSaveEvalResults:
    def test_creates_json_file(self, tmp_path):
        from src.evaluation.results import save_eval_results
        p = str(tmp_path / "results.json")
        save_eval_results({"accuracy": 0.85, "macro_f1": 0.83}, p)
        assert os.path.exists(p)

    def test_content_is_valid_json(self, tmp_path):
        from src.evaluation.results import save_eval_results
        p = str(tmp_path / "results.json")
        data = {"accuracy": 0.9, "mode": "finetune", "dataset": "agnews"}
        save_eval_results(data, p)
        with open(p) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_creates_parent_directories(self, tmp_path):
        from src.evaluation.results import save_eval_results
        p = str(tmp_path / "nested" / "dir" / "results.json")
        save_eval_results({"x": 1}, p)
        assert os.path.exists(p)


# ---------------------------------------------------------------------------
# compute_generation_stats (remains in src/evaluate.py — regression guard)
# ---------------------------------------------------------------------------

class TestComputeGenerationStats:
    def test_empty_returns_empty_dict(self):
        from src.evaluate import compute_generation_stats
        assert compute_generation_stats([]) == {}

    def test_single_example_stats(self):
        from src.evaluate import compute_generation_stats
        ex = _make_example(n_priv=4, n_pub=2)
        stats = compute_generation_stats([ex])
        assert stats["num_examples"] == 1
        assert stats["total_private_tokens"] == 4
        assert stats["total_public_tokens"] == 2
        assert stats["max_private_tokens_in_example"] == 4

    def test_public_token_fraction(self):
        from src.evaluate import compute_generation_stats
        examples = [
            _make_example(n_priv=0, n_pub=4),
            _make_example(n_priv=4, n_pub=0),
        ]
        stats = compute_generation_stats(examples)
        assert stats["public_token_fraction"] == pytest.approx(0.5)
