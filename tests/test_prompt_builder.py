"""
Tests for the prompt layer: PromptBuilder, TextClassificationPromptBuilder,
PROMPT_TEMPLATES, build_prompts (Phase 2 guard).

Contracts protected here:
- PROMPT_TEMPLATES is accessible from both src.prompts and src.config
  (backward-compat re-export).
- TextClassificationPromptBuilder wraps build_prompts with identical output.
- build_prompts output is identical to the old generate.py behavior
  (parity test using the plain-text path, no tokenizer needed).
- TextClassificationPromptBuilder raises for unknown dataset names.
- The public prompt uses the public_seed, not an example text.
- Each private prompt contains the example text.
"""

import pytest

from src.prompts.base import PromptBuilder
from src.prompts.text_classification import (
    PROMPT_TEMPLATES,
    TextClassificationPromptBuilder,
    build_prompts,
    _format_prompt,
)


# ---------------------------------------------------------------------------
# Import parity: PROMPT_TEMPLATES accessible from both src.prompts and src.config
# ---------------------------------------------------------------------------

def test_prompt_templates_importable_from_prompts():
    from src.prompts import PROMPT_TEMPLATES as pt
    assert isinstance(pt, dict)
    assert "agnews" in pt


def test_prompt_templates_importable_from_config():
    """Backward-compat re-export: from src.config import PROMPT_TEMPLATES still works."""
    from src.config import PROMPT_TEMPLATES as pt
    assert isinstance(pt, dict)
    assert "agnews" in pt


def test_prompt_templates_are_same_object():
    """Both import paths resolve to the same dict (not a copy)."""
    from src.prompts import PROMPT_TEMPLATES as pt_prompts
    from src.config import PROMPT_TEMPLATES as pt_config
    assert pt_prompts is pt_config


# ---------------------------------------------------------------------------
# PROMPT_TEMPLATES structure
# ---------------------------------------------------------------------------

REQUIRED_TEMPLATE_KEYS = {"user_message", "response_prefix", "labels"}


def test_all_templates_have_required_keys():
    for dataset_name, template in PROMPT_TEMPLATES.items():
        missing = REQUIRED_TEMPLATE_KEYS - set(template.keys())
        assert not missing, (
            f"PROMPT_TEMPLATES['{dataset_name}'] is missing keys: {missing}"
        )


def test_all_templates_have_public_seed():
    for dataset_name, template in PROMPT_TEMPLATES.items():
        assert "public_seed" in template, (
            f"PROMPT_TEMPLATES['{dataset_name}'] missing 'public_seed'"
        )


# ---------------------------------------------------------------------------
# build_prompts (standalone function, no tokenizer path)
# ---------------------------------------------------------------------------

def test_build_prompts_returns_correct_counts():
    examples = [
        {"text": "first text", "label": 0},
        {"text": "second text", "label": 0},
        {"text": "third text", "label": 0},
    ]
    private_prompts, public_prompt = build_prompts(
        examples, "agnews", "text", label=0
    )
    assert len(private_prompts) == 3, "One private prompt per example"
    assert isinstance(public_prompt, str), "Public prompt must be a string"


def test_build_prompts_private_contains_example_text():
    """Each private prompt must embed the corresponding example text."""
    examples = [
        {"text": "unique_example_alpha", "label": 1},
        {"text": "unique_example_beta", "label": 1},
    ]
    private_prompts, _ = build_prompts(examples, "agnews", "text", label=1)
    assert "unique_example_alpha" in private_prompts[0]
    assert "unique_example_beta" in private_prompts[1]


def test_build_prompts_public_uses_seed_not_example():
    """The public prompt must use the public_seed, not any example text."""
    examples = [{"text": "should_not_appear_in_public", "label": 0}]
    _, public_prompt = build_prompts(examples, "agnews", "text", label=0)
    assert "should_not_appear_in_public" not in public_prompt
    seed = PROMPT_TEMPLATES["agnews"]["public_seed"]
    assert seed in public_prompt


def test_build_prompts_uses_label_name():
    """Private prompts must contain the human-readable label name."""
    examples = [{"text": "some text", "label": 2}]
    label_name = PROMPT_TEMPLATES["agnews"]["labels"][2]  # "Business"
    private_prompts, public_prompt = build_prompts(
        examples, "agnews", "text", label=2
    )
    assert label_name in private_prompts[0], (
        f"Expected label name '{label_name}' in private prompt"
    )
    assert label_name in public_prompt


def test_build_prompts_trec_response_prefix():
    """trec response_prefix is embedded in both public and private prompts."""
    examples = [{"text": "What is it?", "label": 0}]
    private_prompts, public_prompt = build_prompts(
        examples, "trec", "text", label=0
    )
    prefix = PROMPT_TEMPLATES["trec"]["response_prefix"]
    assert prefix in private_prompts[0]
    assert prefix in public_prompt


def test_build_prompts_fallback_without_tokenizer():
    """Without a tokenizer, prompts use the '# [User]' fallback template."""
    examples = [{"text": "test", "label": 0}]
    private_prompts, public_prompt = build_prompts(
        examples, "agnews", "text", label=0, tokenizer=None
    )
    assert "# [User]" in private_prompts[0]
    assert "# [Assistant]" in private_prompts[0]
    assert "# [User]" in public_prompt


# ---------------------------------------------------------------------------
# TextClassificationPromptBuilder
# ---------------------------------------------------------------------------

def test_builder_is_prompt_builder_subclass():
    builder = TextClassificationPromptBuilder("agnews")
    assert isinstance(builder, PromptBuilder)


def test_builder_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        TextClassificationPromptBuilder("unknown_dataset_xyz")


def test_builder_build_prompts_matches_standalone():
    """build_prompts via builder must produce identical output to standalone."""
    examples = [
        {"text": "alpha text", "label": 0},
        {"text": "beta text", "label": 0},
    ]
    builder = TextClassificationPromptBuilder("agnews")

    builder_private, builder_public = builder.build_prompts(
        examples, "text", label=0
    )
    fn_private, fn_public = build_prompts(examples, "agnews", "text", label=0)

    assert builder_private == fn_private
    assert builder_public == fn_public


def test_builder_label_names_property():
    builder = TextClassificationPromptBuilder("dbpedia")
    assert builder.label_names == PROMPT_TEMPLATES["dbpedia"]["labels"]


def test_builder_response_prefix_property():
    builder = TextClassificationPromptBuilder("trec")
    assert builder.response_prefix == PROMPT_TEMPLATES["trec"]["response_prefix"]


def test_builder_for_all_supported_datasets():
    """TextClassificationPromptBuilder can be constructed for every dataset."""
    for name in PROMPT_TEMPLATES:
        builder = TextClassificationPromptBuilder(name)
        assert builder.dataset_name == name
