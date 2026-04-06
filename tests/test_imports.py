"""
Import smoke tests (Phase 1 guard).

Verifies that the current public surface of the src package is importable
without errors and without requiring a GPU or large model weights.

These tests protect the import layer so that Phase 2 restructuring cannot
accidentally break the existing import paths without a test failure.
"""

import pytest


# ---------------------------------------------------------------------------
# Module-level import smoke tests
# ---------------------------------------------------------------------------

def test_import_generate():
    from src.generate import (
        SyntheticExample,
        BatchDescriptor,
        partition_by_label,
        assign_to_batch,
        build_prompts,
        generate_synthetic_dataset,
        generate_batch_examples,
        generate_one_example,
    )


def test_import_clip_utils():
    from src.clip_utils import clip_logits, clip_and_aggregate


def test_import_sparse_vector():
    from src.sparse_vector import (
        compute_distribution_distance,
        sample_noisy_threshold,
        should_use_private_token,
    )


def test_import_config():
    from src.config import (
        PrivacyConfig,
        GenerationConfig,
        ModelConfig,
        DatasetConfig,
        PROMPT_TEMPLATES,
        HYPERPARAM_GRID,
        SVT_SETTINGS,
        compute_max_private_tokens,
    )


def test_import_evaluate():
    from src.evaluate import (
        save_synthetic_data,
        load_synthetic_data,
        compute_generation_stats,
        format_icl_prompt,
        finetune_and_evaluate,
    )


def test_import_privacy_accounting():
    from src.privacy_accounting import (
        compute_rho_per_token,
        compute_total_rho,
        compute_epsilon,
        compute_max_private_tokens,
        privacy_report,
        zcdp_to_approx_dp,
        zcdp_to_dp_tight,
    )


def test_src_package_version():
    import src
    assert hasattr(src, "__version__"), "src package should expose __version__"
    assert src.__version__ == "0.1.0"


# ---------------------------------------------------------------------------
# Dataclass sanity checks (no GPU, no model weights)
# ---------------------------------------------------------------------------

def test_privacy_config_svt_enabled():
    from src.config import PrivacyConfig
    cfg = PrivacyConfig(svt_threshold=0.5, svt_noise=0.2)
    assert cfg.svt_enabled is True


def test_privacy_config_svt_disabled_by_neg_inf():
    from src.config import PrivacyConfig
    cfg = PrivacyConfig(svt_threshold=float("-inf"))
    assert cfg.svt_enabled is False


def test_generation_config_defaults():
    from src.config import GenerationConfig
    cfg = GenerationConfig()
    assert cfg.batch_size == 255
    assert cfg.top_k_vocab == 0
    assert cfg.eos_token_id is None
    assert cfg.max_private_tokens == 100
    assert cfg.max_total_tokens == 256


def test_batch_descriptor_frozen():
    """BatchDescriptor is frozen (immutable), usable as a dict key."""
    from src.generate import BatchDescriptor
    bd = BatchDescriptor(
        batch_id="abc123",
        batch_index=1,
        total_batches=10,
        label=0,
        label_name="World",
        batch_size=255,
    )
    # Should be hashable (frozen dataclass)
    _ = {bd: "value"}


def test_synthetic_example_fields():
    """SyntheticExample holds the expected fields."""
    from src.generate import SyntheticExample
    ex = SyntheticExample(
        text="test text",
        label=2,
        label_name="Business",
        num_private_tokens=7,
        num_public_tokens=3,
        num_total_tokens=10,
    )
    assert ex.text == "test text"
    assert ex.label == 2
    assert ex.num_total_tokens == 10


# ---------------------------------------------------------------------------
# PROMPT_TEMPLATES content smoke test
# ---------------------------------------------------------------------------

def test_prompt_templates_have_required_keys():
    from src.config import PROMPT_TEMPLATES
    required_keys = {"user_message", "response_prefix", "labels"}
    for dataset_name, template in PROMPT_TEMPLATES.items():
        missing = required_keys - set(template.keys())
        assert not missing, (
            f"PROMPT_TEMPLATES['{dataset_name}'] is missing keys: {missing}"
        )


def test_prompt_templates_cover_expected_datasets():
    from src.config import PROMPT_TEMPLATES
    expected = {"agnews", "trec", "dbpedia", "imdb", "yelp"}
    assert expected <= set(PROMPT_TEMPLATES.keys()), (
        f"Missing dataset templates: {expected - set(PROMPT_TEMPLATES.keys())}"
    )


# ---------------------------------------------------------------------------
# Privacy accounting sanity (pure math, no GPU)
# ---------------------------------------------------------------------------

def test_compute_epsilon_positive():
    from src.privacy_accounting import compute_epsilon
    eps = compute_epsilon(
        num_private_tokens=100,
        clip_bound=10.0,
        batch_size=255,
        temperature=2.0,
        delta=1e-5,
        svt_noise=0.2,
    )
    assert eps > 0, "Epsilon should be positive"


def test_compute_rho_per_token_no_svt():
    from src.privacy_accounting import compute_rho_per_token
    rho = compute_rho_per_token(
        clip_bound=10.0, batch_size=255, temperature=2.0, svt_noise=None
    )
    assert rho > 0


def test_privacy_report_keys():
    from src.privacy_accounting import privacy_report
    report = privacy_report(100, 10.0, 255, 2.0, 1e-5, 0.2)
    for key in ("rho_per_token", "total_rho", "epsilon", "delta",
                "num_private_tokens", "clip_bound", "batch_size",
                "temperature", "svt_noise"):
        assert key in report, f"privacy_report missing key: {key}"
