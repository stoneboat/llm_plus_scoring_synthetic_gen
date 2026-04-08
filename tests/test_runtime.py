"""
Tests for src/runtime/generation — run_batch_generation and SyntheticExample.

All tests use mocked mechanism and backend objects; no GPU, no model weights.

run_dataset_generation is integration-heavy (requires PROMPT_TEMPLATES +
partition_by_label + actual batch IDs), so it is not unit-tested here;
its correctness is validated indirectly via the existing test_mechanism.py
and test_batching.py suites plus the end-to-end import smoke tests.
"""

from unittest.mock import MagicMock, call
import pytest

from src.runtime.generation import SyntheticExample, run_batch_generation
from src.config import GenerationConfig, PrivacyConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gen_config(max_private_tokens=10, max_total_tokens=50) -> GenerationConfig:
    cfg = GenerationConfig()
    cfg.max_private_tokens = max_private_tokens
    cfg.max_total_tokens = max_total_tokens
    return cfg


def _make_mechanism(return_values):
    """Return a mock mechanism whose generate_example yields successive values."""
    mech = MagicMock()
    mech.generate_example.side_effect = return_values
    return mech


def _make_backend():
    return MagicMock()


# ---------------------------------------------------------------------------
# SyntheticExample
# ---------------------------------------------------------------------------

class TestSyntheticExample:
    def test_fields_accessible(self):
        ex = SyntheticExample(
            text="hello",
            label=1,
            label_name="Sports",
            num_private_tokens=4,
            num_public_tokens=2,
            num_total_tokens=6,
        )
        assert ex.text == "hello"
        assert ex.label == 1
        assert ex.num_total_tokens == 6


# ---------------------------------------------------------------------------
# run_batch_generation — stopping conditions
# ---------------------------------------------------------------------------

class TestRunBatchGeneration:
    """Tests for the Algorithm 1 outer-loop stopping conditions."""

    def test_budget_exhausted_stops_loop(self):
        """Stops when private_tokens_used >= r."""
        # Each call uses 5 private tokens; budget = 10 → 2 calls
        gen_config = _make_gen_config(max_private_tokens=10)
        mech = _make_mechanism([
            ([1, 2], 5, 1),
            ([3, 4], 5, 1),
            ([5, 6], 5, 1),  # should never be reached
        ])
        results = run_batch_generation(mech, _make_backend(), ["p"], "pub", gen_config)
        assert len(results) == 2
        assert mech.generate_example.call_count == 2

    def test_empty_output_stops_immediately(self):
        """If n_priv == 0 and n_pub == 0, stops regardless of remaining budget."""
        gen_config = _make_gen_config(max_private_tokens=100)
        mech = _make_mechanism([
            ([1], 3, 2),
            ([], 0, 0),    # stop here
            ([2], 1, 1),   # should never be reached
        ])
        results = run_batch_generation(mech, _make_backend(), ["p"], "pub", gen_config)
        assert len(results) == 1
        assert mech.generate_example.call_count == 2

    def test_three_consecutive_no_private_stops(self):
        """Stops after 3 consecutive public-only examples (n_priv == 0, n_pub > 0)."""
        gen_config = _make_gen_config(max_private_tokens=100)
        mech = _make_mechanism([
            ([1], 0, 3),   # consecutive_no_private = 1
            ([2], 0, 2),   # consecutive_no_private = 2
            ([3], 0, 1),   # consecutive_no_private = 3 → stop
            ([4], 1, 0),   # should never be reached
        ])
        results = run_batch_generation(mech, _make_backend(), ["p"], "pub", gen_config)
        assert mech.generate_example.call_count == 3

    def test_private_token_resets_consecutive_counter(self):
        """A private token resets the consecutive counter; loop continues."""
        gen_config = _make_gen_config(max_private_tokens=100)
        mech = _make_mechanism([
            ([1], 0, 2),   # consecutive = 1
            ([2], 0, 2),   # consecutive = 2
            ([3], 1, 0),   # consecutive reset to 0
            ([4], 0, 2),   # consecutive = 1
            ([5], 0, 2),   # consecutive = 2
            ([6], 0, 2),   # consecutive = 3 → stop
        ])
        results = run_batch_generation(mech, _make_backend(), ["p"], "pub", gen_config)
        assert mech.generate_example.call_count == 6

    def test_empty_token_list_not_added_to_results(self):
        """An empty token_ids list from mechanism is not appended to results."""
        gen_config = _make_gen_config(max_private_tokens=100)
        mech = _make_mechanism([
            ([], 1, 0),   # n_priv=1 so budget consumed, but no text
            ([], 0, 0),   # stop
        ])
        results = run_batch_generation(mech, _make_backend(), ["p"], "pub", gen_config)
        assert results == []

    def test_zero_budget_produces_no_results(self):
        """With max_private_tokens=0 the loop body never executes."""
        gen_config = _make_gen_config(max_private_tokens=0)
        mech = _make_mechanism([])
        results = run_batch_generation(mech, _make_backend(), ["p"], "pub", gen_config)
        assert results == []
        mech.generate_example.assert_not_called()

    def test_remaining_budget_decreases(self):
        """mechanism.generate_example is called with correct remaining_budget."""
        gen_config = _make_gen_config(max_private_tokens=10, max_total_tokens=50)
        mech = _make_mechanism([
            ([1], 4, 0),
            ([2], 6, 0),  # uses remaining 6 → loop ends
        ])
        run_batch_generation(mech, _make_backend(), ["p1", "p2"], "pub", gen_config)
        calls = mech.generate_example.call_args_list
        assert len(calls) == 2
        # generate_example(..., remaining_budget=..., max_total_tokens=...)
        assert calls[0].kwargs["remaining_budget"] == 10
        assert calls[1].kwargs["remaining_budget"] == 6
