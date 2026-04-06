"""
Smoke tests for JSONL output schema and checkpoint round-trip (Phase 1 guard).

Contracts protected here:
- save_synthetic_data writes valid JSONL with the expected six record fields.
- load_synthetic_data reconstructs SyntheticExample objects faithfully.
- Metadata header ({"_metadata": ...}) is preserved on write and returned
  separately on load, not included in the example list.
- Batch-complete markers (_batch_complete) are handled correctly: only
  examples from completed batches are returned when markers are present.
- Truncated / malformed trailing lines are tolerated with a warning.
"""

import json
import os
import pytest
from src.generate import SyntheticExample
from src.evaluate import save_synthetic_data, load_synthetic_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = frozenset({
    "text", "label", "label_name",
    "num_private_tokens", "num_public_tokens", "num_total_tokens",
})


def _make_examples():
    return [
        SyntheticExample(
            text="World leaders met today to discuss policy.",
            label=0,
            label_name="World",
            num_private_tokens=5,
            num_public_tokens=3,
            num_total_tokens=8,
        ),
        SyntheticExample(
            text="Local team wins championship in overtime.",
            label=1,
            label_name="Sports",
            num_private_tokens=10,
            num_public_tokens=2,
            num_total_tokens=12,
        ),
    ]


# ---------------------------------------------------------------------------
# Basic save / existence
# ---------------------------------------------------------------------------

def test_save_creates_file(tmp_path):
    path = str(tmp_path / "out.jsonl")
    save_synthetic_data(_make_examples(), path)
    assert os.path.exists(path), "save_synthetic_data did not create the file"


def test_save_creates_parent_dirs(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "out.jsonl")
    save_synthetic_data(_make_examples(), path)
    assert os.path.exists(path), "save_synthetic_data should create missing parent directories"


# ---------------------------------------------------------------------------
# Round-trip fidelity
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_no_metadata(tmp_path):
    examples = _make_examples()
    path = str(tmp_path / "out.jsonl")
    save_synthetic_data(examples, path)
    loaded, meta = load_synthetic_data(path)
    assert meta is None, "Expected no metadata"
    assert len(loaded) == len(examples)
    for orig, rec in zip(examples, loaded):
        assert rec.text == orig.text
        assert rec.label == orig.label
        assert rec.label_name == orig.label_name
        assert rec.num_private_tokens == orig.num_private_tokens
        assert rec.num_public_tokens == orig.num_public_tokens
        assert rec.num_total_tokens == orig.num_total_tokens


def test_save_load_roundtrip_with_metadata(tmp_path):
    path = str(tmp_path / "out.jsonl")
    meta_in = {"dataset": "agnews", "epsilon": 3.0, "batch_size": 255}
    save_synthetic_data(_make_examples(), path, metadata=meta_in)
    loaded, meta_out = load_synthetic_data(path)
    assert meta_out == meta_in, "Metadata not preserved correctly"
    assert len(loaded) == 2


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_jsonl_records_have_required_fields(tmp_path):
    """Every non-metadata record in the JSONL contains the required six fields."""
    path = str(tmp_path / "out.jsonl")
    save_synthetic_data(_make_examples(), path)
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            if "_metadata" in record:
                continue
            missing = REQUIRED_FIELDS - set(record.keys())
            assert not missing, f"Record is missing fields: {missing}"


def test_metadata_not_in_loaded_examples(tmp_path):
    """Metadata header line must not appear in the loaded example list."""
    path = str(tmp_path / "out.jsonl")
    save_synthetic_data(_make_examples(), path, metadata={"key": "value"})
    loaded, _ = load_synthetic_data(path)
    assert all(isinstance(e, SyntheticExample) for e in loaded), (
        "Loaded list should contain only SyntheticExample objects"
    )


# ---------------------------------------------------------------------------
# Checkpoint / resume semantics
# ---------------------------------------------------------------------------

def test_load_handles_batch_complete_markers(tmp_path):
    """
    With _batch_complete markers present, only examples from completed batches
    are returned (resume semantics).
    """
    path = str(tmp_path / "checkpoint.jsonl")
    ex = _make_examples()[0]
    batch_id = "deadbeef1234"
    with open(path, "w") as f:
        # Mark the batch as complete
        f.write(json.dumps({"_batch_complete": {"batch_id": batch_id}}) + "\n")
        # Write one example belonging to that batch
        record = {
            "text": ex.text,
            "label": ex.label,
            "label_name": ex.label_name,
            "num_private_tokens": ex.num_private_tokens,
            "num_public_tokens": ex.num_public_tokens,
            "num_total_tokens": ex.num_total_tokens,
            "batch_id": batch_id,
        }
        f.write(json.dumps(record) + "\n")
    loaded, _ = load_synthetic_data(path)
    assert len(loaded) == 1
    assert loaded[0].text == ex.text


def test_load_excludes_incomplete_batch_examples(tmp_path):
    """
    Examples whose batch_id has no corresponding _batch_complete marker
    are excluded when batch markers are present.
    """
    path = str(tmp_path / "partial.jsonl")
    ex = _make_examples()[0]
    complete_id = "complete_batch"
    incomplete_id = "incomplete_batch"
    with open(path, "w") as f:
        # Only the first batch is marked complete
        f.write(json.dumps({"_batch_complete": {"batch_id": complete_id}}) + "\n")
        for bid, text in [(complete_id, "complete example"), (incomplete_id, "incomplete example")]:
            record = {
                "text": text,
                "label": 0,
                "label_name": "World",
                "num_private_tokens": 5,
                "num_public_tokens": 2,
                "num_total_tokens": 7,
                "batch_id": bid,
            }
            f.write(json.dumps(record) + "\n")
    loaded, _ = load_synthetic_data(path)
    assert len(loaded) == 1
    assert loaded[0].text == "complete example"


def test_load_tolerates_truncated_tail(tmp_path, capsys):
    """
    A truncated (non-parseable) tail line is skipped with a warning,
    and the successfully parsed lines are still returned.
    """
    path = str(tmp_path / "truncated.jsonl")
    ex = _make_examples()[0]
    record = {
        "text": ex.text,
        "label": ex.label,
        "label_name": ex.label_name,
        "num_private_tokens": ex.num_private_tokens,
        "num_public_tokens": ex.num_public_tokens,
        "num_total_tokens": ex.num_total_tokens,
    }
    with open(path, "w") as f:
        f.write(json.dumps(record) + "\n")
        f.write('{"text": "incomplete json...\n')  # truncated line
    loaded, _ = load_synthetic_data(path)
    assert len(loaded) == 1, "Valid lines before truncation should still load"
