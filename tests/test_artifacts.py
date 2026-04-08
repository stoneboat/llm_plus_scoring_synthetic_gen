"""
Tests for src/artifacts/ — JSONL writer, resume, and metadata helpers.

All tests are mocked / in-memory; no GPU, no model weights, no HuggingFace
downloads required.  The JSONL schema tested here is checkpoint_format=1.
"""

import io
import json
import os
import tempfile
from typing import List
from unittest.mock import patch, MagicMock

import pytest

from src.batching.base import BatchDescriptor
from src.runtime.generation import SyntheticExample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_descriptor(
    batch_id="abc123",
    batch_index=1,
    total_batches=4,
    label=0,
    label_name="World",
    batch_size=5,
) -> BatchDescriptor:
    return BatchDescriptor(
        batch_id=batch_id,
        batch_index=batch_index,
        total_batches=total_batches,
        label=label,
        label_name=label_name,
        batch_size=batch_size,
    )


def _make_example(text="hello world", label=0, label_name="World",
                  n_priv=3, n_pub=2) -> SyntheticExample:
    return SyntheticExample(
        text=text,
        label=label,
        label_name=label_name,
        num_private_tokens=n_priv,
        num_public_tokens=n_pub,
        num_total_tokens=n_priv + n_pub,
    )


# ---------------------------------------------------------------------------
# jsonl_writer tests
# ---------------------------------------------------------------------------

class TestAppendLine:
    def test_writes_valid_json_with_newline(self, tmp_path):
        from src.artifacts.jsonl_writer import append_line
        p = tmp_path / "test.jsonl"
        with open(p, "w") as f:
            append_line(f, {"key": "value"})
        line = p.read_text()
        assert line.endswith("\n")
        parsed = json.loads(line.strip())
        assert parsed == {"key": "value"}

    def test_calls_flush_and_fsync(self):
        from src.artifacts.jsonl_writer import append_line
        buf = MagicMock()
        buf.fileno.return_value = 99
        with patch("os.fsync") as mock_fsync:
            append_line(buf, {"k": 1})
        buf.flush.assert_called_once()
        mock_fsync.assert_called_once_with(99)


class TestBatchRecord:
    def test_has_all_required_keys(self):
        from src.artifacts.jsonl_writer import batch_record
        ex = _make_example()
        desc = _make_descriptor()
        rec = batch_record(ex, desc)
        required = {
            "text", "label", "label_name",
            "num_private_tokens", "num_public_tokens", "num_total_tokens",
            "batch_id", "batch_index", "total_batches",
        }
        assert required <= set(rec.keys())

    def test_no_source_batch_size(self):
        from src.artifacts.jsonl_writer import batch_record
        rec = batch_record(_make_example(), _make_descriptor())
        assert "source_batch_size" not in rec

    def test_values_match_inputs(self):
        from src.artifacts.jsonl_writer import batch_record
        ex = _make_example(text="test text", label=2, label_name="Tech")
        desc = _make_descriptor(batch_id="deadbeef", batch_index=3, total_batches=10)
        rec = batch_record(ex, desc)
        assert rec["text"] == "test text"
        assert rec["label"] == 2
        assert rec["label_name"] == "Tech"
        assert rec["batch_id"] == "deadbeef"
        assert rec["batch_index"] == 3
        assert rec["total_batches"] == 10


class TestWriteMetadataHeader:
    def test_creates_file_with_metadata_record(self, tmp_path):
        from src.artifacts.jsonl_writer import write_metadata_header
        out = str(tmp_path / "out.jsonl")
        with patch("os.fsync"):
            write_metadata_header(out, {"dataset": "agnews", "epsilon": 1.0})
        with open(out) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert "_metadata" in parsed
        assert parsed["_metadata"]["dataset"] == "agnews"

    def test_creates_parent_directories(self, tmp_path):
        from src.artifacts.jsonl_writer import write_metadata_header
        nested = str(tmp_path / "a" / "b" / "out.jsonl")
        with patch("os.fsync"):
            write_metadata_header(nested, {"x": 1})
        assert os.path.exists(nested)


class TestAppendCompletedBatch:
    def test_writes_example_records_then_marker(self, tmp_path):
        from src.artifacts.jsonl_writer import write_metadata_header, append_completed_batch
        out = str(tmp_path / "out.jsonl")
        with patch("os.fsync"):
            write_metadata_header(out, {"dataset": "trec"})
            desc = _make_descriptor(batch_id="b1", batch_index=1, total_batches=2)
            examples = [_make_example("text A"), _make_example("text B")]
            append_completed_batch(out, desc, examples)

        with open(out) as f:
            lines = [json.loads(l) for l in f if l.strip()]

        # lines[0] = metadata, lines[1] = example A, lines[2] = example B, lines[3] = marker
        assert len(lines) == 4
        assert "_metadata" in lines[0]
        assert lines[1]["text"] == "text A"
        assert lines[2]["text"] == "text B"
        assert "_batch_complete" in lines[3]

    def test_marker_has_correct_fields(self, tmp_path):
        from src.artifacts.jsonl_writer import write_metadata_header, append_completed_batch
        out = str(tmp_path / "out.jsonl")
        with patch("os.fsync"):
            write_metadata_header(out, {})
            desc = _make_descriptor(batch_id="xyz", batch_size=7)
            append_completed_batch(out, desc, [_make_example()])

        with open(out) as f:
            lines = [json.loads(l) for l in f if l.strip()]

        marker = lines[-1]["_batch_complete"]
        assert marker["batch_id"] == "xyz"
        assert marker["batch_size"] == 7
        assert marker["num_examples"] == 1

    def test_example_records_include_source_batch_size(self, tmp_path):
        from src.artifacts.jsonl_writer import write_metadata_header, append_completed_batch
        out = str(tmp_path / "out.jsonl")
        with patch("os.fsync"):
            write_metadata_header(out, {})
            desc = _make_descriptor(batch_size=13)
            append_completed_batch(out, desc, [_make_example()])

        with open(out) as f:
            lines = [json.loads(l) for l in f if l.strip()]

        example_record = lines[1]  # after metadata
        assert example_record["source_batch_size"] == 13


# ---------------------------------------------------------------------------
# resume tests
# ---------------------------------------------------------------------------

class TestLoadResumeState:
    def _write_checkpoint(self, path: str, metadata: dict,
                          batches: list) -> None:
        """Write a checkpoint file for testing (no fsync needed in tests)."""
        with open(path, "w") as f:
            f.write(json.dumps({"_metadata": metadata}) + "\n")
            for desc, examples, complete in batches:
                for ex in examples:
                    rec = {
                        "text": ex.text,
                        "label": ex.label,
                        "label_name": ex.label_name,
                        "num_private_tokens": ex.num_private_tokens,
                        "num_public_tokens": ex.num_public_tokens,
                        "num_total_tokens": ex.num_total_tokens,
                        "batch_id": desc.batch_id,
                        "batch_index": desc.batch_index,
                        "total_batches": desc.total_batches,
                        "source_batch_size": desc.batch_size,
                    }
                    f.write(json.dumps(rec) + "\n")
                if complete:
                    f.write(json.dumps({"_batch_complete": {
                        "batch_id": desc.batch_id,
                        "batch_index": desc.batch_index,
                        "total_batches": desc.total_batches,
                        "label": desc.label,
                        "label_name": desc.label_name,
                        "batch_size": desc.batch_size,
                        "num_examples": len(examples),
                    }}) + "\n")

    def test_reads_metadata(self, tmp_path):
        from src.artifacts.resume import load_resume_state
        p = str(tmp_path / "ckpt.jsonl")
        self._write_checkpoint(p, {"dataset": "imdb", "epsilon": 2.0}, [])
        meta, _, _, _ = load_resume_state(p)
        assert meta["dataset"] == "imdb"
        assert meta["epsilon"] == 2.0

    def test_only_completed_batches_returned(self, tmp_path):
        from src.artifacts.resume import load_resume_state
        p = str(tmp_path / "ckpt.jsonl")
        desc1 = _make_descriptor(batch_id="b1", batch_index=1, total_batches=2)
        desc2 = _make_descriptor(batch_id="b2", batch_index=2, total_batches=2)
        ex1 = _make_example("text from b1")
        ex2 = _make_example("text from b2")
        # b1 complete, b2 incomplete (no marker)
        self._write_checkpoint(p, {}, [
            (desc1, [ex1], True),
            (desc2, [ex2], False),
        ])
        _, loaded, completed_ids, _ = load_resume_state(p)
        assert completed_ids == {"b1"}
        assert len(loaded) == 1
        assert loaded[0].text == "text from b1"

    def test_tolerates_truncated_tail(self, tmp_path):
        from src.artifacts.resume import load_resume_state
        p = str(tmp_path / "ckpt.jsonl")
        desc = _make_descriptor(batch_id="b1")
        ex = _make_example("hello")
        self._write_checkpoint(p, {"x": 1}, [(desc, [ex], True)])
        # Append a truncated (invalid) JSON line
        with open(p, "a") as f:
            f.write("{invalid json\n")
        _, loaded, completed_ids, _ = load_resume_state(p)
        assert "b1" in completed_ids
        assert len(loaded) == 1

    def test_empty_file_returns_none_metadata(self, tmp_path):
        from src.artifacts.resume import load_resume_state
        p = str(tmp_path / "empty.jsonl")
        with open(p, "w") as f:
            f.write("")
        meta, loaded, completed, _ = load_resume_state(p)
        assert meta is None
        assert loaded == []
        assert completed == set()

    def test_batch_descriptors_populated(self, tmp_path):
        from src.artifacts.resume import load_resume_state
        p = str(tmp_path / "ckpt.jsonl")
        desc = _make_descriptor(batch_id="desc_test", batch_size=10)
        self._write_checkpoint(p, {}, [(desc, [_make_example()], True)])
        _, _, _, descriptors = load_resume_state(p)
        assert "desc_test" in descriptors
        assert descriptors["desc_test"].batch_size == 10


# ---------------------------------------------------------------------------
# metadata tests
# ---------------------------------------------------------------------------

class TestBuildRunMetadata:
    def test_has_checkpoint_format_1(self):
        from src.artifacts.metadata import build_run_metadata
        meta = build_run_metadata(
            dataset="agnews", epsilon=1.0, delta=1e-5,
            batch_size=255, clip_bound=10.0, temperature=2.0,
            public_temperature=1.5, svt_threshold=float("-inf"),
            svt_noise=0.2, top_k_vocab=0, max_private_tokens=100,
            max_total_tokens=256, num_source_examples=10000,
            seed=42, micro_batch_size=32,
            output_path="/tmp/out.jsonl",
        )
        assert meta["checkpoint_format"] == 1

    def test_all_fields_present(self):
        from src.artifacts.metadata import build_run_metadata
        meta = build_run_metadata(
            dataset="trec", epsilon=2.0, delta=1e-4,
            batch_size=127, clip_bound=8.0, temperature=1.5,
            public_temperature=1.0, svt_threshold=0.5,
            svt_noise=0.1, top_k_vocab=1024, max_private_tokens=50,
            max_total_tokens=128, num_source_examples=5000,
            seed=7, micro_batch_size=16,
            output_path="/tmp/sweep.jsonl",
        )
        required = {
            "dataset", "epsilon", "delta", "batch_size", "clip_bound",
            "temperature", "public_temperature", "svt_threshold", "svt_noise",
            "top_k_vocab", "max_private_tokens", "max_total_tokens",
            "num_source_examples", "seed", "micro_batch_size",
            "output_path", "checkpoint_format",
        }
        assert required <= set(meta.keys())

    def test_values_match_inputs(self):
        from src.artifacts.metadata import build_run_metadata
        meta = build_run_metadata(
            dataset="yelp", epsilon=3.0, delta=0.001,
            batch_size=63, clip_bound=5.0, temperature=1.0,
            public_temperature=0.8, svt_threshold=0.3,
            svt_noise=0.05, top_k_vocab=512, max_private_tokens=25,
            max_total_tokens=64, num_source_examples=100,
            seed=99, micro_batch_size=8,
            output_path="/data/out.jsonl",
        )
        assert meta["dataset"] == "yelp"
        assert meta["epsilon"] == 3.0
        assert meta["seed"] == 99
        assert meta["output_path"] == "/data/out.jsonl"
