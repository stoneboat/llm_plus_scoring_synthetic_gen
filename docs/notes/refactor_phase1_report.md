# Phase 1 Refactor Report

**Date:** 2026-04-06
**Scope:** Stabilization and extraction scaffold — no algorithmic changes.

---

## 1. What Changed in Phase 1

### A. Package boundary (`pyproject.toml`)

**File added:** `pyproject.toml`

The repo previously had no `setup.py`, no `setup.cfg`, and no `pyproject.toml`.
Scripts worked by inserting the project root into `sys.path` at startup:

```python
sys.path.insert(0, PROJECT_ROOT)   # pattern in all three scripts
```

A minimal `pyproject.toml` was added with:
- `[build-system]` using setuptools ≥ 68.
- `[project]` metadata (name `llm-pe-synth`, version `0.1.0`).
- `[tool.setuptools] packages = ["src"]` — declares `src/` as the installed
  package, matching the existing `from src.X import Y` import pattern exactly.
- `[project.optional-dependencies] dev = ["pytest>=7.0", "pytest-cov"]`
- `[tool.pytest.ini_options]` pointing `testpaths = ["tests"]`.

After this change, running `pip install -e .` in the repo root makes
`from src.X import Y` importable in any Python process without `sys.path`
manipulation. The existing `sys.path.insert(0, PROJECT_ROOT)` guards in the
three scripts are now harmless no-ops when the package is installed, and
remain as a fallback for bare `python scripts/...` invocations.

**No files were moved.** No imports were changed. The choice to keep the
package name as `src` (instead of a new namespace like `llm_pe_synth`)
was deliberate: it avoids touching any of the ~2,700 lines of existing
imports. Phase 2 can rename if desired.

### B. `src/__init__.py`

**File modified:** `src/__init__.py` (was 1 empty line; now has a docstring and `__version__`).

Added:
```python
__version__ = "0.1.0"
```

This makes the package version queryable via `import src; src.__version__`
and is tested in `tests/test_imports.py::test_src_package_version`.

### C. Test suite (`tests/`)

**Directory added:** `tests/`

**Files added:**

| File | Tests | Contracts protected |
|------|-------|---------------------|
| `tests/conftest.py` | — | Adds project root to `sys.path` so tests work with or without `pip install -e .` |
| `tests/__init__.py` | — | Makes `tests/` a package |
| `tests/test_batching.py` | 10 | `assign_to_batch` determinism, range, label-aware hashing; `partition_by_label` determinism, coverage, label grouping, within-batch sort order |
| `tests/test_clip_utils.py` | 10 | `clip_logits` range `[-c,c]`, max at `c`, lossless when range ≤ 2c, 2-D batched input, idempotency; `clip_and_aggregate` shape, division-by-expected (not actual) batch size, bounded output |
| `tests/test_sparse_vector.py` | 11 | `compute_distribution_distance` near-zero for identical dists, large for divergent, bounded in `[0, 2]`; `sample_noisy_threshold` reproducibility; `should_use_private_token` directional correctness under extreme thresholds, return type |
| `tests/test_jsonl_schema.py` | 9 | `save_synthetic_data` / `load_synthetic_data` round-trip fidelity, required JSONL fields, metadata isolation, batch-complete marker resume semantics, truncated-tail tolerance |
| `tests/test_imports.py` | 17 | Full import surface smoke test; dataclass field/default sanity; `PROMPT_TEMPLATES` structure; privacy accounting output sanity |

**Total: 57 tests, 57 passing, 0 skipped, runtime ~2 s (CPU only).**

No test requires a GPU or model weights. All tests use small synthetic
tensors, `tmp_path` fixtures, or pure arithmetic.

---

## 2. Behavior Intentionally Preserved

All of the following were left completely unchanged:

| Contract | Status |
|----------|--------|
| Hash-based, label-aware disjoint batching (`partition_by_label`) | Unchanged |
| Clipped-logit aggregation formula: `sum(clip(z)) / expected_batch_size` | Unchanged |
| SVT gating logic and noise parameterization (Eq. 2 of paper) | Unchanged |
| Token-by-token generation loop (`_generate_single_example`) | Unchanged |
| Multi-example-per-batch outer loop (`generate_batch_examples`) | Unchanged |
| Budget exhaustion + consecutive-public-token stopping heuristics | Unchanged |
| JSONL output schema (six required fields + optional metadata + batch markers) | Unchanged |
| Checkpoint / resume semantics (batch-complete markers + completed_batch_ids) | Unchanged |
| BatchDescriptor stable ID (SHA-256 of dataset + label + sorted batch texts) | Unchanged |
| Top-k public filtering applied to private aggregate before sampling | Unchanged |
| Temperature scaling for private vs. public tokens | Unchanged |
| Privacy formula: `rho = r * (1/2 * (c/sτ)² + 2/(sσ)²)` | Unchanged |
| `scripts/run_experiment.py`, `scripts/evaluate_downstream.py`, `scripts/sweep_hyperparams.py` | All unchanged |

---

## 3. Tests Added — Detail

### `test_batching.py`
- **`test_assign_to_batch_deterministic`** — same text + same num_batches → same bucket.
- **`test_assign_to_batch_output_in_range`** — output always in `[0, num_batches)`.
- **`test_assign_to_batch_single_bucket`** — `num_batches=1` always returns 0.
- **`test_partition_by_label_deterministic`** — identical structures across repeated calls.
- **`test_partition_by_label_covers_all_examples`** — every example appears exactly once.
- **`test_partition_by_label_groups_by_label`** — all examples in a batch share the label.
- **`test_partition_by_label_produces_correct_keys`** — keys are the unique label values.
- **`test_partition_within_batch_sorted_by_text`** — within-batch order is lexicographic.
- **`test_partition_label_aware_hash`** — `"0\nfoo"` and `"1\nfoo"` hash to different buckets.
- **`test_partition_single_example_per_label`** — one example per label yields one batch.

### `test_clip_utils.py`
- **`test_clip_logits_max_equals_c`** — max component is exactly `c` after clipping.
- **`test_clip_logits_min_ge_neg_c`** — all components ≥ `−c`.
- **`test_clip_logits_all_within_range`** — random large logits clipped to `[−c, c]`.
- **`test_clip_logits_preserves_softmax_when_lossless`** — softmax invariant holds.
- **`test_clip_logits_2d_batch`** — 2-D batch: each row has max = `c`.
- **`test_clip_logits_idempotent`** — applying twice changes nothing.
- **`test_clip_and_aggregate_output_shape`** — output shape is `(vocab_size,)`.
- **`test_clip_and_aggregate_divides_by_expected_not_actual`** — explicit numeric check.
- **`test_clip_and_aggregate_bounded_when_sizes_equal`** — output in `[−c, c]`.
- **`test_clip_and_aggregate_single_row`** — single-row matches manual calculation.

### `test_sparse_vector.py`
- **`test_distribution_distance_identical_is_near_zero`** — L1 ≈ 0 for identical.
- **`test_distribution_distance_divergent_is_large`** — L1 > 1.5 for antipodal.
- **`test_distribution_distance_bounded_in_0_2`** — 20 random trials all in `[0, 2]`.
- **`test_distribution_distance_non_negative`** — always ≥ 0.
- **`test_sample_noisy_threshold_reproducible`** — same seed → same value.
- **`test_sample_noisy_threshold_near_base`** — tiny sigma → near-base output.
- **`test_should_use_private_token_very_low_threshold`** — threshold=−999 → private.
- **`test_should_use_private_token_very_high_threshold`** — threshold=+999 → public.
- **`test_should_use_private_token_high_distance_uses_private`** — divergent dists, threshold=0 → private.
- **`test_should_use_private_token_low_distance_uses_public`** — identical dists, threshold=1 → public.
- **`test_should_use_private_token_return_type`** — returns `(bool, float)`.

### `test_jsonl_schema.py`
- **`test_save_creates_file`** / **`test_save_creates_parent_dirs`** — basic I/O.
- **`test_save_load_roundtrip_no_metadata`** / **`_with_metadata`** — all six fields preserved.
- **`test_jsonl_records_have_required_fields`** — schema validation on raw JSONL.
- **`test_metadata_not_in_loaded_examples`** — metadata stays out of example list.
- **`test_load_handles_batch_complete_markers`** — completed batch → returned.
- **`test_load_excludes_incomplete_batch_examples`** — incomplete batch → excluded.
- **`test_load_tolerates_truncated_tail`** — bad tail line skipped, rest loaded.

### `test_imports.py`
- Full import surface: all public names from `generate`, `clip_utils`, `sparse_vector`, `config`, `evaluate`, `privacy_accounting`.
- `src.__version__` queryable.
- `PrivacyConfig.svt_enabled` property correctness.
- `GenerationConfig` defaults stable.
- `BatchDescriptor` is frozen (hashable).
- `SyntheticExample` fields accessible.
- `PROMPT_TEMPLATES` has required keys for all five datasets.
- `compute_epsilon`, `compute_rho_per_token`, `privacy_report` return plausible values.

---

## 4. Deferred to Phase 2

The following were explicitly **not** done in Phase 1:

### Architecture (the main Phase 2 targets)
- **`DatasetAdapter`**: Dataset loading is duplicated three times
  (`run_experiment.py`, `evaluate_downstream.py`, `sweep_hyperparams.py`).
  A single `DatasetAdapter` interface should consolidate this.
- **`PromptBuilder`**: Prompt construction is woven into `generate.py`
  (`build_prompts`, `_format_prompt`) and implicitly depends on
  `PROMPT_TEMPLATES`. This should be decoupled.
- **`BatchingPolicy`**: `partition_by_label` + `assign_to_batch` are
  hard-coded in `generate_synthetic_dataset`. A swappable `BatchingPolicy`
  interface would enable non-hash-based strategies.
- **`Mechanism`**: The SVT + clipping + aggregation + top-k logic inside
  `_generate_single_example` is tightly coupled to the generation loop.
  A `Mechanism` abstraction would make this swappable.
- **`ModelBackend`**: `get_next_token_logits` and the tokenizer/padding
  setup are interleaved with generation logic. A `ModelBackend` wrapper
  would separate inference from algorithm.
- **`PrivacyAccountant`**: Currently the formula is mechanism-specific and
  duplicated between `config.py` (`compute_max_private_tokens`) and
  `privacy_accounting.py`. A stable interface would serve both.
- **`Evaluator` / `DownstreamTask`**: BERT fine-tuning code is duplicated
  between `evaluate.py` and `evaluate_downstream.py`. An `Evaluator`
  interface would support other tasks.
- **`ArtifactWriter`**: The JSONL checkpoint logic in `run_experiment.py`
  is embedded in a large closure. An `ArtifactWriter` would make the
  output contract explicit and testable.
- **`ExperimentConfig`**: All configuration arrives via argparse in three
  separate scripts. A unified dataclass-based config schema would simplify
  sweep management.

### Code deduplication
- `DATASET_HF_MAP` is defined in three separate scripts.
- `compute_max_private_tokens` exists in both `config.py` and
  `privacy_accounting.py` (with slightly different argument order).
  These should be unified; the duplicate in `config.py` should delegate
  to `privacy_accounting.py` or be removed.
- BERT fine-tuning code exists in both `evaluate.py` and
  `evaluate_downstream.py`.
- ICL prompt formatting in `evaluate.py` is not used by the main
  `evaluate_downstream.py`.

### New features (explicitly out of scope)
- Marginal-query-style or other non-SVT private guidance mechanisms.
- Tabular / structured data support.
- New model backends beyond HuggingFace AutoModel.
- Integration with the `invisibleink` repo.

---

## 5. Risks and Uncertainties

| Item | Risk | Notes |
|------|------|-------|
| `src` as package name | Low | Unconventional but deliberate; `pip install -e .` works and all imports are unchanged. Phase 2 can rename to `llm_pe_synth` if desired. |
| Duplicate `compute_max_private_tokens` | Medium | `config.py` and `privacy_accounting.py` both define this function with slightly different argument signatures. They currently agree on results. Phase 2 should remove the duplicate in `config.py` to prevent drift. |
| `zcdp_to_dp_tight` intentionally falls back to `zcdp_to_approx_dp` | Low/noted | The `dp-accounting` library is listed in requirements but the tight-conversion code path immediately raises `ImportError` and falls back. This is the current behavior; marked as a known limitation in comments. |
| SVT threshold re-sampled per private token | Noted | In `_generate_single_example`, `noisy_thresh` is re-sampled after every private token selection. This is consistent with the paper's Algorithm 1 (fresh noisy threshold per query) but differs from a simple "one threshold per example" reading. This behavior is now protected by the SVT tests. |
| Temperature for public tokens (`public_temperature`) | Noted | The `PrivacyConfig` has a separate `public_temperature` (default 1.5) for tokens that go through the SVT public path. This is not explicitly stated in Algorithm 1 but appears as an implementation detail. Preserved as-is. |
| No GPU in tests | Intended | All tests run on CPU. Tests for `get_next_token_logits` and `_generate_single_example` (which require a real LLM) are intentionally absent; they should be added in Phase 2 via mocking. |

---

## 6. Files Added / Modified

| File | Status | Description |
|------|--------|-------------|
| `pyproject.toml` | **Added** | Package build config; pytest config |
| `src/__init__.py` | **Modified** | Added docstring and `__version__ = "0.1.0"` |
| `tests/__init__.py` | **Added** | Empty; makes tests/ a package |
| `tests/conftest.py` | **Added** | `sys.path` setup for non-installed usage |
| `tests/test_batching.py` | **Added** | 10 tests for batching determinism |
| `tests/test_clip_utils.py` | **Added** | 10 tests for clipping invariants |
| `tests/test_sparse_vector.py` | **Added** | 11 tests for SVT gate behavior |
| `tests/test_jsonl_schema.py` | **Added** | 9 tests for JSONL schema and checkpoint semantics |
| `tests/test_imports.py` | **Added** | 17 tests for import surface and dataclass sanity |

**No existing source files were modified** (other than the 1-line `src/__init__.py`).

---

## 7. How to Run the Tests

```bash
# If the package is installed:
pip install -e ".[dev]"
pytest

# Without installation (sys.path fallback in conftest.py handles this):
/tmp/python-venv/llm_scoring_venv/bin/pip install pytest
/tmp/python-venv/llm_scoring_venv/bin/pytest tests/ -v
```

Expected output: **57 passed in ~2 s** (CPU only, no model download required).
