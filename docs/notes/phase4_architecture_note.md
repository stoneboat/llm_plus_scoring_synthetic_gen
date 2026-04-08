# Phase 4 Architecture Note

**Version:** 0.3.6 → 0.4.0  
**Date:** 2026-04-07

## Motivation

After Phase 3.5b the package had three remaining structural problems:

1. **`src/generate.py` was both an algorithm file and a compat re-export hub.**
   The dataset-level orchestration loop, the batch outer-loop, and the
   `SyntheticExample` dataclass all lived in the same file as a growing set
   of backward-compatibility wrappers.

2. **`scripts/run_experiment.py` was too fat.**
   All JSONL checkpoint writing, resume-state loading, metadata construction,
   and output-path resolution lived as private helpers inside the script,
   making them impossible to reuse in sweep scripts or tests without
   copy-pasting.

3. **No clear artifact boundary.**
   The checkpoint JSONL format was only documented implicitly in the script;
   the functions that wrote and read it were not independently testable.

Phase 4 extracts two new packages:

- **`src/runtime/`** — generation orchestration (the "compute" boundary)
- **`src/artifacts/`** — checkpoint I/O and metadata (the "storage" boundary)

---

## Package Structure After Phase 4

```
src/
├── artifacts/
│   ├── __init__.py          # flat surface: append_line, batch_record,
│   │                        #   write_metadata_header, append_completed_batch,
│   │                        #   load_resume_state, build_run_metadata
│   ├── jsonl_writer.py      # crash-safe JSONL writing
│   ├── resume.py            # checkpoint parsing / resume-state loading
│   └── metadata.py          # build_run_metadata (explicit-param, no argparse)
│
├── runtime/
│   ├── __init__.py          # flat surface: SyntheticExample,
│   │                        #   run_batch_generation, run_dataset_generation
│   └── generation.py        # SyntheticExample dataclass;
│                            #   Algorithm 1 batch loop;
│                            #   dataset-level orchestration loop
│
├── generate.py              # thin compat hub: re-exports from runtime/,
│                            #   backward-compat wrappers (get_next_token_logits,
│                            #   _generate_single_example, generate_batch_examples,
│                            #   generate_one_example);
│                            #   generate_synthetic_dataset delegates to
│                            #   run_dataset_generation
│
├── backends/                # Phase 3 — unchanged
├── mechanisms/              # Phase 3 — unchanged
├── privacy/                 # Phase 3.5/3.5b — unchanged
├── batching/                # Phase 2 — unchanged
├── prompts/                 # Phase 2 — unchanged
└── ...
```

---

## Responsibility Assignments

### `src/runtime/generation.py`

Owns:

| Responsibility | Function |
|---|---|
| Per-example result type | `SyntheticExample` dataclass |
| Algorithm 1 batch outer loop | `run_batch_generation(mechanism, backend, private_prompts, public_prompt, gen_config)` |
| Dataset iteration and orchestration | `run_dataset_generation(backend, mechanism, examples, dataset_name, gen_config, ...)` |

`run_dataset_generation` is responsible for:
- Partitioning examples into disjoint batches (`partition_by_label`)
- Computing stable batch IDs (SHA-256 of dataset/label/texts)
- Constructing `BatchDescriptor` objects
- Skipping already-completed batch IDs
- Building per-batch prompts (`build_prompts`)
- Calling `run_batch_generation` for each batch
- Decoding token IDs into `SyntheticExample` objects
- Calling the `batch_callback` (which triggers checkpoint writing)
- Tracking `max_batch_private_tokens` for the final epsilon print

`run_dataset_generation` does **not** create the backend or mechanism, manage
`padding_side`, or resolve `delta` — those setup steps remain in
`generate_synthetic_dataset` in `src/generate.py`.

### `src/artifacts/jsonl_writer.py`

Owns the crash-safe JSONL write path (checkpoint_format=1):

| Function | Purpose |
|---|---|
| `append_line(handle, record)` | Write one JSON record + fsync |
| `batch_record(ex, descriptor)` | Assemble per-example dict (no source_batch_size) |
| `write_metadata_header(path, metadata)` | Create file, write `_metadata` line |
| `append_completed_batch(path, descriptor, examples)` | Write example records + `_batch_complete` marker |

### `src/artifacts/resume.py`

Owns checkpoint parsing:

| Function | Returns |
|---|---|
| `load_resume_state(path)` | `(metadata, loaded_examples, completed_batch_ids, batch_descriptors)` |

Only examples from **completed** batches (those with a `_batch_complete`
marker) are returned.  Truncated tails (from crashes mid-batch) are silently
ignored.

### `src/artifacts/metadata.py`

Owns run-metadata construction:

| Function | Purpose |
|---|---|
| `build_run_metadata(dataset, epsilon, delta, ...)` | Returns the `_metadata` payload dict |

All parameters are explicit (no `argparse.Namespace`), making the function
reusable from sweep scripts, notebooks, and tests alike.

---

## Dependency Graph

```
scripts/run_experiment.py
  ├── src.generate              (generate_synthetic_dataset, SyntheticExample)
  ├── src.artifacts             (write_metadata_header, append_completed_batch,
  │                              load_resume_state, build_run_metadata)
  └── src.{config,privacy_accounting,evaluate,datasets}

src.generate
  ├── src.runtime.generation    (SyntheticExample, run_batch_generation,
  │                              run_dataset_generation)
  ├── src.batching.*            (re-exports)
  ├── src.prompts.*             (re-exports)
  ├── src.backends.*            (re-exports)
  └── src.mechanisms.*          (re-exports)

src.runtime.generation
  ├── src.batching.*
  ├── src.prompts.*
  ├── src.backends.*
  ├── src.mechanisms.*
  ├── src.privacy_accounting
  └── src.config

src.artifacts.{jsonl_writer,resume}
  ├── src.batching.base         (BatchDescriptor)
  └── src.runtime.generation    (SyntheticExample)

src.artifacts.metadata          (no src imports — pure standard library)
```

No circular dependencies.

---

## What Did Not Change

1. **Algorithm behavior**: the private prediction algorithm (clipped-logit
   aggregation, exponential mechanism, SVT gating) is unchanged.
2. **JSONL schema**: checkpoint_format=1 schema is preserved byte-for-byte.
   Existing checkpoint files can be resumed without modification.
3. **Batch IDs**: the SHA-256 computation is identical.
4. **Public API of `src/generate.py`**: all previously exported names remain
   importable from `src.generate`.
5. **Script behavior**: `scripts/run_experiment.py` and
   `scripts/sweep_hyperparams.py` produce identical output.
6. **All 212 pre-Phase-4 tests**: still pass.

---

## New Tests

| File | Count | Coverage |
|---|---|---|
| `tests/test_artifacts.py` | 18 | jsonl_writer (7), resume (5), metadata (3), imports (3) |
| `tests/test_runtime.py` | 8 | SyntheticExample (1), run_batch_generation (7) |
| `tests/test_imports.py` | +2 | runtime and artifacts import smoke tests |

Total: **240 tests** (up from 212).
