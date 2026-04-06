# Repository Audit Report

## 1. Executive summary

This repository appears to be a **prototype with paper-aligned components**, not yet a reusable framework. The core private-generation path is implemented in `src/generate.py` with explicit support for clipped-logit aggregation, a sparse-vector-style public/private token gate, fixed disjoint batching, checkpointed JSONL outputs, and downstream evaluation scripts for BERT fine-tuning and simple ICL. The current repo is much closer to an executable reproduction script stack than to a modular research framework.

Relative to `paper/local llm + Exponential.pdf`, the codebase implements most of the major ingredients of the **classification-text** version of the method, but not the full experimental scope of the paper. The strongest evidence is in `src/generate.py`, `src/clip_utils.py`, `src/sparse_vector.py`, `src/privacy_accounting.py`, and `scripts/run_experiment.py`. Important gaps remain: no structured JSON generation path, no MIT extraction tasks, no real package/config system, no tests, no dataset/mechanism abstraction boundary, and some implementation choices are ad hoc rather than paper-derived. Confidence is **medium-high** for the implemented AG News style pipeline and **medium** for paper-faithfulness overall, because some behavior is inferred from code comments plus a single saved run artifact, and I did not run a full generation job.

Main uncertainties:
- I did not run end-to-end generation due to cost.
- The PDF text extraction is sufficient for method comparison, but exact page formatting is noisy.
- I lightly inspected the neighboring `invisibleink` repo only to identify integration-relevant structure, not to audit its full algorithm.

## 2. Repository map

- `README.md`
  - High-level project description, setup, example `scripts/run_experiment.py` command, and claimed mapping from source files to method components.
- `requirements.txt`
  - Lightweight dependency list: `torch`, `transformers`, `datasets`, `accelerate`, `huggingface_hub`, `dp-accounting`, `scikit-learn`, Jupyter packages.
- `src/`
  - Core implementation modules.
  - `src/generate.py`: generation algorithm, batching, prompting, token loop.
  - `src/clip_utils.py`: clipped-logit aggregation.
  - `src/sparse_vector.py`: public/private gating via noisy thresholding.
  - `src/privacy_accounting.py`: privacy formulas and reporting.
  - `src/config.py`: dataclasses, prompt templates, sweep constants.
  - `src/evaluate.py`: JSONL IO, generation stats, BERT fine-tuning helper.
  - `src/__init__.py`: empty.
- `scripts/`
  - CLI-oriented orchestration, not a framework layer.
  - `scripts/run_experiment.py`: main end-to-end generation entry point with checkpoint/resume and optional evaluation.
  - `scripts/evaluate_downstream.py`: standalone downstream evaluation on saved synthetic data or real train baseline.
  - `scripts/sweep_hyperparams.py`: small hard-coded sweep runner with optional BERT evaluation.
  - `scripts/local_scripts/install_amazon.sh`, `scripts/local_scripts/install_pace.sh`: machine-specific setup scripts.
- `Notebook/demo.ipynb`
  - Manual notebook walkthrough of the AG News pipeline; repeats CLI logic in cells.
- `paper/`
  - `paper/local llm + Exponential.pdf`: primary method reference.
  - `paper/reports/agnews_eps3.0_s255_20260404_194803.jsonl`: saved generation artifact under `paper/reports/`, apparently duplicating an output-style JSONL.
- `data/`
  - Runtime artifacts, local model cache, HF dataset cache, generation outputs.
  - `data/models/gemma-2-2b-it/`: local Gemma model snapshot.
  - `data/outputs/agnews_eps3.0_s255_20260404_194803.jsonl`: concrete generated dataset artifact.
  - `data/outputs/agnews_eps3.0_s255_20260404_194803_eval_1k.json`: downstream eval result for that run.
- Neighbor repo: `/storage/home/hcoda1/1/ywei368/r-vzikas3-0/Yu-Project/SyncDP/invisibleink`
  - Relevant because it already has a package namespace `src/invink/`, a `pyproject.toml`, and `tests/`, which are absent here.
  - README indicates its core abstractions center around a generation package plus wrapper scripts, suggesting a possible future convergence target at the package/interface level rather than script level.

## 3. Current pipeline reconstruction

### Implemented primary pipeline: classification-text synthetic generation

The practical end-to-end path is:

1. Load dataset examples from Hugging Face in `scripts/run_experiment.py` via `load_dataset_examples()`.
   - Supported datasets are hard-coded in `DATASET_HF_MAP`:
     - `agnews`
     - `dbpedia`
     - `imdb`
     - `yelp`
     - `trec`
   - Rows are normalized to `{"text": ..., "label": ...}`.

2. Resolve privacy and generation configuration in `scripts/run_experiment.py`.
   - `PrivacyConfig` and `GenerationConfig` from `src/config.py`.
   - `delta` defaults to `1 / n`.
   - `max_private_tokens` is either user-provided or computed from privacy formulas using `compute_max_private_tokens(...)`.

3. Load model and tokenizer in `scripts/run_experiment.py:load_model()`.
   - Uses local path if present, otherwise `ModelConfig.hf_model_id`.
   - Uses `AutoTokenizer` and `AutoModelForCausalLM`.
   - Sets `tokenizer.pad_token = tokenizer.eos_token` if missing.

4. Partition source examples into fixed disjoint batches by label in `src/generate.py:partition_by_label()`.
   - Group by label first.
   - For each label group, set `num_batches = ceil(len(label_examples) / batch_size)`.
   - Assign each example to a bucket using SHA-256 hash of `"{label}\n{text}"`.
   - Drop empty buckets and sort examples within each bucket by text.
   - This is an explicit implementation of the paper’s Assumption 1 style fixed assignment, with label-aware hashing.

5. Build private prompts and one public prompt per batch in `src/generate.py:build_prompts()`.
   - Templates come from `src/config.py:PROMPT_TEMPLATES`.
   - If tokenizer is provided, prompts are wrapped with `tokenizer.apply_chat_template(...)`.
   - A dataset-specific `response_prefix` is appended.
   - The public prompt uses `public_seed`, not a true empty example.
   - This is a notable implementation choice because the comment in `src/config.py` says the paper uses an empty string and the seed is a “quality-of-life improvement”.

6. Generate token-by-token in `src/generate.py:generate_batch_examples()` and `_generate_single_example()`.
   - Shared per-batch private budget is `gen_config.max_private_tokens`.
   - The code generates multiple synthetic examples from a single batch until the budget is consumed.
   - For each decoding step:
     - Compute private next-token logits for all private prompts with current generated suffix via `get_next_token_logits()`.
     - If SVT enabled, compute public logits for the public prompt too.
     - Compare public/private distributions with `should_use_private_token(...)` from `src/sparse_vector.py`.
     - If private:
       - clip and aggregate private logits with `clip_and_aggregate(...)`
       - optionally apply top-k public mask via `_apply_top_k_filter(...)`
       - sample from `softmax(z_bar / tau)`
       - increment private-token counter
       - resample noisy threshold
     - If public:
       - optionally top-k restrict public logits
       - sample from `softmax(z_public / tau_public)`
       - no privacy budget consumed
   - Stop an example on EOS, length cap, or budget exhaustion.

7. Decode generated token IDs back to text in `src/generate.py:generate_synthetic_dataset()`.
   - Create `SyntheticExample` objects with per-example token counts.
   - Output examples are appended to an in-memory list.

8. Persist outputs in `scripts/run_experiment.py`.
   - JSONL file starts with `{"_metadata": ...}`.
   - Each generated example line includes text, label metadata, token counts, and batch metadata.
   - A trailing `{"_batch_complete": ...}` marker is written after each finished batch.
   - Resume logic reconstructs completed batches and ignores truncated JSONL tails.

9. Compute generation stats in `src/evaluate.py:compute_generation_stats()`.
   - Number of examples, label distribution, token totals, public-token fraction, max private tokens in an example.

10. Optionally run downstream evaluation in `scripts/run_experiment.py`.
   - Fine-tune BERT via `src.evaluate.finetune_and_evaluate(...)`.
   - Save eval JSON next to output JSONL.

### Implemented secondary pipeline: standalone downstream evaluation

`scripts/evaluate_downstream.py` provides a separate path:

1. Load real test split from HF.
2. Load either:
   - synthetic JSONL via `src.evaluate.load_synthetic_data(...)`, or
   - real train split with `--use_real_train`.
3. Evaluate in one of two modes:
   - `finetune`: BERT-base sequence classifier on synthetic train, evaluate on real test.
   - `icl`: local causal LM classification using synthetic examples as demonstrations.
4. Optionally save results JSON.

### Implemented secondary pipeline: hyperparameter sweep

`scripts/sweep_hyperparams.py`:

1. Loads train and optionally test data.
2. Loads the LLM once.
3. Iterates over `DEFAULT_GRID` of `(temperature, svt_threshold, svt_noise, top_k_vocab)`.
4. Computes `max_private_tokens` from privacy budget.
5. Runs `generate_synthetic_dataset(...)`.
6. Saves each synthetic JSONL plus a summary JSON.
7. Optionally fine-tunes BERT for each run.

### Inferred behavior from saved artifacts

The saved file `data/outputs/agnews_eps3.0_s255_20260404_194803.jsonl` confirms:

- checkpoint format is actively used;
- one run used:
  - `dataset=agnews`
  - `epsilon≈2.9912`
  - `batch_size=255`
  - `temperature=2.0`
  - `public_temperature=1.5`
  - `svt_threshold=0.5`
  - `svt_noise=0.2`
  - `top_k_vocab=1024`
  - `max_private_tokens=177`
- output count loaded by eval was `269` synthetic examples according to `data/outputs/agnews_eps3.0_s255_20260404_194803_eval_1k.json`.

The saved generations also show very mixed quality, with many malformed or assistant-style outputs. That is a quality finding, not an implementation finding, but it matters for future framework design because quality controls and output validators are currently absent.

## 4. Entry points and how to run

### `scripts/run_experiment.py`

- Path: `scripts/run_experiment.py`
- Purpose: main generation entry point
- Command pattern:
  - `python scripts/run_experiment.py --dataset agnews --epsilon 3.0 --batch_size 255 --clip_bound 10.0 --temperature 2.0 --top_k_vocab 1024 --public_temperature 1.5 --svt_threshold 0.5 --svt_noise 0.2`
- Key CLI knobs:
  - dataset/data: `--dataset`, `--num_examples`
  - privacy: `--epsilon`, `--delta`, `--batch_size`, `--clip_bound`, `--temperature`, `--public_temperature`, `--svt_threshold`, `--svt_noise`, `--max_private_tokens`
  - decoding: `--max_total_tokens`, `--top_k_vocab`
  - runtime: `--model_path`, `--device`, `--micro_batch_size`, `--seed`
  - outputs: `--output_dir`, `--output_path`, `--resume`
  - eval: `--evaluate`, `--bert_epochs`, `--max_test`
- Expected outputs:
  - checkpointed JSONL in `data/outputs/`
  - optional `_eval.json`
- Stability judgment: **most stable entry point in repo**

### `scripts/evaluate_downstream.py`

- Path: `scripts/evaluate_downstream.py`
- Purpose: evaluate previously generated synthetic data, or real-train baseline
- Command pattern:
  - `python scripts/evaluate_downstream.py --synthetic_path data/outputs/<run>.jsonl --dataset agnews --mode finetune`
  - `python scripts/evaluate_downstream.py --synthetic_path data/outputs/<run>.jsonl --dataset agnews --mode icl --model_path data/models/gemma-2-2b-it`
- Key CLI knobs:
  - input selection: `--synthetic_path`, `--use_real_train`, `--max_real_train`
  - mode: `--mode finetune|icl`
  - BERT params: `--epochs`, `--bert_batch_size`, `--bert_lr`, `--bert_max_length`, `--bert_model`
  - ICL params: `--num_shots`, `--model_path`
  - runtime/output: `--device`, `--max_test`, `--output_path`
- Expected outputs:
  - printed metrics
  - optional JSON file
- Stability judgment: **usable but partially duplicated**

### `scripts/sweep_hyperparams.py`

- Path: `scripts/sweep_hyperparams.py`
- Purpose: sweep a small hard-coded configuration grid
- Command pattern:
  - `python scripts/sweep_hyperparams.py --dataset agnews --epsilon 3.0`
- Key CLI knobs:
  - privacy/runtime: `--epsilon`, `--delta`, `--batch_size`, `--clip_bound`, `--max_total_tokens`
  - model/runtime: `--model_path`, `--device`, `--micro_batch_size`
  - sweep control: `--output_dir`, `--seed`, `--skip_eval`, `--bert_epochs`
- Expected outputs:
  - per-config JSONL files
  - `*_sweep_summary.json`
- Stability judgment: **experimental / one-off research helper**

### `Notebook/demo.ipynb`

- Path: `Notebook/demo.ipynb`
- Purpose: AG News notebook walkthrough
- Behavior:
  - reimplements core script logic in notebook cells
  - loads model and data, runs generation, saves JSONL, prints examples
- Stability judgment: **demo / exploratory**

## 5. Source-level implementation inventory

### `src/config.py`

- Purpose:
  - dataclass configs and dataset prompt templates
- Key items:
  - `PrivacyConfig`
  - `GenerationConfig`
  - `ModelConfig`
  - `DatasetConfig`
  - `PROMPT_TEMPLATES`
  - `HYPERPARAM_GRID`
  - `SVT_SETTINGS`
  - `compute_max_private_tokens(...)`
- Paper correspondence:
  - maps directly to algorithm hyperparameters and prompt templates from Appendix F / Table 7
- Reusability assessment:
  - **partially reusable**
  - dataclasses are useful seams, but `DatasetConfig` is mostly not used as a first-class contract
  - prompt templates are tightly coupled to a fixed set of text classification datasets
- Notable issue:
  - `compute_max_private_tokens(...)` is duplicated here and also in `src/privacy_accounting.py`

### `src/clip_utils.py`

- Purpose:
  - clipping and aggregation of logits
- Key functions:
  - `clip_logits(...)`
  - `clip_and_aggregate(...)`
- Paper correspondence:
  - strong match to Eq. (1) and Algorithm 1 line 16
- Reusability assessment:
  - **good candidate reusable primitive**
  - currently mechanism-specific, but cleanly isolated

### `src/sparse_vector.py`

- Purpose:
  - public/private token gating via noisy L1 distance
- Key functions:
  - `compute_distribution_distance(...)`
  - `sample_noisy_threshold(...)`
  - `should_use_private_token(...)`
- Paper correspondence:
  - strong match to Eq. (2) and Algorithm 1 gating logic
- Reusability assessment:
  - **moderately reusable**
  - currently assumes this exact SVT-style gate and L1-over-softmax distributions

### `src/privacy_accounting.py`

- Purpose:
  - privacy formulas and report generation
- Key functions:
  - `compute_rho_per_token(...)`
  - `compute_total_rho(...)`
  - `zcdp_to_approx_dp(...)`
  - `zcdp_to_dp_tight(...)`
  - `compute_epsilon(...)`
  - `compute_max_private_tokens(...)`
  - `privacy_report(...)`
- Paper correspondence:
  - implements Theorem 1 style formulas
- Reusability assessment:
  - **potentially reusable**
  - but tightly tied to this mechanism’s parameters
- Notable issue:
  - despite README comments about `dp-accounting`, `zcdp_to_dp_tight(...)` does not actually use the library and intentionally falls back to analytical conversion

### `src/generate.py`

- Purpose:
  - main algorithm implementation
- Key classes/functions:
  - `SyntheticExample`
  - `BatchDescriptor`
  - `assign_to_batch(...)`
  - `partition_by_label(...)`
  - `_format_prompt(...)`
  - `build_prompts(...)`
  - `_apply_top_k_filter(...)`
  - `get_next_token_logits(...)`
  - `_generate_single_example(...)`
  - `generate_batch_examples(...)`
  - `generate_one_example(...)`
  - `generate_synthetic_dataset(...)`
- Paper correspondence:
  - this is the main realization of Algorithm 1 plus batching assumption
- Reusability assessment:
  - **most important future extraction target**
  - currently mixes:
    - dataset semantics
    - prompt rendering
    - batching policy
    - model backend assumptions
    - token-selection mechanism
    - output object construction
- Notable implementation choices:
  - label-aware hash batching is explicit and paper-aligned
  - top-k restriction is deterministic from public logits
  - multiple examples per batch are generated until batch private budget exhausted
  - `max_examples = max(r, 10)` and `consecutive_no_private >= 3` are heuristic stop conditions, not paper-derived
  - noisy threshold is reset per generated example, which may deviate from a strict reading of Algorithm 1 pseudocode where the threshold is initialized once per batch loop

### `src/evaluate.py`

- Purpose:
  - synthetic-data IO, stats, BERT evaluation helper
- Key functions:
  - `format_icl_prompt(...)`
  - `save_synthetic_data(...)`
  - `load_synthetic_data(...)`
  - `compute_generation_stats(...)`
  - `finetune_and_evaluate(...)`
- Paper correspondence:
  - downstream BERT evaluation is aligned to paper goals
  - `format_icl_prompt(...)` claims Figure 2 alignment
- Reusability assessment:
  - **partially reusable**
  - mixes IO with evaluation and is not designed around pluggable tasks
- Notable issue:
  - ICL prompt construction is duplicated, and `format_icl_prompt(...)` is not what `scripts/evaluate_downstream.py` actually uses

### `scripts/run_experiment.py`

- Purpose:
  - orchestration, CLI, output naming, checkpointing, resume
- Reusability assessment:
  - **script-level coupling**
  - useful operational logic, but not a framework abstraction

### `scripts/evaluate_downstream.py`

- Purpose:
  - standalone eval runner
- Reusability assessment:
  - **task-specific**
  - evaluation logic is not abstracted behind task interfaces

### `scripts/sweep_hyperparams.py`

- Purpose:
  - limited hard-coded sweep
- Reusability assessment:
  - **low**
  - useful as a research script, not as framework infrastructure

## 6. Matching against the original approach

### What appears implemented

- Private token selection via clipped-logit aggregation and softmax / exponential-mechanism interpretation.
  - Evidence: `src/clip_utils.py`, `src/generate.py`, `src/privacy_accounting.py`.
  - Matches PDF Method section and Theorem 1 discussion.
- Fixed disjoint batching / partitioning satisfying Assumption 1.
  - Evidence: `src/generate.py:assign_to_batch()` and `partition_by_label()`.
  - Strongly aligned with PDF §4 discussion of hash-based assignment.
- Public-prompt path and sparse-vector-style gating.
  - Evidence: `src/sparse_vector.py`, `src/generate.py:_generate_single_example()`.
- Generation loop with private/public decoding and EOS/length caps.
  - Evidence: `src/generate.py`.
- Hyperparameters exposed in code.
  - Evidence: `src/config.py`, CLI args in `scripts/run_experiment.py`, sweep constants.
- Downstream evaluation and report-like summaries.
  - Evidence: `src/evaluate.py`, `scripts/evaluate_downstream.py`, `scripts/sweep_hyperparams.py`, saved eval JSON.

### What appears simplified

- Privacy accounting uses the simplified analytical conversion `epsilon = rho + sqrt(4 rho log(1/delta))`.
  - The PDF cites sharper zCDP analyses.
  - `src/privacy_accounting.py:zcdp_to_dp_tight()` is effectively a stub.
- Prompting is limited to a small fixed dataset set and mostly classification-style templates.
- The public prompt is a seeded example rather than the paper’s empty-string style, as noted in the comment block in `src/config.py`.
- Evaluation covers only:
  - BERT fine-tuning
  - a simple local-LLM ICL classifier
  - no broader report generation pipeline beyond JSON summaries and console output.

### What appears absent

- Structured data / WikiMoviesJSON generation path from paper §5.3 and Appendix prompts.
  - No JSON schema prompts or structured validators found in current repo.
- MIT-G and MIT-D extraction tasks from paper §5.1.
  - Not present in dataset maps.
- Explicit parsing/validation metrics for structured outputs.
- Strong experiment registry or config schema.
- Tests.
- Generalized mechanism interface for future variants.
- Tabular dataset support.

### Visible deviations from the original method

- The saved and default runnable path is centered on `temperature=2.0`; the paper’s Table 7 discusses `1.5`, `2`, and `2.25`.
- `scripts/sweep_hyperparams.py:DEFAULT_GRID` omits `2.25` despite `src/config.py:HYPERPARAM_GRID` including it.
- The code uses `top_k_vocab=1024` not only for the high-temperature `2.25` case described in the PDF, but also in the README example and saved AG News run at `temperature=2.0`.
- The batching implementation groups by label before hashing. This is consistent with the paper’s note that label-aware batching can improve quality, but is still a concrete design choice.
- `generate_batch_examples()` uses implementation heuristics to stop generation (`max_examples = max(r, 10)`, break after 3 zero-private examples). These do not appear in Algorithm 1.
- The noisy SVT threshold is reinitialized per example in `_generate_single_example()` rather than clearly once per batch loop as written in the pseudocode. This is **uncertain but likely a faithfulness deviation**.

### Implementation choices that may affect faithfulness

- The analytical privacy conversion may be looser than the paper’s sharper accounting.
- Prompt chat templating via `tokenizer.apply_chat_template(...)` is practical, but it changes the literal prompt format from the PDF appendix.
- Public seed examples may improve generation quality but change the exact public-prompt semantics.
- Top-k masking policy is deterministic and public-data-driven, which is privacy-safe, but the conditions under which it is applied differ from the paper’s stated sweep rationale.

## 7. Coupling and extensibility audit

### Where the code is currently coupled

- Dataset-specific formatting and prompt construction are coupled in `src/config.py:PROMPT_TEMPLATES` and `src/generate.py:build_prompts()`.
- Dataset loading is duplicated across scripts:
  - `scripts/run_experiment.py`
  - `scripts/evaluate_downstream.py`
  - `scripts/sweep_hyperparams.py`
- Mechanism logic is embedded directly into the generation loop in `src/generate.py`.
  - No boundary between:
    - public/private gate
    - clipped aggregation
    - candidate filtering
    - sampling policy
- Model backend and tokenization assumptions are embedded in `src/generate.py:get_next_token_logits()` and `scripts/run_experiment.py:load_model()`.
  - Assumes Hugging Face causal LM APIs and tokenizer behavior.
- Privacy accounting is mechanism-specific and not an independent service contract.
- Output schema is tied to `SyntheticExample` plus JSONL checkpoint markers.
  - No artifact abstraction for multiple output types.
- Downstream evaluation is task-specific and duplicated.
  - BERT fine-tuning exists in both `src/evaluate.py` and `scripts/evaluate_downstream.py`.
  - ICL formatting is inconsistent between helper and script.
- Experiment naming and storage conventions are script-local.
  - Output filename conventions live in `scripts/run_experiment.py`.
- There is no config-file driven workflow.
  - Everything is CLI or hard-coded constants.

### Candidate abstraction boundaries for a future refactor

- `DatasetAdapter`
  - Responsibility:
    - load train/test/private/public splits
    - normalize records
    - expose task metadata such as labels or schema
  - Why:
    - current dataset handling is duplicated and classification-centric

- `PromptBuilder`
  - Responsibility:
    - construct private prompts
    - construct public prompts
    - optionally apply chat templates
  - Why:
    - current prompt semantics are welded to dataset templates and generation logic

- `BatchingPolicy`
  - Responsibility:
    - assign records to fixed disjoint batches
    - optionally stratify by label or other attribute
  - Why:
    - current label-aware hash batching is important but should be swappable

- `Mechanism`
  - Responsibility:
    - given private logits and optional public logits, choose next-token distribution and account for privacy usage
  - Why:
    - current approach, InvisibleInk, and marginal-query variants will differ most here

- `PublicPolicy` or `TokenSelectionPolicy`
  - Responsibility:
    - implement SVT-style gate, public-only path, or candidate filtering policy
  - Why:
    - current SVT logic is distinct from clipped aggregation and should be replaceable

- `Generator`
  - Responsibility:
    - generic autoregressive loop over a prompt batch with backend hooks
  - Why:
    - decoding loop is currently tied to one mechanism

- `ModelBackend`
  - Responsibility:
    - tokenizer/model loading
    - batched next-token logits
  - Why:
    - future work may need different HF models or even different backends

- `PrivacyAccountant`
  - Responsibility:
    - mechanism-specific accounting object with stable interface
  - Why:
    - current accounting functions are useful but too entangled with this one theorem

- `DownstreamTask`
  - Responsibility:
    - BERT classification, ICL classification, extraction scoring, structured validity, tabular utility metrics
  - Why:
    - evaluation is currently not extensible

- `ExperimentConfig`
  - Responsibility:
    - single typed schema for dataset, model, mechanism, runtime, outputs
  - Why:
    - current args are spread across scripts and constants

- `ArtifactWriter`
  - Responsibility:
    - JSONL generation outputs
    - metrics JSON
    - sweep summary tables
    - resumable checkpoints
  - Why:
    - output handling is operationally important and currently script-specific

### Pain points to preserve during refactor

- Resume/checkpoint behavior in `scripts/run_experiment.py` is valuable and should not be lost.
- Stable batch identifiers via `BatchDescriptor` are a good operational primitive.
- Public-data-derived top-k filtering is a useful mechanism subcomponent that should remain deterministic and separable.

## 8. Main blockers for the future refactor

### 1. No first-class package architecture

- Type: architecture issue
- Evidence:
  - no local `pyproject.toml`
  - scripts mutate `sys.path` to import `src`
- Why it blocks refactor:
  - hard to create stable interfaces, tests, and multiple runnable methods without a package boundary

### 2. Generation loop hard-codes method behavior

- Type: missing abstraction / algorithm-interface mismatch
- Evidence:
  - `src/generate.py` combines batching, prompts, logits retrieval, SVT, clipping, top-k, decoding, and output creation
- Why it blocks refactor:
  - InvisibleInk and marginal-query variants will likely need different score computations and candidate policies without rewriting the full loop

### 3. Dataset handling is duplicated and classification-specific

- Type: code organization issue
- Evidence:
  - multiple `DATASET_HF_MAP` definitions
  - prompt templates assume labeled text classification
- Why it blocks refactor:
  - tabular datasets, extraction datasets, and structured JSON tasks do not fit this contract cleanly

### 4. Prompting is not abstracted from task semantics

- Type: missing abstraction
- Evidence:
  - `PROMPT_TEMPLATES` mixes labels, response prefixes, and public seed examples in one static dict
- Why it blocks refactor:
  - future tasks need schema prompts, dataset descriptions, different public prompts, or non-text continuations

### 5. Privacy accounting is tied to one theorem and partially duplicated

- Type: architecture issue
- Evidence:
  - duplicated `compute_max_private_tokens(...)`
  - `privacy_accounting.py` is mechanism-specific
- Why it blocks refactor:
  - different mechanisms will need different privacy cost models and accounting objects

### 6. Evaluation is narrow and not task-pluggable

- Type: missing abstraction
- Evidence:
  - only BERT classification and simple ICL implemented
  - no structured-output validator
  - no tabular utility evaluation
- Why it blocks refactor:
  - a reusable framework needs task/evaluator registration, not script duplication

### 7. No durable config schema or experiment registry

- Type: config issue
- Evidence:
  - everything is CLI or hard-coded constants
  - no YAML/TOML/JSON experiment configs
- Why it blocks refactor:
  - comparing many methods and datasets becomes fragile and script-heavy

### 8. No tests around current contracts

- Type: missing test surface
- Evidence:
  - no `tests/` directory in this repo
- Why it blocks refactor:
  - behavior-preserving extraction will be risky, especially around privacy-sensitive batching/accounting logic

### 9. Output schema is operationally useful but method-specific

- Type: code organization issue
- Evidence:
  - JSONL schema assumes `SyntheticExample`, label metadata, batch checkpoint markers
- Why it blocks refactor:
  - structured/tabular outputs or alternate mechanisms may require different per-record metadata without breaking existing tools

### 10. Reproducibility surface is incomplete

- Type: reproducibility issue
- Evidence:
  - machine-specific install scripts
  - no lockfile
  - no package metadata
  - one saved run shows poor quality but there is no validator layer or run manifest beyond JSON header
- Why it blocks refactor:
  - difficult to compare current method and future variants reliably

## 9. Suggested next steps (no implementation yet)

### Step 1: isolate current pipeline contracts without changing behavior

- Document the effective contracts already present:
  - normalized example record
  - batch descriptor
  - synthetic example record
  - privacy report
  - output checkpoint format
- Add thin wrappers only after the contracts are written down.

### Step 2: extract dataset and prompt interfaces

- Introduce a `DatasetAdapter` contract for loading and normalization.
- Introduce a `PromptBuilder` contract for private/public prompt rendering.
- Keep current AG News style behavior as the first adapter implementation.

### Step 3: wrap the current mechanism behind a stable API

- Separate:
  - batching policy
  - next-token mechanism
  - privacy accountant
  - model backend
- Preserve current outputs exactly while moving logic behind interfaces.

### Step 4: isolate downstream evaluation

- Define `DownstreamTask` or `Evaluator` interfaces.
- Move BERT fine-tuning and ICL into task modules.
- Add a path for structured validity metrics even before supporting new generation methods.

### Step 5: create a minimal experiment config schema

- Centralize dataset, model, mechanism, runtime, and artifact settings.
- Use one config object or file format rather than repeated CLI maps across scripts.

### Step 6: preserve compatibility with existing scripts

- Keep `scripts/run_experiment.py` and `scripts/evaluate_downstream.py` as wrappers over the new interfaces.
- Avoid breaking current JSONL checkpoint format in the first pass.

### Step 7: add a minimal test surface before larger method additions

- Priority tests:
  - batch assignment determinism
  - clipping invariants
  - SVT gate behavior under deterministic seeds
  - resume/checkpoint round-trip
  - output schema loading

### Step 8: use the neighboring `invisibleink` repo as an interface benchmark, not as a direct merge target

- `invisibleink` already has:
  - package layout
  - `pyproject.toml`
  - `tests/`
- Compare interface shape, not algorithm internals, when designing the shared framework boundary.

## Appendix: notable concrete findings

- `src/config.py:DatasetConfig` exists but is not the organizing abstraction for data flow.
- `src/generate.py:generate_one_example()` is retained only for backward compatibility; the batch-level multi-example path is the actual current algorithm path.
- `src/evaluate.py:format_icl_prompt()` is not the prompt function actually used by `scripts/evaluate_downstream.py`.
- `data/outputs/agnews_eps3.0_s255_20260404_194803_eval_1k.json` reports `accuracy = 0.246` and the classifier predicts only one class, suggesting the repo has generation implemented but little quality-control infrastructure.
- The current repo has no tests, while the neighboring `invisibleink` repo already has a `tests/` directory and packaged source tree under `src/invink/`.
