# Privacy Architecture Planning Report

## 1. Executive summary

The current repository implements one concrete privacy-aware generation method: the private-prediction algorithm from Amin et al. (2024), centered on `src/generate.py`, `src/mechanisms/private_prediction.py`, `src/sparse_vector.py`, `src/clip_utils.py`, and `src/privacy_accounting.py`. The runtime is already moderately modular at the batching, prompting, backend, and mechanism layers, but the privacy layer is still a flat helper module with theorem-specific formulas and a small reporting/planning surface.

Today’s privacy layer is adequate for the current mechanism because the runtime does not require incremental stepwise accounting. It uses closed-form planning before generation, simple summary reporting during/after generation, and token counts emitted in output metadata. The repository therefore does not yet need a heavy privacy framework.

My recommendation is to move toward a `src/privacy/` package, but to do so in a staged way. The right near-term target is a medium-strength abstraction:

- separate mechanism-specific analysis from generic zCDP or approximate-DP utilities,
- separate planning and reporting from formula definitions,
- keep compatibility wrappers for `src/privacy_accounting.py` and `src/config.compute_max_private_tokens`,
- defer a full event/accountant hierarchy until the repo has a second privacy mechanism or a real need for incremental composition.

In short: adopt a clearer privacy boundary now, but do not force a full abstract bound/event/accountant system into a codebase that currently performs batch-level closed-form accounting for a single theorem.

## 2. Current privacy layer audit

### Current privacy-related files and responsibilities

#### `src/privacy_accounting.py`

This is the current authoritative privacy helper module. It contains six distinct responsibilities in one file:

- mechanism-specific cost formula:
  - `compute_rho_per_token(...)`
  - `compute_total_rho(...)`
- zCDP to approximate-DP conversion:
  - `zcdp_to_approx_dp(...)`
  - `zcdp_to_dp_tight(...)`
- budget planning:
  - `compute_epsilon(...)`
  - `compute_max_private_tokens(...)`
- reporting:
  - `privacy_report(...)`

The file is explicitly theorem-specific. Its module docstring states it implements Theorem 1 from Amin et al. (2024), and the main formula is:

- `rho = r * (1/2 * (c/(s*tau))^2 + 2/(s*sigma)^2)`

This file mixes generic concepts with method-specific ones:

- Generic enough to survive:
  - `zcdp_to_approx_dp(...)`
  - the conceptual role of `privacy_report(...)`, though not necessarily its exact shape
- Mechanism-specific:
  - `compute_rho_per_token(...)`
  - `compute_total_rho(...)`
  - `compute_epsilon(...)` as currently parameterized
  - `compute_max_private_tokens(...)` as currently parameterized

`zcdp_to_dp_tight(...)` is notable because it advertises tighter accounting via `dp-accounting`, but currently falls back to the analytical conversion. So the module already contains a weak implicit split between representation/conversion and mechanism analysis, but it is not expressed structurally.

#### `src/config.py`

This file contains:

- `PrivacyConfig`
- `GenerationConfig`
- model and dataset configs
- `compute_max_private_tokens(...)` as a backward-compat wrapper

The wrapper in `src/config.py` preserves an older argument order and delegates to `src/privacy_accounting.compute_max_private_tokens(...)`. That wrapper is important for migration because it proves there are already compatibility constraints on privacy helper APIs.

#### `src/sparse_vector.py`

This is not an accounting module, but it is privacy-related mechanism logic. It contains:

- `compute_distribution_distance(...)`
- `sample_noisy_threshold(...)`
- `should_use_private_token(...)`

These functions implement the SVT gate used by `PrivatePredictionMechanism`. They are mechanism primitives, not privacy accounting utilities, but they are part of mechanism-specific privacy analysis because they explain why some tokens count as public and others count as private.

#### `src/mechanisms/private_prediction.py`

`PrivatePredictionMechanism.generate_example(...)` uses:

- `clip_and_aggregate(...)` from `src/clip_utils.py`
- SVT helpers from `src/sparse_vector.py`
- per-example counters:
  - `private_token_count`
  - `public_token_count`

This class does not call `src/privacy_accounting.py` directly. Instead, it emits the quantities that the accounting layer consumes later. That means there is already an implicit interface:

- mechanism runtime produces privacy-relevant event counts
- orchestration translates those counts into final guarantees

#### `src/generate.py`

This is the main privacy-orchestration caller. It imports:

- `compute_max_private_tokens`
- `compute_epsilon`
- `privacy_report`

Current uses:

- before generation:
  - `privacy_report(...)` is printed for the configured `max_private_tokens`
- during generation:
  - no incremental accountant is maintained
- after generation:
  - `compute_epsilon(...)` is called using the worst-case batch private-token count

This file also owns the operational notion of the privacy budget:

- `gen_config.max_private_tokens` is the batch private-token budget
- `_run_batch_generation(...)` spends that budget by repeatedly calling `mechanism.generate_example(...)`

This is important: the runtime currently operates on a planned token budget, not on composed privacy objects.

#### `scripts/run_experiment.py`

This is the main CLI-level privacy planning/reporting surface.

It:

- computes `delta`
- resolves `svt_noise` depending on whether SVT is enabled
- computes `max_private_tokens` if not user-specified
- prints a privacy report
- stores privacy metadata in the JSONL header

Its metadata currently includes:

- `epsilon`
- `delta`
- `batch_size`
- `clip_bound`
- `temperature`
- `public_temperature`
- `svt_threshold`
- `svt_noise`
- `top_k_vocab`
- `max_private_tokens`

So reporting already exists in two forms:

- console summary
- serialized run metadata

#### `scripts/sweep_hyperparams.py`

This script directly imports both:

- `src.privacy_accounting.compute_max_private_tokens`
- `src.config.compute_max_private_tokens` as `cfg_max_priv`

but only appears to use the authoritative import for planning. This script confirms:

- planning is done outside the mechanism
- privacy helpers are considered script-safe operational utilities
- there is no accountant object passed through the runtime

#### `src/evaluate.py`

This file is not a privacy module, but it persists privacy-relevant output data:

- per-example `num_private_tokens`
- per-example `num_public_tokens`
- aggregate generation stats such as total public/private tokens

This matters because any future privacy reporting design should remain compatible with these artifacts.

### Current formulas and conversions

Current formulas in `src/privacy_accounting.py`:

- per-private-token zCDP cost:
  - exponential-mechanism term plus optional SVT term
- sequential composition over private tokens:
  - total `rho = r * rho_per_token`
- approximate-DP conversion:
  - `epsilon = rho + sqrt(4 * rho * log(1/delta))`

Current limitations:

- formulas are written for one mechanism family
- parameter names are algorithm-specific
- the “tight” zCDP conversion is a stub/fallback path
- there is no explicit privacy-bound type
- there is no explicit event object
- there is no reusable accountant abstraction

### Current call graph and coupling points

The effective privacy call flow today is:

1. scripts choose `epsilon`, `delta`, mechanism hyperparameters
2. planning helper computes `max_private_tokens`
3. `generate_synthetic_dataset(...)` runs with that budget
4. `PrivatePredictionMechanism.generate_example(...)` returns private/public token counts
5. orchestration aggregates counts
6. `compute_epsilon(...)` is called after generation for the realized worst-case batch
7. metadata and stats are written to JSONL or printed

The main coupling points are:

- `src/generate.py` depends on privacy helper functions by name
- scripts depend on argument-order compatibility in `src/config.compute_max_private_tokens(...)`
- tests depend on `src/privacy_accounting.py` remaining the authoritative implementation

### Is there already an implicit event/accountant/planning split?

Yes, but only implicitly:

- implicit event:
  - one private token
- implicit mechanism analysis:
  - `compute_rho_per_token(...)`
- implicit composition:
  - multiply by `num_private_tokens`
- implicit bound representation:
  - raw `rho` float
- implicit planner:
  - `compute_max_private_tokens(...)`
- implicit reporting:
  - `privacy_report(...)` and script metadata

So the design direction proposed in the prompt is conceptually aligned with what the repo already does. The missing piece is not the concept. The missing piece is a clearer architectural boundary.

## 3. Evaluation of the abstract architecture

### Bounds

Conceptually, a bound is the representation of a privacy guarantee:

- zCDP bound
- RDP profile
- approximate DP bound

This fits the repo in principle because the current code already manipulates:

- `rho` as a zCDP quantity
- `(epsilon, delta)` as an approximate-DP quantity

What problem it solves:

- prevents helper functions from passing around anonymous floats with implicit meaning
- gives conversion routines a natural home
- makes future support for tighter conversions or different privacy representations easier

Is it justified now:

- a lightweight version is justified now
- a large type hierarchy is not justified now

Simplest repo-compatible version:

- a small `bounds.py` with simple dataclasses such as:
  - `ZCDPBound(rho: float)`
  - `ApproxDPBound(epsilon: float, delta: float)`

What is too much for now:

- a rich algebra of bound objects
- full RDP profile support unless a future mechanism actually needs it

Recommendation:

- implement lightweight bound types eventually
- defer `RDPProfile` unless the repo starts using real `dp-accounting`

### Events

Conceptually, an event is one privacy-cost unit before composition:

- one private token
- one noisy query
- one mechanism invocation

What problem it solves:

- makes composition explicit
- makes per-step accounting possible
- provides a path to mechanism-agnostic accounting

Does it fit this repo now:

- only partially

The current runtime does not produce or consume event objects. It consumes closed-form counts. For this repo today, “event = one private token under the private-prediction theorem” is real, but turning that into a generic event framework now would add abstraction that nothing uses.

Simplest justified version now:

- do not implement a general `events.py` first
- if needed, introduce a very small mechanism-specific event record later, such as a private-prediction token event or a simple count summary

What is too much now:

- a reusable event taxonomy
- event subclasses for every possible privacy primitive

Recommendation:

- defer generic events
- treat event abstractions as a phase-2 privacy refactor, not the first extraction

### Accountants

Conceptually, an accountant composes privacy objects and answers queries such as epsilon at delta.

What problem it solves:

- centralizes composition logic
- supports incremental accounting
- supports future multiple-mechanism composition

Does the current repo need it right now:

- not strongly

Current runtime needs:

- closed-form planning before generation
- one final epsilon query after generation

It does not currently need:

- step-by-step composition during decoding
- mixing heterogeneous privacy events
- online budget exhaustion checks based on an accountant state

Simplest justified version now:

- a functional accountant utility, not necessarily a stateful class
- e.g. composition helpers over `ZCDPBound`

Medium-strength version:

- `accountants/zcdp.py` with helpers or a tiny `ZCDPAccountant`

What is too much now:

- a generalized accountant registry
- a multi-backend accountant system
- event-stream accounting when the runtime still uses theorem-specific closed forms

Recommendation:

- introduce an accountant boundary only after splitting out generic zCDP conversion/composition utilities
- keep it minimal at first

### Mechanism-specific analyses

Conceptually, this maps algorithm parameters to privacy cost.

What problem it solves:

- moves theorem-specific formulas out of the generic privacy surface
- makes it easier to add a second mechanism later
- clarifies what part of the privacy layer is paper-specific

Does it fit this repo now:

- yes, strongly

This is the most justified part of the abstract architecture. `src/privacy_accounting.py` is already a mechanism-specific analysis module wearing a generic name.

Simplest justified version now:

- one analysis file for private prediction, e.g.:
  - `src/privacy/analyses/private_prediction.py`

What is too much now:

- a deep class hierarchy for analyses
- trying to generalize before a second mechanism exists

Recommendation:

- this should be the first architectural split

### Planning

Conceptually, planning answers operational questions such as:

- how many private tokens fit in the budget?
- what privacy report should be shown before generation?

Does it fit this repo now:

- yes

The repo already has concrete planning behavior in scripts and `src/privacy_accounting.py`. Planning is one of the strongest justified separations because it is already used independently of runtime generation.

Simplest justified version now:

- `planning.py` with:
  - “budget to max-private-tokens”
  - “planned privacy summary”

Recommendation:

- implement this as an early extraction

### Reporting

Conceptually, reporting serializes or summarizes privacy guarantees for:

- console logs
- output metadata
- sweep summaries

Does it fit this repo now:

- yes

The repo already writes privacy metadata in `scripts/run_experiment.py`, prints privacy summaries in `src/generate.py` and the scripts, and stores per-example token counts in JSONL.

Simplest justified version now:

- a reporting helper that returns structured metadata payloads and human-readable summaries

What is too much now:

- a separate reporting framework
- format registries

Recommendation:

- split reporting from formulas, but keep it lightweight

## 4. Recommended target structure

### Recommended top-level folder name

Use `src/privacy/`.

Why this is the best fit:

- short and clear
- broad enough to cover analysis, planning, conversion, and reporting
- consistent with existing top-level domains like `src/mechanisms/`, `src/backends/`, `src/batching/`, `src/prompts/`

Why not `privacy_analysis/`:

- too narrow for planning and reporting
- suggests read-only theorem work rather than runtime support

Why not `models/`:

- this repo already uses “model” for ML inference objects
- `models/` would be ambiguous and likely misleading

### Recommended staged target tree

The recommended near-term target is:

```text
src/privacy/
├── __init__.py
├── bounds.py
├── planning.py
├── reporting.py
├── conversions.py
└── analyses/
    ├── __init__.py
    └── private_prediction.py
```

Short explanation of each file:

- `src/privacy/__init__.py`
  - stable import surface for the privacy package
- `src/privacy/bounds.py`
  - lightweight privacy-bound dataclasses such as `ZCDPBound` and `ApproxDPBound`
- `src/privacy/conversions.py`
  - generic conversion/composition helpers such as zCDP to approximate-DP
- `src/privacy/planning.py`
  - operational helpers such as max-private-token planning
- `src/privacy/reporting.py`
  - structured privacy summaries and metadata payload helpers
- `src/privacy/analyses/private_prediction.py`
  - theorem-specific formulas for the current mechanism

### What should be deferred

Defer this fuller structure until the repo needs it:

```text
src/privacy/
├── bounds.py
├── events.py
├── accountants/
│   ├── __init__.py
│   ├── base.py
│   └── zcdp.py
└── analyses/
    └── private_prediction.py
```

Reasons to defer:

- there is only one mechanism-specific analysis today
- no runtime caller needs event objects
- no caller needs a stateful accountant
- the current code uses theorem-specific closed-form accounting, not event-stream composition

## 5. Mapping from current code to proposed structure

### Current `src/privacy_accounting.py` to proposed structure

Move or conceptually remap current functions as follows:

#### Mechanism-specific analysis

- current:
  - `src/privacy_accounting.py:compute_rho_per_token`
  - `src/privacy_accounting.py:compute_total_rho`
- proposed:
  - `src/privacy/analyses/private_prediction.py`

Reason:

- these formulas are specific to the private-prediction theorem and its parameters:
  - `clip_bound`
  - `batch_size`
  - `temperature`
  - `svt_noise`
  - `num_private_tokens`

#### Generic conversion

- current:
  - `src/privacy_accounting.py:zcdp_to_approx_dp`
  - `src/privacy_accounting.py:zcdp_to_dp_tight`
- proposed:
  - `src/privacy/conversions.py`

Reason:

- these are generic bound-conversion utilities, not private-prediction-specific logic

#### Planning

- current:
  - `src/privacy_accounting.py:compute_epsilon`
  - `src/privacy_accounting.py:compute_max_private_tokens`
- proposed:
  - `src/privacy/planning.py`
  - potentially with thin delegation into `analyses/private_prediction.py`

Reason:

- these answer operational planning questions, not raw theorem primitives

Possible split:

- `analyses/private_prediction.py`
  - private-prediction cost formulas
- `planning.py`
  - “given target epsilon, compute max tokens”
  - “given observed private tokens, compute epsilon”

#### Reporting

- current:
  - `src/privacy_accounting.py:privacy_report`
- proposed:
  - `src/privacy/reporting.py`

Reason:

- this function packages values for presentation and metadata
- it is neither a primitive theorem formula nor a generic conversion routine

### Bound representation mapping

There is no explicit bound class today. The current implicit representations are:

- `rho: float`
- `epsilon: float`
- `delta: float`

Proposed mapping:

- `rho` should become a `ZCDPBound`
- `(epsilon, delta)` should become an `ApproxDPBound`

This can be introduced gradually without changing runtime behavior first.

### Compatibility wrappers that should remain

These should remain during migration:

- `src/privacy_accounting.py`
  - keep as a compatibility facade that re-exports or delegates to the new `src/privacy/` package
- `src/config.py:compute_max_private_tokens`
  - preserve current argument order for external callers and tests

Why:

- tests explicitly protect wrapper parity and argument order
- scripts and historical notes treat `src/privacy_accounting.py` as the authoritative path

### Mechanism/runtime mapping

Current mechanism/runtime interaction should remain conceptually unchanged:

- `PrivatePredictionMechanism.generate_example(...)`
  - still returns token counts
- `src/generate.py`
  - still decides how to aggregate those counts at batch scope
- planning/reporting layer
  - still consumes counts and parameters

This means the first privacy refactor should not push an accountant into `PrivatePredictionMechanism`. That would be a larger algorithm/runtime redesign than the repo currently needs.

## 6. Migration plan

### Phase 1: Structural split without behavior change

Goal:

- separate theorem-specific formulas from generic conversions, planning, and reporting

Likely files affected:

- new `src/privacy/` package
- `src/privacy_accounting.py`
- imports in `src/generate.py`
- imports in `scripts/run_experiment.py`
- imports in `scripts/sweep_hyperparams.py`
- tests for accounting imports

Expected risk:

- low to medium

Behavior to preserve:

- exact formulas
- exact printed and serialized privacy values
- `src/privacy_accounting.py` import path
- `src/config.compute_max_private_tokens(...)` behavior and argument order

### Phase 2: Introduce lightweight bound objects

Goal:

- replace anonymous `rho` and `(epsilon, delta)` tuples/floats with clearer internal representations where useful

Likely files affected:

- `src/privacy/bounds.py`
- `src/privacy/conversions.py`
- `src/privacy/planning.py`
- tests

Expected risk:

- medium

Behavior to preserve:

- same numeric outputs
- no forced runtime API changes for generation code

Notes:

- this phase should stay internal first
- the scripts can continue receiving plain floats

### Phase 3: Normalize reporting and metadata helpers

Goal:

- centralize privacy metadata construction and summary formatting

Likely files affected:

- `src/privacy/reporting.py`
- `scripts/run_experiment.py`
- possibly `src/generate.py`
- possibly `src/evaluate.py` if metadata serialization helpers are shared

Expected risk:

- low

Behavior to preserve:

- JSONL metadata schema
- current resume compatibility
- current console summaries

### Phase 4: Reassess whether events/accountants are actually needed

Goal:

- decide based on concrete future requirements, not speculation

Trigger conditions for doing this:

- a second DP mechanism is added
- a mechanism requires heterogeneous event composition
- generation needs online budget tracking from accountant state
- the repo actually adopts `dp-accounting` profiles beyond simple zCDP formulas

Expected risk:

- medium to high if done prematurely

Behavior to preserve:

- current private-prediction behavior
- current script interfaces

### What should not be touched yet

- `src/mechanisms/private_prediction.py` runtime logic
- `src/sparse_vector.py` algorithm behavior
- JSONL output format
- batching/prompt/backend package layout

Reason:

- these are not the architectural bottlenecks for the privacy boundary
- moving them now would mix architectural cleanup with algorithm changes

## 7. Recommendation

The repo should not adopt the most abstract privacy architecture immediately. The conceptual model in the prompt is sound, but a full bound/event/accountant architecture would be ahead of the repo’s actual needs.

The right move now is a lighter version:

- create a `src/privacy/` package
- split private-prediction-specific formulas into `analyses/private_prediction.py`
- split generic zCDP conversion into `conversions.py`
- split operational helpers into `planning.py`
- split summary serialization into `reporting.py`
- keep `src/privacy_accounting.py` as a compatibility wrapper

This gives the repo a real privacy boundary without pretending it already needs a generalized event-accountant framework.

Final recommendation:

- adopt the lighter `src/privacy/` package now
- defer generic `events.py` and a formal `accountants/` package
- only add those once there is either a second mechanism or a real runtime need for incremental composition

The next implementation step should therefore be:

- a behavior-preserving extraction of the current `src/privacy_accounting.py` responsibilities into `src/privacy/`, while keeping current import paths and tests working unchanged.
