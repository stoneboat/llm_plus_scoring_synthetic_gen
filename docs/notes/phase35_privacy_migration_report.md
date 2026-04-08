# Phase 3.5 Privacy Migration Report

**Version:** 0.3.0 → 0.3.5  
**Date:** 2026-04-07

## Summary

Phase 3.5 extracted the flat `src/privacy_accounting.py` module into a
structured `src/privacy/` package with explicit bounds, events, analyses,
accountants, planning, and reporting layers.  No algorithmic behavior was
changed.  All existing import paths continue to work.

---

## Files Created

### `src/privacy/__init__.py`
Flat public import surface for the package.  Re-exports all key symbols so
callers can write `from src.privacy import ZCDPBound, ZCDPAccountant, ...`.

### `src/privacy/bounds.py`
`ZCDPBound(rho)` and `ApproxDPBound(epsilon, delta)` — lightweight frozen
dataclasses replacing anonymous floats.

### `src/privacy/events.py`
`PrivacyEvent(rho, label)` and `CompositeEvent(events)` — explicit cost units.

### `src/privacy/conversions.py`
`zcdp_to_approx_dp`, `zcdp_to_dp_tight`, `bound_to_approx_dp` — moved from
`privacy_accounting.py`.  Logic unchanged.

### `src/privacy/analyses/__init__.py`
Module alias re-export for `private_prediction`.

### `src/privacy/analyses/private_prediction.py`
Theorem 1 formulas extracted from `privacy_accounting.py`:
- `rho_per_token(c, s, tau, sigma)` — was `compute_rho_per_token`
- `total_rho(r, c, s, tau, sigma)` — was `compute_total_rho`
- `token_bound(...)` → returns `ZCDPBound` (new)
- `private_token_event(...)` → returns `PrivacyEvent` (new)
- `public_token_event()` → returns zero-cost `PrivacyEvent` (new)

### `src/privacy/accountants/__init__.py`
Re-exports `PrivacyAccountant`, `ZCDPAccountant`.

### `src/privacy/accountants/base.py`
`PrivacyAccountant` ABC with `compose`, `epsilon_at_delta`, `current_bound`.

### `src/privacy/accountants/zcdp.py`
`ZCDPAccountant` — concrete accountant for sequentially composed zCDP events.

### `src/privacy/planning.py`
`compute_epsilon`, `compute_max_private_tokens` — operational helpers extracted
from `privacy_accounting.py`.  Signatures and logic identical.

### `src/privacy/reporting.py`
`privacy_report` — extracted from `privacy_accounting.py`; logic identical.  
`privacy_metadata` — new helper that names the JSONL metadata dict currently
assembled inline in `scripts/run_experiment.py`.

### `tests/test_privacy_layer.py`
61 new tests covering:
- bounds creation, composition, and validation
- event classification, composition, and parity
- analysis formula parity with old module
- conversion correctness and fallback
- accountant composition, counting, epsilon query, and reset
- planning monotonicity and round-trip
- reporting key schema and value parity
- compatibility paths

---

## Files Modified

### `src/privacy_accounting.py`
Replaced implementation with a compatibility facade that re-exports all seven
legacy symbols from their new locations in `src/privacy/`:

| Old symbol | New location |
|---|---|
| `compute_rho_per_token` | `src.privacy.analyses.private_prediction.rho_per_token` |
| `compute_total_rho` | `src.privacy.analyses.private_prediction.total_rho` |
| `zcdp_to_approx_dp` | `src.privacy.conversions.zcdp_to_approx_dp` |
| `zcdp_to_dp_tight` | `src.privacy.conversions.zcdp_to_dp_tight` |
| `compute_epsilon` | `src.privacy.planning.compute_epsilon` |
| `compute_max_private_tokens` | `src.privacy.planning.compute_max_private_tokens` |
| `privacy_report` | `src.privacy.reporting.privacy_report` |

### `src/config.py`
Updated the `compute_max_private_tokens` docstring and delegation target:
- now delegates directly to `src.privacy.planning.compute_max_private_tokens`
  instead of going through the `privacy_accounting` facade.
- Argument-order swap is preserved.
- Added Phase 3.5 note to module docstring.

### `src/__init__.py`
- `__version__` bumped to `"0.3.5"`.
- Docstring updated to mention Phase 3.5.

### `pyproject.toml`
- `version` bumped to `"0.3.5"`.

### `tests/test_imports.py`
- Updated `test_src_package_version` assertion to `"0.3.5"`.

---

## Compatibility Wrappers Preserved

| Path | Status | Behavior |
|---|---|---|
| `src.privacy_accounting.*` | All 7 symbols | exact parity via facade |
| `src.config.compute_max_private_tokens` | preserved | arg order swap preserved |
| `src.generate` imports | unchanged | still imports from `privacy_accounting` facade |

---

## Behavior Changes

None.  All numeric outputs are bit-for-bit identical to Phase 3.0.

---

## Test Results

```
204 passed in 2.79s
```

(143 from Phases 1–3, 61 new from Phase 3.5)

---

## Remaining Technical Debt

1. `scripts/run_experiment.py` still assembles the JSONL metadata dict inline;
   it could delegate to `privacy_metadata()` in a future cleanup pass.
2. `src/generate.py` imports from `src.privacy_accounting` (the facade) rather
   than directly from `src.privacy.*` — low priority, works correctly.
3. `zcdp_to_dp_tight` is still a stub; real `dp-accounting` integration is
   deferred until the conversion is needed for tighter guarantees.
4. `ZCDPAccountant` is not yet wired into the generation runtime;
   `PrivatePredictionMechanism` still returns token counts rather than events.
   This is intentional for Phase 3.5 — no runtime redesign yet.
