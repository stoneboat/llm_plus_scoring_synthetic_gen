# Phase 3.5 Privacy Architecture Note

**Version:** 0.3.5  
**Date:** 2026-04-07

## 1. Goal

Strengthen the privacy-accounting boundary before Phase 4 runtime/artifact
work.  Phase 3.5 creates a real `src/privacy/` package that separates six
distinct concerns that were previously entangled in the single flat module
`src/privacy_accounting.py`.

---

## 2. New Package Structure

```text
src/privacy/
├── __init__.py                    # flat public surface for the package
├── bounds.py                      # privacy guarantee representations
├── events.py                      # privacy cost unit representations
├── conversions.py                 # generic zCDP → approx-DP conversions
├── planning.py                    # operational pre-generation helpers
├── reporting.py                   # summary dicts and metadata payloads
├── analyses/
│   ├── __init__.py
│   └── private_prediction.py      # Theorem 1 formulas (Amin et al. 2024)
└── accountants/
    ├── __init__.py
    ├── base.py                    # PrivacyAccountant ABC
    └── zcdp.py                    # ZCDPAccountant (concrete)
```

---

## 3. Conceptual Split

### 3.1 Bounds (`bounds.py`)

**Responsibility:** represent what a privacy guarantee *is*.

| Type | Meaning |
|---|---|
| `ZCDPBound(rho)` | zero-concentrated DP parameter |
| `ApproxDPBound(epsilon, delta)` | (ε,δ)-DP guarantee |

Both are frozen dataclasses.  `ZCDPBound` has `.compose(other)` and `.scale(n)`
for sequential composition.  `ApproxDPBound` is currently a pure value type.

**Deliberately omitted:** RDP profiles — no current mechanism requires them.

### 3.2 Events (`events.py`)

**Responsibility:** represent one privacy cost *unit* before composition.

| Type | Meaning |
|---|---|
| `PrivacyEvent(rho, label)` | a single privacy cost; rho=0.0 for public tokens |
| `CompositeEvent(events)` | ordered tuple of events with `total_rho`, counts, and `as_bound()` |

Events make composition readable and auditable.  `PrivacyEvent.is_private`
distinguishes zero-cost public tokens from private tokens.

**Not forced into the runtime:** the current generation loop does not emit
event objects.  The accountant and event types exist as a cleaner abstraction
for pre/post-generation analysis and future online accounting.

### 3.3 Conversions (`conversions.py`)

**Responsibility:** convert between privacy representations, independent of any
specific mechanism.

| Function | Converts |
|---|---|
| `zcdp_to_approx_dp(rho, delta)` | rho float → epsilon float (analytical formula) |
| `zcdp_to_dp_tight(rho, delta)` | same, with dp-accounting fallback (currently stubs) |
| `bound_to_approx_dp(bound, delta)` | `ZCDPBound` → `ApproxDPBound` |

### 3.4 Mechanism-specific analysis (`analyses/private_prediction.py`)

**Responsibility:** map algorithm parameters to privacy costs (Theorem 1,
Amin et al. 2024).

This is the place where paper-specific math lives.

| Function | Returns |
|---|---|
| `rho_per_token(c, s, tau, sigma)` | float — per-token zCDP cost |
| `total_rho(r, c, s, tau, sigma)` | float — total zCDP for r tokens |
| `token_bound(c, s, tau, sigma)` | `ZCDPBound` for one private token |
| `private_token_event(c, s, tau, sigma)` | `PrivacyEvent` for one private token |
| `public_token_event()` | zero-cost `PrivacyEvent` |

Formula (Theorem 1):
```
rho_per_token = 1/2 * (c / (s * tau))^2  +  2 / (s * sigma)^2
```
The SVT term (second summand) is omitted when `sigma=None`.

### 3.5 Accountants (`accountants/`)

**Responsibility:** compose events and answer epsilon-at-delta queries.

`PrivacyAccountant` (ABC in `base.py`) defines three methods:
- `compose(event) → self`
- `epsilon_at_delta(delta) → float`
- `current_bound() → ZCDPBound`

`ZCDPAccountant` (`zcdp.py`) implements this for sequentially composed
zCDP events:
- `total_rho` accumulates by addition (sequential composition)
- `epsilon_at_delta` calls `zcdp_to_approx_dp`
- `compose_many(events)` for batch composition
- `reset()` to clear state

### 3.6 Planning (`planning.py`)

**Responsibility:** answer operational pre-generation questions.

| Function | Question answered |
|---|---|
| `compute_epsilon(r, c, s, tau, delta, sigma)` | Given r tokens, what epsilon results? |
| `compute_max_private_tokens(target_eps, delta, c, s, tau, sigma)` | Given a budget, how many tokens fit? |

These internally use `rho_per_token` and `zcdp_to_approx_dp`.

**Note:** argument order is `clip_bound` before `batch_size` (authoritative
order).  The backward-compat wrapper in `src/config.py` swaps these.

### 3.7 Reporting (`reporting.py`)

**Responsibility:** package privacy information for human-readable output
and machine-readable metadata.

| Function | Used by |
|---|---|
| `privacy_report(r, c, s, tau, delta, sigma)` | `src/generate.py`, scripts (console) |
| `privacy_metadata(epsilon, delta, ...)` | scripts (JSONL header) |

`privacy_metadata` gives the existing run-metadata construction a stable,
named home (previously assembled inline in `scripts/run_experiment.py`).

---

## 4. What Was Deliberately Kept Lightweight

1. **No RDP profiles.** One mechanism, one theorem, no need yet.
2. **No event taxonomy.** `PrivacyEvent(rho)` is sufficient.  No subclasses.
3. **No accountant registry.** One concrete accountant is all the repo needs.
4. **No generalized multi-mechanism composition.** Deferred to Phase 4+.
5. **No runtime redesign.** `PrivatePredictionMechanism` and `src/generate.py`
   are unchanged.  The accountant exists as an analysis utility, not a
   generation-time controller.
6. **No script interface changes.** `scripts/` still calls the same helpers
   (now through the facade).

---

## 5. What Remains Deferred

| Concept | Trigger to add |
|---|---|
| Full `ZCDPAccountant`-driven runtime stopping | Second mechanism OR online budget check |
| RDP profiles | Adopting `dp-accounting` for tighter conversions |
| `zcdp_to_dp_tight` via dp-accounting library | Actual library integration |
| Generic accountant registry | Multiple accountant types in the same codebase |
| Event subclasses | More than two event types needed |
| Accountant injected into `PrivatePredictionMechanism` | Per-token runtime tracking |

---

## 6. Import Graph (No Cycles)

```
bounds.py
  ↑
events.py  ← analyses/private_prediction.py
  ↑                  ↑
accountants/base.py  planning.py
  ↑              ↑
accountants/zcdp.py ← conversions.py
                          ↑
                      reporting.py
```

`src/privacy_accounting.py` (facade) → all modules above  
`src/config.py` wrapper → `src/privacy/planning.py` directly
