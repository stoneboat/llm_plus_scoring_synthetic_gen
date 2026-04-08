# Phase 3.5b Privacy Patch Report

**Version:** 0.3.5 → 0.3.6  
**Date:** 2026-04-07

## Files Changed

### `src/privacy/bounds.py`
- Added `PrivacyBound` ABC with `is_trivial: bool` abstract property.
- `ZCDPBound` and `ApproxDPBound` now inherit from `PrivacyBound`.
- `ZCDPBound.is_trivial` = `rho == 0.0`.
- `ApproxDPBound.is_trivial` = `epsilon == 0.0`.

### `src/privacy/events.py`
- `PrivacyEvent`: replaced `rho: float` field with `bound: PrivacyBound`.
- `PrivacyEvent.is_private`: now uses `not self.bound.is_trivial`.
- `PrivacyEvent.as_bound()`: returns `self.bound` directly.
- `PrivacyEvent.rho`: kept as a backward-compat property; raises `TypeError` for non-`ZCDPBound` events.
- `CompositeEvent.as_bound()`: composes via bound objects (accumulates `ZCDPBound.rho` after isinstance check); documents zCDP-only limitation.
- `CompositeEvent.total_rho`: delegates to `self.as_bound().rho`; documents zCDP-specific nature.

### `src/privacy/accountants/base.py`
- `PrivacyAccountant.current_bound()` return type changed from `ZCDPBound` to `PrivacyBound`.
- Added note about explicit bound-family checking in concrete implementations.

### `src/privacy/accountants/zcdp.py`
- `ZCDPAccountant.compose()`: now checks `isinstance(event.bound, ZCDPBound)` and raises `TypeError` for incompatible events.  Extracts rho via `event.bound.rho` (explicit) rather than `event.rho` (implicit).

### `src/privacy/analyses/private_prediction.py`
- `private_token_event()`: now returns `PrivacyEvent(bound=ZCDPBound(rho=...), label=...)`.
- `public_token_event()`: now returns `PrivacyEvent(bound=ZCDPBound(rho=0.0), label=...)`.

### `src/privacy/__init__.py`
- Added `PrivacyBound` to imports and `__all__`.

### `tests/test_privacy_layer.py`
- Updated all direct `PrivacyEvent(rho=...)` constructions to `PrivacyEvent(bound=ZCDPBound(rho=...))`.
- Replaced/expanded `TestPrivacyEvent` tests to cover the new `bound`-carrying interface.
- Added new tests:
  - `test_carries_bound_object` — event stores a `PrivacyBound`, not a scalar.
  - `test_as_bound_returns_same_object` — `as_bound()` is identity.
  - `test_rho_compat_property_for_zcdp` — `.rho` still works for ZCDPBound events.
  - `test_rho_compat_property_raises_for_non_zcdp` — `.rho` raises for ApproxDPBound.
  - `test_is_private_driven_by_bound_is_trivial` — `is_private` works for non-zCDP bounds.
  - `test_as_bound_via_bound_objects` — composite composes via bounds.
  - `test_compose_rejects_non_zcdp_event` — `ZCDPAccountant` raises for non-zCDP events.
  - `test_compose_uses_event_bound_rho_not_event_rho` — verifies explicit extraction path.
  - `test_privacy_bound_is_common_base` — both bound types are `PrivacyBound`.
  - `test_privacy_accountant_current_bound_returns_privacy_bound` — base interface is neutral.

### `src/__init__.py`
- `__version__` bumped to `"0.3.6"`.

### `pyproject.toml`
- `version` bumped to `"0.3.6"`.

### `tests/test_imports.py`
- Version assertion updated to `"0.3.6"`.

---

## Backward Compatibility Status

| Path | Status |
|---|---|
| `src.privacy_accounting.*` | Unchanged (facade still works) |
| `src.config.compute_max_private_tokens` | Unchanged |
| `event.rho` property | Works for ZCDPBound events; raises TypeError for others |
| `event.bound` | New primary interface |
| `ZCDPAccountant.total_rho` | Unchanged |
| `CompositeEvent.total_rho` | Unchanged (zCDP-only, documented) |
| All 204 previously passing tests | Still pass |

The only **breaking change** is that `PrivacyEvent(rho=...)` no longer works
as a direct constructor — the field is now `bound=`.  Code using the
analysis-layer constructors (`private_token_event`, `public_token_event`) is
unaffected.

---

## Behavior Changes

None.  All numeric outputs are identical to Phase 3.5.

---

## New Tests Added

10 new tests (212 total, up from 204).

---

## Remaining Limitations

1. `CompositeEvent.as_bound()` only supports `ZCDPBound` events — raises `TypeError` for mixed or non-zCDP events.
2. `CompositeEvent.total_rho` is zCDP-specific (documented).
3. `PrivacyEvent.rho` compat property is zCDP-specific (documented).
4. Planning and reporting still operate on raw `rho` floats internally.
5. No PLD, RDP, or second mechanism is implemented.
