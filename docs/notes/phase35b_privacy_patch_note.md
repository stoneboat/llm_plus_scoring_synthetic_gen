# Phase 3.5b Privacy Patch Note

**Version:** 0.3.5 → 0.3.6  
**Date:** 2026-04-07

## 1. What Was Conceptually Too Narrow in Phase 3.5

Phase 3.5 introduced `PrivacyEvent(rho: float, label: str)`.  This was an
improvement over raw floats, but it baked a zCDP-native assumption into the
event interface:

- the event always carried a scalar `rho`, the zCDP parameter,
- `is_private` was `rho > 0.0` — a zCDP-specific check,
- `ZCDPAccountant.compose()` called `event.rho` directly — coupling the
  accountant to the zCDP family at the event level,
- `PrivacyAccountant.current_bound()` returned `ZCDPBound` specifically,
  coupling the *base* accountant interface to zCDP.

A future PLD or RDP event would have had to carry a `rho` field that means
nothing to its native representation, or the interface would have had to be
rethought from scratch.

## 2. What Changed

### `bounds.py` — added `PrivacyBound` ABC

```python
class PrivacyBound(ABC):
    @property
    @abstractmethod
    def is_trivial(self) -> bool: ...
```

`ZCDPBound` and `ApproxDPBound` now inherit from `PrivacyBound` and implement
`is_trivial`:
- `ZCDPBound.is_trivial` = `rho == 0.0`
- `ApproxDPBound.is_trivial` = `epsilon == 0.0`

### `events.py` — `PrivacyEvent` carries a `PrivacyBound`

```python
# Before:
PrivacyEvent(rho: float, label: str = "")

# After:
PrivacyEvent(bound: PrivacyBound, label: str = "")
```

Key consequences:
- `is_private = not self.bound.is_trivial` — evaluated via the bound's own
  predicate, not by checking `rho > 0`.
- `as_bound()` returns `self.bound` directly (the native representation).
- A backward-compat `.rho` property extracts `bound.rho` for `ZCDPBound`
  events and raises `TypeError` for other bound types.

`CompositeEvent.as_bound()` now composes via bound objects rather than by
summing `event.rho` directly.

### `accountants/base.py` — `current_bound()` returns `PrivacyBound`

```python
# Before:
def current_bound(self) -> ZCDPBound: ...

# After:
def current_bound(self) -> PrivacyBound: ...
```

The concrete return type (`ZCDPBound`) is still documented in `ZCDPAccountant`,
but the base interface is now guarantee-family-neutral.

### `accountants/zcdp.py` — explicit zCDP extraction in `compose()`

```python
# Before (implicit zCDP assumption):
self._rho += event.rho

# After (explicit, with isinstance guard):
if not isinstance(event.bound, ZCDPBound):
    raise TypeError(...)
self._rho += event.bound.rho
```

This makes the zCDP-family specificity of `ZCDPAccountant` visible and
auditable.  A future `PLDAccountant` would check for `PLDBound` in the same
way.

### `analyses/private_prediction.py` — events constructed with `bound=`

```python
# Before:
return PrivacyEvent(rho=rho, label="private_token")

# After:
return PrivacyEvent(bound=ZCDPBound(rho=rho), label="private_token")
```

## 3. How the New Design Is More Guarantee-Family-Neutral

The key improvement is that the event/accountant interface no longer assumes
every event is fundamentally a `rho: float`:

| Before (3.5) | After (3.5b) |
|---|---|
| `event.rho` — always accessible | `event.bound` — the native representation |
| `is_private = rho > 0` | `is_private = not bound.is_trivial` |
| `accountant.compose()` calls `event.rho` | calls `event.bound.rho` after isinstance check |
| `current_bound() -> ZCDPBound` (base) | `current_bound() -> PrivacyBound` (base) |

A future `PLDAccountant` would:
1. Check `isinstance(event.bound, PLDBound)` in `compose()`.
2. Extract the PLD representation from `event.bound` explicitly.
3. Return a `PLDBound` from `current_bound()` (still satisfies `PrivacyBound`).

No changes to `PrivacyEvent`'s interface would be required.

## 4. What Is Still zCDP-Specific

- `CompositeEvent.as_bound()` currently only supports `ZCDPBound` events
  (documented limitation; raises `TypeError` for mixed families).
- `CompositeEvent.total_rho` is a zCDP-specific convenience property.
- The backward-compat `.rho` property on `PrivacyEvent` only works for
  `ZCDPBound` events.
- Planning and reporting modules operate on raw `rho` floats internally
  (through `rho_per_token`, `total_rho` in analyses).  These are unchanged.

None of these limitations affect the current repo — they are acceptable and
documented.

## 5. What Would Still Need to Change for Real PLD/RDP Support

1. **A `PLDBound` class** inheriting from `PrivacyBound`, implementing
   `is_trivial`.
2. **A `PLDAccountant`** implementing `PrivacyAccountant`, checking for
   `PLDBound` in `compose()`.
3. **Analysis functions** for the PLD family that emit
   `PrivacyEvent(bound=PLDBound(...), ...)`.
4. **`CompositeEvent.as_bound()`** extended to dispatch on bound type, or
   made generic via a `compose(other: PrivacyBound) -> PrivacyBound` method
   on the `PrivacyBound` base.
5. **Planning and reporting** updated to accept PLD inputs (currently
   hard-coded to zCDP formula parameters).

The event and accountant base interfaces would require **no changes** for
items 1–3.  Item 4 would be a small `CompositeEvent` update.  Item 5 is
the largest change and is a natural Phase 4+ concern.
