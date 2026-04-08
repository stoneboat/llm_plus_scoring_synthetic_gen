"""
Lightweight privacy event abstractions.

A privacy event is one unit of privacy cost before composition.  Making events
explicit:
- makes composition readable and auditable,
- distinguishes zero-cost public tokens from private tokens,
- provides a path toward mechanism-agnostic accounting without forcing it now.

Phase 3.5b design improvement:
  PrivacyEvent now carries a PrivacyBound object rather than a raw rho float.
  This decouples the event interface from the zCDP family:

    Before: PrivacyEvent(rho: float, label: str)
    After:  PrivacyEvent(bound: PrivacyBound, label: str)

  Key consequences:
  - is_private is evaluated via bound.is_trivial (bound-driven, not rho-driven).
  - Constructing a zCDP event requires passing a ZCDPBound explicitly.
  - A future PLD/RDP event would pass a different PrivacyBound subclass.
  - The ZCDPAccountant can check isinstance(event.bound, ZCDPBound) to
    make its zCDP-specific extraction explicit and auditable.

  What is still zCDP-specific:
  - The .rho property is a backward-compat accessor for ZCDPBound events.
  - CompositeEvent.total_rho is a zCDP-specific convenience property.
  - CompositeEvent.as_bound() returns ZCDPBound (the only composable family now).

  These limitations are documented in-place and do not need to be resolved
  until a non-zCDP mechanism is added.
"""

from dataclasses import dataclass
from typing import Tuple

from src.privacy.bounds import PrivacyBound, ZCDPBound


@dataclass(frozen=True)
class PrivacyEvent:
    """A single privacy cost unit, carrying a privacy bound in its native family.

    A public token event carries a trivial bound (e.g. ZCDPBound(rho=0.0)).
    A private token event carries a non-trivial bound (e.g. ZCDPBound(rho>0)).

    By carrying a PrivacyBound rather than a raw scalar:
    - is_private is evaluated via the bound's is_trivial predicate,
    - accountants can inspect the bound type to apply family-specific composition,
    - future non-zCDP events can be represented without changing this interface.

    Attributes:
        bound: the privacy cost of this event in its native representation.
        label: optional human-readable label for logging/debugging.
    """
    bound: PrivacyBound
    label: str = ""

    @property
    def is_private(self) -> bool:
        """True if this event has non-trivial privacy cost.

        Evaluated via bound.is_trivial so the check is bound-family-neutral.
        """
        return not self.bound.is_trivial

    def as_bound(self) -> PrivacyBound:
        """Return the event's privacy bound directly."""
        return self.bound

    @property
    def rho(self) -> float:
        """Backward-compat accessor: extract rho from a ZCDPBound event.

        Raises:
            TypeError: if this event's bound is not a ZCDPBound.

        Note: this property is intentionally zCDP-specific.  New code should
        access ``event.bound`` directly and cast as needed.
        """
        if isinstance(self.bound, ZCDPBound):
            return self.bound.rho
        raise TypeError(
            f"The .rho accessor is only defined for ZCDPBound events; "
            f"this event carries a {type(self.bound).__name__}. "
            f"Use event.bound directly."
        )


@dataclass(frozen=True)
class CompositeEvent:
    """Sequential composition of multiple privacy events.

    Represents the total privacy cost of a sequence of events, for example
    all tokens generated for one synthetic example.

    Attributes:
        events: the individual events (ordered).
    """
    events: Tuple[PrivacyEvent, ...]

    @property
    def n_private(self) -> int:
        """Number of events with non-trivial privacy cost."""
        return sum(1 for e in self.events if e.is_private)

    @property
    def n_public(self) -> int:
        """Number of zero-cost events."""
        return sum(1 for e in self.events if not e.is_private)

    def as_bound(self) -> ZCDPBound:
        """Return the composed bound for the entire sequence.

        Currently only ZCDPBound events are supported.  The composition is
        sequential (rho adds).  Raises TypeError if any event carries a
        non-ZCDPBound.

        Future: a family-generic version would dispatch on bound type.
        """
        rho = 0.0
        for e in self.events:
            if not isinstance(e.bound, ZCDPBound):
                raise TypeError(
                    f"CompositeEvent.as_bound() currently only supports "
                    f"ZCDPBound events; got {type(e.bound).__name__}."
                )
            rho += e.bound.rho
        return ZCDPBound(rho=rho)

    @property
    def total_rho(self) -> float:
        """zCDP-specific convenience: sum of rho values across all events.

        Only valid when all events carry ZCDPBound.  Raises TypeError for
        mixed or non-zCDP events.

        Note: this property is zCDP-specific.  Prefer as_bound().rho
        which makes the zCDP specificity explicit.
        """
        return self.as_bound().rho

    def compose(self, other: "CompositeEvent") -> "CompositeEvent":
        """Concatenate two composite events."""
        return CompositeEvent(events=self.events + other.events)
