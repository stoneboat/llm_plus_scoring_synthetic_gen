"""
Abstract base class for privacy accountants.

A privacy accountant composes privacy costs (events or bounds) and answers
guarantee queries such as epsilon-at-delta.

Phase 3.5b design note:
  current_bound() now returns PrivacyBound rather than ZCDPBound.
  This decouples the accountant interface from the zCDP family:
  - ZCDPAccountant still returns a ZCDPBound (a PrivacyBound subclass).
  - A future PLDAccountant would return a PLDBound.
  - Callers that need family-specific access must cast via isinstance.

  The compose() method takes PrivacyEvent (which carries a PrivacyBound).
  The concrete accountant is responsible for verifying that the event's bound
  family matches what it expects (e.g. ZCDPAccountant checks for ZCDPBound).

Design intent (Phase 3.5):
- The interface is small: compose one event, query epsilon, inspect bound.
- The current repo uses a single ZCDPAccountant; this base exists to make
  the boundary explicit, not to build a framework around it.
- Incremental composition during generation is NOT forced yet.
  The accountant is currently used analytically (pre/post generation), not
  step-by-step inside the decoding loop.

Future: if a second mechanism or online budget tracking is added, the accountant
can be extended without changing this interface.
"""

from abc import ABC, abstractmethod

from src.privacy.bounds import PrivacyBound
from src.privacy.events import PrivacyEvent


class PrivacyAccountant(ABC):
    """Interface for a privacy accountant that composes events and answers queries."""

    @abstractmethod
    def compose(self, event: PrivacyEvent) -> "PrivacyAccountant":
        """Record one privacy event and update internal state.

        The concrete accountant may raise TypeError if the event's bound
        family is incompatible with its composition logic.

        Args:
            event: the PrivacyEvent to compose.

        Returns:
            self (for chaining).
        """

    @abstractmethod
    def epsilon_at_delta(self, delta: float) -> float:
        """Return the epsilon value such that the composed mechanism is
        (epsilon, delta)-DP.

        Args:
            delta: the target failure probability.

        Returns:
            epsilon (float).
        """

    @abstractmethod
    def current_bound(self) -> PrivacyBound:
        """Return the current accumulated privacy bound.

        The concrete type of the returned bound depends on the accountant
        family (e.g. ZCDPAccountant returns ZCDPBound).

        Returns:
            PrivacyBound representing the composed guarantee so far.
        """
