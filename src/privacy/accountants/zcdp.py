"""
zCDP accountant for the private-prediction algorithm.

Maintains a running total of zCDP rho by composing PrivacyEvent objects.
Answers epsilon-at-delta queries using the analytical zCDP → approx-DP
conversion.

Sequential composition under zCDP:
    rho_total = rho_1 + rho_2 + ... + rho_n

This is exact (not an upper bound) for sequentially adaptive mechanisms.

Phase 3.5b note:
  compose() now explicitly checks that each event carries a ZCDPBound and
  extracts rho via event.bound.rho rather than event.rho.  This makes the
  zCDP-family specificity of this accountant visible and auditable:

    Before: self._rho += event.rho          (implicit zCDP assumption)
    After:  self._rho += event.bound.rho    (explicit, after isinstance check)

  A future PLDAccountant would check for PLDBound in the same way.

Usage example (post-generation summary):
    from src.privacy.accountants.zcdp import ZCDPAccountant
    from src.privacy.analyses.private_prediction import private_token_event, public_token_event

    acct = ZCDPAccountant()
    for _ in range(n_private):
        acct.compose(private_token_event(clip_bound, batch_size, temperature, svt_noise))
    for _ in range(n_public):
        acct.compose(public_token_event())

    eps = acct.epsilon_at_delta(delta)
"""

from typing import List

from src.privacy.accountants.base import PrivacyAccountant
from src.privacy.bounds import PrivacyBound, ZCDPBound
from src.privacy.conversions import zcdp_to_approx_dp
from src.privacy.events import PrivacyEvent


class ZCDPAccountant(PrivacyAccountant):
    """Accountant for mechanisms with zCDP guarantees.

    Composes PrivacyEvent objects by extracting and summing their ZCDPBound
    rho values (sequential composition) and converts the accumulated zCDP
    bound to approximate DP on demand.

    Only events carrying a ZCDPBound (including trivial zero-cost public-token
    events) are accepted.  Raises TypeError for events with other bound types.

    Attributes:
        _rho: accumulated zCDP parameter (running sum of rho from ZCDPBound events).
        _events: list of composed events (for inspection / auditing).
    """

    def __init__(self) -> None:
        self._rho: float = 0.0
        self._events: List[PrivacyEvent] = []

    # ------------------------------------------------------------------
    # PrivacyAccountant interface
    # ------------------------------------------------------------------

    def compose(self, event: PrivacyEvent) -> "ZCDPAccountant":
        """Add one privacy event via sequential zCDP composition.

        Explicitly checks that the event carries a ZCDPBound and extracts
        the rho cost via event.bound.rho.  This makes the zCDP-specificity
        of this accountant visible rather than implicitly relying on event.rho.

        Args:
            event: the PrivacyEvent to compose.  Must carry a ZCDPBound.

        Returns:
            self (for chaining).

        Raises:
            TypeError: if event.bound is not a ZCDPBound.
        """
        if not isinstance(event.bound, ZCDPBound):
            raise TypeError(
                f"ZCDPAccountant only accepts events with ZCDPBound; "
                f"got {type(event.bound).__name__}. "
                f"Use an accountant compatible with {type(event.bound).__name__}."
            )
        self._rho += event.bound.rho  # explicit zCDP extraction
        self._events.append(event)
        return self

    def epsilon_at_delta(self, delta: float) -> float:
        """Return epsilon such that the composed mechanism is (eps, delta)-DP.

        Uses the analytical formula:
            epsilon = rho + sqrt(4 * rho * log(1/delta))

        Args:
            delta: failure probability.

        Returns:
            epsilon (float).
        """
        return zcdp_to_approx_dp(self._rho, delta)

    def current_bound(self) -> ZCDPBound:
        """Return the accumulated zCDP bound.

        Returns:
            ZCDPBound(rho=sum of all composed event rhos).
        """
        return ZCDPBound(rho=self._rho)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def total_rho(self) -> float:
        """Accumulated zCDP parameter."""
        return self._rho

    @property
    def n_events(self) -> int:
        """Total number of composed events (private + public)."""
        return len(self._events)

    @property
    def n_private(self) -> int:
        """Number of events with non-trivial privacy cost."""
        return sum(1 for e in self._events if e.is_private)

    @property
    def n_public(self) -> int:
        """Number of zero-cost events."""
        return sum(1 for e in self._events if not e.is_private)

    def compose_many(self, events) -> "ZCDPAccountant":
        """Compose an iterable of PrivacyEvents.

        Args:
            events: iterable of PrivacyEvent objects (must all carry ZCDPBound).

        Returns:
            self (for chaining).
        """
        for e in events:
            self.compose(e)
        return self

    def reset(self) -> "ZCDPAccountant":
        """Reset the accountant to zero."""
        self._rho = 0.0
        self._events = []
        return self
