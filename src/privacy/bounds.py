"""
Lightweight privacy-bound representations.

A bound is the representation of a privacy guarantee.  Making these explicit
prevents helper functions from passing anonymous floats with implicit meaning
and gives conversion routines a natural home.

Phase 3.5 introduces two concrete bound types used in this repo:
- ZCDPBound  — a zero-concentrated DP guarantee parameterized by rho
- ApproxDPBound — an (epsilon, delta)-DP guarantee

Phase 3.5b adds:
- PrivacyBound — lightweight ABC that all bound types inherit from.
  Provides is_trivial so that PrivacyEvent.is_private can be evaluated
  without assuming the bound is a ZCDPBound.

RDP profiles are deliberately omitted until the repo adopts a second mechanism
that needs order-by-order composition (see planning doc for rationale).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class PrivacyBound(ABC):
    """Abstract base for all privacy guarantee representations.

    All concrete bound types (ZCDPBound, ApproxDPBound, future PLDBound …)
    inherit from this class.  The only behaviour required of every bound is
    the ability to report whether the guarantee is *trivial* — i.e. whether
    it imposes no constraint at all (zero privacy cost).

    This base type is used by PrivacyEvent so that events can carry any bound
    family without baking zCDP assumptions into the event interface.
    """

    @property
    @abstractmethod
    def is_trivial(self) -> bool:
        """True if this bound represents zero privacy cost.

        For ZCDPBound: rho == 0.0.
        For ApproxDPBound: epsilon == 0.0.
        """


@dataclass(frozen=True)
class ZCDPBound(PrivacyBound):
    """Zero-concentrated differential privacy bound.

    A mechanism is rho-zCDP if its privacy loss satisfies the Rényi
    divergence condition at every order alpha > 1 (Bun & Steinke, 2016).

    Sequential composition:  rho_total = rho_1 + rho_2 + ...
    Parallel composition:    same as worst component.

    Attributes:
        rho: the zCDP parameter (non-negative).
    """
    rho: float

    def __post_init__(self) -> None:
        if self.rho < 0:
            raise ValueError(f"rho must be non-negative, got {self.rho}")

    @property
    def is_trivial(self) -> bool:
        return self.rho == 0.0

    def compose(self, other: "ZCDPBound") -> "ZCDPBound":
        """Sequential composition: rho adds."""
        return ZCDPBound(rho=self.rho + other.rho)

    def scale(self, n: int) -> "ZCDPBound":
        """Repeat n times sequentially (e.g., n private tokens)."""
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        return ZCDPBound(rho=self.rho * n)


@dataclass(frozen=True)
class ApproxDPBound(PrivacyBound):
    """Approximate differential privacy guarantee: (epsilon, delta)-DP.

    A mechanism is (epsilon, delta)-DP if for all neighboring datasets D, D'
    and all events S:
        Pr[M(D) in S] <= exp(epsilon) * Pr[M(D') in S] + delta

    Attributes:
        epsilon: the DP epsilon parameter (non-negative).
        delta: the failure probability (in [0, 1]).
    """
    epsilon: float
    delta: float

    def __post_init__(self) -> None:
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {self.epsilon}")
        if not (0.0 <= self.delta <= 1.0):
            raise ValueError(f"delta must be in [0, 1], got {self.delta}")

    @property
    def is_trivial(self) -> bool:
        return self.epsilon == 0.0
