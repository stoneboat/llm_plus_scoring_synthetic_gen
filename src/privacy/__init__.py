"""
Privacy layer for the private-prediction synthetic text generation project.

Phase 3.5 extracted the privacy boundary from the flat src/privacy_accounting.py
module into a structured package.  The key conceptual split is:

  bounds.py         — what a privacy guarantee IS (ZCDPBound, ApproxDPBound)
  events.py         — what one privacy cost UNIT is (PrivacyEvent, CompositeEvent)
  conversions.py    — how bounds are CONVERTED (zCDP → approx-DP)
  analyses/         — how mechanism parameters MAP to costs (theorem-specific)
  accountants/      — how costs COMPOSE and how queries are answered
  planning.py       — operational PRE-generation helpers (token budget)
  reporting.py      — POST-generation summaries and metadata payloads

Backward compatibility:
  src.privacy_accounting  — legacy module, now a facade that re-exports from here
  src.config.compute_max_private_tokens — preserved with original argument order

Public flat surface (importable directly from src.privacy):
"""

# Bounds
from src.privacy.bounds import PrivacyBound, ZCDPBound, ApproxDPBound  # noqa: F401

# Events
from src.privacy.events import PrivacyEvent, CompositeEvent  # noqa: F401

# Conversions
from src.privacy.conversions import (  # noqa: F401
    zcdp_to_approx_dp,
    zcdp_to_dp_tight,
    bound_to_approx_dp,
)

# Mechanism-specific analysis
from src.privacy.analyses.private_prediction import (  # noqa: F401
    rho_per_token,
    total_rho,
    token_bound,
    private_token_event,
    public_token_event,
)

# Accountants
from src.privacy.accountants.zcdp import ZCDPAccountant  # noqa: F401

# Planning
from src.privacy.planning import compute_epsilon, compute_max_private_tokens  # noqa: F401

# Reporting
from src.privacy.reporting import privacy_report, privacy_metadata  # noqa: F401

__all__ = [
    # Bounds
    "PrivacyBound",
    "ZCDPBound",
    "ApproxDPBound",
    # Events
    "PrivacyEvent",
    "CompositeEvent",
    # Conversions
    "zcdp_to_approx_dp",
    "zcdp_to_dp_tight",
    "bound_to_approx_dp",
    # Analysis
    "rho_per_token",
    "total_rho",
    "token_bound",
    "private_token_event",
    "public_token_event",
    # Accountants
    "ZCDPAccountant",
    # Planning
    "compute_epsilon",
    "compute_max_private_tokens",
    # Reporting
    "privacy_report",
    "privacy_metadata",
]
