"""
Backward-compatibility facade for the privacy accounting layer.

Phase 3.5 moved all privacy logic into the src/privacy/ package.
This module re-exports every symbol under its original name so that existing
call sites (src/generate.py, scripts/, tests/) continue to work unchanged.

Do not add new functionality here.  New callers should import directly from
src.privacy.* for clarity.

Mapping to new locations:
    compute_rho_per_token   → src.privacy.analyses.private_prediction.rho_per_token
    compute_total_rho       → src.privacy.analyses.private_prediction.total_rho
    zcdp_to_approx_dp       → src.privacy.conversions.zcdp_to_approx_dp
    zcdp_to_dp_tight        → src.privacy.conversions.zcdp_to_dp_tight
    compute_epsilon         → src.privacy.planning.compute_epsilon
    compute_max_private_tokens → src.privacy.planning.compute_max_private_tokens
    privacy_report          → src.privacy.reporting.privacy_report
"""

# Mechanism-specific cost formulas
from src.privacy.analyses.private_prediction import (   # noqa: F401
    rho_per_token as compute_rho_per_token,
    total_rho as compute_total_rho,
)

# Generic zCDP conversions
from src.privacy.conversions import (                   # noqa: F401
    zcdp_to_approx_dp,
    zcdp_to_dp_tight,
)

# Operational planning helpers
from src.privacy.planning import (                      # noqa: F401
    compute_epsilon,
    compute_max_private_tokens,
)

# Reporting
from src.privacy.reporting import privacy_report        # noqa: F401

__all__ = [
    "compute_rho_per_token",
    "compute_total_rho",
    "zcdp_to_approx_dp",
    "zcdp_to_dp_tight",
    "compute_epsilon",
    "compute_max_private_tokens",
    "privacy_report",
]
