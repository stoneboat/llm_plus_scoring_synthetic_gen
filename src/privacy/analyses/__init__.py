"""
Mechanism-specific privacy analyses (Phase 3.5).

Each sub-module owns the theorem-specific formulas for one privacy-aware
generation algorithm.  Currently only one mechanism is implemented.

Public surface:
    private_prediction  -- module alias for src.privacy.analyses.private_prediction
"""

from src.privacy.analyses import private_prediction  # noqa: F401 (module alias)

__all__ = ["private_prediction"]
