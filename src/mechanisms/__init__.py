"""
Generation mechanism abstractions (Phase 3).

Public surface:
    Mechanism                    -- abstract base class
    PrivatePredictionMechanism   -- Algorithm 1 (Amin et al. 2024)
"""

from src.mechanisms.base import Mechanism
from src.mechanisms.private_prediction import PrivatePredictionMechanism

__all__ = ["Mechanism", "PrivatePredictionMechanism"]
