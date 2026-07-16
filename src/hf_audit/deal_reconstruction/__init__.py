"""Exchange-native deal reconstruction.

The package is intentionally independent from:

- trading strategies;
- allocation logic;
- live reconciliation;
- CCXT-specific data structures;
- lifecycle simulation;
- reporting formats.

External data must first be converted into the domain models exposed here.
"""

from hf_audit.deal_reconstruction.domain.enums import (
    DealCloseReason,
    FillAction,
    FillOrigin,
    PositionSide,
)
from hf_audit.deal_reconstruction.domain.models import (
    Deal,
    DealLeg,
    NormalizedFill,
    PositionState,
)

__all__ = [
    "Deal",
    "DealCloseReason",
    "DealLeg",
    "FillAction",
    "FillOrigin",
    "NormalizedFill",
    "PositionSide",
    "PositionState",
]
