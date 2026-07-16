"""Domain types supporting exchange-native deal reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Tuple

from hf_audit.deal_reconstruction.domain.enums import PositionSide
from hf_audit.deal_reconstruction.domain.models import Deal, NormalizedFill


ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class ReconstructionAnomaly:
    """A fill that cannot safely be incorporated into a completed deal."""

    anomaly_type: str
    symbol: str
    position_side: PositionSide
    trade_id: str
    timestamp_iso: str
    quantity: Decimal
    message: str

    def __post_init__(self) -> None:
        if not self.anomaly_type.strip():
            raise ValueError("anomaly_type cannot be empty")

        if not self.symbol.strip():
            raise ValueError("symbol cannot be empty")

        if not self.trade_id.strip():
            raise ValueError("trade_id cannot be empty")

        if self.quantity <= ZERO:
            raise ValueError("anomaly quantity must be positive")


@dataclass(frozen=True, slots=True)
class OpenPositionSnapshot:
    """Immutable end-of-data snapshot for an incomplete round trip."""

    exchange: str
    symbol: str
    position_side: PositionSide
    open_quantity: Decimal
    entry_fills: Tuple[NormalizedFill, ...]
    exit_fills: Tuple[NormalizedFill, ...]

    def __post_init__(self) -> None:
        if self.open_quantity <= ZERO:
            raise ValueError("open_quantity must be positive")

        if not self.entry_fills:
            raise ValueError("an open position must contain entry fills")

    @property
    def opened_at(self):
        return min(fill.timestamp for fill in self.entry_fills)

    @property
    def origins(self):
        return tuple(
            dict.fromkeys(
                fill.origin
                for fill in self.entry_fills + self.exit_fills
            )
        )


@dataclass(frozen=True, slots=True)
class DealReconstructionResult:
    """Complete output from one deterministic reconstruction pass."""

    deals: Tuple[Deal, ...]
    open_positions: Tuple[OpenPositionSnapshot, ...]
    anomalies: Tuple[ReconstructionAnomaly, ...]
    processed_fill_count: int

    def __post_init__(self) -> None:
        if self.processed_fill_count < 0:
            raise ValueError("processed_fill_count cannot be negative")

    @property
    def completed_deal_count(self) -> int:
        return len(self.deals)

    @property
    def open_position_count(self) -> int:
        return len(self.open_positions)

    @property
    def anomaly_count(self) -> int:
        return len(self.anomalies)
