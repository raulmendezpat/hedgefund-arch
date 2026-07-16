"""Strongly typed domain entities for exchange-native deal reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Mapping, Tuple

from hf_audit.deal_reconstruction.domain.enums import (
    DealCloseReason,
    FillAction,
    FillOrigin,
    PositionSide,
)
from hf_audit.deal_reconstruction.domain.exceptions import DomainValidationError


ZERO = Decimal("0")


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise DomainValidationError(f"{field_name} must be timezone-aware")

    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise DomainValidationError(f"{field_name} must use UTC")


def _require_positive(value: Decimal, field_name: str) -> None:
    if value <= ZERO:
        raise DomainValidationError(f"{field_name} must be greater than zero")


def _require_non_negative(value: Decimal, field_name: str) -> None:
    if value < ZERO:
        raise DomainValidationError(f"{field_name} cannot be negative")


@dataclass(frozen=True, slots=True)
class NormalizedFill:
    """Exchange-independent representation of one executed fill.

    ``realised_pnl`` is expected to contain the exchange-reported realised
    profit for closing fills. Opening fills normally contain zero.
    ``fee`` is represented as a non-negative cost.
    """

    exchange: str
    symbol: str
    trade_id: str
    order_id: str
    timestamp: datetime
    position_side: PositionSide
    action: FillAction
    quantity: Decimal
    price: Decimal
    fee: Decimal = ZERO
    realised_pnl: Decimal = ZERO
    origin: FillOrigin = FillOrigin.UNKNOWN
    raw_reference: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.exchange.strip():
            raise DomainValidationError("exchange cannot be empty")

        if not self.symbol.strip():
            raise DomainValidationError("symbol cannot be empty")

        if not self.trade_id.strip():
            raise DomainValidationError("trade_id cannot be empty")

        if not self.order_id.strip():
            raise DomainValidationError("order_id cannot be empty")

        _require_utc(self.timestamp, "timestamp")
        _require_positive(self.quantity, "quantity")
        _require_positive(self.price, "price")
        _require_non_negative(self.fee, "fee")

        if self.action is FillAction.OPEN and self.realised_pnl != ZERO:
            raise DomainValidationError(
                "opening fills cannot contain non-zero realised_pnl"
            )

    @property
    def notional(self) -> Decimal:
        """Absolute quote-currency value of the fill."""

        return self.quantity * self.price


@dataclass(frozen=True, slots=True)
class DealLeg:
    """Aggregated entry or exit component of a deal."""

    fills: Tuple[NormalizedFill, ...]

    def __post_init__(self) -> None:
        if not self.fills:
            raise DomainValidationError("DealLeg must contain at least one fill")

        symbols = {fill.symbol for fill in self.fills}
        sides = {fill.position_side for fill in self.fills}
        actions = {fill.action for fill in self.fills}

        if len(symbols) != 1:
            raise DomainValidationError("DealLeg fills must share one symbol")

        if len(sides) != 1:
            raise DomainValidationError(
                "DealLeg fills must share one position side"
            )

        if len(actions) != 1:
            raise DomainValidationError("DealLeg fills must share one action")

    @property
    def quantity(self) -> Decimal:
        return sum((fill.quantity for fill in self.fills), ZERO)

    @property
    def quote_notional(self) -> Decimal:
        return sum((fill.notional for fill in self.fills), ZERO)

    @property
    def vwap(self) -> Decimal:
        quantity = self.quantity
        if quantity <= ZERO:
            raise DomainValidationError("DealLeg quantity must be positive")
        return self.quote_notional / quantity

    @property
    def fees(self) -> Decimal:
        return sum((fill.fee for fill in self.fills), ZERO)

    @property
    def realised_pnl(self) -> Decimal:
        return sum((fill.realised_pnl for fill in self.fills), ZERO)

    @property
    def started_at(self) -> datetime:
        return min(fill.timestamp for fill in self.fills)

    @property
    def ended_at(self) -> datetime:
        return max(fill.timestamp for fill in self.fills)


@dataclass(frozen=True, slots=True)
class Deal:
    """One reconstructed round trip from flat to flat."""

    deal_id: str
    exchange: str
    symbol: str
    position_side: PositionSide
    entry_leg: DealLeg
    exit_leg: DealLeg
    close_reason: DealCloseReason
    origins: Tuple[FillOrigin, ...]

    def __post_init__(self) -> None:
        if not self.deal_id.strip():
            raise DomainValidationError("deal_id cannot be empty")

        if self.entry_leg.fills[0].action is not FillAction.OPEN:
            raise DomainValidationError("entry_leg must contain opening fills")

        if self.exit_leg.fills[0].action is not FillAction.CLOSE:
            raise DomainValidationError("exit_leg must contain closing fills")

        if self.entry_leg.quantity != self.exit_leg.quantity:
            raise DomainValidationError(
                "A completed Deal must have equal entry and exit quantities"
            )

        if self.entry_leg.started_at > self.exit_leg.ended_at:
            raise DomainValidationError(
                "Deal entry cannot occur after the final exit"
            )

    @property
    def quantity(self) -> Decimal:
        return self.entry_leg.quantity

    @property
    def opened_at(self) -> datetime:
        return self.entry_leg.started_at

    @property
    def closed_at(self) -> datetime:
        return self.exit_leg.ended_at

    @property
    def duration_seconds(self) -> int:
        return int((self.closed_at - self.opened_at).total_seconds())

    @property
    def exchange_realised_pnl(self) -> Decimal:
        """Authoritative realised P/L reported by closing fills."""

        return self.exit_leg.realised_pnl

    @property
    def total_fees(self) -> Decimal:
        return self.entry_leg.fees + self.exit_leg.fees

    @property
    def net_realised_pnl(self) -> Decimal:
        """Realised P/L after explicitly reported fees."""

        return self.exchange_realised_pnl - self.total_fees

    @property
    def includes_manual_close(self) -> bool:
        manual_origins = {
            FillOrigin.IOS,
            FillOrigin.WEB,
            FillOrigin.ANDROID,
        }
        return any(origin in manual_origins for origin in self.origins)


@dataclass(frozen=True, slots=True)
class PositionState:
    """Immutable snapshot of one symbol/side reconstruction state."""

    symbol: str
    position_side: PositionSide
    open_quantity: Decimal = ZERO
    entry_fills: Tuple[NormalizedFill, ...] = ()
    exit_fills: Tuple[NormalizedFill, ...] = ()

    def __post_init__(self) -> None:
        _require_non_negative(self.open_quantity, "open_quantity")

        for fill in self.entry_fills:
            if fill.action is not FillAction.OPEN:
                raise DomainValidationError(
                    "entry_fills can contain only opening fills"
                )

        for fill in self.exit_fills:
            if fill.action is not FillAction.CLOSE:
                raise DomainValidationError(
                    "exit_fills can contain only closing fills"
                )

    @property
    def is_flat(self) -> bool:
        return self.open_quantity == ZERO
