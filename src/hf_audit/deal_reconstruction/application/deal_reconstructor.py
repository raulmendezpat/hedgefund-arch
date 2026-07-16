"""Application service for reconstructing exchange-native deals."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Iterable

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
)
from hf_audit.deal_reconstruction.domain.reconstruction import (
    DealReconstructionResult,
    OpenPositionSnapshot,
    ReconstructionAnomaly,
)


ZERO = Decimal("0")


def _decimal_close(
    left: Decimal,
    right: Decimal,
    *,
    tolerance: Decimal,
) -> bool:
    return abs(left - right) <= tolerance


def _allocate_decimal(
    total: Decimal,
    *,
    allocated_quantity: Decimal,
    original_quantity: Decimal,
) -> Decimal:
    """Allocate a monetary value proportionally to part of a fill."""

    if original_quantity <= ZERO:
        raise ValueError("original_quantity must be positive")

    if allocated_quantity <= ZERO:
        raise ValueError("allocated_quantity must be positive")

    if allocated_quantity > original_quantity:
        raise ValueError(
            "allocated_quantity cannot exceed original_quantity"
        )

    return total * allocated_quantity / original_quantity


def _slice_fill(
    fill: NormalizedFill,
    *,
    quantity: Decimal,
    allocation_index: int,
) -> NormalizedFill:
    """Create an immutable proportional slice of a normalized fill."""

    if quantity <= ZERO:
        raise ValueError("fill slice quantity must be positive")

    if quantity > fill.quantity:
        raise ValueError("fill slice cannot exceed source quantity")

    if quantity == fill.quantity:
        return fill

    fee = _allocate_decimal(
        fill.fee,
        allocated_quantity=quantity,
        original_quantity=fill.quantity,
    )

    realised_pnl = _allocate_decimal(
        fill.realised_pnl,
        allocated_quantity=quantity,
        original_quantity=fill.quantity,
    )

    metadata = dict(fill.metadata)
    metadata.update(
        {
            "source_trade_id": fill.trade_id,
            "allocated_quantity": str(quantity),
            "source_quantity": str(fill.quantity),
            "allocation_index": str(allocation_index),
        }
    )

    return replace(
        fill,
        trade_id=f"{fill.trade_id}#allocation-{allocation_index}",
        quantity=quantity,
        fee=fee,
        realised_pnl=realised_pnl,
        metadata=metadata,
    )


@dataclass(slots=True)
class _PositionAccumulator:
    exchange: str
    symbol: str
    position_side: PositionSide
    sequence_number: int
    quantity_tolerance: Decimal
    open_quantity: Decimal = ZERO
    entry_fills: list[NormalizedFill] = field(default_factory=list)
    exit_fills: list[NormalizedFill] = field(default_factory=list)

    def apply_open(self, fill: NormalizedFill) -> None:
        self._validate_identity(fill)

        if fill.action is not FillAction.OPEN:
            raise ValueError("apply_open requires an OPEN fill")

        self.entry_fills.append(fill)
        self.open_quantity += fill.quantity

    def apply_close(
        self,
        fill: NormalizedFill,
    ) -> tuple[NormalizedFill | None, NormalizedFill | None]:
        """Apply as much of a closing fill as this position can consume.

        Returns:
            ``(consumed_fill, remainder_fill)``.
        """

        self._validate_identity(fill)

        if fill.action is not FillAction.CLOSE:
            raise ValueError("apply_close requires a CLOSE fill")

        if self.open_quantity <= self.quantity_tolerance:
            return None, fill

        consumed_quantity = min(self.open_quantity, fill.quantity)

        consumed_fill = _slice_fill(
            fill,
            quantity=consumed_quantity,
            allocation_index=1,
        )
        self.exit_fills.append(consumed_fill)
        self.open_quantity -= consumed_quantity

        if abs(self.open_quantity) <= self.quantity_tolerance:
            self.open_quantity = ZERO

        remainder_quantity = fill.quantity - consumed_quantity

        if remainder_quantity <= self.quantity_tolerance:
            return consumed_fill, None

        remainder_fill = _slice_fill(
            fill,
            quantity=remainder_quantity,
            allocation_index=2,
        )
        return consumed_fill, remainder_fill

    @property
    def is_flat(self) -> bool:
        return self.open_quantity == ZERO

    def build_deal(self) -> Deal:
        if not self.is_flat:
            raise ValueError("cannot build a completed deal while position is open")

        if not self.entry_fills:
            raise ValueError("completed deal has no entry fills")

        if not self.exit_fills:
            raise ValueError("completed deal has no exit fills")

        entry_quantity = sum(
            (fill.quantity for fill in self.entry_fills),
            ZERO,
        )
        exit_quantity = sum(
            (fill.quantity for fill in self.exit_fills),
            ZERO,
        )

        if not _decimal_close(
            entry_quantity,
            exit_quantity,
            tolerance=self.quantity_tolerance,
        ):
            raise ValueError(
                "completed deal entry and exit quantities do not match: "
                f"entry={entry_quantity} exit={exit_quantity}"
            )

        origins = tuple(
            dict.fromkeys(
                fill.origin
                for fill in self.entry_fills + self.exit_fills
            )
        )

        first_entry = min(
            fill.timestamp
            for fill in self.entry_fills
        )

        deal_id = (
            f"{self.exchange}:"
            f"{self.symbol}:"
            f"{self.position_side.value}:"
            f"{first_entry.isoformat()}:"
            f"{self.sequence_number}"
        )

        return Deal(
            deal_id=deal_id,
            exchange=self.exchange,
            symbol=self.symbol,
            position_side=self.position_side,
            entry_leg=DealLeg(fills=tuple(self.entry_fills)),
            exit_leg=DealLeg(fills=tuple(self.exit_fills)),
            close_reason=DealCloseReason.FULL_CLOSE,
            origins=origins,
        )

    def snapshot(self) -> OpenPositionSnapshot:
        if self.is_flat:
            raise ValueError("flat position cannot produce an open snapshot")

        return OpenPositionSnapshot(
            exchange=self.exchange,
            symbol=self.symbol,
            position_side=self.position_side,
            open_quantity=self.open_quantity,
            entry_fills=tuple(self.entry_fills),
            exit_fills=tuple(self.exit_fills),
        )

    def _validate_identity(self, fill: NormalizedFill) -> None:
        if fill.exchange != self.exchange:
            raise ValueError("fill exchange does not match accumulator")

        if fill.symbol != self.symbol:
            raise ValueError("fill symbol does not match accumulator")

        if fill.position_side is not self.position_side:
            raise ValueError("fill position side does not match accumulator")


class DealReconstructor:
    """Reconstruct flat-to-flat deals from normalized exchange fills.

    The service is deterministic and exchange-independent. It depends only on
    the ``NormalizedFill`` domain contract.

    A position is tracked independently for every:

        ``(exchange, symbol, position_side)``

    This matches hedge-mode exchange semantics where long and short positions
    can coexist.
    """

    def __init__(
        self,
        *,
        quantity_tolerance: Decimal = Decimal("0.000000000001"),
    ) -> None:
        if quantity_tolerance < ZERO:
            raise ValueError("quantity_tolerance cannot be negative")

        self._quantity_tolerance = quantity_tolerance

    def reconstruct(
        self,
        fills: Iterable[NormalizedFill],
    ) -> DealReconstructionResult:
        ordered_fills = sorted(
            tuple(fills),
            key=lambda fill: (
                fill.timestamp,
                fill.exchange,
                fill.symbol,
                fill.position_side.value,
                fill.trade_id,
            ),
        )

        active: dict[
            tuple[str, str, PositionSide],
            _PositionAccumulator,
        ] = {}

        sequence_by_key: dict[
            tuple[str, str, PositionSide],
            int,
        ] = {}

        deals: list[Deal] = []
        anomalies: list[ReconstructionAnomaly] = []

        for fill in ordered_fills:
            key = (
                fill.exchange,
                fill.symbol,
                fill.position_side,
            )

            accumulator = active.get(key)

            if fill.action is FillAction.OPEN:
                if accumulator is None:
                    sequence = sequence_by_key.get(key, 0) + 1
                    sequence_by_key[key] = sequence

                    accumulator = _PositionAccumulator(
                        exchange=fill.exchange,
                        symbol=fill.symbol,
                        position_side=fill.position_side,
                        sequence_number=sequence,
                        quantity_tolerance=self._quantity_tolerance,
                    )
                    active[key] = accumulator

                accumulator.apply_open(fill)
                continue

            if accumulator is None:
                anomalies.append(
                    self._orphan_close_anomaly(
                        fill,
                        message=(
                            "Closing fill arrived without a reconstructed "
                            "open position. The audit window may start after "
                            "the original entry, or the position may have "
                            "been created outside the available dataset."
                        ),
                    )
                )
                continue

            _, remainder = accumulator.apply_close(fill)

            if accumulator.is_flat:
                deals.append(accumulator.build_deal())
                del active[key]

            if remainder is not None:
                anomalies.append(
                    self._orphan_close_anomaly(
                        remainder,
                        message=(
                            "Closing fill quantity exceeded the reconstructed "
                            "open quantity. The consumed portion completed the "
                            "deal; the remainder could not be attributed."
                        ),
                    )
                )

        open_positions = tuple(
            accumulator.snapshot()
            for _, accumulator in sorted(
                active.items(),
                key=lambda item: (
                    item[0][0],
                    item[0][1],
                    item[0][2].value,
                ),
            )
            if not accumulator.is_flat
        )

        return DealReconstructionResult(
            deals=tuple(deals),
            open_positions=open_positions,
            anomalies=tuple(anomalies),
            processed_fill_count=len(ordered_fills),
        )

    @staticmethod
    def _orphan_close_anomaly(
        fill: NormalizedFill,
        *,
        message: str,
    ) -> ReconstructionAnomaly:
        return ReconstructionAnomaly(
            anomaly_type="orphan_close",
            symbol=fill.symbol,
            position_side=fill.position_side,
            trade_id=fill.trade_id,
            timestamp_iso=fill.timestamp.isoformat(),
            quantity=fill.quantity,
            message=message,
        )
