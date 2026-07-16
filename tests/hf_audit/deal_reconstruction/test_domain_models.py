from __future__ import annotations

import unittest
from datetime import datetime, timezone
from decimal import Decimal

from hf_audit.deal_reconstruction.domain.enums import (
    DealCloseReason,
    FillAction,
    FillOrigin,
    PositionSide,
)
from hf_audit.deal_reconstruction.domain.exceptions import (
    DomainValidationError,
)
from hf_audit.deal_reconstruction.domain.models import (
    Deal,
    DealLeg,
    NormalizedFill,
)


def make_fill(
    *,
    trade_id: str,
    action: FillAction,
    quantity: str,
    price: str,
    realised_pnl: str = "0",
    fee: str = "0",
    origin: FillOrigin = FillOrigin.API,
) -> NormalizedFill:
    return NormalizedFill(
        exchange="bitget",
        symbol="BTC/USDT:USDT",
        trade_id=trade_id,
        order_id=f"order-{trade_id}",
        timestamp=datetime(
            2026,
            7,
            16,
            0,
            0,
            tzinfo=timezone.utc,
        ),
        position_side=PositionSide.LONG,
        action=action,
        quantity=Decimal(quantity),
        price=Decimal(price),
        realised_pnl=Decimal(realised_pnl),
        fee=Decimal(fee),
        origin=origin,
    )


class DomainModelsTest(unittest.TestCase):
    def test_normalized_fill_notional(self) -> None:
        fill = make_fill(
            trade_id="1",
            action=FillAction.OPEN,
            quantity="0.5",
            price="60000",
        )

        self.assertEqual(
            fill.notional,
            Decimal("30000.0"),
        )

    def test_open_fill_rejects_realised_pnl(self) -> None:
        with self.assertRaises(DomainValidationError):
            make_fill(
                trade_id="1",
                action=FillAction.OPEN,
                quantity="1",
                price="100",
                realised_pnl="2",
            )

    def test_deal_leg_calculates_vwap(self) -> None:
        first = make_fill(
            trade_id="1",
            action=FillAction.OPEN,
            quantity="1",
            price="100",
        )
        second = make_fill(
            trade_id="2",
            action=FillAction.OPEN,
            quantity="2",
            price="110",
        )

        leg = DealLeg(
            fills=(
                first,
                second,
            )
        )

        self.assertEqual(
            leg.quantity,
            Decimal("3"),
        )
        self.assertEqual(
            leg.vwap,
            Decimal("106.6666666666666666666666667"),
        )

    def test_completed_deal_supports_manual_close(self) -> None:
        entry = make_fill(
            trade_id="entry",
            action=FillAction.OPEN,
            quantity="2",
            price="100",
            fee="0.04",
        )
        close = make_fill(
            trade_id="close",
            action=FillAction.CLOSE,
            quantity="2",
            price="110",
            realised_pnl="20",
            fee="0.05",
            origin=FillOrigin.IOS,
        )

        deal = Deal(
            deal_id="BTC-long-1",
            exchange="bitget",
            symbol="BTC/USDT:USDT",
            position_side=PositionSide.LONG,
            entry_leg=DealLeg(
                fills=(entry,)
            ),
            exit_leg=DealLeg(
                fills=(close,)
            ),
            close_reason=DealCloseReason.FULL_CLOSE,
            origins=(
                FillOrigin.API,
                FillOrigin.IOS,
            ),
        )

        self.assertEqual(
            deal.exchange_realised_pnl,
            Decimal("20"),
        )
        self.assertEqual(
            deal.total_fees,
            Decimal("0.09"),
        )
        self.assertEqual(
            deal.net_realised_pnl,
            Decimal("19.91"),
        )
        self.assertTrue(
            deal.includes_manual_close
        )

    def test_completed_deal_rejects_quantity_mismatch(self) -> None:
        entry = make_fill(
            trade_id="entry",
            action=FillAction.OPEN,
            quantity="2",
            price="100",
        )
        close = make_fill(
            trade_id="close",
            action=FillAction.CLOSE,
            quantity="1",
            price="110",
            realised_pnl="10",
        )

        with self.assertRaises(DomainValidationError):
            Deal(
                deal_id="invalid",
                exchange="bitget",
                symbol="BTC/USDT:USDT",
                position_side=PositionSide.LONG,
                entry_leg=DealLeg(
                    fills=(entry,)
                ),
                exit_leg=DealLeg(
                    fills=(close,)
                ),
                close_reason=DealCloseReason.FULL_CLOSE,
                origins=(FillOrigin.API,),
            )


if __name__ == "__main__":
    unittest.main()
