from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from hf_audit.deal_reconstruction.application import DealReconstructor
from hf_audit.deal_reconstruction.domain.enums import (
    FillAction,
    FillOrigin,
    PositionSide,
)
from hf_audit.deal_reconstruction.domain.models import NormalizedFill


BASE_TIME = datetime(2026, 7, 16, 0, 0, tzinfo=timezone.utc)


def make_fill(
    *,
    trade_id: str,
    minute: int,
    action: FillAction,
    position_side: PositionSide,
    quantity: str,
    price: str,
    realised_pnl: str = "0",
    fee: str = "0",
    origin: FillOrigin = FillOrigin.API,
    symbol: str = "BTC/USDT:USDT",
) -> NormalizedFill:
    order_side = (
        "buy"
        if (
            action is FillAction.OPEN
            and position_side is PositionSide.LONG
        )
        or (
            action is FillAction.CLOSE
            and position_side is PositionSide.SHORT
        )
        else "sell"
    )

    return NormalizedFill(
        exchange="bitget",
        symbol=symbol,
        trade_id=trade_id,
        order_id=f"order-{trade_id}",
        timestamp=BASE_TIME + timedelta(minutes=minute),
        position_side=position_side,
        action=action,
        quantity=Decimal(quantity),
        price=Decimal(price),
        fee=Decimal(fee),
        realised_pnl=Decimal(realised_pnl),
        origin=origin,
        metadata={"order_side": order_side},
    )


class DealReconstructorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reconstructor = DealReconstructor()

    def test_simple_open_and_close(self) -> None:
        fills = [
            make_fill(
                trade_id="open",
                minute=0,
                action=FillAction.OPEN,
                position_side=PositionSide.LONG,
                quantity="2",
                price="100",
                fee="0.02",
            ),
            make_fill(
                trade_id="close",
                minute=10,
                action=FillAction.CLOSE,
                position_side=PositionSide.LONG,
                quantity="2",
                price="110",
                realised_pnl="20",
                fee="0.03",
            ),
        ]

        result = self.reconstructor.reconstruct(fills)

        self.assertEqual(result.processed_fill_count, 2)
        self.assertEqual(result.completed_deal_count, 1)
        self.assertEqual(result.open_position_count, 0)
        self.assertEqual(result.anomaly_count, 0)

        deal = result.deals[0]

        self.assertEqual(deal.quantity, Decimal("2"))
        self.assertEqual(deal.entry_leg.vwap, Decimal("100"))
        self.assertEqual(deal.exit_leg.vwap, Decimal("110"))
        self.assertEqual(deal.exchange_realised_pnl, Decimal("20"))
        self.assertEqual(deal.total_fees, Decimal("0.05"))
        self.assertEqual(deal.net_realised_pnl, Decimal("19.95"))

    def test_multiple_entries_and_partial_closes(self) -> None:
        fills = [
            make_fill(
                trade_id="open-1",
                minute=0,
                action=FillAction.OPEN,
                position_side=PositionSide.LONG,
                quantity="1",
                price="100",
            ),
            make_fill(
                trade_id="open-2",
                minute=1,
                action=FillAction.OPEN,
                position_side=PositionSide.LONG,
                quantity="2",
                price="110",
            ),
            make_fill(
                trade_id="close-1",
                minute=2,
                action=FillAction.CLOSE,
                position_side=PositionSide.LONG,
                quantity="1",
                price="115",
                realised_pnl="10",
            ),
            make_fill(
                trade_id="close-2",
                minute=3,
                action=FillAction.CLOSE,
                position_side=PositionSide.LONG,
                quantity="2",
                price="120",
                realised_pnl="20",
            ),
        ]

        result = self.reconstructor.reconstruct(fills)

        self.assertEqual(result.completed_deal_count, 1)
        self.assertEqual(result.open_position_count, 0)
        self.assertEqual(result.anomaly_count, 0)

        deal = result.deals[0]

        self.assertEqual(deal.quantity, Decimal("3"))
        self.assertEqual(
            deal.entry_leg.vwap,
            Decimal("106.6666666666666666666666667"),
        )
        self.assertEqual(
            deal.exit_leg.vwap,
            Decimal("118.3333333333333333333333333"),
        )
        self.assertEqual(deal.exchange_realised_pnl, Decimal("30"))

    def test_manual_close_origin_is_preserved(self) -> None:
        fills = [
            make_fill(
                trade_id="open",
                minute=0,
                action=FillAction.OPEN,
                position_side=PositionSide.SHORT,
                quantity="5",
                price="10",
            ),
            make_fill(
                trade_id="close",
                minute=5,
                action=FillAction.CLOSE,
                position_side=PositionSide.SHORT,
                quantity="5",
                price="9",
                realised_pnl="5",
                origin=FillOrigin.IOS,
            ),
        ]

        result = self.reconstructor.reconstruct(fills)

        self.assertEqual(result.completed_deal_count, 1)
        self.assertTrue(result.deals[0].includes_manual_close)
        self.assertIn(FillOrigin.IOS, result.deals[0].origins)

    def test_long_and_short_are_independent_in_hedge_mode(self) -> None:
        fills = [
            make_fill(
                trade_id="long-open",
                minute=0,
                action=FillAction.OPEN,
                position_side=PositionSide.LONG,
                quantity="1",
                price="100",
            ),
            make_fill(
                trade_id="short-open",
                minute=1,
                action=FillAction.OPEN,
                position_side=PositionSide.SHORT,
                quantity="2",
                price="101",
            ),
            make_fill(
                trade_id="short-close",
                minute=2,
                action=FillAction.CLOSE,
                position_side=PositionSide.SHORT,
                quantity="2",
                price="99",
                realised_pnl="4",
            ),
            make_fill(
                trade_id="long-close",
                minute=3,
                action=FillAction.CLOSE,
                position_side=PositionSide.LONG,
                quantity="1",
                price="103",
                realised_pnl="3",
            ),
        ]

        result = self.reconstructor.reconstruct(fills)

        self.assertEqual(result.completed_deal_count, 2)
        self.assertEqual(result.open_position_count, 0)
        self.assertEqual(result.anomaly_count, 0)

        sides = {
            deal.position_side
            for deal in result.deals
        }

        self.assertEqual(
            sides,
            {
                PositionSide.LONG,
                PositionSide.SHORT,
            },
        )

    def test_open_position_is_reported_at_end_of_data(self) -> None:
        fills = [
            make_fill(
                trade_id="open",
                minute=0,
                action=FillAction.OPEN,
                position_side=PositionSide.LONG,
                quantity="3",
                price="100",
            ),
            make_fill(
                trade_id="partial-close",
                minute=1,
                action=FillAction.CLOSE,
                position_side=PositionSide.LONG,
                quantity="1",
                price="101",
                realised_pnl="1",
            ),
        ]

        result = self.reconstructor.reconstruct(fills)

        self.assertEqual(result.completed_deal_count, 0)
        self.assertEqual(result.open_position_count, 1)
        self.assertEqual(result.anomaly_count, 0)
        self.assertEqual(
            result.open_positions[0].open_quantity,
            Decimal("2"),
        )

    def test_orphan_close_is_reported_as_anomaly(self) -> None:
        fills = [
            make_fill(
                trade_id="close",
                minute=0,
                action=FillAction.CLOSE,
                position_side=PositionSide.LONG,
                quantity="2",
                price="100",
                realised_pnl="1",
            )
        ]

        result = self.reconstructor.reconstruct(fills)

        self.assertEqual(result.completed_deal_count, 0)
        self.assertEqual(result.open_position_count, 0)
        self.assertEqual(result.anomaly_count, 1)
        self.assertEqual(
            result.anomalies[0].anomaly_type,
            "orphan_close",
        )

    def test_over_close_completes_deal_and_reports_remainder(self) -> None:
        fills = [
            make_fill(
                trade_id="open",
                minute=0,
                action=FillAction.OPEN,
                position_side=PositionSide.LONG,
                quantity="2",
                price="100",
                fee="0.20",
            ),
            make_fill(
                trade_id="close",
                minute=1,
                action=FillAction.CLOSE,
                position_side=PositionSide.LONG,
                quantity="3",
                price="110",
                realised_pnl="30",
                fee="0.30",
            ),
        ]

        result = self.reconstructor.reconstruct(fills)

        self.assertEqual(result.completed_deal_count, 1)
        self.assertEqual(result.open_position_count, 0)
        self.assertEqual(result.anomaly_count, 1)

        deal = result.deals[0]

        self.assertEqual(deal.quantity, Decimal("2"))
        self.assertEqual(
            deal.exchange_realised_pnl,
            Decimal("20"),
        )
        self.assertEqual(
            deal.exit_leg.fees,
            Decimal("0.20"),
        )
        self.assertEqual(
            result.anomalies[0].quantity,
            Decimal("1"),
        )

    def test_input_order_does_not_change_result(self) -> None:
        open_fill = make_fill(
            trade_id="open",
            minute=0,
            action=FillAction.OPEN,
            position_side=PositionSide.LONG,
            quantity="1",
            price="100",
        )
        close_fill = make_fill(
            trade_id="close",
            minute=1,
            action=FillAction.CLOSE,
            position_side=PositionSide.LONG,
            quantity="1",
            price="105",
            realised_pnl="5",
        )

        result = self.reconstructor.reconstruct(
            [close_fill, open_fill]
        )

        self.assertEqual(result.completed_deal_count, 1)
        self.assertEqual(result.anomaly_count, 0)


if __name__ == "__main__":
    unittest.main()
