from __future__ import annotations

import unittest
from datetime import timezone
from decimal import Decimal

from hf_audit.deal_reconstruction.domain.enums import (
    FillAction,
    FillOrigin,
    PositionSide,
)
from hf_audit.deal_reconstruction.infrastructure.bitget_fill_normalizer import (
    BitgetFillNormalizer,
)


def make_trade(
    *,
    trade_id: str = "trade-1",
    order_id: str = "order-1",
    symbol: str = "BTC/USDT:USDT",
    trade_side: str = "open",
    side: str = "buy",
    source: str = "api",
    amount: str = "0.5",
    price: str = "60000",
    profit: str = "0",
    fee_detail=None,
    timestamp: int = 1784160000000,
):
    info = {
        "tradeId": trade_id,
        "orderId": order_id,
        "symbol": "BTCUSDT",
        "tradeSide": trade_side,
        "side": side,
        "enterPointSource": source,
        "baseVolume": amount,
        "price": price,
        "profit": profit,
        "cTime": str(timestamp),
        "tradeScope": "taker",
        "posMode": "hedge_mode",
    }

    if fee_detail is not None:
        info["feeDetail"] = fee_detail

    return {
        "id": trade_id,
        "order": order_id,
        "symbol": symbol,
        "timestamp": timestamp,
        "datetime": "2026-07-16T00:00:00.000Z",
        "side": side,
        "amount": float(amount),
        "price": float(price),
        "fee": None,
        "info": info,
    }


class BitgetFillNormalizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = BitgetFillNormalizer()

    def test_open_long_api(self) -> None:
        fill = self.normalizer.normalize(
            make_trade(
                trade_side="open",
                side="buy",
            )
        )

        self.assertEqual(fill.action, FillAction.OPEN)
        self.assertEqual(fill.position_side, PositionSide.LONG)
        self.assertEqual(fill.origin, FillOrigin.API)
        self.assertEqual(fill.quantity, Decimal("0.5"))
        self.assertEqual(fill.price, Decimal("60000.0"))
        self.assertEqual(fill.realised_pnl, Decimal("0"))
        self.assertEqual(fill.timestamp.tzinfo, timezone.utc)

    def test_open_short_api(self) -> None:
        fill = self.normalizer.normalize(
            make_trade(
                trade_side="open",
                side="sell",
            )
        )

        self.assertEqual(fill.action, FillAction.OPEN)
        self.assertEqual(fill.position_side, PositionSide.SHORT)

    def test_close_long_sell(self) -> None:
        fill = self.normalizer.normalize(
            make_trade(
                trade_side="close",
                side="sell",
                profit="12.75",
            )
        )

        self.assertEqual(fill.action, FillAction.CLOSE)
        self.assertEqual(fill.position_side, PositionSide.LONG)
        self.assertEqual(fill.realised_pnl, Decimal("12.75"))

    def test_close_short_buy(self) -> None:
        fill = self.normalizer.normalize(
            make_trade(
                trade_side="close",
                side="buy",
                profit="-2.5",
            )
        )

        self.assertEqual(fill.action, FillAction.CLOSE)
        self.assertEqual(fill.position_side, PositionSide.SHORT)
        self.assertEqual(fill.realised_pnl, Decimal("-2.5"))

    def test_ios_manual_close(self) -> None:
        fill = self.normalizer.normalize(
            make_trade(
                trade_side="close",
                side="buy",
                source="ios",
                profit="0.18160",
            )
        )

        self.assertEqual(fill.origin, FillOrigin.IOS)
        self.assertEqual(fill.action, FillAction.CLOSE)

    def test_json_fee_detail_is_converted_to_positive_cost(self) -> None:
        fill = self.normalizer.normalize(
            make_trade(
                trade_side="close",
                side="sell",
                fee_detail='{"fee": "-0.025"}',
            )
        )

        self.assertEqual(fill.fee, Decimal("0.025"))

    def test_ccxt_fee_is_supported(self) -> None:
        trade = make_trade(
            trade_side="close",
            side="sell",
        )
        trade["fee"] = {
            "cost": -0.04,
            "currency": "USDT",
        }

        fill = self.normalizer.normalize(trade)

        self.assertEqual(fill.fee, Decimal("0.04"))
        self.assertEqual(fill.metadata["ccxt_fee_currency"], "USDT")

    def test_open_profit_is_forced_to_zero(self) -> None:
        fill = self.normalizer.normalize(
            make_trade(
                trade_side="open",
                side="buy",
                profit="999",
            )
        )

        self.assertEqual(fill.realised_pnl, Decimal("0"))

    def test_unknown_origin_is_supported(self) -> None:
        fill = self.normalizer.normalize(
            make_trade(
                source="desktop_terminal",
            )
        )

        self.assertEqual(fill.origin, FillOrigin.UNKNOWN)

    def test_invalid_trade_side_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.normalizer.normalize(
                make_trade(
                    trade_side="unexpected",
                )
            )

    def test_invalid_order_side_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.normalizer.normalize(
                make_trade(
                    side="hold",
                )
            )

    def test_missing_trade_id_is_rejected(self) -> None:
        trade = make_trade()
        trade["id"] = ""
        trade["info"]["tradeId"] = ""

        with self.assertRaises(ValueError):
            self.normalizer.normalize(trade)


if __name__ == "__main__":
    unittest.main()
