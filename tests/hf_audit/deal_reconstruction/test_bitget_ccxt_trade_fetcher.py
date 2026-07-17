"""Tests for the read-only CCXT Bitget trade fetcher."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from hf_audit.deal_reconstruction.infrastructure.bitget_ccxt_trade_fetcher import (
    BitgetCcxtTradeFetcher,
)


UTC = timezone.utc


class FakeExchange:
    rateLimit = 0

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def fetch_my_trades(self, *, symbol, since, limit):
        self.calls.append(
            {
                "symbol": symbol,
                "since": since,
                "limit": limit,
            }
        )

        if not self._responses:
            return []

        return self._responses.pop(0)


class BitgetCcxtTradeFetcherTest(unittest.TestCase):
    def setUp(self):
        self.start = datetime(
            2026,
            7,
            16,
            0,
            0,
            0,
            tzinfo=UTC,
        )
        self.end = datetime(
            2026,
            7,
            16,
            1,
            0,
            0,
            tzinfo=UTC,
        )

    def test_fetches_and_orders_records_chronologically(self):
        exchange = FakeExchange(
            [
                [
                    {
                        "id": "trade-2",
                        "order": "order-2",
                        "timestamp": 1784161800000,
                        "symbol": "BTC/USDT:USDT",
                        "info": {
                            "tradeId": "trade-2",
                            "tradeSide": "close",
                        },
                    },
                    {
                        "id": "trade-1",
                        "order": "order-1",
                        "timestamp": 1784160900000,
                        "symbol": "BTC/USDT:USDT",
                        "info": {
                            "tradeId": "trade-1",
                            "tradeSide": "open",
                        },
                    },
                ]
            ]
        )

        fetcher = BitgetCcxtTradeFetcher(
            exchange,
            page_limit=100,
        )

        result = list(
            fetcher.fetch_trades(
                symbols=["BTC/USDT:USDT"],
                start=self.start,
                end=self.end,
            )
        )

        self.assertEqual(
            [record["id"] for record in result],
            ["trade-1", "trade-2"],
        )
        self.assertEqual(
            result[0]["requested_symbol"],
            "BTC/USDT:USDT",
        )
        self.assertEqual(
            result[0]["_source"],
            "fetch_my_trades",
        )
        self.assertEqual(len(exchange.calls), 1)

    def test_paginates_and_removes_duplicate_boundary_trade(self):
        first_timestamp = 1784160900000
        second_timestamp = 1784161800000
        third_timestamp = 1784162700000

        duplicate = {
            "id": "trade-2",
            "order": "order-2",
            "timestamp": second_timestamp,
            "symbol": "ETH/USDT:USDT",
            "info": {
                "tradeId": "trade-2",
                "orderId": "order-2",
            },
        }

        exchange = FakeExchange(
            [
                [
                    {
                        "id": "trade-1",
                        "order": "order-1",
                        "timestamp": first_timestamp,
                        "symbol": "ETH/USDT:USDT",
                        "info": {
                            "tradeId": "trade-1",
                            "orderId": "order-1",
                        },
                    },
                    duplicate,
                ],
                [
                    duplicate,
                    {
                        "id": "trade-3",
                        "order": "order-3",
                        "timestamp": third_timestamp,
                        "symbol": "ETH/USDT:USDT",
                        "info": {
                            "tradeId": "trade-3",
                            "orderId": "order-3",
                        },
                    },
                ],
                [],
            ]
        )

        fetcher = BitgetCcxtTradeFetcher(
            exchange,
            page_limit=2,
            rate_limit_seconds=0,
        )

        result = list(
            fetcher.fetch_trades(
                symbols=["ETH/USDT:USDT"],
                start=self.start,
                end=self.end,
            )
        )

        self.assertEqual(
            [record["id"] for record in result],
            ["trade-1", "trade-2", "trade-3"],
        )
        self.assertEqual(len(exchange.calls), 3)
        self.assertEqual(
            exchange.calls[1]["since"],
            second_timestamp + 1,
        )

    def test_excludes_rows_outside_requested_window(self):
        exchange = FakeExchange(
            [
                [
                    {
                        "id": "before",
                        "order": "order-before",
                        "timestamp": 1784159999000,
                        "symbol": "SOL/USDT:USDT",
                        "info": {},
                    },
                    {
                        "id": "inside",
                        "order": "order-inside",
                        "timestamp": 1784161800000,
                        "symbol": "SOL/USDT:USDT",
                        "info": {},
                    },
                    {
                        "id": "at-end",
                        "order": "order-at-end",
                        "timestamp": 1784163600000,
                        "symbol": "SOL/USDT:USDT",
                        "info": {},
                    },
                ]
            ]
        )

        fetcher = BitgetCcxtTradeFetcher(exchange)

        result = list(
            fetcher.fetch_trades(
                symbols=["SOL/USDT:USDT"],
                start=self.start,
                end=self.end,
            )
        )

        self.assertEqual(
            [record["id"] for record in result],
            ["inside"],
        )

    def test_uses_nested_bitget_ctime_when_ccxt_timestamp_missing(self):
        exchange = FakeExchange(
            [
                [
                    {
                        "id": "trade-info-time",
                        "order": "order-info-time",
                        "symbol": "BNB/USDT:USDT",
                        "info": {
                            "tradeId": "trade-info-time",
                            "orderId": "order-info-time",
                            "cTime": "1784161800000",
                        },
                    }
                ]
            ]
        )

        fetcher = BitgetCcxtTradeFetcher(exchange)

        result = list(
            fetcher.fetch_trades(
                symbols=["BNB/USDT:USDT"],
                start=self.start,
                end=self.end,
            )
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0]["id"],
            "trade-info-time",
        )

    def test_rejects_invalid_time_range(self):
        exchange = FakeExchange([])
        fetcher = BitgetCcxtTradeFetcher(exchange)

        with self.assertRaisesRegex(
            ValueError,
            "end must be later than start",
        ):
            list(
                fetcher.fetch_trades(
                    symbols=["BTC/USDT:USDT"],
                    start=self.end,
                    end=self.start,
                )
            )

        self.assertEqual(exchange.calls, [])

    def test_rejects_empty_symbol_collection(self):
        exchange = FakeExchange([])
        fetcher = BitgetCcxtTradeFetcher(exchange)

        with self.assertRaisesRegex(
            ValueError,
            "at least one symbol is required",
        ):
            list(
                fetcher.fetch_trades(
                    symbols=[],
                    start=self.start,
                    end=self.end,
                )
            )

        self.assertEqual(exchange.calls, [])


if __name__ == "__main__":
    unittest.main()
