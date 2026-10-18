from __future__ import annotations

import sys
import tempfile
import types
import unittest
from unittest import mock

import pandas as pd

from hf.data import ohlcv
from hf_core.trade_lifecycle.engine import TradeLifecycleEngine

H = 3_600_000


def _bars(start_ts: int, n: int, close: float) -> list[list[float]]:
    return [[start_ts + i * H, close, close, close, close, 1.0] for i in range(n)]


class FakeExchange:
    def __init__(self, rows: list[list[float]]):
        self.rows = rows

    def load_markets(self):
        return {}

    def fetch_ohlcv(self, symbol, timeframe="1h", since=0, limit=500):
        return [r for r in self.rows if r[0] >= since][:limit]


class TimeframeTests(unittest.TestCase):
    def test_timeframe_to_ms(self):
        self.assertEqual(ohlcv.timeframe_to_ms("1h"), H)
        self.assertEqual(ohlcv.timeframe_to_ms("15m"), 15 * 60_000)
        self.assertEqual(ohlcv.timeframe_to_ms("1d"), 24 * H)
        with self.assertRaises(ValueError):
            ohlcv.timeframe_to_ms("1x")

    def test_drop_unclosed_candles(self):
        df = pd.DataFrame({"timestamp": [0, H, 2 * H], "close": [1.0, 2.0, 3.0]})
        # At 2h30m only the 0h and 1h candles have closed.
        out = ohlcv.drop_unclosed_candles(df, "1h", now_ms=2 * H + H // 2)
        self.assertEqual(out["timestamp"].tolist(), [0, H])


class FetchCacheTests(unittest.TestCase):
    def _run_fetch(self, exchange_rows, cached_rows, now_ms, start_ms=0, end_ms=None):
        fake_ccxt = types.SimpleNamespace(fakeex=lambda cfg: FakeExchange(exchange_rows))
        with tempfile.TemporaryDirectory() as tmp:
            path = ohlcv._cache_path(tmp, "fakeex", "TST/USDT:USDT", "1h")
            path.parent.mkdir(parents=True, exist_ok=True)
            if cached_rows:
                pd.DataFrame(cached_rows, columns=["timestamp", "open", "high", "low", "close", "volume"]).to_csv(
                    path, index=False, compression="gzip"
                )
            with mock.patch.dict(sys.modules, {"ccxt": fake_ccxt}), mock.patch.object(
                ohlcv.time, "time", return_value=now_ms / 1000
            ):
                out = ohlcv.fetch_ohlcv_ccxt(
                    "TST/USDT:USDT", "1h", start_ms, end_ms, exchange_id="fakeex", cache_dir=tmp, refresh_if_no_end=True
                )
            stored = pd.read_csv(path)
        return out, stored

    def test_forming_candle_is_neither_returned_nor_persisted(self):
        rows = _bars(0, 5, 10.0)  # candle 4h is still forming at 4h20m
        out, stored = self._run_fetch(rows, None, now_ms=4 * H + 20 * 60_000)
        self.assertEqual(int(out["timestamp"].max()), 3 * H)
        self.assertEqual(int(stored["timestamp"].max()), 3 * H)

    def test_fresh_exchange_rows_overwrite_partial_cached_candle(self):
        # Cache holds a partial 3h candle (close 9.0); the exchange now has the final value.
        cached = _bars(0, 3, 10.0) + [[3 * H, 10.0, 10.0, 9.0, 9.0, 0.1]]
        exchange = _bars(0, 6, 10.0)
        out, stored = self._run_fetch(exchange, cached, now_ms=6 * H)
        row = stored[stored["timestamp"] == 3 * H].iloc[0]
        self.assertEqual(float(row["close"]), 10.0)
        self.assertEqual(float(row["volume"]), 1.0)
        self.assertEqual(int(stored["timestamp"].max()), 5 * H)


class ExitSlippageTests(unittest.TestCase):
    def _close(self, side: str, slippage_bps: float) -> float:
        eng = TradeLifecycleEngine(taker_fee=0.0, exit_slippage_bps=slippage_bps)
        eng.open_position(
            symbol="TST", strategy_id="s", side=side, entry_ts=0, entry_px=100.0, qty=1.0, entry_atr=1.0
        )
        rec = eng.force_close(symbol="TST", exit_ts=H, exit_price=100.0)
        return rec.exit_px

    def test_slippage_is_adverse_for_both_sides(self):
        self.assertAlmostEqual(self._close("long", 10.0), 99.9)
        self.assertAlmostEqual(self._close("short", 10.0), 100.1)

    def test_zero_slippage_keeps_price(self):
        self.assertEqual(self._close("long", 0.0), 100.0)


if __name__ == "__main__":
    unittest.main()
