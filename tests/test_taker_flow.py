from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hf_research.oi_signals import rank_weights, taker_flow
from hf_research.taker_flow import daily_log_ratio, fetch_rows, scores_for_day, target_weights

_spec = importlib.util.spec_from_file_location(
    "paper_taker_flow", Path(__file__).resolve().parents[1] / "scripts" / "paper_taker_flow.py"
)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
SYMS = [f"S{i}/USDT:USDT" for i in range(10)]


class FakeBinance:
    """Constant buy/sell ratio per symbol (S0 most selling ... S9 most buying), 5-minute rows."""

    def __init__(self, missing=()):
        self.missing = set(missing)
        self.calls = 0

    def fapiDataGetTakerlongshortRatio(self, p):
        self.calls += 1
        sym = p["symbol"].replace("USDT", "/USDT:USDT")
        if sym in self.missing:
            return []
        ratio = 0.8 + 0.05 * SYMS.index(sym)
        ts = range(p["startTime"], p["endTime"] + 1, 300_000)
        assert len(ts) <= 500
        return [{"timestamp": str(t), "buySellRatio": str(ratio)} for t in ts]


class TakerFlowTests(unittest.TestCase):
    def test_daily_mean_of_log_ratio_by_utc_day(self):
        rows = [{"timestamp": T("2026-10-18 23:55").value // 1_000_000, "buySellRatio": "2.0"},
                {"timestamp": T("2026-10-19 00:00").value // 1_000_000, "buySellRatio": "1.0"},
                {"timestamp": T("2026-10-19 00:05").value // 1_000_000, "buySellRatio": "4.0"},
                {"timestamp": T("2026-10-19 00:10").value // 1_000_000, "buySellRatio": "0"}]
        s = daily_log_ratio(rows)
        self.assertAlmostEqual(s[T("2026-10-18")], np.log(2.0))
        self.assertAlmostEqual(s[T("2026-10-19")], np.log(4.0) / 2)

    def test_paged_fetch_covers_the_window_without_gaps(self):
        ex = FakeBinance()
        rows = fetch_rows(ex, SYMS[3], T("2026-10-19"), T("2026-10-25 23:55"))
        self.assertEqual(len(rows), 7 * 288)
        self.assertEqual(len({r["timestamp"] for r in rows}), 7 * 288)

    def test_scores_match_the_backtest_definition(self):
        days = pd.date_range("2026-10-13", "2026-10-25", freq="D", tz="UTC")
        rng = np.random.default_rng(0)
        daily = pd.DataFrame(rng.normal(size=(len(days), 3)), index=days, columns=["A", "B", "C"])
        daily.iloc[-3:, 2] = np.nan  # C has only 4 of the last 7 days: no score
        got = scores_for_day(daily, T("2026-10-25"))
        want = taker_flow(daily, 7).loc[T("2026-10-25")]
        self.assertTrue(np.allclose(got[["A", "B"]], want[["A", "B"]]))
        self.assertTrue(np.isnan(got["C"]))

    def test_targets_long_buyers_short_sellers(self):
        w = target_weights(pd.Series(np.arange(10.0), index=SYMS))
        self.assertEqual(sorted(w[w > 0].index), SYMS[8:])
        self.assertEqual(sorted(w[w < 0].index), SYMS[:2])
        ref = rank_weights(pd.Series(np.arange(10.0), index=SYMS).to_frame().T,
                           pd.DataFrame(True, index=[0], columns=SYMS), select_frac=0.2, long_high=True).iloc[0]
        self.assertTrue(np.allclose(w.reindex(SYMS, fill_value=0.0), ref))


class ComputeTargetsTests(unittest.TestCase):
    def setUp(self):
        self._orig = (runner.xs_signal.binance_exchange, runner.update_cache)
        self.ex = FakeBinance(missing=[SYMS[9]])
        runner.xs_signal.binance_exchange = lambda: self.ex
        runner.update_cache = lambda *a, **k: None

    def tearDown(self):
        runner.xs_signal.binance_exchange, runner.update_cache = self._orig

    def test_targets_skip_symbols_without_data_and_not_on_bitget(self):
        listed = {s for s in SYMS if s != SYMS[0]}
        tgt = runner.compute_targets(T("2026-10-25"), SYMS, listed, "unused", lambda m: None)
        w = pd.Series(tgt["weights"])
        # Scored: S1..S8 (S0 not on Bitget, S9 no data) -> long S7, S8; short S1, S2.
        self.assertEqual(sorted(w[w > 0].index), SYMS[7:9])
        self.assertEqual(sorted(w[w < 0].index), SYMS[1:3])
        self.assertEqual(tgt["not_on_bitget"], [SYMS[0]])
        self.assertAlmostEqual(tgt["coverage"], 8 / 9)

    def test_low_coverage_postpones(self):
        self.ex.missing = set(SYMS)
        with self.assertRaises(RuntimeError):
            runner.compute_targets(T("2026-10-25"), SYMS, set(SYMS), "unused", lambda m: None)


if __name__ == "__main__":
    unittest.main()
