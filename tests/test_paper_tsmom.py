from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from hf_research.paper_tsmom import TsmomPaperConfig, daily_returns, due_daily_decision
from hf_research.paper_xs import PaperBook

_spec = importlib.util.spec_from_file_location(
    "paper_tsmom", Path(__file__).resolve().parents[1] / "scripts" / "paper_tsmom.py"
)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

CFG = TsmomPaperConfig()
T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
BTC, ETH = "BTC/USDT:USDT", "ETH/USDT:USDT"


class ScheduleTests(unittest.TestCase):
    def test_daily_fill_at_0100_after_the_decision_day(self):
        start = T("2026-10-21 08:00")
        self.assertEqual(due_daily_decision(T("2026-10-21 23:30"), start, None, CFG)[1], "idle")
        self.assertEqual(due_daily_decision(T("2026-10-22 00:59"), start, None, CFG)[1], "idle")
        self.assertEqual(due_daily_decision(T("2026-10-22 01:10"), start, None, CFG)[:2], (T("2026-10-21"), "due", )[:2])
        self.assertEqual(due_daily_decision(T("2026-10-22 05:00"), start, T("2026-10-21"), CFG)[1], "idle")

    def test_late_fill_is_missed_and_gaps_are_reported(self):
        d, status, skipped = due_daily_decision(T("2026-10-25 14:00"), T("2026-10-21 08:00"), T("2026-10-21"), CFG)
        self.assertEqual((d, status), (T("2026-10-24"), "missed"))  # fill 10-25 01:00, 13 h late
        self.assertEqual(skipped, [T("2026-10-22"), T("2026-10-23")])

    def test_first_window_never_reaches_before_the_start(self):
        d, status, skipped = due_daily_decision(T("2026-10-24 02:00"), T("2026-10-21 08:00"), None, CFG)
        self.assertEqual((d, status), (T("2026-10-23"), "due"))
        self.assertEqual(skipped, [T("2026-10-21"), T("2026-10-22")])


class DailyReturnsTests(unittest.TestCase):
    def test_first_mark_after_the_fill_hour_closes_the_previous_row(self):
        eq = pd.DataFrame({"ts": ["2026-10-22T00:10:00Z", "2026-10-22T01:10:00Z", "2026-10-22T05:10:00Z",
                                  "2026-10-23T01:10:00Z"], "equity": [10_000.0, 9_990.0, 9_000.0, 10_989.0]})
        r = daily_returns(eq, 10_000.0, 1)
        self.assertEqual(list(r.index), [T("2026-10-21"), T("2026-10-22")])
        self.assertAlmostEqual(r.iloc[0], -0.001)
        self.assertAlmostEqual(r.iloc[1], 0.1)


class FakeBitget:
    def __init__(self):
        self.mid = {BTC: 100_000.0, ETH: 4_000.0}
        self.funding = []

    def fetch_tickers(self, symbols):
        return {s: {"bid": self.mid[s] * 0.99995, "ask": self.mid[s] * 1.00005, "last": self.mid[s]} for s in symbols}

    def fetch_order_book(self, s, limit=100):
        m = self.mid[s]
        return {"bids": [[m * 0.99995, 1e6]], "asks": [[m * 1.00005, 1e6]]}

    def amount_to_precision(self, s, x):
        return f"{x:.4f}"

    def fetch_funding_rate_history(self, s, since=None, limit=100):
        return [{"timestamp": t, "fundingRate": r} for sym, t, r in self.funding if sym == s and t >= since]


class RunCycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ex = FakeBitget()
        self.weights = {BTC: 0.4, ETH: -0.3}
        self.calls = 0

        def fake_targets(d, cfg, cache_dir, log):
            self.calls += 1
            return {"weights": dict(self.weights), "signal_close": {}}

        self._orig = (runner._bitget, runner.compute_targets)
        runner._bitget = lambda: (self.ex, set(self.ex.mid))
        runner.compute_targets = fake_targets
        self.store = runner.Store(Path(self.tmp.name))
        start = T("2026-10-21 08:00")
        self.store.save_state({"config": runner.asdict(CFG), "strategy": runner.STRATEGY.to_dict(),
                               "start": runner._iso(start), "book": PaperBook(equity=10_000.0).to_dict(),
                               "last_decision_day": None, "last_run_ms": int(start.value // 1_000_000),
                               "funding_cursor": {}})

    def tearDown(self):
        runner._bitget, runner.compute_targets = self._orig
        self.tmp.cleanup()

    def _run(self, now: str) -> dict:
        runner.run_cycle(self.store, SimpleNamespace(now=now, dry_run=False, cache_dir="unused"))
        return self.store.load_state()

    def test_daily_rebalance_band_and_funding(self):
        s = self._run("2026-10-22 01:10")
        self.assertAlmostEqual(s["book"]["qty"][BTC], 0.04, places=4)
        self.assertAlmostEqual(s["book"]["qty"][ETH], -0.75, places=4)
        self.assertEqual(self._run("2026-10-22 02:10")["last_decision_day"], "2026-10-21")
        self.assertEqual(self.calls, 1)

        # Next day: BTC moves less than the 0.05 band (skipped), ETH flips (always trades).
        self.weights = {BTC: 0.43, ETH: 0.2}
        self.ex.funding.append((ETH, int(T("2026-10-22 08:00").value // 1_000_000), 0.0001))
        s = self._run("2026-10-23 01:10")
        self.assertEqual(self.calls, 2)
        self.assertAlmostEqual(s["book"]["qty"][BTC], 0.04, places=4)
        self.assertGreater(s["book"]["qty"][ETH], 0)
        fund = self.store.read("funding.jsonl")
        self.assertAlmostEqual(fund["pnl_usdt"].iloc[0], 0.75 * 4_000.0 * 0.0001)  # the short receives
        types = self.store.read("events.jsonl")["type"].tolist()
        self.assertEqual(types, ["rebalance", "rebalance"])

    def test_gap_days_are_logged_as_missed(self):
        self._run("2026-10-22 01:10")
        s = self._run("2026-10-25 02:00")
        self.assertEqual(s["last_decision_day"], "2026-10-24")
        types = self.store.read("events.jsonl")["type"].tolist()
        self.assertEqual(types, ["rebalance", "missed_rebalance", "missed_rebalance", "rebalance"])


if __name__ == "__main__":
    unittest.main()
