from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from hf_research.paper_xs import (
    PaperBook,
    PaperConfig,
    due_decision_day,
    first_decision_day_after,
    plan_weights,
    scheduled_fill,
    walk_book,
)

_spec = importlib.util.spec_from_file_location(
    "paper_xs_momentum", Path(__file__).resolve().parents[1] / "scripts" / "paper_xs_momentum.py"
)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

CFG = PaperConfig()
T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


class ScheduleTests(unittest.TestCase):
    def test_decision_on_sunday_fill_monday_0100(self):
        self.assertEqual(scheduled_fill(T("2026-10-25"), CFG), T("2026-10-26 01:00"))
        self.assertEqual(first_decision_day_after(T("2026-10-19 06:55"), CFG), T("2026-10-25"))
        self.assertEqual(first_decision_day_after(T("2026-10-19 00:30"), CFG), T("2026-10-18"))

    def test_due_idle_and_missed(self):
        start = T("2026-10-19 07:00")
        self.assertEqual(due_decision_day(T("2026-10-20 10:00"), start, None, CFG), (None, "idle"))
        self.assertEqual(due_decision_day(T("2026-10-26 00:59"), start, None, CFG), (None, "idle"))
        self.assertEqual(due_decision_day(T("2026-10-26 01:05"), start, None, CFG), (T("2026-10-25"), "due"))
        self.assertEqual(due_decision_day(T("2026-10-26 03:05"), start, T("2026-10-25"), CFG), (None, "idle"))
        self.assertEqual(due_decision_day(T("2026-10-27 02:00"), start, None, CFG), (T("2026-10-25"), "missed"))


class BookTests(unittest.TestCase):
    def test_walk_book_averages_levels_and_flags_thin_books(self):
        asks = [[100.0, 1.0], [101.0, 1.0], [103.0, 5.0]]
        avg, got, exhausted = walk_book(asks, 2.5)
        self.assertAlmostEqual(avg, (100 + 101 + 0.5 * 103) / 2.5)
        self.assertFalse(exhausted)
        avg, got, exhausted = walk_book(asks, 10.0)
        self.assertTrue(exhausted)
        self.assertAlmostEqual(got, 7.0)
        self.assertAlmostEqual(avg, (100 + 101 + 5 * 103 + 3 * 103) / 10.0)

    def test_accounting_round_trip(self):
        b = PaperBook(equity=1000.0)
        r = b.trade("A", 5.0, 100.5, 100.0, 0.001)  # buy 500 notional, 0.5 above mid
        self.assertAlmostEqual(r["slippage_usdt"], 2.5)
        self.assertAlmostEqual(r["slippage_bps"], 50.0)
        self.assertAlmostEqual(b.equity, 1000 - 2.5 - 5 * 100.5 * 0.001)
        e0 = b.equity
        b.mark({"A": 110.0})
        self.assertAlmostEqual(b.equity, e0 + 50.0)
        self.assertAlmostEqual(b.funding("A", 0.0001, 110.0), -5 * 110 * 0.0001)  # long pays
        r = b.trade("A", -5.0, 109.0, 110.0, 0.0)  # sell 1 below mid
        self.assertAlmostEqual(r["slippage_bps"], 1 / 110 * 1e4)
        self.assertEqual(b.qty, {})
        self.assertAlmostEqual(b.equity, e0 + 50.0 - 0.055 - 5.0)

    def test_short_receives_positive_funding(self):
        b = PaperBook(equity=1000.0, qty={"A": -2.0}, last_px={"A": 50.0})
        self.assertAlmostEqual(b.funding("A", 0.001, 50.0), 0.1)
        self.assertAlmostEqual(b.equity, 1000.1)

    def test_plan_weights_band_matches_backtest_rule(self):
        plan = plan_weights({"A": 0.05, "B": -0.05, "C": 0.04}, {"A": 0.04, "B": 0.03, "D": -0.02}, band=0.02)
        self.assertNotIn("A", plan)  # small same-side resize skipped
        self.assertEqual(plan["B"], (0.03, -0.05))  # flip trades
        self.assertEqual(plan["C"], (0.0, 0.04))  # open trades even below the band
        self.assertEqual(plan["D"], (-0.02, 0.0))  # close trades


class RebalanceOnTests(unittest.TestCase):
    def test_backtest_rebalances_only_on_given_days(self):
        from hf_research.xs_momentum import MomentumConfig, backtest

        idx = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
        rng = np.random.default_rng(0)
        px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (60, 8)), axis=0)), index=idx,
                          columns=[f"A{k}" for k in range(8)])
        panels = {"signal_close": px, "exec_close": px, "funding": px * 0.0}
        cfg = MomentumConfig(lookback_days=5, rebalance_days=1, select_frac=0.25, min_assets=4, vol_window_days=10)
        days = [idx[20], idx[40]]
        d = backtest(panels, cfg, rebalance_on=days)
        self.assertEqual(list(d.index[d["turnover"] > 0]), days)


class FakeBitget:
    """Two markets with a static book around a moving mid and one funding event per call window."""

    def __init__(self):
        self.mid = {"AAA/USDT:USDT": 10.0, "BBB/USDT:USDT": 20.0}
        self.funding = []  # (symbol, ts_ms, rate)

    def fetch_tickers(self, symbols):
        return {s: {"bid": self.mid[s] * 0.9999, "ask": self.mid[s] * 1.0001, "last": self.mid[s]} for s in symbols}

    def fetch_order_book(self, s, limit=100):
        m = self.mid[s]
        return {"bids": [[m * 0.9999, 1e6]], "asks": [[m * 1.0001, 1e6]]}

    def amount_to_precision(self, s, x):
        return f"{x:.3f}"

    def fetch_funding_rate_history(self, s, since=None, limit=100):
        return [{"timestamp": t, "fundingRate": r} for sym, t, r in self.funding if sym == s and t >= since]


class RunCycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ex = FakeBitget()
        self.calls = 0

        def fake_targets(d, members, listed, cache_dir, log):
            self.calls += 1
            return {"weights": {"AAA/USDT:USDT": 0.5, "BBB/USDT:USDT": -0.5}, "eligible": members,
                    "not_on_bitget": [], "signal_close": {}}

        self._orig = (runner._bitget, runner.compute_targets, runner.ensure_membership)
        runner._bitget = lambda: (self.ex, set(self.ex.mid))
        runner.compute_targets = fake_targets
        runner.ensure_membership = lambda store, month, cfg, cache_dir, log: pd.DataFrame(
            {"month": month, "symbol": ["AAA/USDT:USDT", "BBB/USDT:USDT"], "rank": [1, 2]})
        self.store = runner.Store(Path(self.tmp.name))
        start = T("2026-10-19 07:00")
        self.store.save_state({"config": runner.asdict(CFG), "strategy": runner.PRIMARY_STRATEGY, "start": runner._iso(start),
                               "book": PaperBook(equity=10_000.0).to_dict(), "last_decision_day": None,
                               "last_run_ms": int(start.value // 1_000_000), "funding_cursor": {}})

    def tearDown(self):
        runner._bitget, runner.compute_targets, runner.ensure_membership = self._orig
        self.tmp.cleanup()

    def _run(self, now: str):
        runner.run_cycle(self.store, SimpleNamespace(now=now, dry_run=False, cache_dir="unused"))
        return self.store.load_state()

    def test_rebalance_once_then_funding_and_marks(self):
        s = self._run("2026-10-20 10:05")
        self.assertEqual(s["book"]["qty"], {})
        s = self._run("2026-10-26 01:05")
        self.assertEqual(self.calls, 1)
        self.assertAlmostEqual(s["book"]["qty"]["AAA/USDT:USDT"], 500.0, places=2)
        self.assertAlmostEqual(s["book"]["qty"]["BBB/USDT:USDT"], -250.0, places=2)
        e_after = s["book"]["equity"]
        self.assertAlmostEqual(e_after, 10_000 - 10_000 * (0.0001 + 0.0006), delta=0.05)

        s = self._run("2026-10-26 02:05")  # idempotent within the week
        self.assertEqual(self.calls, 1)

        self.ex.funding.append(("AAA/USDT:USDT", int(T("2026-10-26 08:00").value // 1_000_000), 0.001))
        self.ex.mid["AAA/USDT:USDT"] = 11.0
        s = self._run("2026-10-26 08:05")
        # +500 from AAA rising 1.0 on 500 units; AAA long pays 500 * 11 * 0.001 = 5.5.
        self.assertAlmostEqual(s["book"]["equity"], e_after + 500.0 - 5.5, delta=0.05)
        s = self._run("2026-10-26 09:05")  # the same funding event is not booked twice
        self.assertAlmostEqual(s["book"]["equity"], e_after + 500.0 - 5.5, delta=0.05)

        fills = self.store.read("fills.jsonl")
        self.assertEqual(len(fills), 2)
        self.assertTrue(np.allclose(fills["slippage_bps"], 1.0, atol=1e-6))
        events = self.store.read("events.jsonl")
        self.assertEqual(events["type"].tolist(), ["rebalance"])
        eq = [json.loads(x) for x in self.store.path("equity.jsonl").read_text().splitlines()]
        self.assertEqual(len(eq), 5)

    def test_missed_rebalance_is_logged_and_not_traded(self):
        s = self._run("2026-10-27 03:00")
        self.assertEqual(self.calls, 0)
        self.assertEqual(s["last_decision_day"], "2026-10-25")
        self.assertEqual(self.store.read("events.jsonl")["type"].tolist(), ["missed_rebalance"])


class ReportAlignmentTests(unittest.TestCase):
    def test_paper_that_follows_the_model_tracks_it_exactly(self):
        from hf_research.xs_momentum import MomentumConfig, backtest, load_daily_panels

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            (cache / "binanceusdm" / "1h").mkdir(parents=True)
            t = pd.date_range("2026-08-01", "2026-10-31 23:00", freq="h", tz="UTC")
            rng = np.random.default_rng(5)
            syms = [f"S{k}/USDT:USDT" for k in range(10)]
            for k, sym in enumerate(syms):
                px = 100 * np.exp(np.cumsum(rng.normal(0.0002 * (k - 5), 0.004, len(t))))
                pd.DataFrame({"timestamp": t.as_unit("ms").asi8, "open": px, "high": px, "low": px, "close": px,
                              "volume": 1.0}).to_csv(
                    cache / "binanceusdm" / "1h" / f"{sym.replace('/', '_').replace(':', '_')}.csv.gz",
                    index=False, compression="gzip")
            days = [T("2026-10-04"), T("2026-10-11"), T("2026-10-18")]
            panels = load_daily_panels(syms, cache_dir=str(cache), exchange="binanceusdm")
            elig = pd.DataFrame(True, index=panels["signal_close"].index, columns=syms)
            model = backtest(dict(panels, eligible=elig), MomentumConfig(**runner.PRIMARY_STRATEGY), rebalance_on=days)
            model = model[model.index >= days[0]]["net"]

            store = runner.Store(Path(tmp) / "paper")
            store.root.mkdir()
            store.save_state({"config": {**runner.asdict(CFG), "capital_usdt": 1.0}, "start": "2026-10-04T00:00:00Z",
                              "book": {"equity": 1.0}, "last_run": None})
            # Equity snapshot on day t at 01:05 closes model row t-1.
            eq = (1.0 + model).cumprod()
            store.append("equity.jsonl", [{"ts": runner._iso(d + pd.Timedelta(days=1, hours=1, minutes=5)), "equity": float(v)}
                                          for d, v in eq.items()])
            store.append("events.jsonl", [{"type": "rebalance", "decision_day": str(d.date()), "delay_hours": 0.0,
                                           "eligible": syms, "bitget_map": {}} for d in days])
            out = runner.report(store, SimpleNamespace(cache_dir=str(cache)))
        vm = out["vs_model"]
        self.assertGreaterEqual(vm["days"], 10)  # synthetic bars after the real clock are dropped as unclosed
        self.assertAlmostEqual(vm["daily_corr"], 1.0, places=9)
        self.assertAlmostEqual(vm["implementation_gap_ann"], 0.0, places=9)


class ReplayDataTests(unittest.TestCase):
    def test_missing_held_prices_flags_only_rows_of_held_symbols(self):
        from hf_research.replay_data import held_symbols, missing_held_prices

        idx = pd.date_range("2026-10-25", "2026-11-03", freq="D", tz="UTC")
        exe = pd.DataFrame({"A": 1.0, "B": 1.0}, index=idx)
        exe.loc[T("2026-10-30"):, "B"] = np.nan  # B left the universe: its cache stopped
        days = pd.Series([T("2026-10-25"), T("2026-11-01")])
        targets = [{"A": 0.5, "B": -0.5}, {"A": 0.5, "C": -0.5}]
        rows = pd.date_range("2026-10-25", "2026-11-02", freq="D", tz="UTC")
        got = missing_held_prices(exe, days, targets, rows)
        self.assertEqual(list(got), ["2026-10-30", "2026-10-31", "2026-11-01", "2026-11-02"])
        self.assertEqual(got["2026-10-30"], ["B"])
        self.assertEqual(got["2026-11-02"], ["C"])  # C is held after the 11-01 fill and was never cached
        self.assertEqual(held_symbols(pd.DataFrame({"targets": targets})), ["A", "B", "C"])
        self.assertEqual(held_symbols(pd.DataFrame({"eligible": [["A"]]})), [])


class ReportMissingPriceTests(unittest.TestCase):
    def test_comparison_stops_where_a_held_symbol_lacks_prices(self):
        from hf_research.xs_momentum import MomentumConfig, backtest, load_daily_panels

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            (cache / "binanceusdm" / "1h").mkdir(parents=True)
            t = pd.date_range("2026-08-01", "2026-10-31 23:00", freq="h", tz="UTC")
            rng = np.random.default_rng(5)
            syms = [f"S{k}/USDT:USDT" for k in range(10)]
            paths = {}
            for k, sym in enumerate(syms):
                px = 100 * np.exp(np.cumsum(rng.normal(0.0002 * (k - 5), 0.004, len(t))))
                paths[sym] = cache / "binanceusdm" / "1h" / f"{sym.replace('/', '_').replace(':', '_')}.csv.gz"
                pd.DataFrame({"timestamp": t.as_unit("ms").asi8, "open": px, "high": px, "low": px, "close": px,
                              "volume": 1.0}).to_csv(paths[sym], index=False, compression="gzip")
            days = [T("2026-10-04"), T("2026-10-11"), T("2026-10-18")]
            panels = load_daily_panels(syms, cache_dir=str(cache), exchange="binanceusdm")
            elig = pd.DataFrame(True, index=panels["signal_close"].index, columns=syms)
            cfg = MomentumConfig(**runner.PRIMARY_STRATEGY)
            from hf_research.xs_momentum import target_weights
            w = target_weights(panels["signal_close"], cfg, elig)
            targets = [{s: float(v) for s, v in w.loc[d].items() if v} for d in days]
            model = backtest(dict(panels, eligible=elig), cfg, rebalance_on=days)
            model = model[model.index >= days[0]]["net"]
            # A symbol held from the 10-11 decision stops being cached on 10-14, as when it leaves the universe.
            gone = sorted(set(targets[1]) - set(targets[0]) or set(targets[1]))[0]
            df = pd.read_csv(paths[gone])
            df[df["timestamp"] < T("2026-10-14").value // 1_000_000].to_csv(paths[gone], index=False, compression="gzip")

            store = runner.Store(Path(tmp) / "paper")
            store.root.mkdir()
            store.save_state({"config": {**runner.asdict(CFG), "capital_usdt": 1.0}, "start": "2026-10-04T00:00:00Z",
                              "book": {"equity": 1.0}, "last_run": None})
            eq = (1.0 + model).cumprod()
            store.append("equity.jsonl", [{"ts": runner._iso(d + pd.Timedelta(days=1, hours=1, minutes=5)), "equity": float(v)}
                                          for d, v in eq.items()])
            store.append("events.jsonl", [{"type": "rebalance", "decision_day": str(d.date()), "delay_hours": 0.0,
                                           "eligible": syms, "bitget_map": {}, "targets": tg} for d, tg in zip(days, targets)])
            refreshed = []
            orig = runner.refresh_replay_cache
            runner.refresh_replay_cache = lambda ex, s, since, *a, **k: refreshed.extend(s)
            orig_ex = runner.xs_signal.binance_exchange
            runner.xs_signal.binance_exchange = lambda: None
            try:
                out = runner.report(store, SimpleNamespace(cache_dir=str(cache)))
            finally:
                runner.refresh_replay_cache, runner.xs_signal.binance_exchange = orig, orig_ex
        vm = out["vs_model"]
        self.assertEqual(sorted(set(refreshed)), sorted({s for tg in targets for s in tg}))
        self.assertEqual(min(vm["missing_prices"]), "2026-10-13")  # row 10-13 needs the 10-13 exec close (10-14 00:00)
        self.assertIn(gone, vm["missing_prices"]["2026-10-13"])
        self.assertEqual(vm["days"], 9)  # rows 10-04 .. 10-12
        self.assertAlmostEqual(vm["implementation_gap_ann"], 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
