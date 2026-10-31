from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from test_live_xs import FakeAccount

from hf_research.live_xs import SanityError, build_orders

_spec = importlib.util.spec_from_file_location(
    "live_tsmom", Path(__file__).resolve().parents[1] / "scripts" / "live_tsmom.py"
)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

BTC, ETH = "BTC/USDT:USDT", "ETH/USDT:USDT"
CFG = runner.live_config()


class TsmomAccount(FakeAccount):
    def __init__(self, equity=1_000.0):
        super().__init__(equity)
        self.px = {BTC: 80_000.0, ETH: 2_500.0, "AAA/USDT:USDT": 10.0}
        self.listed = set(self.px)

    def to_precision(self, s, q):
        step = 0.0001 if s == BTC else 0.01
        return int(q / step + 1e-9) * step


class ConfigTests(unittest.TestCase):
    def test_limits_allow_the_strategy_and_nothing_more(self):
        mids = {BTC: 80_000.0, ETH: 2_500.0}
        orders, _, _ = build_orders({BTC: 1.0, ETH: -1.0}, {}, mids, 1_000.0, CFG, 0.05, lambda s, q: q)
        self.assertEqual(len(orders), 2)
        with self.assertRaises(SanityError):
            build_orders({BTC: 1.2}, {}, mids, 1_000.0, CFG, 0.05, lambda s, q: q)

    def test_refuses_the_momentum_runners_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            keys = "BITGET_API_KEY=k\nBITGET_API_SECRET=s\nBITGET_API_PASSPHRASE=p\n"
            for name, text in (("xs.env", keys), ("ts.env", keys), ("ts2.env", keys.replace("=k", "=other"))):
                p = Path(tmp) / name
                p.write_text(text)
                p.chmod(0o600)
            saved = {k: os.environ.get(k) for k in ("BITGET_ENV_FILE", "BITGET_TSMOM_ENV_FILE")}
            try:
                os.environ["BITGET_ENV_FILE"] = str(Path(tmp) / "xs.env")
                os.environ["BITGET_TSMOM_ENV_FILE"] = str(Path(tmp) / "ts.env")
                with self.assertRaises(SystemExit):
                    runner.load_credentials("")
                os.environ["BITGET_TSMOM_ENV_FILE"] = str(Path(tmp) / "ts2.env")
                self.assertEqual(runner.load_credentials("")["apiKey"], "other")
                os.environ["BITGET_TSMOM_ENV_FILE"] = str(Path(tmp) / "missing.env")
                with self.assertRaises(SystemExit):  # never falls back to the momentum runner's file
                    runner.load_credentials("")
            finally:
                for k, v in saved.items():
                    os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.acct = TsmomAccount()
        self.weights = {BTC: 0.2, ETH: -0.25}
        self.calls = []
        self._orig = runner.compute_targets

        def fake_targets(d, cache_dir, log):
            self.calls.append(d)
            return {"weights": dict(self.weights), "signal_close": {}}

        runner.compute_targets = fake_targets
        self.env = os.environ.pop("LIVE_TRADING", None)

    def tearDown(self):
        runner.compute_targets = self._orig
        if self.env is not None:
            os.environ["LIVE_TRADING"] = self.env
        self.tmp.cleanup()

    def _run(self, now, mode="live"):
        store = runner.Store(Path(self.tmp.name) / mode)
        store.root.mkdir(exist_ok=True)
        if not store.path("state.json").exists():
            store.save_state({"config": CFG.to_dict(), "schedule": "daily", "start": "2026-11-02T08:00:00Z",
                              "managed_symbols": [], "risk": None, "last_decision_day": None})
        runner.run_cycle(store, SimpleNamespace(now=now, mode=mode, cache_dir="unused"),
                         account_factory=lambda args: self.acct)
        return store

    def test_dry_run_sends_nothing(self):
        store = self._run("2026-11-03 00:30", mode="dry-run")
        self.assertEqual(self.calls, [])  # the first fill is 2026-11-03 01:00
        store = self._run("2026-11-03 01:17", mode="dry-run")
        self.assertEqual(self.calls, [pd.Timestamp("2026-11-02", tz="UTC")])
        self.assertEqual(self.acct.calls, [])
        orders = store.read("orders.jsonl")
        self.assertEqual(sorted(orders["symbol"]), [BTC, ETH])
        self.assertFalse(orders["sent"].any())
        self.assertTrue(orders["client_oid"].str.startswith("ts20261102").all())
        self.assertEqual(store.load_state()["managed_symbols"], [])

    def test_live_rebalances_daily_with_the_band(self):
        with self.assertRaises(SystemExit):
            self._run("2026-11-03 01:17")
        os.environ["LIVE_TRADING"] = "1"
        store = self._run("2026-11-03 01:17")
        self.assertAlmostEqual(self.acct.pos[BTC], 0.0025)  # 0.2 * 1000 / 80000
        self.assertAlmostEqual(self.acct.pos[ETH], -0.1)
        self.assertEqual(store.read("events.jsonl").iloc[-1]["drift"], {})
        n = len(self.acct.calls)
        self._run("2026-11-03 02:17")  # same day: nothing to do
        self.assertEqual(len(self.acct.calls), n)

        self.weights = {BTC: 0.22, ETH: 0.25}  # BTC moves less than the 0.05 band; ETH flips
        self._run("2026-11-04 01:17")
        orders = [c for c in self.acct.calls[n:] if c[0] == "order"]
        self.assertEqual([(c[1], c[2], c[4]) for c in orders], [(ETH, "buy", True), (ETH, "buy", False)])
        self.assertAlmostEqual(self.acct.pos[BTC], 0.0025)
        self.assertAlmostEqual(self.acct.pos[ETH], 0.1)

    def test_days_the_host_was_down_are_missed_once(self):
        os.environ["LIVE_TRADING"] = "1"
        self._run("2026-11-03 01:17")
        store = self._run("2026-11-06 01:17")  # 03 and 04 never ran
        ev = store.read("events.jsonl")
        self.assertEqual(list(ev.loc[ev["type"] == "missed_rebalance", "decision_day"]), ["2026-11-03", "2026-11-04"])
        self.assertEqual(store.load_state()["last_decision_day"], "2026-11-05")

    def test_positions_from_another_runner_abort(self):
        os.environ["LIVE_TRADING"] = "1"
        self.acct.pos, self.acct.entry = {"AAA/USDT:USDT": 10.0}, {"AAA/USDT:USDT": 10.0}
        with self.assertRaises(SystemExit):
            self._run("2026-11-03 01:17")
        self.assertEqual([c for c in self.acct.calls if c[0] == "order"], [])

    def test_kill_switch_flattens(self):
        os.environ["LIVE_TRADING"] = "1"
        self._run("2026-11-03 01:17")
        self.acct.cash -= 400.0  # 40% drawdown
        store = self._run("2026-11-03 05:17")
        self.assertEqual(self.acct.pos, {})
        self.assertIn("kill_switch", list(store.read("events.jsonl")["type"]))


if __name__ == "__main__":
    unittest.main()
