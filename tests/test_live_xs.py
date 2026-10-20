from __future__ import annotations

import importlib.util
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from hf_research import xs_signal
from hf_research.live_xs import (
    LiveConfig,
    SanityError,
    build_orders,
    check_risk,
    flatten_orders,
    new_risk_state,
    position_drift,
    position_losses,
    reduce_only_targets,
)

_spec = importlib.util.spec_from_file_location(
    "live_xs_momentum", Path(__file__).resolve().parents[1] / "scripts" / "live_xs_momentum.py"
)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

CFG = LiveConfig()
T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
P3 = lambda s, q: math.floor(q * 1000) / 1000  # noqa: E731  (0.001 step)


class RiskTests(unittest.TestCase):
    def test_drawdown_halts_and_stays_halted(self):
        r = new_risk_state(1000.0, T("2026-11-02 01:00"))
        self.assertEqual(check_risk(r, 1100.0, T("2026-11-03 01:00"), CFG)[0], "ok")
        self.assertEqual(r["peak_equity"], 1100.0)
        self.assertEqual(check_risk(r, 1100 * 0.66, T("2026-11-10 01:00"), CFG)[0], "ok")
        self.assertEqual(check_risk(r, 1100 * 0.65, T("2026-11-11 01:00"), CFG)[0], "halt")
        self.assertEqual(check_risk(r, 2000.0, T("2026-11-12 01:00"), CFG)[0], "halted")

    def test_daily_loss_is_reduce_only_until_next_day(self):
        r = new_risk_state(1000.0, T("2026-11-02 00:10"))
        self.assertEqual(check_risk(r, 919.0, T("2026-11-02 05:00"), CFG)[0], "reduce_only")
        self.assertEqual(check_risk(r, 919.0, T("2026-11-03 00:05"), CFG)[0], "ok")  # new day start

    def test_reduce_only_targets_never_grow_open_or_flip(self):
        t = reduce_only_targets({"A": 0.05, "B": 0.02, "C": -0.04, "D": 0.03}, {"A": 0.03, "B": 0.04, "C": 0.04})
        self.assertEqual(t, {"A": 0.03, "B": 0.02, "C": 0.0})  # A kept, B shrinks, C closes, D not opened


class PositionLossTests(unittest.TestCase):
    def test_short_squeeze_and_long_drop_past_threshold(self):
        qty = {"S": -100.0, "L": 10.0, "OK": 5.0}
        ref = {"S": 1.0, "L": 10.0, "OK": 10.0}
        px = {"S": 1.47, "L": 8.0, "OK": 9.0}  # S +47% short: -47 USDT; L -20%: -20; OK: -5
        self.assertEqual(position_losses(qty, ref, px, 2000.0, 0.015), {"S": 47.0 / 2000.0})
        self.assertEqual(set(position_losses(qty, ref, px, 1000.0, 0.015)), {"S", "L"})
        self.assertEqual(position_losses(qty, ref, {"S": float("nan")}, 1000.0, 0.015), {})


class BuildOrdersTests(unittest.TestCase):
    mids = {"A": 10.0, "B": 20.0, "C": 50.0}

    def test_open_from_flat_sizes_by_equity(self):
        orders, skipped, intended = build_orders({"A": 0.05, "B": -0.05}, {}, self.mids, 10_000.0, CFG, 0.02, P3)
        self.assertEqual([(o.symbol, o.side, o.qty, o.reduce_only) for o in orders],
                         [("A", "buy", 50.0, False), ("B", "sell", 25.0, False)])
        self.assertEqual(intended, {"A": 0.05, "B": -0.05})

    def test_flip_closes_exact_then_opens_and_reductions_come_first(self):
        orders, _, intended = build_orders({"A": -0.05, "C": 0.05}, {"A": 37.123, "C": 12.0}, self.mids, 10_000.0,
                                           CFG, 0.0, P3)
        self.assertEqual([(o.symbol, o.side, o.qty, o.reduce_only, o.reason) for o in orders],
                         [("A", "sell", 37.123, True, "flip_close"),
                          ("C", "sell", 2.0, True, "reduce"),
                          ("A", "sell", 50.0, False, "flip_open")])  # every reduction before any open
        self.assertAlmostEqual(intended["A"], -0.05)

    def test_band_and_min_order_skips_are_reflected_in_intended(self):
        orders, skipped, intended = build_orders({"A": 0.045, "B": 0.0002}, {"A": 40.0}, self.mids, 10_000.0,
                                                 CFG, 0.02, P3)
        self.assertEqual(orders, [])  # A: 0.04 -> 0.045 inside the band; B: 2 USDT below min order
        self.assertEqual(skipped[0]["symbol"], "B")
        self.assertEqual(intended, {"A": 0.04})

    def test_sanity_limits_abort(self):
        with self.assertRaises(SanityError):
            build_orders({"A": 0.2}, {}, self.mids, 10_000.0, CFG, 0.02, P3)  # weight cap
        with self.assertRaises(SanityError):
            build_orders({k: 0.1 for k in "ABCDEFGHIJK"}, {}, {k: 1.0 for k in "ABCDEFGHIJK"}, 10_000.0, CFG, 0.0, P3)
        big = LiveConfig(max_order_usdt=100.0)
        with self.assertRaises(SanityError):
            build_orders({"A": 0.05}, {}, self.mids, 10_000.0, big, 0.0, P3)  # 500 USDT order
        with self.assertRaises(SanityError):
            build_orders({"Z": 0.05}, {}, self.mids, 10_000.0, CFG, 0.0, P3)  # no price
        # Closing a large position is never blocked by the order cap.
        orders, _, _ = build_orders({}, {"A": 1000.0}, self.mids, 10_000.0, big, 0.0, P3)
        self.assertEqual(orders[0].reason, "close")

    def test_reduce_only_mode_only_shrinks(self):
        orders, _, intended = build_orders({"A": 0.05, "B": -0.05}, {"A": 100.0}, self.mids, 10_000.0, CFG, 0.0, P3,
                                           reduce_only_mode=True)
        self.assertEqual([(o.symbol, o.side, o.qty, o.reduce_only) for o in orders], [("A", "sell", 50.0, True)])
        self.assertEqual(intended, {"A": 0.05})

    def test_flatten_and_drift(self):
        f = flatten_orders({"A": 3.0, "B": -2.0}, self.mids)
        self.assertEqual([(o.symbol, o.side, o.qty, o.reduce_only) for o in f], [("A", "sell", 3.0, True), ("B", "buy", 2.0, True)])
        d = position_drift({"A": 0.05, "B": -0.05}, {"A": 50.0, "B": -10.0}, self.mids, 10_000.0, CFG)
        self.assertEqual(set(d), {"B"})
        self.assertAlmostEqual(d["B"], 0.03)


class CredentialFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = {k: os.environ.pop(k, None) for k in (*runner.ENV_KEYS.values(), "BITGET_ENV_FILE")}

    def tearDown(self):
        for k, v in self.saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        self.tmp.cleanup()

    def _write(self, text, mode=0o600):
        p = Path(self.tmp.name) / "bitget.env"
        p.write_text(text)
        p.chmod(mode)
        os.environ["BITGET_ENV_FILE"] = str(p)
        return p

    def test_values_are_data_not_shell(self):
        marker = Path(self.tmp.name) / "pwned"
        self._write(f"# comment\nexport BITGET_API_KEY=k$HOME\nBITGET_API_SECRET='s ec;ret'\n"
                    f"BITGET_API_PASSPHRASE=\"p$(touch {marker})\"\n\ntouch {marker}\n")
        creds = runner.load_credentials("")
        self.assertEqual(creds, {"apiKey": "k$HOME", "secret": "s ec;ret", "password": f"p$(touch {marker})"})
        self.assertFalse(marker.exists())

    def test_environment_variables_take_precedence(self):
        self._write("BITGET_API_KEY=file\nBITGET_API_SECRET=file\nBITGET_API_PASSPHRASE=file\n")
        os.environ["BITGET_API_KEY"] = "env"
        self.assertEqual(runner.load_credentials("")["apiKey"], "env")

    def test_refuses_readable_file_and_missing_keys(self):
        self._write("BITGET_API_KEY=a\nBITGET_API_SECRET=b\nBITGET_API_PASSPHRASE=c\n", mode=0o644)
        with self.assertRaises(SystemExit):
            runner.load_credentials("")
        self._write("BITGET_API_KEY=a\n")
        with self.assertRaises(SystemExit):
            runner.load_credentials("")


class FakeAccount:
    def __init__(self, equity=10_000.0):
        self.cash = equity
        self.pos: dict[str, float] = {}
        self.entry: dict[str, float] = {}
        self.px = {"AAA/USDT:USDT": 10.0, "BBB/USDT:USDT": 20.0, "CCC/USDT:USDT": 40.0}
        self.listed = set(self.px)
        self.calls: list[tuple] = []
        self.fail_symbols: set[str] = set()
        self.flows: list[dict] = []  # transfer bills: {"id", "ts" (ms), "type", "amount"}
        self.flows_error: Exception | None = None
        self.fills: dict[str, dict] = {}
        self.fills_pending: set[str] = set()  # order ids whose fill is not visible yet

    def equity(self):
        return self.cash + sum(q * (self.px[s] - self.entry[s]) for s, q in self.pos.items())

    def positions(self):
        return dict(self.pos), [{"symbol": s, "hedged": False} for s in self.pos]

    def mids(self, symbols):
        return {s: self.px[s] for s in symbols}

    def to_precision(self, s, q):
        return P3(s, q)

    def prepare_symbol(self, s, cfg):
        self.calls.append(("prepare", s))
        return []

    def ensure_one_way(self):
        self.calls.append(("one_way",))

    def market_order(self, o, oid, cfg):
        self.calls.append(("order", o.symbol, o.side, o.qty, o.reduce_only, oid))
        if o.symbol in self.fail_symbols:
            raise RuntimeError("rejected")
        q0 = self.pos.get(o.symbol, 0.0)
        dq = o.qty if o.side == "buy" else -o.qty
        if q0:  # realize P&L on the old position at the current price
            self.cash += q0 * (self.px[o.symbol] - self.entry[o.symbol])
        q1 = q0 + dq
        if abs(q1) < 1e-12:
            self.pos.pop(o.symbol, None)
            self.entry.pop(o.symbol, None)
        else:
            self.pos[o.symbol] = q1
            self.entry[o.symbol] = self.px[o.symbol]
        avg = self.px[o.symbol] * (1.001 if o.side == "buy" else 0.999)  # 10 bps worse than the mid
        self.fills[oid] = {"id": oid, "filled": o.qty, "average": avg, "fee_usdt": 0.0006 * o.qty * avg}
        return {"id": oid, "status": "closed"}

    def order_fill(self, order_id, symbol):
        self.calls.append(("fill", order_id))
        return None if order_id in self.fills_pending else self.fills.get(order_id)

    def cash_flows(self, since_ms, until_ms):
        if self.flows_error:
            raise self.flows_error
        return [f for f in self.flows if since_ms <= f["ts"] <= until_ms]


WEIGHTS = {"AAA/USDT:USDT": 0.05, "BBB/USDT:USDT": -0.05}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.acct = FakeAccount()
        self.weights = dict(WEIGHTS)
        self._orig = (xs_signal.ensure_membership, xs_signal.compute_targets)
        xs_signal.ensure_membership = lambda path, month, cfg, cache_dir, log: pd.DataFrame(
            {"month": month, "symbol": list(self.acct.px), "rank": [1, 2, 3]})
        xs_signal.compute_targets = lambda d, members, listed, cache_dir, log, min_coverage=0.5: {
            "weights": dict(self.weights), "eligible": members, "not_on_bitget": [], "coverage": 1.0, "signal_close": {}}
        self.env = os.environ.pop("LIVE_TRADING", None)

    def tearDown(self):
        xs_signal.ensure_membership, xs_signal.compute_targets = self._orig
        if self.env is not None:
            os.environ["LIVE_TRADING"] = self.env
        self.tmp.cleanup()

    def _store(self, mode):
        store = runner.Store(Path(self.tmp.name) / mode)
        store.root.mkdir(exist_ok=True)
        if not store.path("state.json").exists():
            store.save_state({"config": CFG.to_dict(), "start": "2026-10-20T00:00:00Z", "managed_symbols": [],
                              "risk": None, "last_decision_day": None})
        return store

    def _run(self, now, mode="live"):
        store = self._store(mode)
        runner.run_cycle(store, SimpleNamespace(now=now, mode=mode, cache_dir="unused"),
                         account_factory=lambda args: self.acct)
        return store

    def test_dry_run_sends_nothing_and_changes_no_settings(self):
        store = self._run("2026-10-26 01:07", mode="dry-run")
        self.assertEqual(self.acct.calls, [])
        orders = store.read("orders.jsonl")
        self.assertEqual(len(orders), 2)
        self.assertFalse(orders["sent"].any())
        self.assertEqual(store.load_state()["last_decision_day"], "2026-10-25")
        self.assertEqual(store.load_state()["managed_symbols"], [])

    def test_live_requires_env_flag(self):
        with self.assertRaises(SystemExit):
            self._run("2026-10-26 01:07")
        self.assertEqual(self.acct.calls, [])

    def test_live_rebalance_is_verified_and_idempotent(self):
        os.environ["LIVE_TRADING"] = "1"
        store = self._run("2026-10-26 01:07")
        self.assertEqual(self.acct.pos, {"AAA/USDT:USDT": 50.0, "BBB/USDT:USDT": -25.0})
        self.assertIn(("one_way",), self.acct.calls)
        ev = store.read("events.jsonl").iloc[-1]
        self.assertEqual(ev["type"], "rebalance")
        self.assertEqual(ev["drift"], {})
        self.assertEqual(store.load_state()["managed_symbols"], ["AAA/USDT:USDT", "BBB/USDT:USDT"])
        n = len(self.acct.calls)
        self._run("2026-10-26 02:07")  # same week: nothing to do
        self.assertEqual(len(self.acct.calls), n)

        # Next week: AAA flips short, BBB closes, CCC opens.
        self.weights = {"AAA/USDT:USDT": -0.05, "CCC/USDT:USDT": 0.05}
        self._run("2026-11-02 01:07")
        orders = [c for c in self.acct.calls[n:] if c[0] == "order"]
        self.assertEqual([(c[1], c[2], c[4]) for c in orders[:2]],
                         [("AAA/USDT:USDT", "sell", True), ("BBB/USDT:USDT", "buy", True)])  # reductions first
        self.assertAlmostEqual(self.acct.pos["AAA/USDT:USDT"], -50.0)
        self.assertNotIn("BBB/USDT:USDT", self.acct.pos)
        self.assertAlmostEqual(self.acct.pos["CCC/USDT:USDT"], 12.5)

    def test_foreign_positions_abort_without_trading(self):
        os.environ["LIVE_TRADING"] = "1"
        self.acct.pos = {"CCC/USDT:USDT": 1.0}
        self.acct.entry = {"CCC/USDT:USDT": 40.0}
        with self.assertRaises(SystemExit):
            self._run("2026-10-26 01:07")
        self.assertEqual([c for c in self.acct.calls if c[0] == "order"], [])
        self.assertEqual(self._store("live").read("events.jsonl")["type"].tolist(), ["abort"])

    def test_kill_switch_flattens_then_stays_halted_until_reset(self):
        os.environ["LIVE_TRADING"] = "1"
        self._run("2026-10-26 01:07")
        self.acct.px["AAA/USDT:USDT"] = 0.0  # the 50 AAA long goes to zero: -500 USDT
        self.acct.cash -= 3500.0  # plus 3500 more lost: equity ~6000, a 40% drawdown from the peak
        store = self._run("2026-10-27 05:07")
        self.assertEqual(self.acct.pos, {})
        self.assertTrue(store.load_state()["risk"]["halted"])
        self.assertIn("kill_switch", store.read("events.jsonl")["type"].tolist())
        n = len(self.acct.calls)
        self._run("2026-11-02 01:07")  # halted: no rebalance
        self.assertEqual(len(self.acct.calls), n)

    def test_kill_switch_retries_flatten_while_positions_remain(self):
        os.environ["LIVE_TRADING"] = "1"
        self._run("2026-10-26 01:07")
        self.acct.px["AAA/USDT:USDT"] = 0.0
        self.acct.cash -= 3500.0
        self.acct.fail_symbols = {"BBB/USDT:USDT"}  # the BBB close is rejected during the flatten
        store = self._run("2026-10-27 05:07")
        self.assertEqual(list(self.acct.pos), ["BBB/USDT:USDT"])
        self.acct.fail_symbols = set()
        self._run("2026-10-27 06:07")  # still halted, but the leftover position is closed
        self.assertEqual(self.acct.pos, {})
        self.assertIn("kill_switch_retry", store.read("events.jsonl")["type"].tolist())
        self.assertTrue(store.load_state()["risk"]["halted"])

    def test_failed_post_trade_check_keeps_ownership_and_order_log(self):
        os.environ["LIVE_TRADING"] = "1"
        acct, reads = self.acct, {"n": 0}
        real_positions = acct.positions

        def flaky_positions():
            reads["n"] += 1
            if reads["n"] == 2:  # the post-trade read
                raise ConnectionError("timeout")
            return real_positions()

        acct.positions = flaky_positions
        store = self._run("2026-10-26 01:07")
        self.assertEqual(store.load_state()["managed_symbols"], ["AAA/USDT:USDT", "BBB/USDT:USDT"])
        self.assertEqual(len(store.read("orders.jsonl")), 2)
        self.assertIn("verify_failed", store.read("events.jsonl")["type"].tolist())
        self._run("2026-10-26 02:07")  # its own positions are not treated as foreign
        self.assertNotIn("abort", store.read("events.jsonl")["type"].tolist())

    @staticmethod
    def _ms(s):
        return int(pd.Timestamp(s, tz="UTC").value // 1_000_000)

    def test_withdrawal_shifts_risk_baselines_instead_of_tripping_kill_switch(self):
        os.environ["LIVE_TRADING"] = "1"
        self._run("2026-10-26 01:07")
        self.acct.cash -= 4000.0  # a 40% withdrawal would otherwise read as a 40% drawdown
        self.acct.flows.append({"id": "b1", "ts": self._ms("2026-10-26 01:30"), "type": "trans_to_contract",
                                "amount": -4000.0})
        store = self._run("2026-10-26 02:07")
        risk = store.load_state()["risk"]
        self.assertFalse(risk["halted"])
        self.assertAlmostEqual(risk["peak_equity"], 6000.0, places=0)
        self.assertEqual(self.acct.pos["AAA/USDT:USDT"], 50.0)  # nothing was flattened
        ev = store.read("events.jsonl")
        self.assertEqual(ev[ev["type"] == "cash_flow"]["net_usdt"].tolist(), [-4000.0])
        self._run("2026-10-26 03:07")  # still inside the overlap window: not counted twice
        self.assertAlmostEqual(store.load_state()["risk"]["peak_equity"], 6000.0, places=0)

    def test_transfers_before_the_first_run_are_history(self):
        self.acct.flows.append({"id": "b0", "ts": self._ms("2026-10-26 00:30"), "type": "trans_from_contract",
                                "amount": 10_000.0})
        store = self._run("2026-10-26 00:50", mode="dry-run")
        store = self._run("2026-10-26 01:50", mode="dry-run")
        self.assertNotIn("cash_flow", store.read("events.jsonl")["type"].tolist())
        self.assertAlmostEqual(store.load_state()["risk"]["peak_equity"], 10_000.0)

    def test_failed_transfer_query_is_logged_and_retried(self):
        store = self._run("2026-10-26 00:07", mode="dry-run")
        self.acct.flows_error = ConnectionError("timeout")
        self.acct.flows.append({"id": "b2", "ts": self._ms("2026-10-26 00:30"), "type": "trans_from_contract",
                                "amount": 500.0})
        self._run("2026-10-26 01:07", mode="dry-run")
        self.assertIn("cash_flow_check_failed", store.read("events.jsonl")["type"].tolist())
        self.acct.flows_error = None
        self._run("2026-10-26 02:07", mode="dry-run")  # the cursor did not move: the deposit is found
        ev = store.read("events.jsonl")
        self.assertEqual(ev[ev["type"] == "cash_flow"]["net_usdt"].tolist(), [500.0])

    def test_fills_are_recorded_late_and_reported(self):
        os.environ["LIVE_TRADING"] = "1"
        self.acct.fills_pending = {runner.client_oid("2026-10-25", runner.OrderIntent(
            "BBB/USDT:USDT", "sell", 25.0, False, 20.0, 0.0, -0.05, "open"))}
        store = self._run("2026-10-26 01:07")
        self.assertEqual(store.read("fills.jsonl")["symbol"].tolist(), ["AAA/USDT:USDT"])
        self.acct.fills_pending = set()
        self._run("2026-10-26 02:07")  # the late fill is picked up, the recorded one is not re-fetched
        fills = store.read("fills.jsonl")
        self.assertEqual(sorted(fills["symbol"]), ["AAA/USDT:USDT", "BBB/USDT:USDT"])
        self.assertTrue(((fills["slippage_bps"] - 10.0).abs() < 1e-6).all())
        self.assertTrue(((fills["fee_bps"] - 6.0).abs() < 1e-6).all())
        rep = runner.report(store)
        self.assertEqual(rep["n_fills"], 2)
        self.assertAlmostEqual(rep["by_rebalance"]["2026-10-25"]["slippage_bps_weighted"], 10.0)

    def test_position_loss_alert_once_per_rebalance(self):
        os.environ["LIVE_TRADING"] = "1"
        store = self._run("2026-10-26 01:07")  # short 25 BBB at 20
        self.acct.px["BBB/USDT:USDT"] = 30.0  # +50%: -250 USDT = 2.5% of equity
        self._run("2026-10-26 05:07")
        self._run("2026-10-26 06:07")
        ev = store.read("events.jsonl")
        alerts = ev[ev["type"] == "position_loss"]
        self.assertEqual(alerts["symbol"].tolist(), ["BBB/USDT:USDT"])
        self.assertAlmostEqual(alerts["loss_frac"].iloc[0], 0.025)
        self.acct.px["BBB/USDT:USDT"] = 20.0
        self._run("2026-11-02 01:07")  # new rebalance: new reference, alerts re-armed
        self.assertEqual(store.load_state()["position_ref"]["alerted"], [])

    def test_failed_order_is_retried_next_hour_from_actual_positions(self):
        os.environ["LIVE_TRADING"] = "1"
        self.acct.fail_symbols = {"BBB/USDT:USDT"}
        store = self._run("2026-10-26 01:07")
        self.assertEqual(self.acct.pos, {"AAA/USDT:USDT": 50.0})
        self.assertIsNone(store.load_state()["last_decision_day"])
        self.acct.fail_symbols = set()
        self._run("2026-10-26 02:07")
        orders = [c for c in self.acct.calls if c[0] == "order"]
        self.assertEqual([(c[1], c[2]) for c in orders[-1:]], [("BBB/USDT:USDT", "sell")])  # only the missing leg
        self.assertEqual(self.acct.pos["BBB/USDT:USDT"], -25.0)
        self.assertEqual(store.load_state()["last_decision_day"], "2026-10-25")


if __name__ == "__main__":
    unittest.main()
