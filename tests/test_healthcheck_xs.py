from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import healthcheck_xs as hc  # noqa: E402

from hf_research.paper_xs import PaperConfig  # noqa: E402


def ts(s):
    return pd.Timestamp(s, tz="UTC")


class HealthcheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.runner = hc.Runner("paper", root / "state", root / "logs")
        self.runner.state_dir.mkdir()
        self.runner.log_dir.mkdir()
        self.state = {"config": PaperConfig().__dict__, "start": "2026-10-19T07:00:00Z",
                      "last_decision_day": None, "last_run": "2026-10-20T10:05:00Z"}
        self._save()

    def tearDown(self):
        self.tmp.cleanup()

    def _save(self):
        (self.runner.state_dir / "state.json").write_text(json.dumps(self.state))

    def _events(self, *events):
        with open(self.runner.state_dir / "events.jsonl", "a") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

    def _check(self, now, hc_state):
        return hc.run_check([self.runner], hc_state, ts(now), heartbeat_hour=9, local_tz="UTC")

    def texts(self, notices):
        return [(n.level, n.text) for n in notices]

    def test_stale_alerts_once_then_clears(self):
        notices, st = self._check("2026-10-20 10:20", {})
        self.assertEqual([n for n in notices if n.runner == "paper"], [])
        notices, st = self._check("2026-10-20 13:20", st)
        self.assertEqual([n.level for n in notices if n.runner == "paper"], ["alert"])
        self.assertIn("sin correr", notices[0].text)
        notices, st = self._check("2026-10-20 14:20", st)  # still stale: no repeat
        self.assertEqual([n for n in notices if n.runner == "paper"], [])
        self.state["last_run"] = "2026-10-20 15:05:00Z"
        self._save()
        notices, st = self._check("2026-10-20 15:20", st)
        self.assertEqual([n.level for n in notices if n.runner == "paper"], ["ok"])

    def test_overdue_rebalance_and_new_events(self):
        self.state["last_run"] = "2026-10-26T04:05:00Z"
        self._save()
        _, st = self._check("2026-10-26 00:20", {})
        notices, st = self._check("2026-10-26 04:20", st)  # fill was due at 01:00, now 3.3 h late
        self.assertTrue(any("pendiente" in n.text for n in notices))
        self._events({"type": "rebalance", "decision_day": "2026-10-25", "n_orders": 24, "equity_after": 9990.4})
        self.state["last_decision_day"] = "2026-10-25"
        self._save()
        notices, st = self._check("2026-10-26 05:20", st)
        got = self.texts([n for n in notices if n.runner == "paper"])
        self.assertIn(("info", "rebalanceo 2026-10-25: 24 órdenes, equity 9,990.40"), got)
        self.assertTrue(any(level == "ok" for level, _ in got))
        notices, _ = self._check("2026-10-26 06:20", st)  # nothing new
        self.assertEqual([n for n in notices if n.runner == "paper"], [])

    def test_existing_history_is_not_replayed_and_tracebacks_alert(self):
        self._events({"type": "missed_rebalance", "decision_day": "2026-10-18"})
        log = self.runner.log_dir / "x.log"
        log.write_text("Traceback (most recent call last)\n")
        notices, st = self._check("2026-10-20 10:20", {})
        self.assertEqual([n for n in notices if n.runner == "paper"], [])
        log.write_text(log.read_text() + "ok\nTraceback (most recent call last)\n")
        notices, _ = self._check("2026-10-20 11:20", st)
        self.assertEqual([n.level for n in notices if n.runner == "paper"], ["alert"])

    def test_paper_position_loss_alerts_once_per_rebalance(self):
        self.state.update(last_decision_day="2026-10-25", last_run="2026-10-26T02:05:00Z",
                          book={"equity": 10_000.0, "qty": {"BR/USDT:USDT": -400.0}, "last_px": {"BR/USDT:USDT": 1.0}})
        self._save()
        notices, st = self._check("2026-10-26 01:20", {})  # reference taken at the fill-time mark
        self.assertEqual([n for n in notices if n.runner == "paper"], [])
        self.state["book"]["last_px"]["BR/USDT:USDT"] = 1.47  # short squeeze: -188 USDT = 1.9%
        self._save()
        notices, st = self._check("2026-10-26 02:20", st)
        self.assertEqual(self.texts([n for n in notices if n.runner == "paper"]),
                         [("alert", "BR/USDT:USDT pierde 1.9% de la cartera desde el último rebalanceo")])
        notices, st = self._check("2026-10-26 03:20", st)  # already alerted this week
        self.assertEqual([n for n in notices if n.runner == "paper"], [])
        self.state["last_decision_day"] = "2026-11-01"
        self._save()
        _, st = self._check("2026-11-02 01:20", st)
        self.assertEqual(st["runners"]["paper"]["pos_ref"]["px"], {"BR/USDT:USDT": 1.47})

    def test_daily_heartbeat_once_per_day(self):
        notices, st = self._check("2026-10-20 08:20", {})
        self.assertEqual([n for n in notices if n.runner == "monitor"], [])
        notices, st = self._check("2026-10-20 10:20", st)
        self.assertEqual([n.text for n in notices if n.runner == "monitor"], ["chequeo diario: paper OK"])
        notices, st = self._check("2026-10-20 11:20", st)
        self.assertEqual([n for n in notices if n.runner == "monitor"], [])


if __name__ == "__main__":
    unittest.main()
