from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "event_review_xs", Path(__file__).resolve().parents[1] / "scripts" / "event_review_xs.py"
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def ts(s):
    return pd.Timestamp(s, tz="UTC")


class EventReviewTests(unittest.TestCase):
    def test_due_once_per_week_after_the_sunday_close(self):
        self.assertEqual(m.due_decision_day(ts("2026-11-01 23:30"), "2026-10-25"), (None, False))
        self.assertEqual(m.due_decision_day(ts("2026-11-02 00:12"), "2026-10-25"), (ts("2026-11-01"), False))
        self.assertEqual(m.due_decision_day(ts("2026-11-02 03:12"), "2026-10-25"), (ts("2026-11-01"), True))
        self.assertEqual(m.due_decision_day(ts("2026-11-02 05:12"), "2026-11-01"), (None, False))
        self.assertEqual(m.due_decision_day(ts("2026-11-04 01:12"), "2026-10-25"), (None, False))  # too late

    def test_mapping_ratio_follows_the_prefixes(self):
        pairs = {"1000PEPE/USDT:USDT": "PEPE/USDT:USDT", "BTC/USDT:USDT": "BTC/USDT:USDT",
                 "NEIRO/USDT:USDT": "NEIRO/USDT:USDT", "XMR/USDT:USDT": "XMR/USDT:USDT"}
        bn = {"1000PEPE/USDT:USDT": 0.0042, "BTC/USDT:USDT": 100_000.0, "NEIRO/USDT:USDT": 0.0005, "XMR/USDT:USDT": 595.0}
        bg = {"PEPE/USDT:USDT": 0.0000042, "BTC/USDT:USDT": 100_050.0, "NEIRO/USDT:USDT": 0.5}
        got = {i["binance"]: i["issue"] for i in m.mapping_issues(pairs, bn, bg)}
        self.assertEqual(set(got), {"NEIRO/USDT:USDT", "XMR/USDT:USDT"})
        self.assertIn("1000.000x", got["NEIRO/USDT:USDT"])
        self.assertIn("sin precio", got["XMR/USDT:USDT"])

    def test_parse_and_notices(self):
        review = {"risk": "high", "summary": "Listing en Upbit el 30-oct", "events": []}
        content = [SimpleNamespace(type="server_tool_use"), SimpleNamespace(type="text", text=json.dumps(review))]
        self.assertEqual(m.parse_review(content)["risk"], "high")
        with self.assertRaises(ValueError):
            m.parse_review([SimpleNamespace(type="text", text=json.dumps(dict(review, risk="huge")))])
        rv = {"decision_day": "2026-11-01", "post_fill": False, "cost_usd": 1.9,
              "mapping_issues": [{"binance": "A/USDT:USDT", "bitget": "A/USDT:USDT", "issue": "x"}],
              "positions": [{"binance": "B/USDT:USDT", "bitget": "1000B/USDT:USDT", "side": "short", **review},
                            {"binance": "C/USDT:USDT", "bitget": "C/USDT:USDT", "side": "long", "risk": "low", "summary": ""},
                            {"binance": "D/USDT:USDT", "bitget": "D/USDT:USDT", "side": "long", "error": "boom"}]}
        got = [(n.level, n.text) for n in m.notices(rv)]
        self.assertEqual(got[0][0], "alert")
        self.assertEqual(got[1], ("alert", "B corto: riesgo high — Listing en Upbit el 30-oct"))
        self.assertEqual(got[2], ("info", "revisión 2026-11-01 (antes del fill): 3 posiciones, 1 con riesgo, "
                                          "mapeo con problemas, 1 sin revisar, costo 1.90 USD"))


if __name__ == "__main__":
    unittest.main()
