from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "build_liquid_universe", Path(__file__).resolve().parents[1] / "scripts" / "build_liquid_universe.py"
)
blu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(blu)

DAY = 86_400_000


def _daily(start: str, days: int, qv: float) -> pd.DataFrame:
    t0 = pd.Timestamp(start, tz="UTC").value // 1_000_000
    return pd.DataFrame({"timestamp": [t0 + k * DAY for k in range(days)], "close": 1.0, "volume": qv})


class MembershipTests(unittest.TestCase):
    def test_ranking_uses_only_days_before_the_month(self):
        # B is tiny until it explodes on Mar 1: it must not lead the March ranking.
        b = _daily("2024-01-01", 120, 1.0)
        b.loc[pd.to_datetime(b["timestamp"], unit="ms", utc=True) >= pd.Timestamp("2024-03-01", tz="UTC"), "volume"] = 1e9
        daily = {"A": _daily("2024-01-01", 120, 100.0), "B": b}
        m = blu.monthly_membership(daily, top_n=1, window_days=30, min_history_days=20)
        self.assertEqual(m.loc[m["month"] == "2024-03", "symbol"].tolist(), ["A"])
        self.assertEqual(m.loc[m["month"] == "2024-04", "symbol"].tolist(), ["B"])

    def test_delisted_symbol_drops_out_and_young_symbol_waits_for_history(self):
        daily = {
            "OLD": _daily("2024-01-01", 45, 1e6),  # stops trading mid-February
            "NEW": _daily("2024-02-20", 70, 10.0),
            "MID": _daily("2024-01-01", 130, 50.0),
        }
        m = blu.monthly_membership(daily, top_n=3, window_days=30, min_history_days=20)
        self.assertIn("OLD", m.loc[m["month"] == "2024-02", "symbol"].tolist())
        self.assertNotIn("OLD", m.loc[m["month"] == "2024-04", "symbol"].tolist())
        self.assertNotIn("NEW", m.loc[m["month"] == "2024-03", "symbol"].tolist())
        self.assertIn("NEW", m.loc[m["month"] == "2024-04", "symbol"].tolist())


if __name__ == "__main__":
    unittest.main()
