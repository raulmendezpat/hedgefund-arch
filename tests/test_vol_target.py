from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from hf_research.vol_target import account_path, intraday_paths, vol_scale

DAYS = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")


class VolScaleTests(unittest.TestCase):
    def test_uses_only_previous_rows_and_caps(self):
        idx = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
        net = pd.Series([0.01, -0.01] * 3 + [0.5, -0.5], index=idx)
        k = vol_scale(net, pd.Series(True, index=idx), target=0.25, window=4, cap=2.0)
        self.assertEqual(k.iloc[:2].tolist(), [1.0, 1.0])  # fewer than window // 2 prior rows
        sigma6 = net.iloc[2:6].std() * np.sqrt(365.0)  # row 6 sees rows 2..5, not its own 0.5
        self.assertAlmostEqual(k.iloc[6], min(2.0, 0.25 / sigma6))
        self.assertLess(k.iloc[7], 0.25)

    def test_inactive_rows_do_not_count(self):
        net = pd.Series([0.0, 0.0, 0.0, 0.02], index=DAYS)
        k = vol_scale(net, pd.Series([False, False, False, True], index=DAYS), target=0.25, window=4, cap=2.0)
        self.assertTrue((k == 1.0).all())


class IntradayTests(unittest.TestCase):
    def test_hours_after_the_fill_until_the_next_fill(self):
        hours = pd.date_range(DAYS[0], periods=24 * 5, freq="h")
        px = pd.Series(100.0, index=hours)
        px[DAYS[1] + pd.Timedelta(hours=5)] = 80.0  # a dip inside row 1
        px[DAYS[1]] = 50.0  # row 0's last mark and row 1's fill
        w = pd.DataFrame({"A": [-1.0, -1.0, 0.0, 0.0]}, index=DAYS)
        ret, notional = intraday_paths(w, pd.DataFrame({"A": px}))
        self.assertAlmostEqual(ret[0, -1], 0.5)  # short from 100 to 50
        self.assertAlmostEqual(ret[1, 4], -0.6)  # short from 50 to 80, 5 bars after the fill
        self.assertAlmostEqual(notional[0, -1], 0.5)
        self.assertTrue((ret[2:] == 0).all())


class AccountTests(unittest.TestCase):
    def test_liquidation_stops_the_path(self):
        net = pd.Series([0.01, -0.5, 0.2, 0.0], index=DAYS)
        ret = np.zeros((4, 24))
        ret[1, 10] = -0.95
        notional = np.full((4, 24), 4.0)
        out = account_path(net, ret, notional, maintenance=0.02)
        self.assertEqual(out["liquidated"], "2024-01-02")
        self.assertEqual(out["final_equity"], 0.0)

    def test_kill_switch_uses_the_intraday_low(self):
        net = pd.Series([0.0, -0.1, 0.0, 0.0], index=DAYS)
        ret = np.zeros((4, 24))
        ret[1, 3] = -0.3
        out = account_path(net, ret, np.ones((4, 24)))
        self.assertIsNone(out["liquidated"])
        self.assertEqual(out["kill_switch_first_hit"], "2024-01-02")
        self.assertAlmostEqual(out["max_dd_hourly"], -0.3)
        self.assertAlmostEqual(out["max_dd_daily"], -0.1)


if __name__ == "__main__":
    unittest.main()
