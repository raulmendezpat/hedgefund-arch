from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from hf_research.engine import backtest_weights
from hf_research.funding_carry import CarryConfig, carry_targets, fee_series, spot_col, trailing_funding_ann

IDX = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")


class CarryTargetTests(unittest.TestCase):
    def test_entry_exit_hysteresis_and_capacity(self):
        fa = pd.DataFrame({"A": [0.12, 0.05, 0.02, 0.2, 0.2, 0.2],
                           "B": [0.30, 0.30, 0.30, 0.30, 0.01, 0.01],
                           "C": [0.50, 0.50, 0.50, 0.50, 0.50, np.nan]}, index=IDX)
        ok = pd.DataFrame(True, index=IDX, columns=fa.columns)
        ok.loc[IDX[5], "C"] = False
        w = carry_targets(fa, ok, CarryConfig(max_positions=2, pair_weight=0.1))
        held = [sorted(c for c in fa.columns if w.at[d, c] < 0) for d in IDX]
        # day0: top-2 of A,B,C by funding = C,B. A waits. day2: still full.
        # day4: B < exit -> slot for A (0.2 >= entry). day5: C untradable -> out; no other candidate.
        self.assertEqual(held, [["B", "C"], ["B", "C"], ["B", "C"], ["B", "C"], ["A", "C"], ["A"]])
        for d in IDX:
            for p in fa.columns:
                self.assertEqual(w.at[d, p], -w.at[d, spot_col(p)])

    def test_open_pair_survives_between_exit_and_entry(self):
        fa = pd.DataFrame({"A": [0.12, 0.05, 0.04, 0.02, 0.2, 0.2]}, index=IDX)
        w = carry_targets(fa, pd.DataFrame(True, index=IDX, columns=["A"]), CarryConfig())
        self.assertEqual((w["A"] < 0).tolist(), [True, True, True, False, True, True])

    def test_trailing_window_has_no_partial_values(self):
        daily = pd.DataFrame({"A": [0.0001] * 6}, index=IDX)
        fa = trailing_funding_ann(daily, 3)
        self.assertTrue(fa["A"].iloc[:2].isna().all())
        self.assertAlmostEqual(fa["A"].iloc[2], 0.0001 * 365)


class CarryEngineTests(unittest.TestCase):
    def test_hedged_pair_earns_funding_minus_costs_regardless_of_price(self):
        px = pd.Series([100.0, 150.0, 90.0, 120.0, 60.0, 100.0], index=IDX)
        panels = {"exec_close": pd.DataFrame({"A": px, spot_col("A"): px}),
                  "funding": pd.DataFrame({"A": [0.0, 0.001, 0.001, 0.001, 0.001, 0.001],
                                           spot_col("A"): 0.0}, index=IDX)}
        targets = pd.DataFrame({"A": -0.5, spot_col("A"): 0.5}, index=IDX)
        cfg = CarryConfig()
        daily = backtest_weights(panels, targets, fee=fee_series(panels["exec_close"].columns, cfg),
                                 slippage_bps=0.0, band=0.5)
        self.assertAlmostEqual(daily["cost"].iloc[0], 0.5 * cfg.perp_fee + 0.5 * cfg.spot_fee)
        # Price moves cancel between the legs; what is left is the short's funding.
        self.assertAlmostEqual(daily["gross"].abs().max(), 0.0, places=12)
        self.assertTrue((daily["funding"].iloc[1:] > 0).all())


if __name__ == "__main__":
    unittest.main()
