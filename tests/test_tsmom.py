from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from hf_research.tsmom import TsmomConfig, tsmom_targets

IDX = pd.date_range("2024-01-01", periods=12, freq="D", tz="UTC")


class TsmomTargetTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        up = 100 * np.cumprod(1 + 0.01 + 0.005 * rng.standard_normal(len(IDX)))
        self.px = pd.DataFrame({"UP": up, "DOWN": 10000.0 / up}, index=IDX)
        self.cfg = TsmomConfig(lookbacks=(2, 4), vol_window=3)

    def test_sign_and_vol_sizing(self):
        w = tsmom_targets(self.px, self.cfg)
        vol = self.px.pct_change().rolling(3).std() * np.sqrt(365.0)
        d = IDX[8]
        self.assertAlmostEqual(w.at[d, "UP"], min(1.0, 0.20 / vol.at[d, "UP"]))
        self.assertAlmostEqual(w.at[d, "DOWN"], max(-1.0, -0.20 / vol.at[d, "DOWN"]))
        self.assertTrue((w.iloc[:4] == 0).all().all())  # longest lookback not yet available

    def test_no_lookahead(self):
        w = tsmom_targets(self.px, self.cfg)
        shocked = self.px.copy()
        shocked.iloc[9:] *= 0.1
        self.assertTrue(tsmom_targets(shocked, self.cfg).iloc[:9].equals(w.iloc[:9]))

    def test_long_only_never_shorts(self):
        w = tsmom_targets(self.px, TsmomConfig(lookbacks=(2, 4), vol_window=3, mode="long_only"))
        self.assertTrue((w["DOWN"] == 0).all())
        self.assertTrue((w["UP"].iloc[5:] > 0).all())


if __name__ == "__main__":
    unittest.main()
