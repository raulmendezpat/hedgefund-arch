from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hf_research.oi_signals import load_metrics, oi_growth, rank_weights

IDX = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")


class OiSignalTests(unittest.TestCase):
    def test_growth_uses_contracts_and_needs_both_ends(self):
        oi = pd.DataFrame({"A": [100.0, 110.0, np.nan, 121.0, 0.0]}, index=IDX)
        g = oi_growth(oi, 1)
        self.assertAlmostEqual(g["A"].iloc[1], np.log(1.1))
        self.assertTrue(np.isnan(g["A"].iloc[2]) and np.isnan(g["A"].iloc[3]) and np.isnan(g["A"].iloc[4]))

    def test_contrarian_book_shorts_the_highest_scores(self):
        score = pd.DataFrame([np.arange(10.0)], index=IDX[:1], columns=[f"S{i}" for i in range(10)])
        ok = pd.DataFrame(True, index=score.index, columns=score.columns)
        w = rank_weights(score, ok, select_frac=0.2, long_high=False).iloc[0]
        self.assertEqual(sorted(w[w < 0].index), ["S8", "S9"])
        self.assertEqual(sorted(w[w > 0].index), ["S0", "S1"])
        self.assertAlmostEqual(w.sum(), 0.0)
        self.assertAlmostEqual(w.abs().sum(), 1.0)
        ok.loc[:, "S9"] = False  # ineligible symbols never enter the book
        w2 = rank_weights(score, ok, select_frac=0.2, long_high=False).iloc[0]
        self.assertEqual(w2["S9"], 0.0)

    def test_load_skips_missing_days(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "binance_metrics").mkdir()
            pd.DataFrame({"day": ["2024-01-01", "2024-01-02"], "oi": [5.0, np.nan], "count_ls": [1.5, np.nan],
                          "taker_ratio_log_mean": [0.1, np.nan], "missing": [False, True]}
                         ).to_csv(Path(d, "binance_metrics", "AUSDT.csv.gz"), index=False)
            m = load_metrics(["A/USDT:USDT", "B/USDT:USDT"], IDX, d)
        self.assertEqual(m["oi"].loc[IDX[0], "A/USDT:USDT"], 5.0)
        self.assertTrue(m["oi"].iloc[1:].isna().all().all())


if __name__ == "__main__":
    unittest.main()
