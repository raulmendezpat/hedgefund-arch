from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from hf_score.features import MODEL_FEATURES, build_model_frame
from hf_score.walkforward import monthly_folds, run_walkforward

H_MS = 3_600_000


def _candidates(n_bars: int = 6) -> pd.DataFrame:
    rows = []
    for k in range(n_bars):
        for sym, side, ret in (("AAA", "long", 0.02), ("BBB", "short", 0.02)):
            rows.append(
                {
                    "timestamp": k * H_MS, "symbol": sym, "side": side,
                    "adx": 25.0, "atrp": 0.01, "atrp_zscore": 0.0, "range_pct": 0.01,
                    "rolling_vol_24h": 0.01, "rolling_vol_72h": 0.01, "btc_rolling_vol_24h": 0.01,
                    "btc_atrp": 0.01, "btc_adx": 20.0, "ret_1h_lag": ret, "ret_4h_lag": ret,
                    "ret_12h_lag": ret, "ret_24h_lag": ret, "ema_gap_fast_slow": 0.01,
                    "dist_close_ema_fast": 0.0, "dist_close_ema_slow": 0.0, "pullback_depth": 0.0,
                    "btc_ret_24h_lag": 0.0, "breakout_distance_up": 0.03, "breakout_distance_down": 0.01,
                    "stop_atr_mult": 1.5,
                }
            )
    return pd.DataFrame(rows)


class FeatureTests(unittest.TestCase):
    def test_directional_features_are_signed_by_side(self):
        x = build_model_frame(_candidates(), round_trip_cost=0.0018)
        self.assertEqual(list(x.columns), MODEL_FEATURES)
        longs, shorts = x.iloc[0::2], x.iloc[1::2]
        self.assertTrue((longs["sgn_ret_24h_lag"] > 0).all())
        self.assertTrue((shorts["sgn_ret_24h_lag"] < 0).all())  # +2% move is against a short
        self.assertAlmostEqual(float(longs["room_to_extreme"].iloc[0]), 0.03)
        self.assertAlmostEqual(float(shorts["room_to_extreme"].iloc[0]), 0.01)
        self.assertAlmostEqual(float(x["expected_cost_R"].iloc[0]), 0.0018 / 0.015)
        self.assertTrue((x["xs_n_candidates"] == 2).all())
        self.assertTrue((x["xs_long_share"] == 0.5).all())


class WalkForwardTests(unittest.TestCase):
    def test_folds_are_monthly_with_embargo(self):
        ts = pd.Series(pd.date_range("2022-01-01", "2022-03-15", freq="h", tz="UTC"))
        folds = monthly_folds(ts, first_test="2022-02-01", embargo_bars=48)
        self.assertEqual([f.test_start.month for f in folds], [2, 3])
        self.assertEqual(folds[0].train_end, pd.Timestamp("2022-01-30", tz="UTC"))

    def test_training_never_sees_labels_realized_after_the_boundary(self):
        seen_max_exit = []

        class Spy:
            def fit(self, X, y):
                seen_max_exit.append(exit_ts[X.index].max())
                return self

            def predict(self, X):
                return np.zeros(len(X))

        ts = pd.Series(pd.date_range("2021-01-01", "2022-03-31", freq="h", tz="UTC"))
        exit_ts = ts + pd.Timedelta(hours=30)
        X = pd.DataFrame({"f": np.arange(len(ts), dtype=float)})
        y = pd.Series(np.ones(len(ts)))
        pred = run_walkforward(X, y, ts, exit_ts, first_test="2022-01-01", embargo_bars=48, model_factory=Spy)
        self.assertEqual(len(seen_max_exit), 3)
        for fold_start, max_exit in zip(["2022-01-01", "2022-02-01", "2022-03-01"], seen_max_exit):
            self.assertLess(max_exit, pd.Timestamp(fold_start, tz="UTC") - pd.Timedelta(hours=48))
        self.assertTrue(pred[ts < pd.Timestamp("2022-01-01", tz="UTC")].isna().all())
        self.assertTrue(pred[ts >= pd.Timestamp("2022-01-01", tz="UTC")].notna().all())


if __name__ == "__main__":
    unittest.main()
