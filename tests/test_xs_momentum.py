from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from hf_research.xs_momentum import MomentumConfig, backtest, target_weights


def _panels(n_days: int = 80, n_assets: int = 8, seed: int = 3) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    drift = np.linspace(-0.01, 0.01, n_assets)  # asset k trends at drift[k] per day
    rets = drift + rng.normal(0, 0.002, (n_days, n_assets))
    px = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=[f"A{k}" for k in range(n_assets)])
    return {"signal_close": px, "exec_close": px * 1.0, "funding": pd.DataFrame(0.0, index=idx, columns=px.columns)}


CFG = MomentumConfig(lookback_days=5, rebalance_days=1, select_frac=0.25, min_assets=4, vol_window_days=10, fee=0.0, slippage_bps=0.0)


class TargetWeightTests(unittest.TestCase):
    def test_weights_at_day_d_ignore_later_prices(self):
        p = _panels()
        w = target_weights(p["signal_close"], CFG)
        future = p["signal_close"].copy()
        future.iloc[50:] = future.iloc[50:] * np.linspace(0.1, 10, future.shape[1])
        w2 = target_weights(future, CFG)
        pd.testing.assert_frame_equal(w.iloc[:50], w2.iloc[:50])

    def test_long_short_is_dollar_neutral_and_picks_trend(self):
        w = target_weights(_panels()["signal_close"], CFG).iloc[-1]
        self.assertAlmostEqual(w.sum(), 0.0)
        self.assertAlmostEqual(w.abs().sum(), 1.0)
        self.assertGreater(w["A7"], 0)  # strongest uptrend is long
        self.assertLess(w["A0"], 0)  # strongest downtrend is short


class BacktestTests(unittest.TestCase):
    def test_trending_universe_is_profitable_without_costs(self):
        daily = backtest(_panels(), CFG)
        self.assertGreater(daily["net"].sum(), 0.0)

    def test_costs_charge_turnover(self):
        cfg = MomentumConfig(**{**CFG.to_dict(), "fee": 0.001, "slippage_bps": 0.0})
        daily = backtest(_panels(), cfg)
        first = daily[daily["turnover"] > 0].iloc[0]
        self.assertAlmostEqual(first["turnover"], 1.0)  # from flat to gross 1.0
        self.assertAlmostEqual(first["cost"], 0.001)

    def test_funding_paid_by_longs_received_by_shorts(self):
        p = _panels()
        p["funding"] = pd.DataFrame(0.001, index=p["funding"].index, columns=p["funding"].columns)
        long_only = MomentumConfig(**{**CFG.to_dict(), "mode": "long_only"})
        d = backtest(p, long_only)
        held_days = d[d["gross_exposure"].shift(1, fill_value=0) > 0]
        self.assertTrue((held_days["funding"] < 0).all())
        d_ls = backtest(p, CFG)
        held_ls = d_ls[d_ls["gross_exposure"].shift(1, fill_value=0) > 0]
        self.assertTrue(np.allclose(held_ls["funding"], 0.0, atol=1e-4))  # neutral book nets out


if __name__ == "__main__":
    unittest.main()
