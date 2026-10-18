from __future__ import annotations

import unittest
from pathlib import Path

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


class UniverseAndBandTests(unittest.TestCase):
    def test_ineligible_assets_never_get_weight(self):
        from hf_research.xs_momentum import eligible_from_membership

        p = _panels()
        idx = p["signal_close"].index
        months = sorted(set(idx.strftime("%Y-%m")))
        membership = pd.DataFrame(
            [{"month": m, "symbol": c} for m in months for c in p["signal_close"].columns if c != "A7"]
        )
        elig = eligible_from_membership(membership, idx, p["signal_close"].columns)
        self.assertFalse(elig["A7"].any())
        w = target_weights(p["signal_close"], CFG, elig)
        self.assertTrue((w["A7"] == 0.0).all())
        self.assertGreater(w["A6"].iloc[-1], 0)  # next-best trend takes its place

    def test_band_reduces_turnover(self):
        p = _panels(seed=11)
        p["signal_close"] = p["signal_close"] * np.exp(np.random.default_rng(1).normal(0, 0.02, p["signal_close"].shape))
        no_band = backtest(p, CFG)["turnover"].sum()
        banded = backtest(p, MomentumConfig(**{**CFG.to_dict(), "band": 0.05}))["turnover"].sum()
        self.assertLess(banded, no_band)

    def test_band_never_blocks_opening_positions_smaller_than_the_band(self):
        p = _panels()
        cfg = MomentumConfig(**{**CFG.to_dict(), "band": 0.5})  # far larger than any single weight
        d = backtest(p, cfg)
        self.assertGreater(d["gross_exposure"].max(), 0.99)

    def test_delisted_position_is_closed(self):
        p = _panels()
        long_only = MomentumConfig(**{**CFG.to_dict(), "mode": "long_only", "rebalance_days": 30})
        first = backtest(p, long_only)
        held_day = first[first["turnover"] > 0].index[0]
        w0 = target_weights(p["signal_close"], long_only).loc[held_day]
        victim = w0[w0 > 0].index[0]
        p["exec_close"].loc[p["exec_close"].index > held_day + pd.Timedelta(days=5), victim] = np.nan
        d = backtest(p, long_only)
        closing = d.loc[held_day + pd.Timedelta(days=6)]
        self.assertGreater(closing["turnover"], 0.0)
        self.assertLess(d.loc[d.index[-1], "gross_exposure"], 1.0 - 1e-9)


class AttributionAndCostPanelTests(unittest.TestCase):
    def test_detail_contributions_add_up_to_daily(self):
        p = _panels()
        p["funding"] = pd.DataFrame(0.0005, index=p["funding"].index, columns=p["funding"].columns)
        cfg = MomentumConfig(**{**CFG.to_dict(), "fee": 0.0006, "slippage_bps": 3.0})
        daily, parts = backtest(p, cfg, detail=True)
        for k in ("gross", "funding", "cost"):
            np.testing.assert_allclose(parts[k].sum(axis=1), daily[k], atol=1e-12)
        pd.testing.assert_frame_equal(backtest(p, cfg), daily)

    def test_per_asset_slippage_panel_overrides_flat_slippage(self):
        p = _panels()
        cfg = MomentumConfig(**{**CFG.to_dict(), "fee": 0.0, "slippage_bps": 3.0})
        flat = backtest(p, cfg)
        p_same = dict(p, slippage_bps=pd.DataFrame(3.0, index=p["exec_close"].index, columns=p["exec_close"].columns))
        pd.testing.assert_frame_equal(backtest(p_same, cfg), flat)
        p_wide = dict(p, slippage_bps=pd.DataFrame(30.0, index=p["exec_close"].index, columns=p["exec_close"].columns))
        np.testing.assert_allclose(backtest(p_wide, cfg)["cost"], flat["cost"] * 10.0)


class FundingAlignmentTests(unittest.TestCase):
    def test_funding_events_are_charged_to_the_position_held_between_fills(self):
        import tempfile

        from hf_research.xs_momentum import load_daily_panels

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ex"
            (root / "1h").mkdir(parents=True)
            (root / "funding").mkdir()
            t = pd.date_range("2024-01-01", periods=24 * 4, freq="h", tz="UTC")
            ms = t.as_unit("ms").asi8
            pd.DataFrame({"timestamp": ms, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}).to_csv(
                root / "1h" / "X_USDT_USDT.csv.gz", index=False, compression="gzip")
            # Events at 00:00 (+3ms jitter) and 08:00 of Jan 2, and 00:00 of Jan 3.
            ev = pd.to_datetime(["2024-01-02 00:00:00.003", "2024-01-02 08:00:00.000", "2024-01-03 00:00:00.000"], utc=True)
            pd.DataFrame({"timestamp": ev.as_unit("ms").asi8, "funding_rate": [1.0, 10.0, 100.0]}).to_csv(
                root / "funding" / "X_USDT_USDT.csv.gz", index=False, compression="gzip")
            f = load_daily_panels(["X/USDT:USDT"], cache_dir=tmp, exchange="ex")["funding"]["X/USDT:USDT"]
        # Fill at 01:00. Row Jan 1 holds 01:00 Jan 1 -> 01:00 Jan 2 (00:00 Jan 2 event);
        # row Jan 2 holds 01:00 Jan 2 -> 01:00 Jan 3 (08:00 Jan 2 and 00:00 Jan 3 events).
        self.assertEqual(float(f[f.index == pd.Timestamp("2024-01-01", tz="UTC")].iloc[0]), 1.0)
        self.assertEqual(float(f[f.index == pd.Timestamp("2024-01-02", tz="UTC")].iloc[0]), 110.0)


if __name__ == "__main__":
    unittest.main()
