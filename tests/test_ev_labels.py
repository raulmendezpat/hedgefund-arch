from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from hf_score.config import LabelConfig
from hf_score.labels import _simulate_policy_trade, label_candidates
from hf_score.market_data import build_feature_frame

H = 3_600_000
EXIT_CFG = {
    "family": "trend_atr_dynamic",
    "params": {
        "stop_atr_mult": 1.0,
        "tp_atr_mult": 2.0,
        "breakeven_activate_atr": 99.0,
        "trail_activate_atr": 99.0,
        "max_hold_bars": 0,
    },
}
NO_COST = LabelConfig(taker_fee=0.0, slippage_bps=0.0, ctx_time_stop_bars=6, max_horizon_bars=10)


def _bars(closes, *, atr=1.0, wick=0.2):
    """Continuous bars (open = previous close) with fixed ATR and a small wick."""
    opens = [closes[0]] + list(closes[:-1])
    return [
        {
            "timestamp": k * H,
            "open": o,
            "high": max(o, c) + wick,
            "low": min(o, c) - wick,
            "close": c,
            "atr": atr,
            "adx": 30.0,
            "atrp": atr / c,
            "ema_fast": c,
            "ema_slow": c - 1.0,
        }
        for k, (o, c) in enumerate(zip(opens, closes))
    ]


def _sim(bars, side="long", cfg=NO_COST):
    return _simulate_policy_trade(bars=bars, i=0, symbol="TST", strategy_id="t", side=side, exit_cfg=EXIT_CFG, cfg=cfg)


class PolicyLabelTests(unittest.TestCase):
    def test_take_profit_long(self):
        bars = _bars([100.0, 100.5, 101.0, 101.5, 102.5, 103.0])
        res = _sim(bars)
        self.assertEqual(res["exit_reason"], "tp")
        self.assertAlmostEqual(res["exit_px"], 102.0)
        self.assertAlmostEqual(res["net_R"], (102.0 / 100.0 - 1.0) / 0.01)  # +2 ATR = +2R

    def test_stop_loss_short(self):
        bars = _bars([100.0, 100.4, 101.2, 101.5])
        res = _sim(bars, side="short")
        self.assertEqual(res["exit_reason"], "sl")
        self.assertAlmostEqual(res["exit_px"], 101.0)
        self.assertLess(res["net_R"], -0.99)

    def test_costs_reduce_return(self):
        # Flat prices exit on the time stop at a fixed close, so the gap is exactly the costs.
        bars = _bars([100.0] * 12)
        gross = _sim(bars)["net_ret"]
        cfg = LabelConfig(taker_fee=0.0006, slippage_bps=3.0, ctx_time_stop_bars=6, max_horizon_bars=10)
        net = _sim(bars, cfg=cfg)["net_ret"]
        self.assertAlmostEqual(gross, 0.0)
        self.assertAlmostEqual(gross - net, cfg.round_trip_cost, delta=1e-5)

    def test_time_stop_and_incomplete(self):
        flat = _bars([100.0] * 12)
        self.assertEqual(_sim(flat)["exit_reason"], "time_stop")
        self.assertFalse(_sim(_bars([100.0] * 3))["label_complete"])

    def test_label_ignores_bars_after_exit_and_before_entry(self):
        bars = _bars([100.0, 100.5, 101.0, 101.5, 102.5, 103.0, 90.0, 80.0])
        base = _sim(bars)
        changed = [dict(b) for b in bars]
        changed[7]["low"] = 1.0  # after exit
        self.assertEqual(_sim(changed), base)


class NoLookaheadTests(unittest.TestCase):
    def test_features_are_point_in_time(self):
        rng = np.random.default_rng(7)
        n = 900
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
        bars = pd.DataFrame(
            {
                "timestamp": (idx.asi8 // 1_000_000).astype("int64"),
                "open": close,
                "high": close * 1.004,
                "low": close * 0.996,
                "close": close,
                "volume": 1.0,
            },
            index=idx,
        )
        full = build_feature_frame(bars)
        for t in (300, 650, n - 1):
            part = build_feature_frame(bars.iloc[: t + 1])
            pd.testing.assert_series_equal(full.iloc[t], part.iloc[-1], check_names=False)

    def test_label_candidates_uses_bars_after_signal_only(self):
        bars = _bars([100.0, 100.5, 101.0, 101.5, 102.5, 103.0, 103.0, 103.0])
        idx = pd.to_datetime([b["timestamp"] for b in bars], unit="ms", utc=True)
        frame = pd.DataFrame(bars, index=idx)
        cands = pd.DataFrame(
            [{"ts": idx[2], "timestamp": bars[2]["timestamp"], "symbol": "TST", "strategy_id": "t", "side": "long"}]
        )
        registry = {"default": EXIT_CFG}
        out = label_candidates(cands, {"TST": frame}, NO_COST, exit_registry=registry)
        frame_changed = frame.copy()
        frame_changed.iloc[0, frame_changed.columns.get_loc("close")] = 50.0  # before the signal bar
        out2 = label_candidates(cands, {"TST": frame_changed}, NO_COST, exit_registry=registry)
        self.assertAlmostEqual(out["net_R"].iloc[0], out2["net_R"].iloc[0])
        self.assertAlmostEqual(out["entry_px"].iloc[0], 101.0)


if __name__ == "__main__":
    unittest.main()
