from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hf_research.tradfi import bitget_rate_drag, drop_bad_ticks

_spec = importlib.util.spec_from_file_location(
    "run_tradfi_study", Path(__file__).resolve().parents[1] / "scripts" / "run_tradfi_study.py"
)
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)


def walk(n=120, vol=0.005, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, vol, n))), index=idx)


class BadTickTests(unittest.TestCase):
    def test_isolated_fx_spike_that_reverts_is_dropped(self):
        s = walk()
        s.iloc[90] *= 1.08
        out = drop_bad_ticks(s, fx=True)
        self.assertEqual(list(s.index.difference(out.index)), [s.index[90]])

    def test_exchange_traded_shock_that_reverts_is_kept_unless_absurd(self):
        s = walk(vol=0.02)
        s.iloc[90] *= 0.81  # a real -19% day that bounces back
        self.assertEqual(len(drop_bad_ticks(s)), len(s))
        s.iloc[90] *= 0.6  # now -51%: a bad print
        self.assertEqual(list(s.index.difference(drop_bad_ticks(s).index)), [s.index[90]])

    def test_moves_that_do_not_revert_are_kept(self):
        s = walk()
        s.iloc[90:] *= 1.10
        self.assertEqual(len(drop_bad_ticks(s, fx=True)), len(s))


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.saved = study.START, study.END
        study.START, study.END = "2024-01-01", "2024-01-14"

    def tearDown(self):
        study.START, study.END = self.saved

    def test_friday_decision_fills_at_the_next_trading_close(self):
        days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08",
                               "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12", "2024-01-16"])
        c = pd.DataFrame({"A": np.arange(100.0, 110.0)}, index=days)
        p = study.build_panels(c, pd.Series({"A": 0.5}), pd.Series({"A": 3.65}))
        sig, exe = p["signal_close"]["A"], p["exec_close"]["A"]
        fri, sat, mon = (pd.Timestamp(d, tz="UTC") for d in ("2024-01-05", "2024-01-06", "2024-01-08"))
        self.assertEqual((sig[fri], exe[fri]), (103.0, 104.0))  # Friday's close decides, Monday's fills
        self.assertEqual((sig[sat], exe[sat]), (103.0, 104.0))  # weekends add no return
        self.assertEqual(exe[mon], 105.0)
        self.assertTrue(np.isnan(sig[pd.Timestamp("2024-01-01", tz="UTC")]))
        self.assertAlmostEqual(p["funding"]["A"].iloc[0], 0.0365 / 365)
        self.assertEqual(p["slippage_bps"]["A"].iloc[0], study.MIN_SLIP_BPS)  # floored

    def test_conservative_funding_never_credits_the_receiving_side(self):
        idx = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
        panels = {"signal_close": pd.DataFrame({"L": 100.0, "S": 100.0}, index=idx),
                  "exec_close": pd.DataFrame({"L": 100.0, "S": 100.0}, index=idx),
                  "funding": bitget_rate_drag(pd.Series({"L": 0.365, "S": 0.365}), idx, ["L", "S"]),
                  "slippage_bps": pd.DataFrame(0.0, index=idx, columns=["L", "S"])}
        targets = pd.DataFrame({"L": 0.5, "S": -0.5}, index=idx)
        daily, _ = study.run(panels, targets, band=0.0, weekly=False, fee=0.0)
        self.assertAlmostEqual(daily["funding_symmetric"].iloc[5], 0.0)  # long pays what the short earns
        self.assertAlmostEqual(daily["funding"].iloc[5], -0.5 * 0.001)  # only the payment counts


if __name__ == "__main__":
    unittest.main()
