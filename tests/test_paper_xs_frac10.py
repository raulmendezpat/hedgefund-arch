from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from hf_research.xs_momentum import MomentumConfig, backtest, load_daily_panels, target_weights

_spec = importlib.util.spec_from_file_location(
    "paper_xs_frac10", Path(__file__).resolve().parents[1] / "scripts" / "paper_xs_frac10.py"
)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


class Frac10Tests(unittest.TestCase):
    def test_only_the_fraction_per_side_differs_from_the_primary(self):
        diff = {k: v for k, v in runner.STRATEGY.items() if runner.PRIMARY_STRATEGY.get(k) != v}
        self.assertEqual(diff, {"select_frac": 0.1})
        self.assertEqual(runner.compute_targets.keywords, {"strategy": runner.STRATEGY})
        self.assertEqual(runner.model_replay.keywords, {"strategy": runner.STRATEGY})

    def test_sixty_assets_give_six_per_side(self):
        idx = pd.date_range("2026-01-01", periods=80, freq="D", tz="UTC")
        rng = np.random.default_rng(3)
        px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.03, (80, 60)), axis=0)), index=idx,
                          columns=[f"S{k}/USDT:USDT" for k in range(60)])
        w = target_weights(px, MomentumConfig(**runner.STRATEGY)).iloc[-1]
        self.assertEqual(int((w > 0).sum()), 6)
        self.assertEqual(int((w < 0).sum()), 6)
        self.assertTrue(np.allclose(w[w != 0].abs(), 0.5 / 6))

    def test_model_replay_uses_the_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            (cache / "binanceusdm" / "1h").mkdir(parents=True)
            t = pd.date_range("2026-08-01", "2026-10-31 23:00", freq="h", tz="UTC")
            rng = np.random.default_rng(5)
            syms = [f"S{k}/USDT:USDT" for k in range(20)]
            for k, sym in enumerate(syms):
                px = 100 * np.exp(np.cumsum(rng.normal(0.0002 * (k - 10), 0.004, len(t))))
                pd.DataFrame({"timestamp": t.as_unit("ms").asi8, "open": px, "high": px, "low": px, "close": px,
                              "volume": 1.0}).to_csv(
                    cache / "binanceusdm" / "1h" / f"{sym.replace('/', '_').replace(':', '_')}.csv.gz",
                    index=False, compression="gzip")
            days = pd.Series([T("2026-10-04"), T("2026-10-11"), T("2026-10-18")])
            reb = pd.DataFrame({"eligible": [syms] * 3})
            args = SimpleNamespace(cache_dir=str(cache))
            got = runner.model_replay(reb, days, args)["net"]
            panels = load_daily_panels(syms, cache_dir=str(cache), exchange="binanceusdm")
            elig = pd.DataFrame(False, index=panels["signal_close"].index, columns=syms)
            elig.loc[list(days)] = True
            want = {f: backtest(dict(panels, eligible=elig), MomentumConfig(**{**runner.STRATEGY, "select_frac": f}),
                                rebalance_on=days)["net"] for f in (0.1, 0.2)}
        pd.testing.assert_series_equal(got, want[0.1])
        self.assertFalse(np.allclose(got, want[0.2]))


if __name__ == "__main__":
    unittest.main()
