"""Time-series momentum on a few liquid perpetuals (pre-registered in
docs/plans/tsmom_btc_eth_hypothesis.md).

Each asset is held long or short by the sign of its own trailing returns over several lookbacks,
sized to a per-asset volatility target. Timing follows hf_research.xs_momentum: targets for day d
use signal closes <= d and are filled at exec_close[d] by hf_research.engine.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TsmomConfig:
    lookbacks: tuple[int, ...] = (20, 60, 120)
    vol_window: int = 60
    asset_vol_target: float = 0.20  # annualized, per asset
    max_weight: float = 1.0
    band: float = 0.05
    mode: str = "long_short"  # long_short | long_only
    fee: float = 0.0006
    slippage_bps: float = 3.0

    def to_dict(self) -> dict:
        return asdict(self)


def tsmom_targets(signal_close: pd.DataFrame, cfg: TsmomConfig) -> pd.DataFrame:
    """w = mean(sign of each lookback return) * asset_vol_target / trailing vol, capped; 0 until
    the longest lookback and the vol window are available."""
    px = signal_close
    signal = sum(np.sign(px / px.shift(lb) - 1.0) for lb in cfg.lookbacks) / len(cfg.lookbacks)
    if cfg.mode == "long_only":
        signal = signal.clip(lower=0.0)
    vol = px.pct_change(fill_method=None).rolling(cfg.vol_window, min_periods=cfg.vol_window).std() * np.sqrt(365.0)
    w = (signal * cfg.asset_vol_target / vol).clip(-cfg.max_weight, cfg.max_weight)
    return w.where(px.shift(max(cfg.lookbacks)).notna()).fillna(0.0)
