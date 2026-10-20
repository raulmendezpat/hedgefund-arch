"""Volatility targeting and leveraged-account simulation for daily-rebalanced books
(pre-registered in docs/plans/vol_targeting_hypothesis.md).

- vol_scale: exposure multiplier per decision day from the trailing realized volatility of a
  shadow (unscaled) strategy, using only rows whose fill precedes the decision.
- intraday_paths: hour-by-hour portfolio return and gross notional within each daily row, from
  the weights held in that row (engine detail["weight"]) and 1h closes.
- account_path: equity path of a margin account through those hours, with liquidation (cross
  margin, maintenance as a fraction of notional) and the drawdown kill-switch level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hf.data.ohlcv import _cache_path, drop_unclosed_candles


def vol_scale(shadow_net: pd.Series, active: pd.Series, *, target: float, window: int, cap: float) -> pd.Series:
    """k_d = min(cap, target / sigma_d); sigma_d from active rows <= d-1, k = 1 until window // 2 rows exist.

    Row d's return ends at the day-d fill (d+1, 01:00 UTC), after the day-d decision (23:00), so the
    estimate at d uses rows up to d-1 only."""
    net = shadow_net.where(active)
    sigma = net.rolling(window, min_periods=window // 2).std().shift(1) * np.sqrt(365.0)
    return (target / sigma).clip(upper=cap).fillna(1.0)


def load_hourly_close(symbols: list[str], cache_dir: str = "data/cache", exchange: str = "binanceusdm") -> pd.DataFrame:
    """1h closes indexed by bar open time (UTC), one column per symbol."""
    out = {}
    for s in symbols:
        path = _cache_path(cache_dir, exchange, s, "1h")
        if not path.exists():
            continue
        df = drop_unclosed_candles(pd.read_csv(path), "1h").drop_duplicates("timestamp", keep="last")
        t = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        out[s] = pd.Series(df["close"].astype(float).to_numpy(), index=t).sort_index()
    return pd.DataFrame(out)


def intraday_paths(weights: pd.DataFrame, hourly: pd.DataFrame, exec_hour: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Per daily row d and hour h=1..24 after the fill: portfolio return and gross notional, as
    fractions of the equity at the fill.

    Row d holds the weights filled at the close of the exec_hour bar of day d; its 24 hourly marks
    are the closes of the following 24 bars (the last one is the next fill). Missing closes carry
    the previous mark; a symbol without a fill price contributes nothing."""
    days = weights.index
    if not (days[1:] - days[:-1] == pd.Timedelta(days=1)).all():
        raise ValueError("weights must have one row per consecutive UTC day")
    held = [c for c in weights.columns if c in hourly.columns and (weights[c] != 0).any()]
    w = weights[held].fillna(0.0).to_numpy(np.float64)
    start = days[0] + pd.Timedelta(hours=exec_hour)
    grid = pd.date_range(start, periods=(len(days) + 1) * 24, freq="h")
    arr = hourly[held].reindex(grid).to_numpy(np.float64).reshape(len(days) + 1, 24, len(held))
    base = arr[:-1, 0, :]
    path = np.concatenate([arr[:-1, 1:, :], arr[1:, :1, :]], axis=1)
    ratio = path / base[:, None, :] - 1.0
    for h in range(1, 24):
        gap = np.isnan(ratio[:, h, :])
        ratio[:, h, :][gap] = ratio[:, h - 1, :][gap]
    ratio = np.nan_to_num(ratio, nan=0.0)
    ret = np.einsum("dhn,dn->dh", ratio, w)
    notional = np.einsum("dhn,dn->dh", 1.0 + ratio, np.abs(w))
    return ret, notional


def account_path(net: pd.Series, ret: np.ndarray, notional: np.ndarray, *, maintenance: float = 0.02,
                 kill_dd: float = -0.25) -> dict:
    """Hourly equity of a margin account starting at 1. Within a row the equity moves with the
    held book; the row closes at (1 + net), which adds funding and costs. Stops at liquidation:
    equity <= maintenance * gross notional at any hour."""
    eq, peak, worst_dd = 1.0, 1.0, 0.0
    kill_day = liq_day = None
    closes = []
    for i, d in enumerate(net.index):
        hourly = eq * (1.0 + ret[i])
        if np.any(1.0 + ret[i] <= maintenance * notional[i]):
            liq_day, eq = d, 0.0
            closes.append(0.0)
            worst_dd = -1.0
            break
        running = np.maximum.accumulate(np.concatenate([[peak], hourly]))[1:]
        dd = float((hourly / running - 1.0).min())
        worst_dd = min(worst_dd, dd)
        if kill_day is None and dd <= kill_dd:
            kill_day = d
        peak = max(peak, float(hourly.max()))
        eq *= 1.0 + float(net.iloc[i])
        peak = max(peak, eq)
        closes.append(eq)
    eq_s = pd.Series(closes, index=net.index[: len(closes)])
    roll = (eq_s / eq_s.shift(30) - 1.0).dropna()
    return {
        "final_equity": float(eq),
        "liquidated": None if liq_day is None else str(liq_day.date()),
        "kill_switch_first_hit": None if kill_day is None else str(kill_day.date()),
        "max_dd_hourly": worst_dd,
        "max_dd_daily": float((eq_s / eq_s.cummax() - 1.0).min()),
        "worst_day": float(net.iloc[: len(closes)].min()),
        "roll30": {"median": float(roll.median()), "p05": float(roll.quantile(0.05)),
                   "p95": float(roll.quantile(0.95)), "worst": float(roll.min()),
                   "share_ge_10pct": float((roll >= 0.10).mean()), "share_le_minus10pct": float((roll <= -0.10).mean())},
    }
