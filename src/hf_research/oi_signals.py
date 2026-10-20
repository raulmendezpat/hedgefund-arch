"""Cross-sectional positioning signals from Binance futures metrics (pre-registered in
docs/plans/open_interest_hypothesis.md).

Metrics come from scripts/fetch_binance_metrics.py: one row per symbol and UTC day with the
day's last open interest / long-short ratios and the day's mean ln(taker buy/sell ratio). Row d
is known by the day-d signal close (24:00 UTC), so scores for day d use rows <= d only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_FIELDS = ("oi", "count_ls", "taker_ratio_log_mean")


def binance_id(symbol: str) -> str:
    base, rest = symbol.split("/", 1)
    return base + rest.split(":")[0]


def load_metrics(perps: list[str], index: pd.DatetimeIndex, cache_dir: str = "data/cache") -> dict[str, pd.DataFrame]:
    """{field: day x perp panel} aligned with `index`; NaN where the archive had no data."""
    cols: dict[str, dict[str, pd.Series]] = {f: {} for f in METRIC_FIELDS}
    for p in perps:
        path = f"{cache_dir}/binance_metrics/{binance_id(p)}.csv.gz"
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            continue
        df = df[df["missing"].astype(str) == "False"]
        days = pd.to_datetime(df["day"]).dt.tz_localize("UTC")
        for f in METRIC_FIELDS:
            cols[f][p] = pd.Series(pd.to_numeric(df[f], errors="coerce").to_numpy(), index=days)
    return {f: pd.DataFrame(v).reindex(index=index, columns=perps) for f, v in cols.items()}


def oi_growth(oi: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """ln(OI_d / OI_{d-lookback}) in contracts; NaN when either end is missing or not positive."""
    pos = oi.where(oi > 0)
    return np.log(pos / pos.shift(lookback))


def taker_flow(taker_log_mean: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    return taker_log_mean.rolling(window, min_periods=window - 2).mean()


def rank_weights(score: pd.DataFrame, eligible: pd.DataFrame, *, select_frac: float, long_high: bool,
                 min_assets: int = 8) -> pd.DataFrame:
    """Equal-weight long/short book: 0.5 spread over the top select_frac by score on the long side
    when long_high (else the bottom), -0.5 over the opposite tail. Same sizing as the momentum book."""
    s_all = score.where(eligible.reindex(index=score.index, columns=score.columns).fillna(False).astype(bool))
    w = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    for d in score.index:
        s = s_all.loc[d].dropna()
        if len(s) < min_assets:
            continue
        k = max(2, int(round(len(s) * select_frac)))
        ranked = s.sort_values()
        high, low = ranked.index[-k:], ranked.index[:k]
        longs, shorts = (high, low) if long_high else (low, high)
        w.loc[d, longs] = 0.5 / k
        w.loc[d, shorts] = -0.5 / k
    return w
