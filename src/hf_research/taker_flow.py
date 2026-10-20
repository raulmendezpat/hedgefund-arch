"""Cross-sectional taker-flow signal for the paper runner (frozen in docs/plans/taker_flow_hypothesis.md).

Same score as the backtest secondary S2 (hf_research.oi_signals.taker_flow): per UTC day the mean
of ln(taker buy/sell volume ratio) over its 5-minute rows, then the 7-day mean (at least 5 days).
Long the top 20% by score, short the bottom 20%, equal weights 0.5 / -0.5.

Live data: Binance /futures/data/takerlongshortRatio (5m rows, last 30 days). Its buySellRatio at
timestamp T equals the archive's sum_taker_long_short_vol_ratio at create_time T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hf_research.oi_signals import binance_id, rank_weights, taker_flow

SELECT_FRAC = 0.2
WINDOW_DAYS = 7
MIN_ASSETS = 8
PAGE = pd.Timedelta(hours=40)  # 480 five-minute rows per request (API limit 500)


def daily_log_ratio(rows: list[dict]) -> pd.Series:
    """Mean ln(buySellRatio) per UTC day from API rows."""
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows).drop_duplicates("timestamp")
    t = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    r = pd.to_numeric(df["buySellRatio"], errors="coerce")
    ok = r > 0
    return np.log(r[ok]).groupby(t[ok].dt.floor("D")).mean()


def fetch_rows(ex, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    """5-minute taker rows with timestamps in [start, end] for a ccxt Binance USDT-M symbol."""
    rows, cur = [], start
    while cur <= end:
        stop = min(cur + PAGE, end)
        rows += ex.fapiDataGetTakerlongshortRatio({"symbol": binance_id(symbol), "period": "5m", "limit": 500,
                                                   "startTime": int(cur.value // 1_000_000),
                                                   "endTime": int(stop.value // 1_000_000)})
        cur = stop + pd.Timedelta(minutes=5)
    return [r for r in rows if start.value // 1_000_000 <= int(r["timestamp"]) <= end.value // 1_000_000]


def scores_for_day(daily: pd.DataFrame, decision_day: pd.Timestamp) -> pd.Series:
    """7-day taker-flow score per symbol at the decision day (NaN with fewer than 5 days of data)."""
    days = pd.date_range(decision_day - pd.Timedelta(days=WINDOW_DAYS - 1), decision_day, freq="D")
    return taker_flow(daily.reindex(days), WINDOW_DAYS).loc[decision_day]


def target_weights(scores: pd.Series) -> pd.Series:
    s = scores.dropna()
    frame = s.to_frame().T
    w = rank_weights(frame, pd.DataFrame(True, index=frame.index, columns=frame.columns),
                     select_frac=SELECT_FRAC, long_high=True, min_assets=MIN_ASSETS).iloc[0]
    return w[w != 0.0]
