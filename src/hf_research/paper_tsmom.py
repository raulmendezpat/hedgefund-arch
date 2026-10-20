"""Paper trading of time-series momentum on BTC/ETH (plan: docs/plans/tsmom_paper_plan.md).

Fixed before the paper run starts (do not tune while it runs):
- Strategy: TsmomConfig() primary (lookbacks 20/60/120, 20% vol per asset, band 0.05,
  long/short) on Binance 1h closes at 23:00 UTC.
- Schedule: decision every day d, fill on d+1 at 01:00 UTC; a fill later than
  max_fill_delay_hours is skipped and logged as missed.
- Execution and accounting: hf_research.paper_xs (PaperBook, walk_book, plan_weights) against the
  live Bitget order book, taker fee 0.06%, actual Bitget funding.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

ASSETS = ("BTC/USDT:USDT", "ETH/USDT:USDT")
COMBO_TSMOM_SCALE = 1.0  # backtest scale to match momentum's in-sample vol was 0.98


@dataclass(frozen=True)
class TsmomPaperConfig:
    name: str = "tsmom_btc_eth_v1"
    capital_usdt: float = 10_000.0
    taker_fee: float = 0.0006
    fill_hour_utc: int = 1
    max_fill_delay_hours: int = 12
    min_order_usdt: float = 5.0
    history_days: int = 200  # 120-day lookback + 60-day vol window + slack


def due_daily_decision(now: pd.Timestamp, start: pd.Timestamp, last_decision_day: pd.Timestamp | None,
                       cfg: TsmomPaperConfig) -> tuple[pd.Timestamp | None, str, list[pd.Timestamp]]:
    """(latest decision day whose fill time has passed, "due" | "missed" | "idle", skipped days).

    Skipped days are earlier decision days after the last handled one that were never traded
    (the host was down past their fill window); the caller logs them as missed."""
    fill_offset = pd.Timedelta(days=1, hours=cfg.fill_hour_utc)
    d = (now - fill_offset).floor("D")
    fill = d + fill_offset
    if fill < start or (last_decision_day is not None and last_decision_day >= d):
        return None, "idle", []
    first = (start - fill_offset).ceil("D")  # first decision day whose fill is at or after start
    if last_decision_day is not None:
        first = max(first, last_decision_day + pd.Timedelta(days=1))
    skipped = list(pd.date_range(first, d - pd.Timedelta(days=1), freq="D"))
    if now - fill > pd.Timedelta(hours=cfg.max_fill_delay_hours):
        return d, "missed", skipped
    return d, "due", skipped


def daily_returns(equity: pd.DataFrame, capital: float, fill_hour: int) -> pd.Series:
    """Paper daily returns aligned with the model rows: the first mark at or after the fill hour
    each UTC day closes the model row of the previous day (fill d-1 -> fill d)."""
    if equity.empty:
        return pd.Series(dtype=float)
    e = equity.assign(ts=pd.to_datetime(equity["ts"], utc=True)).set_index("ts")["equity"]
    e = e[e.index.hour >= fill_hour]
    snap = e.groupby(e.index.floor("D")).first()
    if snap.empty:
        return pd.Series(dtype=float)
    snap.loc[snap.index[0] - pd.Timedelta(days=1)] = float(capital)
    r = snap.sort_index().pct_change().dropna()
    r.index = r.index - pd.Timedelta(days=1)
    return r
