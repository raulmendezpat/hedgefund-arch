"""Daily data for the traditional-asset perpetuals on Bitget (stocks, ETFs, metals, energy, FX).

Bitget lists these as USDT-M perpetuals ("isRwa": "YES") that trade around the clock, but its
history only starts in 2026-08. Backtests therefore use the underlying's daily closes from Yahoo
Finance as the price, and model the perpetual's funding separately (see
docs/plans/tradfi_trend_hypothesis.md).

Cache: <cache_dir>/yahoo/<TICKER>.csv with columns date (exchange-local trading day), close
(split-adjusted, without dividends: a perpetual holder does not receive them) and adjclose.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def yahoo_cache_path(cache_dir: str | Path, ticker: str) -> Path:
    return Path(cache_dir) / "yahoo" / f"{ticker.replace('^', '_').replace('=', '_')}.csv"


def fetch_yahoo_daily(ticker: str, start: str = "1995-01-01", session: requests.Session | None = None) -> pd.DataFrame:
    """Full daily history of one Yahoo ticker: date, close, adjclose (NaN rows dropped)."""
    s = session or requests.Session()
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(time.time()) + 86400
    r = s.get(YAHOO_CHART.format(ticker=ticker), headers=_HEADERS, timeout=30,
              params={"period1": p1, "period2": p2, "interval": "1d", "events": "div,split"})
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    tz = res["meta"].get("exchangeTimezoneName", "UTC")
    ts = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz)
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
    df = pd.DataFrame({"date": ts.strftime("%Y-%m-%d"), "close": q["close"], "adjclose": adj})
    return df.dropna(subset=["close"]).drop_duplicates("date", keep="last").reset_index(drop=True)


def update_yahoo(tickers, cache_dir: str | Path, start: str = "1995-01-01", max_age_hours: float = 12.0, log=print) -> None:
    """Refresh the cached history of each ticker older than max_age_hours (full re-download: Yahoo
    revises split adjustments retroactively, so appending would mix bases)."""
    s = requests.Session()
    for t in tickers:
        path = yahoo_cache_path(cache_dir, t)
        if path.exists() and time.time() - path.stat().st_mtime < max_age_hours * 3600:
            continue
        df = fetch_yahoo_daily(t, start, s)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        log(f"{t}: {len(df)} days {df['date'].iloc[0]} .. {df['date'].iloc[-1]}")
        time.sleep(0.3)


def load_closes(tickers, cache_dir: str | Path, column: str = "close") -> pd.DataFrame:
    """Trading-day closes, one column per ticker, indexed by date (tz-naive, union of calendars)."""
    cols = {}
    for t in tickers:
        df = pd.read_csv(yahoo_cache_path(cache_dir, t))
        cols[t] = df.set_index(pd.to_datetime(df["date"]))[column].astype(float)
    return pd.DataFrame(cols).sort_index()


def to_calendar(closes: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Calendar-day UTC panel: weekends and holidays carry the last close forward (zero return),
    matching a perpetual that is held every day. Leading NaNs (before listing) are kept."""
    idx = pd.date_range(start or closes.index.min(), end or closes.index.max(), freq="D", tz="UTC")
    c = closes.copy()
    c.index = pd.DatetimeIndex(c.index).tz_localize("UTC")
    return c.reindex(idx.union(c.index)).ffill().reindex(idx).where(c.reindex(idx).notna().cummax())


def bitget_rate_drag(funding_annual: pd.Series | float, index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """Daily funding panel for the engine (fraction per day, longs pay when positive) from an
    annualised rate per asset (Series by column or one float), constant over time."""
    rate = (funding_annual.reindex(columns).fillna(0.0) if isinstance(funding_annual, pd.Series)
            else pd.Series(float(funding_annual), index=columns))
    return pd.DataFrame(np.tile((rate / 365.0).to_numpy(), (len(index), 1)), index=index, columns=columns)
