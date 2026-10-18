from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Optional

import pandas as pd


# Bars re-fetched behind the cached max on every refresh, so candles that were
# stored while still forming (older cache versions) get overwritten.
REFETCH_OVERLAP_BARS = 3

_TF_UNIT_MS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}


def timeframe_to_ms(timeframe: str) -> int:
    tf = str(timeframe).strip()
    unit = tf[-1].lower()
    if unit not in _TF_UNIT_MS or not tf[:-1].isdigit():
        raise ValueError(f"unsupported timeframe: {timeframe!r}")
    return int(tf[:-1]) * _TF_UNIT_MS[unit]


def drop_unclosed_candles(df: pd.DataFrame, timeframe: str, now_ms: Optional[int] = None) -> pd.DataFrame:
    """Keep only candles whose period has fully elapsed (timestamp is the bar open)."""
    if df is None or df.empty or "timestamp" not in df.columns:
        return df
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    tf_ms = timeframe_to_ms(timeframe)
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    return df[(ts + tf_ms) <= now_ms].copy()


def _cache_path(cache_dir: str, exchange_id: str, symbol: str, timeframe: str) -> Path:
    # Mirror legacy format: CSV.GZ per (exchange, symbol, timeframe)
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    return Path(cache_dir) / exchange_id / timeframe / f"{safe_symbol}.csv.gz"


def load_ohlcv_cached(symbol: str, timeframe: str, exchange_id: str = "bitget", cache_dir: str = ".cache/ohlcv") -> pd.DataFrame:
    p = _cache_path(cache_dir, exchange_id, symbol, timeframe)
    if not p.exists():
        raise FileNotFoundError(f"OHLCV cache not found: {p}")
    df = pd.read_csv(p)
    # legacy columns usually: timestamp, open, high, low, close, volume
    # normalize
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype("int64")
    return df


def fetch_ohlcv_ccxt(
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: Optional[int],
    exchange_id: str = "bitget",
    cache_dir: str = ".cache/ohlcv",
    use_cache: bool = True,
    refresh_if_no_end: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV via ccxt with local CSV.GZ cache.

    - Cache keyed by (exchange_id, symbol, timeframe)
    - Stored as CSV.GZ under cache_dir/<exchange>/<timeframe>/<symbol>.csv.gz
    """
    import ccxt  # local import to keep base deps small

    cache_path = _cache_path(cache_dir, exchange_id, symbol, timeframe) if use_cache else None
    cache_path.parent.mkdir(parents=True, exist_ok=True) if cache_path else None

    def _read_cache() -> Optional[pd.DataFrame]:
        if cache_path and cache_path.exists():
            try:
                df = pd.read_csv(cache_path)
                if "timestamp" in df.columns:
                    df["timestamp"] = df["timestamp"].astype("int64")
                return df
            except Exception:
                return None
        return None

    cached = _read_cache() if use_cache else None
    if cached is not None and not cached.empty:
        have_min = int(cached["timestamp"].min())
        have_max = int(cached["timestamp"].max())

        # For bounded windows, only return directly from cache if the sliced range is complete and continuous.
        if end_ms is not None:
            if have_min <= int(start_ms) and have_max >= int(end_ms):
                sliced = cached[(cached["timestamp"] >= int(start_ms)) & (cached["timestamp"] <= int(end_ms))].copy()
                if not sliced.empty:
                    sliced = sliced.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                    tf_ms = 60 * 60 * 1000  # research_runtime uses 1h
                    expected_rows = ((int(end_ms) - int(start_ms)) // tf_ms) + 1
                    deltas_ok = True
                    if len(sliced) > 1:
                        deltas_ok = bool((sliced["timestamp"].diff().dropna() == tf_ms).all())
                    if len(sliced) == expected_rows and deltas_ok:
                        return sliced

        # For open-ended windows without refresh, return cached data from start onward.
        if end_ms is None and not refresh_if_no_end:
            sliced = cached[cached["timestamp"] >= int(start_ms)].copy()
            if not sliced.empty:
                return sliced

    # Fetch from exchange
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    ex.load_markets()

    limit = 500
    cur = int(start_ms)
    rows = []

    tf_ms = timeframe_to_ms(timeframe)
    overlap_ms = REFETCH_OVERLAP_BARS * tf_ms

    # If bounded window and cache already covers the beginning, continue from the cached max
    # (minus a small overlap so previously stored partial candles are repaired).
    if cached is not None and not cached.empty and end_ms is not None:
        have_min = int(cached["timestamp"].min())
        have_max = int(cached["timestamp"].max())
        if have_min <= int(start_ms) and have_max < int(end_ms):
            cur = max(int(start_ms), have_max + 1 - overlap_ms)

    # If we have cache, restart a few bars behind the last cached candle when refreshing without end_ms.
    if cached is not None and not cached.empty and end_ms is None and refresh_if_no_end:
        cur = max(int(start_ms), int(cached["timestamp"].max()) + 1 - overlap_ms)

    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cur, limit=limit)
        if not batch:
            break
        rows.extend(batch)

        last_ts = int(batch[-1][0])
        if last_ts <= cur:
            break
        cur = last_ts + 1

        if end_ms is not None and cur > int(end_ms):
            break

    df_new = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if not df_new.empty:
        df_new["timestamp"] = df_new["timestamp"].astype("int64")
    # The exchange returns the still-forming candle last; never use or persist it.
    df_new = drop_unclosed_candles(df_new, timeframe)

    # Merge with cache and write back. Fresh exchange rows win over cached ones.
    if use_cache and cache_path:
        if cached is not None and not cached.empty:
            merged = pd.concat([cached, df_new], ignore_index=True)
            merged = merged.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
        else:
            merged = df_new
        merged.to_csv(cache_path, index=False, compression="gzip")
        df_out = merged
    else:
        df_out = df_new

    # Slice to requested window
    if end_ms is not None and not df_out.empty:
        df_out = df_out[(df_out["timestamp"] >= int(start_ms)) & (df_out["timestamp"] <= int(end_ms))].copy()
    elif not df_out.empty:
        df_out = df_out[df_out["timestamp"] >= int(start_ms)].copy()

    return df_out


def fetch_closed_window(ex, symbol: str, timeframe: str, since_ms: int, limit: int = 1000) -> pd.DataFrame:
    """Paginate ex.fetch_ohlcv from since_ms to now and keep closed candles only."""
    rows = []
    cur = int(since_ms)
    now_ms = int(time.time() * 1000)
    step = timeframe_to_ms(timeframe)
    while cur <= now_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cur, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        last_ts = int(batch[-1][0])
        if last_ts < cur:
            break
        cur = last_ts + step
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if not df.empty:
        df["timestamp"] = df["timestamp"].astype("int64")
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
    return drop_unclosed_candles(df, timeframe)


def dt_to_ms_utc(s: str) -> int:
    # expects "YYYY-MM-DD HH:MM:SS"
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)
