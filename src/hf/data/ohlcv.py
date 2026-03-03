from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


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
        # If end_ms is set, just slice from cached (best effort).
        if end_ms is not None:
            sliced = cached[(cached["timestamp"] >= int(start_ms)) & (cached["timestamp"] <= int(end_ms))].copy()
            if not sliced.empty:
                return sliced
        # If no end is provided and refresh requested, we will extend cache to latest.
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

    # If we have cache, start from last+1 candle when refreshing without end_ms.
    if cached is not None and not cached.empty and end_ms is None and refresh_if_no_end:
        cur = int(cached["timestamp"].max()) + 1

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

    # Merge with cache and write back
    if use_cache and cache_path:
        if cached is not None and not cached.empty:
            merged = pd.concat([cached, df_new], ignore_index=True)
            merged = merged.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
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


def dt_to_ms_utc(s: str) -> int:
    # expects "YYYY-MM-DD HH:MM:SS"
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)
