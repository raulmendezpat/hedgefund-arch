"""Data for the paper reports' model replays.

The replays read the Binance cache. The hourly runners only refresh the symbols of the current
month's universe, so a symbol that drops out of it keeps a stale cache while the book still holds
it until the next rebalance. The backtest treats a missing price as a zero return, which hid a
+30% move of a held short (LSK, week of 2026-10-26) from the replay. Hence:
- refresh_replay_cache: refresh 1h closes and funding of every symbol the book has held;
- missing_held_prices: the replay rows where a held symbol still lacks a price, so the report
  can stop the comparison there instead of comparing against a wrong model.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from hf_research.liquid_universe import update_cache


def funding_cache_path(cache_dir: str, exchange: str, symbol: str) -> Path:
    safe = symbol.replace("/", "_").replace(":", "_")
    return Path(cache_dir) / exchange / "funding" / f"{safe}.csv.gz"


def update_funding(ex, symbol: str, since_ms: int, cache_dir: str, exchange: str) -> pd.DataFrame:
    """Append funding rows after the cached max (or from since_ms) and write the merged file."""
    path = funding_cache_path(cache_dir, exchange, symbol)
    cached = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["timestamp", "funding_rate"])
    cur = int(cached["timestamp"].max()) + 1 if len(cached) else int(since_ms)
    now_ms = int(time.time() * 1000)
    fetched = []
    while cur < now_ms:
        batch = ex.fetch_funding_rate_history(symbol, since=cur, limit=1000)
        if not batch:
            break
        fetched.extend((int(b["timestamp"]), float(b["fundingRate"])) for b in batch)
        last = int(batch[-1]["timestamp"])
        if last < cur:
            break
        cur = last + 1
    new = pd.DataFrame(fetched, columns=["timestamp", "funding_rate"])
    merged = (
        pd.concat([cached, new], ignore_index=True)
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False, compression="gzip")
    return merged


def held_symbols(reb: pd.DataFrame) -> list[str]:
    """Every symbol (Binance) the logged rebalances targeted."""
    if "targets" not in reb:
        return []
    return sorted({s for t in reb["targets"] if isinstance(t, dict) for s in t})


def refresh_replay_cache(ex, symbols: list[str], since: pd.Timestamp, cache_dir: str, exchange: str, log=print) -> None:
    since_ms = int(since.value // 1_000_000)
    for s in symbols:
        try:
            update_cache(ex, s, "1h", since_ms, cache_dir, exchange)
            update_funding(ex, s, since_ms, cache_dir, exchange)
        except Exception as exc:  # noqa: BLE001 - missing_held_prices reports what is still absent
            log(f"replay cache refresh failed {s}: {type(exc).__name__}")


def missing_held_prices(exec_close: pd.DataFrame, days: pd.Series, targets: list[dict], rows: pd.Index) -> dict[str, list[str]]:
    """{row day: symbols} for replay rows whose return needs a price the cache lacks.

    Row r earns exec_close[r] / exec_close[r-1] on the targets of the last decision day before r."""
    days = list(pd.DatetimeIndex(days))
    out = {}
    for r in rows:
        k = sum(d < r for d in days) - 1
        if k < 0:
            continue
        held = [s for s, w in targets[k].items() if w]
        prev = r - pd.Timedelta(days=1)
        miss = [s for s in held
                if s not in exec_close.columns or r not in exec_close.index or prev not in exec_close.index
                or pd.isna(exec_close.at[r, s]) or pd.isna(exec_close.at[prev, s])]
        if miss:
            out[str(r.date())] = sorted(miss)
    return out
