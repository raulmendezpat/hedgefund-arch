"""Re-download a window of OHLCV and overwrite the cached candles with exchange values.

Older cache versions persisted the still-forming candle of every refresh and never
corrected it afterwards, so a cache refreshed hourly can hold a partial candle per hour.
This script re-fetches [--since, now) for every symbol in a strategy registry, lets the
exchange values win, drops unclosed candles and reports how many cached rows changed.

Example:
  PYTHONPATH=src python scripts/repair_ohlcv_cache.py \
    --strategy-registry artifacts/asset_universe_ab/rt_registry_base_plus_first_crypto_discovery.json \
    --since "2026-01-01 00:00:00"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hf.data.ohlcv import _cache_path, drop_unclosed_candles, dt_to_ms_utc, fetch_closed_window, timeframe_to_ms


def _registry_symbols(path: str) -> list[str]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[str] = []
    for r in rows:
        s = str(r.get("symbol", "") or "")
        if s and s not in out:
            out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy-registry", required=True)
    ap.add_argument("--since", required=True, help="UTC 'YYYY-MM-DD HH:MM:SS'")
    ap.add_argument("--exchange", default="binanceusdm")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--dry-run", action="store_true", help="Report differences without writing the cache")
    args = ap.parse_args()

    import ccxt

    ex = getattr(ccxt, args.exchange)({"enableRateLimit": True})
    ex.load_markets()
    since_ms = dt_to_ms_utc(args.since)
    tf_ms = timeframe_to_ms(args.timeframe)
    cols = ["open", "high", "low", "close", "volume"]

    for symbol in _registry_symbols(args.strategy_registry):
        path = _cache_path(args.cache_dir, args.exchange, symbol, args.timeframe)
        cached = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["timestamp", *cols])
        if not cached.empty:
            cached["timestamp"] = cached["timestamp"].astype("int64")
        fresh = fetch_closed_window(ex, symbol, args.timeframe, since_ms)

        both = cached.merge(fresh, on="timestamp", suffixes=("_cached", "_fresh"))
        changed = int(
            (
                (both[[f"{c}_cached" for c in cols]].to_numpy() != both[[f"{c}_fresh" for c in cols]].to_numpy())
                .any(axis=1)
            ).sum()
        )

        merged = pd.concat([cached, fresh], ignore_index=True)
        merged = merged.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
        merged = drop_unclosed_candles(merged, args.timeframe)
        gaps = int((merged["timestamp"].diff().dropna() != tf_ms).sum()) if len(merged) > 1 else 0
        last = pd.to_datetime(merged["timestamp"].iloc[-1], unit="ms", utc=True) if len(merged) else None

        print(
            f"{symbol}: compared={len(both)} changed={changed} fresh_rows={len(fresh)} "
            f"total_rows={len(merged)} last_closed={last} gaps={gaps}"
        )
        if not args.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(path, index=False, compression="gzip")


if __name__ == "__main__":
    main()
