"""Download perpetual funding-rate history into the cache (incremental).

Writes <cache-dir>/<exchange>/funding/<SYMBOL>.csv.gz with columns timestamp (ms, funding
time) and funding_rate (per funding interval; longs pay when positive).

Example:
  PYTHONPATH=src python scripts/fetch_funding_history.py \
    --strategy-registry artifacts/asset_universe_ab/rt_registry_base_plus_first_crypto_discovery.json \
    --since "2021-01-01 00:00:00"
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from hf.data.ohlcv import dt_to_ms_utc


def funding_cache_path(cache_dir: str, exchange: str, symbol: str) -> Path:
    safe = symbol.replace("/", "_").replace(":", "_")
    return Path(cache_dir) / exchange / "funding" / f"{safe}.csv.gz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy-registry", required=True)
    ap.add_argument("--since", required=True)
    ap.add_argument("--exchange", default="binanceusdm")
    ap.add_argument("--cache-dir", default="data/cache")
    args = ap.parse_args()

    import ccxt

    ex = getattr(ccxt, args.exchange)({"enableRateLimit": True})
    ex.load_markets()
    rows = json.loads(Path(args.strategy_registry).read_text(encoding="utf-8"))
    symbols = sorted({str(r["symbol"]) for r in rows if r.get("symbol")})
    now_ms = int(time.time() * 1000)

    for symbol in symbols:
        path = funding_cache_path(args.cache_dir, args.exchange, symbol)
        cached = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["timestamp", "funding_rate"])
        cur = int(cached["timestamp"].max()) + 1 if len(cached) else dt_to_ms_utc(args.since)
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
        first = pd.to_datetime(merged["timestamp"].min(), unit="ms", utc=True) if len(merged) else None
        print(f"{symbol}: rows={len(merged)} new={len(new)} first={first} mean_rate={merged['funding_rate'].mean():.6f}")


if __name__ == "__main__":
    main()
