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
from pathlib import Path

import pandas as pd

from hf.data.ohlcv import dt_to_ms_utc
from hf_research.replay_data import funding_cache_path, update_funding  # noqa: F401  (re-exported)


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

    for symbol in symbols:
        merged = update_funding(ex, symbol, dt_to_ms_utc(args.since), args.cache_dir, args.exchange)
        first = pd.to_datetime(merged["timestamp"].min(), unit="ms", utc=True) if len(merged) else None
        print(f"{symbol}: rows={len(merged)} first={first} mean_rate={merged['funding_rate'].mean():.6f}")


if __name__ == "__main__":
    main()
