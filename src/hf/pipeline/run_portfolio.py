from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Dict, Optional

import pandas as pd

from hf.core.types import Candle, Allocation
from hf.engines.regime_csv import CSVRegimeEngine

from hf.engines.legacy_wrappers import (
    LEGACY_SYMBOLS,
    PlaceholderSignalEngine,
    StaticRegimeEngine,
    DynamicAllocator,
)
from hf.data.ohlcv import fetch_ohlcv_ccxt, dt_to_ms_utc


def _row_to_candle(ts: int, row: pd.Series) -> Candle:
    # ts in ms
    return Candle(
        ts=pd.to_datetime(int(ts), unit="ms", utc=True),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row.get("volume", 0.0)),
    )


def run(name: str, start: str, end: Optional[str], exchange: str, cache_dir: str, refresh_cache: bool, regime_csv: str) -> pd.DataFrame:
    start_ms = dt_to_ms_utc(start)
    end_ms = dt_to_ms_utc(end) if end else None

    btc_sym = LEGACY_SYMBOLS["BTC"]
    sol_sym = LEGACY_SYMBOLS["SOL"]

    btc = fetch_ohlcv_ccxt(btc_sym, "1h", start_ms, end_ms, exchange_id=exchange, cache_dir=cache_dir, use_cache=True, refresh_if_no_end=refresh_cache)
    sol = fetch_ohlcv_ccxt(sol_sym, "1h", start_ms, end_ms, exchange_id=exchange, cache_dir=cache_dir, use_cache=True, refresh_if_no_end=refresh_cache)

    if btc.empty or sol.empty:
        raise SystemExit("OHLCV empty for BTC or SOL. Check cache_dir/exchange/symbol/timeframe.")

    btc = btc.set_index("timestamp").sort_index()
    sol = sol.set_index("timestamp").sort_index()

    # common timestamps only (safe)
    common_ts = btc.index.intersection(sol.index)
    if len(common_ts) < 10:
        raise SystemExit(f"Not enough overlapping candles: {len(common_ts)}")

    sig_engine = PlaceholderSignalEngine()
    reg_engine = CSVRegimeEngine(csv_path=regime_csv)
    allocator = DynamicAllocator(both_btc_weight=0.75, sticky_when_both_off=True)

    prev_alloc: Optional[Allocation] = None
    rows = []

    for ts in common_ts:
        candles: Dict[str, Candle] = {
            btc_sym: _row_to_candle(ts, btc.loc[ts]),
            sol_sym: _row_to_candle(ts, sol.loc[ts]),
        }
        signals = sig_engine.generate(candles)
        regimes = reg_engine.evaluate(candles, signals)
        alloc = allocator.allocate(candles, signals, regimes, prev_alloc)
        prev_alloc = alloc

        rows.append({
            "ts": int(ts),
            "ts_utc": pd.to_datetime(int(ts), unit="ms", utc=True).isoformat(),
            "w_btc": float(alloc.weights.get(btc_sym, 0.0)),
            "w_sol": float(alloc.weights.get(sol_sym, 0.0)),
            "case": (alloc.meta or {}).get("case", ""),
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"results/pipeline_allocations_{name}.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pipeline")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--exchange", default="bitget")
    ap.add_argument("--cache-dir", default=".cache/ohlcv")
    ap.add_argument("--regime-csv", default="results/portfolio_regime_flags_v8ml_from_newrepo.csv")
    ap.add_argument("--refresh-cache", action="store_true")
    args = ap.parse_args()

    df = run(args.name, args.start, args.end, args.exchange, args.cache_dir, args.refresh_cache, args.regime_csv)
    print(f"Saved -> results/pipeline_allocations_{args.name}.csv (rows={len(df)})")


if __name__ == "__main__":
    main()
