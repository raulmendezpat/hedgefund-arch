"""Cache Binance spot 1h candles for every perpetual in the liquid-universe membership.

The spot pair of a perpetual is <BASE>/USDT, or the base without a 1000 / 1000000 / 1M size
prefix (1000PEPE -> PEPE). Writes the perp -> spot mapping (and the perps without a spot pair)
to <cache_dir>/binance/spot_map.json. Candles start two months before the symbol's first month
in the universe.

  PYTHONPATH=src python scripts/fetch_binance_spot.py
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

from hf_research.liquid_universe import update_cache

SPOT_EXCHANGE = "binance"


def spot_candidates(perp_symbol: str) -> list[str]:
    base = perp_symbol.split("/")[0]
    out = [f"{base}/USDT"]
    stripped = re.sub(r"^(1000000|1000|1M)(?=[A-Z])", "", base)
    if stripped != base:
        out.append(f"{stripped}/USDT")
    return out


def spot_symbol(perp_symbol: str, spot_markets: set[str]) -> str | None:
    return next((c for c in spot_candidates(perp_symbol) if c in spot_markets), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", default="artifacts/research/universe_top60_v1/membership.csv")
    ap.add_argument("--cache-dir", default="data/cache")
    args = ap.parse_args()

    import ccxt

    ex = ccxt.binance({"enableRateLimit": True, "timeout": 30000})
    markets = ex.load_markets()
    spot = {k for k, v in markets.items() if v.get("spot") and v.get("quote") == "USDT"}
    m = pd.read_csv(args.membership)
    first_month = m.groupby("symbol")["month"].min()
    mapping, missing = {}, []
    t0 = time.time()
    for k, (perp, month) in enumerate(sorted(first_month.items()), start=1):
        s = spot_symbol(perp, spot)
        if s is None:
            missing.append(perp)
            continue
        since = pd.Timestamp(f"{month}-01", tz="UTC") - pd.DateOffset(months=2)
        try:
            df = update_cache(ex, s, "1h", int(since.value // 1_000_000), args.cache_dir, SPOT_EXCHANGE)
            mapping[perp] = s
            print(f"[{k}/{len(first_month)}] {perp} -> {s}: {len(df)} bars ({time.time() - t0:.0f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001 - delisted spot pairs can reject history requests
            missing.append(perp)
            print(f"[{k}/{len(first_month)}] {perp} -> {s}: failed {type(exc).__name__}", flush=True)
    out = Path(args.cache_dir) / SPOT_EXCHANGE / "spot_map.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mapping": mapping, "no_spot": sorted(missing)}, indent=2))
    print(f"done: {len(mapping)} with spot, {len(missing)} without; {out}")


if __name__ == "__main__":
    main()
