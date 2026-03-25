import json
from pathlib import Path
import pandas as pd

from hf.data.ohlcv import fetch_ohlcv_ccxt, dt_to_ms_utc

REGISTRY = "artifacts/strategy_registry.json"
EXCHANGE = "binanceusdm"
CACHE_DIR = "data/cache"

rows = json.loads(Path(REGISTRY).read_text(encoding="utf-8"))
symbols = []
seen = set()
for r in rows:
    s = str(r.get("symbol", "") or "")
    if s and s not in seen:
        seen.add(s)
        symbols.append(s)

out = []
for sym in symbols:
    df = fetch_ohlcv_ccxt(
        symbol=sym,
        timeframe="1h",
        start_ms=dt_to_ms_utc("2024-01-01 00:00:00"),
        end_ms=None,
        exchange_id=EXCHANGE,
        cache_dir=CACHE_DIR,
        use_cache=True,
        refresh_if_no_end=True,
    ).copy()
    ts = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True, errors="coerce").dropna()
    out.append({
        "symbol": sym,
        "rows": len(ts),
        "min_ts": ts.min(),
        "max_ts": ts.max(),
    })

cov = pd.DataFrame(out).sort_values("symbol")
print(cov.to_string(index=False))

common_start = cov["min_ts"].max()
common_end = cov["max_ts"].min()
print("\ncommon_start:", common_start)
print("common_end  :", common_end)
print("common_hours:", (common_end - common_start).total_seconds() / 3600.0)
