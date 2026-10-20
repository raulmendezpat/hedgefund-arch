"""Cache daily Binance USDT-M positioning metrics (open interest, long/short ratios, taker flow)
for the liquid-universe membership, from the public archive at data.binance.vision.

Each archived day holds 5-minute rows; per symbol and UTC day this keeps the last row (open
interest and long/short ratios as of the day's end) and the day's mean of ln(taker buy/sell
volume ratio). Days come from the symbol's membership months plus the month before.
Incremental: days already cached (or known missing) are not requested again.

  PYTHONPATH=src python scripts/fetch_binance_metrics.py
Writes <cache_dir>/binance_metrics/<ID>.csv.gz (ID = BTCUSDT, ...).
"""
from __future__ import annotations

import argparse
import io
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://data.binance.vision/data/futures/um/daily/metrics/{id}/{id}-metrics-{day}.zip"
COLUMNS = ["day", "oi", "oi_value", "count_ls", "top_ls", "taker_ratio_log_mean", "last_ts", "missing"]


def binance_id(symbol: str) -> str:
    base, rest = symbol.split("/", 1)
    return base + rest.split(":")[0]


def metrics_path(cache_dir: str, bid: str) -> Path:
    return Path(cache_dir) / "binance_metrics" / f"{bid}.csv.gz"


def summarize_day(raw: pd.DataFrame, day: str) -> dict:
    t = pd.to_datetime(raw["create_time"])
    raw = raw[t.dt.strftime("%Y-%m-%d") == day].assign(_t=t).sort_values("_t")
    if raw.empty:
        return {"day": day, "missing": True}
    last = raw.iloc[-1]
    taker = pd.to_numeric(raw["sum_taker_long_short_vol_ratio"], errors="coerce")
    taker = taker[taker > 0]
    return {"day": day, "oi": float(last["sum_open_interest"]), "oi_value": float(last["sum_open_interest_value"]),
            "count_ls": float(last["count_long_short_ratio"]), "top_ls": float(last["sum_toptrader_long_short_ratio"]),
            "taker_ratio_log_mean": float(np.log(taker).mean()) if len(taker) else np.nan,
            "last_ts": str(last["_t"]), "missing": False}


def fetch_day(bid: str, day: str, retries: int = 3) -> dict:
    url = URL.format(id=urllib.parse.quote(bid), day=day)
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                raw = pd.read_csv(z.open(z.namelist()[0]))
            return summarize_day(raw, day)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {"day": day, "missing": True}
            time.sleep(2 ** k)
        except Exception:  # noqa: BLE001 - network hiccups and truncated archives are retried
            time.sleep(2 ** k)
    return {"day": day, "missing": None}  # unknown: retried on the next run


def wanted_days(membership: pd.DataFrame, end: pd.Timestamp) -> dict[str, list[str]]:
    out = {}
    for sym, months in membership.groupby("symbol")["month"]:
        days = set()
        for m in months:
            start = pd.Timestamp(f"{m}-01") - pd.DateOffset(months=1)
            stop = min(pd.Timestamp(f"{m}-01") + pd.DateOffset(months=1) - pd.Timedelta(days=1), end)
            days |= {d.strftime("%Y-%m-%d") for d in pd.date_range(start, stop, freq="D")}
        out[sym] = sorted(days)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", default="artifacts/research/universe_top60_v1/membership.csv")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--end", default="", help="last day (default: yesterday UTC)")
    args = ap.parse_args()
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D") - pd.Timedelta(days=1)

    want = wanted_days(pd.read_csv(args.membership), end)
    jobs, cached = [], {}
    for sym, days in want.items():
        bid = binance_id(sym)
        path = metrics_path(args.cache_dir, bid)
        have = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
        known = set(have.loc[have["missing"].notna(), "day"]) if len(have) else set()
        cached[bid] = have
        jobs += [(bid, d) for d in days if d not in known]
    print(f"{len(want)} symbols, {len(jobs)} days to fetch", flush=True)

    t0, results = time.time(), {}
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(fetch_day, bid, d): bid for bid, d in jobs}
        for k, f in enumerate(as_completed(futs), start=1):
            results.setdefault(futs[f], []).append(f.result())
            if k % 2000 == 0:
                print(f"{k}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)

    unknown = 0
    for bid, rows in results.items():
        new = pd.DataFrame(rows)
        unknown += int(new["missing"].isna().sum())
        merged = pd.concat([cached[bid], new[new["missing"].notna()]], ignore_index=True)
        merged = merged.drop_duplicates("day", keep="last").sort_values("day").reindex(columns=COLUMNS)
        path = metrics_path(args.cache_dir, bid)
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(path, index=False, compression="gzip")
    print(f"done in {time.time() - t0:.0f}s; {unknown} days failed and will be retried next run")


if __name__ == "__main__":
    main()
