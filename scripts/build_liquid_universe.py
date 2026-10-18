"""Point-in-time liquid universe of Binance USDT-M crypto perpetuals, including delisted ones.

1. Lists every linear USDT crypto PERPETUAL market, active or SETTLING (delisted contracts that
   the exchange still serves), excluding stablecoins and index contracts.
2. Caches daily candles for all of them (cheap) and, at every month start, ranks the symbols by
   their trailing median daily quote volume using days strictly before that month.
3. Writes the monthly membership table and, with --fetch-hourly, caches 1h candles and funding
   for every symbol that ever entered the top N, so strategies can use the universe as it was.

Example:
  PYTHONPATH=src python scripts/build_liquid_universe.py --top-n 60 --since "2021-01-01 00:00:00" --fetch-hourly
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hf.data.ohlcv import _cache_path, dt_to_ms_utc, fetch_closed_window

EXCLUDED_BASES = {
    "USDC", "BUSD", "TUSD", "FDUSD", "DAI", "USDP", "USDE", "PYUSD", "EUR", "BTCDOM", "DEFI", "FOOTBALL", "BLUEBIRD",
}


def crypto_perpetuals(ex) -> list[dict]:
    out = []
    for m in ex.load_markets().values():
        info = m.get("info", {}) or {}
        if not (m.get("swap") and m.get("linear") and m.get("quote") == "USDT" and m.get("settle") == "USDT"):
            continue
        if info.get("contractType") != "PERPETUAL" or info.get("status") not in {"TRADING", "SETTLING"}:
            continue
        if str(m.get("base", "")).upper() in EXCLUDED_BASES:
            continue
        out.append({"symbol": m["symbol"], "status": info.get("status"), "onboard_ms": int(info.get("onboardDate") or 0)})
    return sorted(out, key=lambda r: r["symbol"])


def update_cache(ex, symbol: str, timeframe: str, since_ms: int, cache_dir: str, exchange: str) -> pd.DataFrame:
    """Fetch from the cached max (minus 3 bars) or since_ms, merge with fresh rows winning, write back."""
    path = _cache_path(cache_dir, exchange, symbol, timeframe)
    cached = pd.read_csv(path) if path.exists() else pd.DataFrame()
    start = since_ms
    if len(cached):
        cached["timestamp"] = cached["timestamp"].astype("int64")
        if int(cached["timestamp"].min()) <= since_ms:
            start = int(cached["timestamp"].max()) - 3 * (int(cached["timestamp"].diff().median()) or 3_600_000)
    fresh = fetch_closed_window(ex, symbol, timeframe, start)
    merged = pd.concat([cached, fresh], ignore_index=True) if len(cached) else fresh
    if len(merged):
        merged = merged.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(path, index=False, compression="gzip")
    return merged


def monthly_membership(daily: dict[str, pd.DataFrame], *, top_n: int, window_days: int, min_history_days: int) -> pd.DataFrame:
    qv = {}
    for sym, df in daily.items():
        if df.empty:
            continue
        t = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        qv[sym] = pd.Series((df["close"] * df["volume"]).to_numpy(), index=t)
    panel = pd.DataFrame(qv).sort_index()
    rows = []
    for month in pd.date_range(panel.index.min().normalize() + pd.offsets.MonthBegin(1), panel.index.max(), freq="MS"):
        hist = panel[panel.index < month]
        recent = hist.tail(window_days)
        eligible = hist.notna().sum() >= min_history_days
        # A symbol that stopped trading before the window has no recent rows and drops out.
        med = recent.median().where(eligible & recent.notna().sum().ge(window_days // 2)).dropna()
        for rank, (sym, v) in enumerate(med.sort_values(ascending=False).head(top_n).items(), start=1):
            rows.append({"month": month.strftime("%Y-%m"), "symbol": sym, "rank": rank, "median_quote_volume": float(v)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=60)
    ap.add_argument("--since", default="2021-01-01 00:00:00")
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--min-history-days", type=int, default=60)
    ap.add_argument("--exchange", default="binanceusdm")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--fetch-hourly", action="store_true")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    import ccxt

    from fetch_funding_history import update_funding  # noqa: E402  (sibling script)

    ex = getattr(ccxt, args.exchange)({"enableRateLimit": True})
    markets = crypto_perpetuals(ex)
    since_ms = dt_to_ms_utc(args.since)
    print(f"markets: {len(markets)} ({sum(m['status'] == 'SETTLING' for m in markets)} delisted)", flush=True)

    daily = {}
    for k, m in enumerate(markets, start=1):
        try:
            daily[m["symbol"]] = update_cache(ex, m["symbol"], "1d", since_ms, args.cache_dir, args.exchange)
        except Exception as exc:  # a few delisted contracts reject history requests
            print(f"skip 1d {m['symbol']}: {type(exc).__name__}: {exc}", flush=True)
        if k % 100 == 0:
            print(f"daily {k}/{len(markets)}", flush=True)

    membership = monthly_membership(daily, top_n=args.top_n, window_days=args.window_days, min_history_days=args.min_history_days)
    members = sorted(membership["symbol"].unique())
    print(f"symbols ever in top {args.top_n}: {len(members)}", flush=True)

    hourly_status = {}
    if args.fetch_hourly:
        for k, sym in enumerate(members, start=1):
            try:
                h = update_cache(ex, sym, "1h", since_ms, args.cache_dir, args.exchange)
                f = update_funding(ex, sym, since_ms, args.cache_dir, args.exchange)
                hourly_status[sym] = {"bars_1h": int(len(h)), "funding_rows": int(len(f))}
            except Exception as exc:
                hourly_status[sym] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"hourly {k}/{len(members)} {sym} {hourly_status[sym]}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir or f"artifacts/research/universe_top{args.top_n}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    membership.to_csv(out_dir / "membership.csv", index=False)
    status = {m["symbol"]: m["status"] for m in markets}
    manifest = {
        "created_utc": stamp,
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "exchange": args.exchange,
        "top_n": args.top_n,
        "window_days": args.window_days,
        "min_history_days": args.min_history_days,
        "markets_listed": len(markets),
        "members_ever": len(members),
        "members_delisted": [s for s in members if status.get(s) == "SETTLING"],
        "excluded_bases": sorted(EXCLUDED_BASES),
        "hourly_status": hourly_status,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
