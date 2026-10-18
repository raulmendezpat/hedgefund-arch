"""Check whether the point-in-time liquid universe (built on Binance) is tradable on Bitget.

For the symbols in the most recent months of the membership file it reports:
- listing on Bitget (linear USDT perpetual, active),
- Bitget 24h quote volume, current bid/ask spread,
- order-book cost (half spread + depth walk, in bps) to fill several notionals,
- Bitget vs Binance funding over the last --funding-days (mean daily rate, correlation).

Order books and funding are snapshots of today: they calibrate costs, they are not history.

Example:
  PYTHONPATH=src python scripts/check_bitget_universe.py \
    --membership artifacts/research/universe_top60_v1/membership.csv --months 3
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

NOTIONALS = (2_000, 10_000, 50_000)
# Same asset under a different ticker on Bitget (Binance base -> Bitget base).
BASE_ALIASES = {"RAYSOL": "RAY"}


def bitget_symbol(symbol: str, listed: set[str]) -> str | None:
    """Bitget market for a Binance symbol: same name, 1000x-prefix variants, or a known alias."""
    base, rest = symbol.split("/", 1)
    candidates = [base, base.removeprefix("1000"), f"1000{base}", BASE_ALIASES.get(base, base)]
    for b in candidates:
        if f"{b}/{rest}" in listed:
            return f"{b}/{rest}"
    return None


def book_cost_bps(book: dict, notional: float, side: str) -> float:
    """Average fill price vs mid, in bps, for a market order of `notional` USDT (NaN if too thin)."""
    bids, asks = book.get("bids") or [], book.get("asks") or []
    if not bids or not asks:
        return float("nan")
    mid = (float(bids[0][0]) + float(asks[0][0])) / 2.0
    levels = asks if side == "buy" else bids
    left, qty, spent = notional, 0.0, 0.0
    for px, amt, *_ in levels:
        px, amt = float(px), float(amt)
        take = min(left, px * amt)
        qty += take / px
        spent += take
        left -= take
        if left <= 1e-9:
            break
    if left > 1e-9:
        return float("nan")
    avg = spent / qty
    return float(abs(avg / mid - 1.0) * 10000.0)


def funding_history(ex, symbol: str, since_ms: int) -> pd.DataFrame:
    rows, cur = [], since_ms
    now_ms = int(time.time() * 1000)
    while cur < now_ms:
        batch = ex.fetch_funding_rate_history(symbol, since=cur, limit=100)
        if not batch:
            break
        rows.extend((int(b["timestamp"]), float(b["fundingRate"])) for b in batch)
        last = int(batch[-1]["timestamp"])
        if last <= cur:
            break
        cur = last + 1
    return pd.DataFrame(rows, columns=["timestamp", "funding_rate"]).drop_duplicates("timestamp")


def daily_funding(df: pd.DataFrame) -> pd.Series:
    t = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.round("min").dt.floor("D")
    return df.groupby(t)["funding_rate"].sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", required=True)
    ap.add_argument("--months", type=int, default=3, help="recent membership months to check")
    ap.add_argument("--funding-days", type=int, default=60)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    import ccxt

    from hf.data.ohlcv import _cache_path

    membership = pd.read_csv(args.membership)
    months = sorted(membership["month"].unique())[-args.months:]
    recent = membership[membership["month"].isin(months)]
    latest = recent[recent["month"] == months[-1]].set_index("symbol")["rank"]
    symbols = sorted(set(recent["symbol"]))

    ex = ccxt.bitget({"enableRateLimit": True, "timeout": 20000, "options": {"defaultType": "swap"}})
    markets = ex.load_markets()
    listed = {
        k for k, v in markets.items()
        if v.get("swap") and v.get("linear") and v.get("settle") == "USDT" and v.get("active", True)
    }
    mapping = {s: bitget_symbol(s, listed) for s in symbols}
    on_bitget = sorted({b for b in mapping.values() if b})
    tickers = ex.fetch_tickers(on_bitget) if on_bitget else {}

    since_ms = int((time.time() - args.funding_days * 86400) * 1000)
    rows = []
    for s in symbols:
        b = mapping[s]
        r = {"symbol": s, "bitget_symbol": b, "rank_latest": latest.get(s, np.nan), "on_bitget": b is not None,
             "months_in_top": int((recent["symbol"] == s).sum())}
        if b is not None:
            t = tickers.get(b, {})
            bid, ask = t.get("bid"), t.get("ask")
            r["bitget_quote_vol_24h"] = t.get("quoteVolume")
            r["spread_bps"] = (ask / bid - 1.0) * 10000.0 if bid and ask else np.nan
            try:
                book = ex.fetch_order_book(b, limit=100)
                for n in NOTIONALS:
                    both_sides = [book_cost_bps(book, n, "buy"), book_cost_bps(book, n, "sell")]
                    # NaN when the visible book (100 levels) cannot fill the notional on either side.
                    r[f"cost_bps_{n}"] = float(np.nanmean(both_sides)) if not np.isnan(both_sides).all() else np.nan
            except Exception as e:  # noqa: BLE001 - one bad symbol must not stop the survey
                r["error"] = f"book: {e}"
            try:
                bg = daily_funding(funding_history(ex, b, since_ms))
                bpath = _cache_path(args.cache_dir, "binanceusdm", s, "1h").parent.parent / "funding" / _cache_path(
                    args.cache_dir, "binanceusdm", s, "1h").name
                if bpath.exists():
                    bn = daily_funding(pd.read_csv(bpath))
                    both = pd.concat([bg.rename("bitget"), bn.rename("binance")], axis=1, sort=True).dropna()
                    both = both[both.index >= pd.Timestamp(since_ms, unit="ms", tz="UTC").ceil("D")]
                    both = both[both.index < pd.Timestamp.now(tz="UTC").floor("D")]
                    if len(both) >= 10:
                        r["funding_days"] = len(both)
                        r["bitget_funding_daily_mean"] = float(both["bitget"].mean())
                        r["binance_funding_daily_mean"] = float(both["binance"].mean())
                        r["funding_corr"] = float(both["bitget"].corr(both["binance"]))
            except Exception as e:  # noqa: BLE001
                r["error"] = f"{r.get('error', '')} funding: {e}".strip()
        rows.append(r)
        print(f"{s}: {b or 'MISSING'} {r.get('spread_bps', '')}", flush=True)

    df = pd.DataFrame(rows).sort_values(["rank_latest", "symbol"])
    cur = df[df["rank_latest"].notna()]
    on = df[df["on_bitget"]]
    summary = {
        "months": months,
        "symbols_checked": len(df),
        "listed_on_bitget": int(df["on_bitget"].sum()),
        "latest_month_top60_listed": int(cur["on_bitget"].sum()),
        "latest_month_top60_missing": cur.loc[~cur["on_bitget"], "symbol"].tolist(),
        "spread_bps_median": float(on["spread_bps"].median()),
        **{f"cost_bps_{n}_median": float(on[f"cost_bps_{n}"].median()) for n in NOTIONALS if f"cost_bps_{n}" in on},
        **{f"cost_bps_{n}_p90": float(on[f"cost_bps_{n}"].quantile(0.9)) for n in NOTIONALS if f"cost_bps_{n}" in on},
        "cost_bps_by_rank_tier": {
            tier: {f"{n}": float(g[f"cost_bps_{n}"].median()) for n in NOTIONALS if f"cost_bps_{n}" in g}
            for tier, g in on[on["rank_latest"].notna()].groupby(
                pd.cut(on.loc[on["rank_latest"].notna(), "rank_latest"], [0, 10, 30, 60], labels=["1-10", "11-30", "31-60"]),
                observed=True)
        },
        "bitget_to_binance_quote_volume_note": "Bitget 24h volume is a single-day snapshot",
        "bitget_quote_vol_24h_median": float(on["bitget_quote_vol_24h"].median()),
        "funding_daily_mean_bitget_avg": float(on["bitget_funding_daily_mean"].mean()) if "bitget_funding_daily_mean" in on else None,
        "funding_daily_mean_binance_avg": float(on["binance_funding_daily_mean"].mean()) if "binance_funding_daily_mean" in on else None,
        "funding_corr_median": float(on["funding_corr"].median()) if "funding_corr" in on else None,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir or f"artifacts/research/bitget_universe_check_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "symbols.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({"created_utc": stamp, **summary}, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
