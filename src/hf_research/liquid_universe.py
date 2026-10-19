"""Point-in-time liquid universe helpers shared by research and paper trading.

- crypto_perpetuals: linear USDT crypto perpetuals on Binance USDT-M (active or SETTLING).
- update_cache: incremental closed-candle cache refresh.
- monthly_membership: top-N by trailing median daily quote volume, using days before each month.
- bitget_symbol: the Bitget market for a Binance symbol (handles 1000x tickers and renames).
"""
from __future__ import annotations

import pandas as pd

from hf.data.ohlcv import _cache_path, fetch_closed_window

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


def monthly_membership(
    daily: dict[str, pd.DataFrame], *, top_n: int, window_days: int, min_history_days: int, months: list[str] | None = None
) -> pd.DataFrame:
    """Top-N per month by trailing median daily quote volume, using only days before the month.

    By default every month after the first data month up to the last data day; pass `months`
    ("YYYY-MM") to rank specific months, e.g. the current one right after it starts."""
    qv = {}
    for sym, df in daily.items():
        if df.empty:
            continue
        t = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        qv[sym] = pd.Series((df["close"] * df["volume"]).to_numpy(), index=t)
    panel = pd.DataFrame(qv).sort_index()
    rows = []
    if months is None:
        month_starts = pd.date_range(panel.index.min().normalize() + pd.offsets.MonthBegin(1), panel.index.max(), freq="MS")
    else:
        month_starts = [pd.Timestamp(f"{m}-01", tz="UTC") for m in months]
    for month in month_starts:
        hist = panel[panel.index < month]
        recent = hist.tail(window_days)
        eligible = hist.notna().sum() >= min_history_days
        # A symbol that stopped trading before the window has no recent rows and drops out.
        med = recent.median().where(eligible & recent.notna().sum().ge(window_days // 2)).dropna()
        for rank, (sym, v) in enumerate(med.sort_values(ascending=False).head(top_n).items(), start=1):
            rows.append({"month": month.strftime("%Y-%m"), "symbol": sym, "rank": rank, "median_quote_volume": float(v)})
    return pd.DataFrame(rows)


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
