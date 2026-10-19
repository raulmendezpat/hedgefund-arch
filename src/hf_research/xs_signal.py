"""Universe and target weights of the cross-sectional momentum hypothesis, shared by the paper
and live runners so both trade exactly the same signal as the backtest.

- Universe: monthly top N Binance USDT-M perpetuals by trailing median quote volume (days
  strictly before the month), restricted to symbols listed on Bitget.
- Signal: hf_research.xs_momentum.target_weights on Binance 1h closes at 23:00 UTC.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from hf_research.liquid_universe import bitget_symbol, crypto_perpetuals, monthly_membership, update_cache
from hf_research.paper_xs import PRIMARY_STRATEGY
from hf_research.xs_momentum import MomentumConfig, load_daily_panels, target_weights

BINANCE = "binanceusdm"


def binance_exchange():
    import ccxt

    return ccxt.binanceusdm({"enableRateLimit": True, "timeout": 20000})


def ensure_membership(path: Path, month: str, cfg, cache_dir: str, log) -> pd.DataFrame:
    """Membership of `month` (cached in `path`), from Binance daily candles strictly before the month.

    `cfg` needs top_n, window_days and min_history_days."""
    have = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["month", "symbol", "rank"])
    if (have["month"] == month).any():
        return have[have["month"] == month]
    ex = binance_exchange()
    month_start = pd.Timestamp(f"{month}-01", tz="UTC")
    since_ms = int((month_start - pd.Timedelta(days=cfg.min_history_days + cfg.window_days + 60)).value // 1_000_000)
    daily = {}
    markets = crypto_perpetuals(ex)
    for k, m in enumerate(markets, start=1):
        try:
            daily[m["symbol"]] = update_cache(ex, m["symbol"], "1d", since_ms, cache_dir, BINANCE)
        except Exception as exc:  # noqa: BLE001 - some delisted contracts reject history requests
            log(f"skip 1d {m['symbol']}: {type(exc).__name__}")
        if k % 100 == 0:
            log(f"daily candles {k}/{len(markets)}")
    m_new = monthly_membership(daily, top_n=cfg.top_n, window_days=cfg.window_days,
                               min_history_days=cfg.min_history_days, months=[month])
    if m_new.empty:
        raise RuntimeError(f"could not build membership for {month}")
    pd.concat([have, m_new], ignore_index=True).to_csv(path, index=False)
    log(f"membership {month}: {len(m_new)} symbols")
    return m_new


def compute_targets(decision_day: pd.Timestamp, members: list[str], listed: set[str], cache_dir: str, log,
                    *, min_coverage: float = 0.5) -> dict:
    """Target weights (Binance symbols) for the decision day plus the data used to build them.

    Raises RuntimeError when fewer than `min_coverage` of the tradable members have a signal
    close for the decision day (data not refreshed yet, or an exchange outage)."""
    ex = binance_exchange()
    since_ms = int((decision_day - pd.Timedelta(days=120)).value // 1_000_000)
    for s in members:
        try:
            update_cache(ex, s, "1h", since_ms, cache_dir, BINANCE)
        except Exception as exc:  # noqa: BLE001
            log(f"skip 1h {s}: {type(exc).__name__}")
    tradable = [s for s in members if bitget_symbol(s, listed)]
    panels = load_daily_panels(tradable, cache_dir=cache_dir, exchange=BINANCE)
    sig = panels["signal_close"]
    coverage = float(sig.loc[decision_day].notna().mean()) if decision_day in sig.index and len(tradable) else 0.0
    if coverage < min_coverage:
        raise RuntimeError(f"signal closes for {decision_day.date()} cover {coverage:.0%} of members "
                           f"(need {min_coverage:.0%})")
    eligible = pd.DataFrame(True, index=sig.index, columns=sig.columns)
    w = target_weights(sig, MomentumConfig(**PRIMARY_STRATEGY), eligible).loc[decision_day]
    w = w[w != 0.0]
    return {"weights": {s: float(v) for s, v in w.items()},
            "eligible": tradable,
            "not_on_bitget": [s for s in members if s not in tradable],
            "coverage": coverage,
            "signal_close": {s: float(sig.loc[decision_day, s]) for s in w.index}}
