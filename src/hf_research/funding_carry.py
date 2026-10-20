"""Delta-neutral funding carry: short the perpetual, long the same notional in spot, while the
perpetual's recent funding stays high (pre-registered in docs/plans/funding_carry_hypothesis.md).

Panels follow hf_research.engine: perpetual columns are the perp symbols ("BTC/USDT:USDT"); the
spot leg of each pair is a separate column named spot_col(perp) with no funding. Timing is the
momentum one: decide on the day-d 23:00 UTC close, fill at the next exec bar.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from hf.data.ohlcv import _cache_path, drop_unclosed_candles

SPOT_PREFIX = "SPOT|"


@dataclass(frozen=True)
class CarryConfig:
    lookback_days: int = 7
    entry_ann: float = 0.10  # enter when trailing funding annualizes to at least this
    exit_ann: float = 0.03  # exit when it falls below this (hysteresis)
    max_positions: int = 10
    pair_weight: float = 0.07  # spot long = perp short = this fraction of equity per pair
    perp_fee: float = 0.0006
    spot_fee: float = 0.0010
    band: float = 0.02

    def key(self) -> str:
        return f"L{self.lookback_days}|in{self.entry_ann}|out{self.exit_ann}|N{self.max_positions}"

    def to_dict(self) -> dict:
        return asdict(self)


def spot_col(perp: str) -> str:
    return SPOT_PREFIX + perp


def signal_funding(perps: list[str], index: pd.DatetimeIndex, cache_dir: str = "data/cache",
                   exchange: str = "binanceusdm") -> pd.DataFrame:
    """Funding realized on each UTC day (events stamped that day), known at the day's 23:00 close."""
    out = {}
    for s in perps:
        path = _cache_path(cache_dir, exchange, s, "1h")
        fpath = path.parent.parent / "funding" / path.name
        if not fpath.exists():
            continue
        f = pd.read_csv(fpath)
        ts = pd.to_datetime(f["timestamp"], unit="ms", utc=True).dt.round("min")
        out[s] = f.groupby(ts.dt.floor("D"))["funding_rate"].sum()
    return pd.DataFrame(out).reindex(index=index).fillna(0.0)


def trailing_funding_ann(daily_funding: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Sum of the last `lookback_days` days of funding, annualized; NaN until the window is full."""
    return daily_funding.rolling(lookback_days, min_periods=lookback_days).sum() * (365.0 / lookback_days)


def carry_targets(fund_ann: pd.DataFrame, tradable: pd.DataFrame, cfg: CarryConfig) -> pd.DataFrame:
    """Target weights per decision day for perp and spot columns.

    `tradable[d, perp]` is True when the perp is in the universe that day and both legs have a
    price. Open pairs exit when funding drops below exit_ann or they stop being tradable; free
    slots go to the highest-funding candidates at or above entry_ann."""
    perps = list(fund_ann.columns)
    cols = perps + [spot_col(p) for p in perps]
    w = pd.DataFrame(0.0, index=fund_ann.index, columns=cols)
    trad = tradable.reindex(index=fund_ann.index, columns=perps).fillna(False).astype(bool)
    held: set[str] = set()
    for d in fund_ann.index:
        f, ok = fund_ann.loc[d], trad.loc[d]
        held = {p for p in held if ok[p] and pd.notna(f[p]) and f[p] >= cfg.exit_ann}
        free = cfg.max_positions - len(held)
        if free > 0:
            cand = f[ok & f.ge(cfg.entry_ann)].drop(labels=list(held), errors="ignore").sort_values(ascending=False)
            held |= set(cand.index[:free])
        for p in held:
            w.at[d, p] = -cfg.pair_weight
            w.at[d, spot_col(p)] = cfg.pair_weight
    return w


def load_spot_exec(spot_map: dict[str, str], index: pd.DatetimeIndex, cache_dir: str = "data/cache",
                   exchange: str = "binance", exec_hour: int = 0) -> pd.DataFrame:
    """Spot fill prices aligned with the perp exec_close (the exec_hour bar of day d+1 -> row d)."""
    out = {}
    for perp, spot in spot_map.items():
        path = _cache_path(cache_dir, exchange, spot, "1h")
        if not path.exists():
            continue
        df = drop_unclosed_candles(pd.read_csv(path), "1h").drop_duplicates("timestamp", keep="last")
        t = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        c = pd.Series(df["close"].astype(float).to_numpy(), index=t).sort_index()
        e = c[c.index.hour == exec_hour]
        out[spot_col(perp)] = e.set_axis(e.index.floor("D") - pd.Timedelta(days=1))
    return pd.DataFrame(out).reindex(index=index)


def fee_series(columns, cfg: CarryConfig) -> pd.Series:
    return pd.Series([cfg.spot_fee if c.startswith(SPOT_PREFIX) else cfg.perp_fee for c in columns], index=columns)
