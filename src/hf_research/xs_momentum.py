"""Cross-sectional momentum on perpetual futures, daily rebalancing.

Timing (no lookahead):
- signal_close[d]: close of the 23:00 UTC 1h bar of day d (the daily close). Momentum and
  volatility at day d use signal closes <= d only.
- exec_close[d]:   close of the <exec_hour> UTC bar of day d+1 (default 00:00, i.e. one hour
  after the signal). Weights decided at day d are filled at exec_close[d] and earn
  exec_close[d+1]/exec_close[d].
- Costs and funding: the shared execution model in hf_research.engine (fee + slippage on
  traded weight; funding events between two fills charged to the position held in between).
- Universe: optionally restricted per day by a boolean "eligible" panel (e.g. the point-in-time
  top-N liquidity membership of that month). A held asset that loses its price (delisting) is
  closed at its last available price.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from hf.data.ohlcv import _cache_path, drop_unclosed_candles
from hf_research.engine import backtest_weights, summarize  # noqa: F401 - summarize re-exported


@dataclass(frozen=True)
class MomentumConfig:
    lookback_days: int = 14
    skip_days: int = 0
    rebalance_days: int = 7
    weighting: str = "equal"  # equal | inv_vol
    mode: str = "long_short"  # long_short | long_only
    select_frac: float = 0.3
    min_assets: int = 8
    vol_window_days: int = 30
    vol_adjust_signal: bool = False
    # Skip trades whose weight change is below this (absolute weight units); 0 = always trade.
    band: float = 0.0
    fee: float = 0.0006
    slippage_bps: float = 3.0

    @property
    def cost_per_unit_turnover(self) -> float:
        return self.fee + self.slippage_bps / 10000.0

    def key(self) -> str:
        return (
            f"{self.mode}|L{self.lookback_days}|S{self.skip_days}|R{self.rebalance_days}|"
            f"{self.weighting}|f{self.select_frac}|va{int(self.vol_adjust_signal)}|b{self.band}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def load_daily_panels(
    symbols: list[str], *, cache_dir: str = "data/cache", exchange: str = "binanceusdm", exec_hour: int = 0
) -> dict[str, pd.DataFrame]:
    """signal_close / exec_close / funding panels indexed by UTC day, one column per symbol."""
    # Fill for decision day d happens at the close of the exec bar: day d+1, (exec_hour + 1):00.
    fill_offset = pd.Timedelta(hours=exec_hour + 1)
    sig, exe, fund = {}, {}, {}
    for s in symbols:
        df = pd.read_csv(_cache_path(cache_dir, exchange, s, "1h"))
        df = drop_unclosed_candles(df, "1h").drop_duplicates("timestamp", keep="last")
        t = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        c = pd.Series(df["close"].astype(float).to_numpy(), index=t).sort_index()
        day = c.index.floor("D")
        sig[s] = c[c.index.hour == 23].set_axis(day[c.index.hour == 23])
        # The exec_hour bar of day d+1 belongs to decision day d.
        e = c[c.index.hour == exec_hour]
        exe[s] = e.set_axis(e.index.floor("D") - pd.Timedelta(days=1))

        fpath = _cache_path(cache_dir, exchange, s, "1h").parent.parent / "funding" / _cache_path(cache_dir, exchange, s, "1h").name
        if fpath.exists():
            f = pd.read_csv(fpath)
            # Row d holds the position from fill(d-1) to fill(d], i.e. funding events in
            # (d + fill_offset, d + 1 day + fill_offset]. Timestamps carry a few ms of jitter.
            ft = pd.to_datetime(f["timestamp"], unit="ms", utc=True).dt.round("min")
            bucket = (ft - fill_offset - pd.Timedelta(milliseconds=1)).dt.floor("D")
            fund[s] = f.groupby(bucket)["funding_rate"].sum()
    signal = pd.DataFrame(sig).sort_index()
    execp = pd.DataFrame(exe).reindex(signal.index)
    funding = pd.DataFrame(fund).reindex(signal.index).fillna(0.0)
    return {"signal_close": signal, "exec_close": execp, "funding": funding}


def eligible_from_membership(membership: pd.DataFrame, index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """Daily boolean panel: True when the symbol is in that calendar month's membership list.
    Membership for a month must be computed from data before the month starts."""
    months = pd.Series(index.strftime("%Y-%m"), index=index)
    by_month = membership.groupby("month")["symbol"].apply(set).to_dict()
    return pd.DataFrame(
        {c: [c in by_month.get(m, set()) for m in months] for c in columns}, index=index
    )


def target_weights(signal_close: pd.DataFrame, cfg: MomentumConfig, eligible: pd.DataFrame | None = None) -> pd.DataFrame:
    """Target weights decided at each day's close (rows = decision day). NaN-safe, no lookahead."""
    px = signal_close
    mom = px.shift(cfg.skip_days) / px.shift(cfg.skip_days + cfg.lookback_days) - 1.0
    rets = px.pct_change(fill_method=None)
    vol = rets.rolling(cfg.vol_window_days, min_periods=cfg.vol_window_days // 2).std()
    score = mom / vol if cfg.vol_adjust_signal else mom
    score = score.where(vol.notna())
    if eligible is not None:
        score = score.where(eligible.reindex(index=score.index, columns=score.columns).fillna(False).astype(bool))

    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for d in px.index:
        s = score.loc[d].dropna()
        if len(s) < cfg.min_assets:
            continue
        k = max(2, int(round(len(s) * cfg.select_frac)))
        ranked = s.sort_values()
        longs, shorts = ranked.index[-k:], ranked.index[:k]
        if cfg.weighting == "inv_vol":
            iv = 1.0 / vol.loc[d]
            wl = iv[longs] / iv[longs].sum()
            ws = iv[shorts] / iv[shorts].sum()
        else:
            wl = pd.Series(1.0 / k, index=longs)
            ws = pd.Series(1.0 / k, index=shorts)
        if cfg.mode == "long_short":
            w.loc[d, longs] = 0.5 * wl
            w.loc[d, shorts] = -0.5 * ws
        else:
            w.loc[d, longs] = wl
    return w


def backtest(
    panels: dict[str, pd.DataFrame], cfg: MomentumConfig, *, detail: bool = False, rebalance_on=None
):
    """Daily P&L of the momentum targets under the shared execution model (hf_research.engine).

    Rebalances every cfg.rebalance_days rows from the first row, or exactly on the decision
    days in `rebalance_on` when given (e.g. the days a paper/live book actually traded).
    With detail=True also returns per-asset daily contributions (see engine.backtest_weights)."""
    targets = target_weights(panels["signal_close"], cfg, panels.get("eligible"))
    return backtest_weights(panels, targets, fee=cfg.fee, slippage_bps=cfg.slippage_bps, band=cfg.band,
                            rebalance_days=cfg.rebalance_days, rebalance_on=rebalance_on, detail=detail)
