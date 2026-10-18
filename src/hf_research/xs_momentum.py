"""Cross-sectional momentum on perpetual futures, daily rebalancing.

Timing (no lookahead):
- signal_close[d]: close of the 23:00 UTC 1h bar of day d (the daily close). Momentum and
  volatility at day d use signal closes <= d only.
- exec_close[d]:   close of the 00:00 UTC bar of day d+1, i.e. one hour after the signal.
  Weights decided at day d are filled at exec_close[d] and earn exec_close[d+1]/exec_close[d].
- Costs: |w_new - w_held| * (taker fee + slippage) at every fill, where w_held are the
  previous weights after drifting with prices.
- Funding: the position held over day d+1 pays sum(funding_rate) of that day times its weight
  (longs pay positive funding, shorts receive it).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from hf.data.ohlcv import _cache_path, drop_unclosed_candles


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
    fee: float = 0.0006
    slippage_bps: float = 3.0

    @property
    def cost_per_unit_turnover(self) -> float:
        return self.fee + self.slippage_bps / 10000.0

    def key(self) -> str:
        return (
            f"{self.mode}|L{self.lookback_days}|S{self.skip_days}|R{self.rebalance_days}|"
            f"{self.weighting}|f{self.select_frac}|va{int(self.vol_adjust_signal)}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def load_daily_panels(
    symbols: list[str], *, cache_dir: str = "data/cache", exchange: str = "binanceusdm"
) -> dict[str, pd.DataFrame]:
    """signal_close / exec_close / funding panels indexed by UTC day, one column per symbol."""
    sig, exe, fund = {}, {}, {}
    for s in symbols:
        df = pd.read_csv(_cache_path(cache_dir, exchange, s, "1h"))
        df = drop_unclosed_candles(df, "1h").drop_duplicates("timestamp", keep="last")
        t = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        c = pd.Series(df["close"].astype(float).to_numpy(), index=t).sort_index()
        day = c.index.floor("D")
        sig[s] = c[c.index.hour == 23].set_axis(day[c.index.hour == 23])
        # The 00:00 bar of day d+1 belongs to decision day d.
        e = c[c.index.hour == 0]
        exe[s] = e.set_axis(e.index.floor("D") - pd.Timedelta(days=1))

        fpath = _cache_path(cache_dir, exchange, s, "1h").parent.parent / "funding" / _cache_path(cache_dir, exchange, s, "1h").name
        if fpath.exists():
            f = pd.read_csv(fpath)
            ft = pd.to_datetime(f["timestamp"], unit="ms", utc=True).dt.floor("D")
            fund[s] = f.groupby(ft)["funding_rate"].sum()
    signal = pd.DataFrame(sig).sort_index()
    execp = pd.DataFrame(exe).reindex(signal.index)
    funding = pd.DataFrame(fund).reindex(signal.index).fillna(0.0)
    return {"signal_close": signal, "exec_close": execp, "funding": funding}


def target_weights(signal_close: pd.DataFrame, cfg: MomentumConfig) -> pd.DataFrame:
    """Target weights decided at each day's close (rows = decision day). NaN-safe, no lookahead."""
    px = signal_close
    mom = px.shift(cfg.skip_days) / px.shift(cfg.skip_days + cfg.lookback_days) - 1.0
    rets = px.pct_change(fill_method=None)
    vol = rets.rolling(cfg.vol_window_days, min_periods=cfg.vol_window_days // 2).std()
    score = mom / vol if cfg.vol_adjust_signal else mom
    score = score.where(vol.notna())

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


def backtest(panels: dict[str, pd.DataFrame], cfg: MomentumConfig) -> pd.DataFrame:
    """Daily P&L. Row d = return earned from exec_close[d-1] to exec_close[d]."""
    exe = panels["exec_close"]
    funding = panels["funding"]
    targets = target_weights(panels["signal_close"], cfg)
    asset_ret = (exe / exe.shift(1) - 1.0).fillna(0.0)

    days = list(exe.index)
    held = pd.Series(0.0, index=exe.columns)
    rows = []
    for n, d in enumerate(days):
        # 1) P&L of the position held since yesterday's fill.
        gross = float((held * asset_ret.loc[d]).sum())
        fund = float(-(held * funding.loc[d]).sum())
        drifted = held * (1.0 + asset_ret.loc[d])
        total = float(1.0 + gross)
        held = drifted / total if total > 0 else drifted
        # 2) Rebalance at today's exec close on the schedule (decided from today's signal close).
        cost = 0.0
        turnover = 0.0
        if n % cfg.rebalance_days == 0:
            new = targets.loc[d].fillna(0.0)
            # Assets without an exec price today cannot be traded.
            new = new.where(exe.loc[d].notna(), 0.0)
            turnover = float((new - held).abs().sum())
            cost = turnover * cfg.cost_per_unit_turnover
            held = new
        rows.append(
            {"day": d, "gross": gross, "funding": fund, "cost": cost, "net": gross + fund - cost,
             "turnover": turnover, "gross_exposure": float(held.abs().sum()), "n_positions": int((held != 0).sum())}
        )
    return pd.DataFrame(rows).set_index("day")


def summarize(daily: pd.DataFrame, start: str | None = None, end: str | None = None) -> dict:
    d = daily
    if start:
        d = d[d.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        d = d[d.index < pd.Timestamp(end, tz="UTC")]
    d = d[d["gross_exposure"].shift(1, fill_value=0.0).gt(0) | d["turnover"].gt(0)]
    if len(d) < 30:
        return {"days": int(len(d))}
    net = d["net"]
    eq = (1.0 + net).cumprod()
    years = len(d) / 365.0
    monthly = (1.0 + net).groupby(d.index.strftime("%Y-%m")).prod() - 1.0
    return {
        "days": int(len(d)),
        "cagr": float(eq.iloc[-1] ** (1.0 / years) - 1.0),
        "vol": float(net.std() * np.sqrt(365.0)),
        "sharpe": float(net.mean() / net.std() * np.sqrt(365.0)) if net.std() > 0 else float("nan"),
        "max_dd": float((eq / eq.cummax() - 1.0).min()),
        "months_positive": float((monthly > 0).mean()),
        "gross_ann": float(d["gross"].mean() * 365.0),
        "funding_ann": float(d["funding"].mean() * 365.0),
        "cost_ann": float(d["cost"].mean() * 365.0),
        "turnover_ann": float(d["turnover"].sum() / years),
    }
