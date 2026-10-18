"""Cross-sectional momentum on perpetual futures, daily rebalancing.

Timing (no lookahead):
- signal_close[d]: close of the 23:00 UTC 1h bar of day d (the daily close). Momentum and
  volatility at day d use signal closes <= d only.
- exec_close[d]:   close of the <exec_hour> UTC bar of day d+1 (default 00:00, i.e. one hour
  after the signal). Weights decided at day d are filled at exec_close[d] and earn
  exec_close[d+1]/exec_close[d].
- Costs: |w_new - w_held| * (taker fee + slippage) at every fill, where w_held are the
  previous weights after drifting with prices. Slippage is cfg.slippage_bps, or a per-day,
  per-asset panel when panels["slippage_bps"] is given.
- Funding: every funding event between two consecutive fills is charged to the position held
  in between (longs pay positive funding, shorts receive it).
- Universe: optionally restricted per day by a boolean "eligible" panel (e.g. the point-in-time
  top-N liquidity membership of that month). A held asset that loses its price (delisting) is
  closed at its last available price.
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


def backtest(panels: dict[str, pd.DataFrame], cfg: MomentumConfig, *, detail: bool = False):
    """Daily P&L. Row d = return earned from exec_close[d-1] to exec_close[d].

    With detail=True also returns per-asset daily contributions {"gross", "funding", "cost",
    "weight"} (each day x asset) so P&L can be attributed to legs and symbols."""
    exe = panels["exec_close"]
    funding = panels["funding"]
    slip = panels.get("slippage_bps")
    if slip is not None:
        unit_cost = (cfg.fee + slip.reindex(index=exe.index, columns=exe.columns).fillna(cfg.slippage_bps) / 10000.0)
    else:
        unit_cost = pd.DataFrame(cfg.cost_per_unit_turnover, index=exe.index, columns=exe.columns)
    targets = target_weights(panels["signal_close"], cfg, panels.get("eligible"))
    asset_ret = (exe / exe.shift(1) - 1.0).fillna(0.0)
    has_price = exe.notna()

    days = list(exe.index)
    held = pd.Series(0.0, index=exe.columns)
    rows = []
    contrib = {k: [] for k in ("gross", "funding", "cost", "weight")}
    for n, d in enumerate(days):
        # 1) P&L of the position held since yesterday's fill.
        g_asset = held * asset_ret.loc[d]
        f_asset = -(held * funding.loc[d])
        gross = float(g_asset.sum())
        fund = float(f_asset.sum())
        held_before = held
        drifted = held * (1.0 + asset_ret.loc[d])
        total = float(1.0 + gross)
        held = drifted / total if total > 0 else drifted
        # 2) Positions in assets that lost their price (delisted) are closed at the last price.
        traded = pd.Series(0.0, index=exe.columns)
        dead = (held != 0.0) & ~has_price.loc[d]
        if dead.any():
            traded = traded + held.where(dead, 0.0).abs()
            held = held.where(~dead, 0.0)
        # 3) Rebalance at today's exec close on the schedule (decided from today's signal close).
        if n % cfg.rebalance_days == 0:
            new = targets.loc[d].fillna(0.0)
            new = new.where(has_price.loc[d], 0.0)
            if cfg.band > 0.0:
                # Only resizes of an open position on the same side are skipped when small;
                # opening, closing and flipping always trade.
                resize = (held != 0.0) & (new != 0.0) & (np.sign(new) == np.sign(held))
                small = resize & ((new - held).abs() < cfg.band)
                new = new.where(~small, held)
            traded = traded + (new - held).abs()
            held = new
        turnover = float(traded.sum())
        c_asset = traded * unit_cost.loc[d]
        cost = float(c_asset.sum())
        if detail:
            contrib["gross"].append(g_asset)
            contrib["funding"].append(f_asset)
            contrib["cost"].append(c_asset)
            contrib["weight"].append(held_before)
        rows.append(
            {"day": d, "gross": gross, "funding": fund, "cost": cost, "net": gross + fund - cost,
             "turnover": turnover, "gross_exposure": float(held.abs().sum()), "n_positions": int((held != 0).sum())}
        )
    daily = pd.DataFrame(rows).set_index("day")
    if not detail:
        return daily
    return daily, {k: pd.DataFrame(v, index=daily.index) for k, v in contrib.items()}


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
