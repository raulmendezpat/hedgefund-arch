"""Strategy-agnostic daily portfolio backtest: target weights in, daily P&L out.

Any strategy (momentum, funding carry, ...) produces a panel of target weights per decision
day; this engine applies the shared execution model so every strategy is measured the same way.

Timing (no lookahead), panels indexed by UTC decision day, one column per asset:
- exec_close[d]: the fill price for weights decided at day d (after the day-d signal close).
  Row d of the result is the return earned from exec_close[d-1] to exec_close[d].
- funding[d]: funding events while the day-(d-1) fill was held, i.e. up to the day-d fill
  (longs pay positive funding, shorts receive it). Assets without funding (spot) have zeros.
- Costs: |w_new - w_held| * (fee + slippage) at every fill, where w_held are the previous
  weights after drifting with prices. `fee` is a float or a per-asset Series (spot and perps
  pay different fees); slippage is `slippage_bps`, or a per-day, per-asset panel in
  panels["slippage_bps"].
- A held asset that loses its price (delisting) is closed at its last available price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_weights(
    panels: dict[str, pd.DataFrame],
    targets: pd.DataFrame,
    *,
    fee: float | pd.Series = 0.0006,
    slippage_bps: float = 3.0,
    band: float = 0.0,
    rebalance_days: int = 1,
    rebalance_on=None,
    detail: bool = False,
):
    """Daily P&L of trading toward `targets` (rows = decision day, NaN = 0).

    Rebalances every `rebalance_days` rows from the first row, or exactly on the decision days in
    `rebalance_on` when given. With band > 0, resizing an open position on the same side is
    skipped when the change is below `band`; opening, closing and flipping always trade.

    With detail=True also returns per-asset daily contributions {"gross", "funding", "cost",
    "weight"} (each day x asset) so P&L can be attributed to legs and symbols."""
    exe = panels["exec_close"]
    funding = panels.get("funding")
    funding = (funding.reindex(index=exe.index, columns=exe.columns).fillna(0.0) if funding is not None
               else pd.DataFrame(0.0, index=exe.index, columns=exe.columns))
    fee_by_asset = (fee.reindex(exe.columns).astype(float) if isinstance(fee, pd.Series)
                    else pd.Series(float(fee), index=exe.columns))
    slip = panels.get("slippage_bps")
    if slip is not None:
        unit_cost = fee_by_asset + slip.reindex(index=exe.index, columns=exe.columns).fillna(slippage_bps) / 10000.0
    else:
        unit_cost = pd.DataFrame(np.tile((fee_by_asset + slippage_bps / 10000.0).to_numpy(), (len(exe), 1)),
                                 index=exe.index, columns=exe.columns)
    targets = targets.reindex(index=exe.index, columns=exe.columns)
    asset_ret = (exe / exe.shift(1) - 1.0).fillna(0.0)
    has_price = exe.notna()

    days = list(exe.index)
    if rebalance_on is not None:
        rebalance_on = set(pd.DatetimeIndex(rebalance_on))
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
        due = (d in rebalance_on) if rebalance_on is not None else (n % rebalance_days == 0)
        if due:
            new = targets.loc[d].fillna(0.0)
            new = new.where(has_price.loc[d], 0.0)
            if band > 0.0:
                resize = (held != 0.0) & (new != 0.0) & (np.sign(new) == np.sign(held))
                small = resize & ((new - held).abs() < band)
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
