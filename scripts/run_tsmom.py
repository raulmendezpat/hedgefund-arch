"""Time-series momentum on BTC and ETH (pre-registered in docs/plans/tsmom_btc_eth_hypothesis.md).

  PYTHONPATH=src python scripts/run_tsmom.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from hf_research.engine import backtest_weights, summarize
from hf_research.paper_xs import PRIMARY_STRATEGY
from hf_research.tsmom import TsmomConfig, tsmom_targets
from hf_research.vol_target import account_path, intraday_paths, load_hourly_close
from hf_research.xs_momentum import MomentumConfig, backtest, eligible_from_membership, load_daily_panels

ASSETS = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
CRITERIA = {"oos_sharpe_min": 0.7, "oos_max_dd_min": -0.30, "is_sharpe_min": 0.5, "per_asset_oos_sharpe_gt": 0.0}
COMBO = {"sharpe_gain_min": 0.15, "max_gross": 2.0, "scale_start": "2021-03-01"}
KILL_DD = -0.35
TIERS = {10: 3.0, 30: 6.0, 60: 12.0}
GRID = {"lookbacks": [(20,), (60,), (120,), (20, 60, 120)], "band": [0.02, 0.05, 0.10], "mode": ["long_short", "long_only"]}


def slippage_panel(membership: pd.DataFrame, index: pd.DatetimeIndex, columns, tiers: dict[int, float]) -> pd.DataFrame:
    worst = max(tiers.values())

    def bps(rank: float) -> float:
        return next((tiers[c] for c in sorted(tiers) if rank <= c), worst)

    by_key = {(m, s): bps(r) for m, s, r in membership[["month", "symbol", "rank"]].itertuples(index=False)}
    months = index.strftime("%Y-%m")
    return pd.DataFrame({c: [by_key.get((m, c), worst) for m in months] for c in columns}, index=index)


def bootstrap_p_sharpe_le_0(net: pd.Series, block: int = 20, reps: int = 2000, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    x, n = net.to_numpy(), len(net)
    out = []
    for _ in range(reps):
        starts = rng.integers(0, n, size=n // block + 1)
        s = x[(starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n]
        out.append(s.mean() / s.std())
    return float((np.array(out) <= 0).mean())


def max_dd(net: pd.Series) -> float:
    eq = (1.0 + net).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", default="artifacts/research/universe_top60_v1/membership.csv")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--is-start", default="2020-04-01")
    ap.add_argument("--oos-start", default="2024-01-01")
    ap.add_argument("--out-dir", default="artifacts/research/tsmom_btc_eth_v1")
    args = ap.parse_args()
    t0 = time.perf_counter()
    ts = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
    spans = {"is": (args.is_start, args.oos_start), "oos": (args.oos_start, None)}

    panels = load_daily_panels(ASSETS, cache_dir=args.cache_dir)

    def run(cfg: TsmomConfig, assets=ASSETS, detail: bool = False, targets: pd.DataFrame | None = None):
        p = {k: v[assets] for k, v in panels.items()}
        t = tsmom_targets(p["signal_close"], cfg) if targets is None else targets[assets]
        return backtest_weights(p, t, fee=cfg.fee, slippage_bps=cfg.slippage_bps, band=cfg.band, detail=detail)

    def per(daily: pd.DataFrame) -> dict:
        return {k: summarize(daily, s, e) for k, (s, e) in spans.items()} | {
            "sharpe_by_year": {str(y): summarize(daily, f"{y}-01-01", f"{y + 1}-01-01").get("sharpe")
                               for y in range(2020, 2027)}}

    cfg = TsmomConfig()
    daily, parts = run(cfg, detail=True)
    oos = daily.index >= ts(args.oos_start)
    rep: dict = {"config": cfg.to_dict(), "criteria": CRITERIA, "primary": per(daily),
                 "primary_cost_x2": per(run(replace(cfg, fee=2 * cfg.fee, slippage_bps=2 * cfg.slippage_bps))),
                 "per_asset": {a: per(run(cfg, [a])) for a in ASSETS}}
    rep["primary"]["exposure_oos"] = {"avg_gross": float(daily.loc[oos, "gross_exposure"].mean()),
                                      "p95_gross": float(daily.loc[oos, "gross_exposure"].quantile(0.95)),
                                      "share_net_long": float((parts["weight"].loc[oos].sum(axis=1) > 0).mean())}
    # P&L by side: price + funding of the weights held, split by their sign (costs not split).
    pnl = parts["gross"] + parts["funding"]
    long_side, short_side = pnl.where(parts["weight"] > 0, 0.0).sum(axis=1), pnl.where(parts["weight"] < 0, 0.0).sum(axis=1)
    rep["by_side_ann"] = {p: {"long": float(long_side[m].mean() * 365), "short": float(short_side[m].mean() * 365),
                              "cost": float(daily.loc[m, "cost"].mean() * 365)}
                          for p, m in (("is", (daily.index >= ts(args.is_start)) & ~oos), ("oos", oos))}

    # Buy and hold 50/50, compared at the same realized volatility as the strategy per period.
    hold = run(cfg, targets=pd.DataFrame(0.5, index=daily.index, columns=ASSETS))
    rep["buy_and_hold"] = {}
    for p, (s, e) in spans.items():
        m = (daily.index >= ts(s)) & ((daily.index < ts(e)) if e else True)
        a, h = daily.loc[m, "net"], hold.loc[m, "net"]
        rep["buy_and_hold"][p] = {"sharpe": summarize(hold, s, e).get("sharpe"), "corr_with_strategy": float(a.corr(h)),
                                  "max_dd_at_strategy_vol": max_dd(h * a.std() / h.std()),
                                  "strategy_max_dd": max_dd(a)}

    o, i = rep["primary"]["oos"], rep["primary"]["is"]
    rep["verdict"] = {"oos_sharpe": o["sharpe"] >= CRITERIA["oos_sharpe_min"],
                      "oos_max_dd": o["max_dd"] >= CRITERIA["oos_max_dd_min"],
                      "is_sharpe": i["sharpe"] >= CRITERIA["is_sharpe_min"],
                      "per_asset_oos_positive": all(rep["per_asset"][a]["oos"]["sharpe"] > CRITERIA["per_asset_oos_sharpe_gt"]
                                                    for a in ASSETS)}
    rep["verdict"]["pass"] = all(rep["verdict"].values())
    rep["bootstrap_oos_p_sharpe_le_0"] = bootstrap_p_sharpe_le_0(daily.loc[oos, "net"])

    hourly = load_hourly_close(ASSETS, args.cache_dir)
    ret, notional = intraday_paths(parts["weight"], hourly)
    rep["account"] = {}
    for p, (s, e) in spans.items():
        m = (daily.index >= ts(s)) & ((daily.index < ts(e)) if e else True)
        rep["account"][p] = account_path(daily.loc[m, "net"], ret[m], notional[m], kill_dd=KILL_DD)
    print(f"primary done ({time.perf_counter() - t0:.0f}s)", flush=True)

    # Equal-risk combination with the primary cross-sectional momentum.
    membership = pd.read_csv(args.membership)
    symbols = sorted(set(membership["symbol"]))
    xs_p = load_daily_panels(symbols, cache_dir=args.cache_dir)
    xidx = xs_p["signal_close"].index
    xs_p["eligible"] = eligible_from_membership(membership, xidx, symbols)
    xs_p["slippage_bps"] = slippage_panel(membership, xidx, symbols, TIERS)
    mom = backtest(xs_p, MomentumConfig(**PRIMARY_STRATEGY))
    t_net = daily["net"].reindex(mom.index).fillna(0.0)
    sm = (mom.index >= ts(COMBO["scale_start"])) & (mom.index < ts(args.oos_start))
    scale = float(mom.loc[sm, "net"].std() / t_net[sm].std())
    combo = mom.copy()
    for c in ("net", "gross", "funding", "cost", "turnover"):
        combo[c] = 0.5 * mom[c] + 0.5 * scale * daily[c].reindex(mom.index).fillna(0.0)
    combo["gross_exposure"] = mom["gross_exposure"] + daily["gross_exposure"].reindex(mom.index).fillna(0.0)
    moos = mom.index >= ts(args.oos_start)
    tsmom_gross_scaled = scale * daily["gross_exposure"].reindex(mom.index).fillna(0.0)[moos]
    c_spans = {"is": (COMBO["scale_start"], args.oos_start), "oos": (args.oos_start, None)}
    rep["combination"] = {"tsmom_scale": scale, "tsmom_gross_at_scale_oos_avg": float(tsmom_gross_scaled.mean()),
                          "tsmom_gross_at_scale_oos_p95": float(tsmom_gross_scaled.quantile(0.95)),
                          "oos_daily_corr": float(mom.loc[moos, "net"].corr(t_net[moos])),
                          "momentum": {k: summarize(mom, s, e) for k, (s, e) in c_spans.items()},
                          "combined": {k: summarize(combo, s, e) for k, (s, e) in c_spans.items()}}
    cm, mm = rep["combination"]["combined"]["oos"], rep["combination"]["momentum"]["oos"]
    rep["combination"]["verdict"] = {"sharpe_gain": cm["sharpe"] - mm["sharpe"] >= COMBO["sharpe_gain_min"],
                                     "max_dd_better": cm["max_dd"] > mm["max_dd"],
                                     "implementable": rep["combination"]["tsmom_gross_at_scale_oos_p95"] <= COMBO["max_gross"]}
    rep["combination"]["verdict"]["pass"] = all(rep["combination"]["verdict"].values())
    print(f"combination done ({time.perf_counter() - t0:.0f}s)", flush=True)

    grid = []
    for lbs, band, mode in itertools.product(GRID["lookbacks"], GRID["band"], GRID["mode"]):
        s = per(run(replace(cfg, lookbacks=lbs, band=band, mode=mode)))
        grid.append({"lookbacks": "/".join(map(str, lbs)), "band": band, "mode": mode, "is_sharpe": s["is"].get("sharpe"),
                     "oos_sharpe": s["oos"].get("sharpe"), "oos_cagr": s["oos"].get("cagr"), "oos_max_dd": s["oos"].get("max_dd")})
    gdf = pd.DataFrame(grid)
    rep["grid"] = {"share_oos_sharpe_ge_0_5": float((gdf["oos_sharpe"] >= 0.5).mean()),
                   "oos_sharpe_median": float(gdf["oos_sharpe"].median()),
                   "is_sharpe_median": float(gdf["is_sharpe"].median())}
    rep["seconds"] = round(time.perf_counter() - t0, 1)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    gdf.to_csv(out / "grid.csv", index=False)
    pd.DataFrame({"tsmom": daily["net"], "buy_and_hold": hold["net"]}).join(
        pd.DataFrame({"momentum": mom["net"], "combined": combo["net"]}), how="outer").to_csv(out / "daily_net.csv")
    print(json.dumps({k: rep[k] for k in ("verdict",)} | {"combination": rep["combination"]["verdict"]}, indent=2))
    print(f"wrote {out} in {rep['seconds']}s")


if __name__ == "__main__":
    main()
