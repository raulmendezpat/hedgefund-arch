"""Backtest of the pre-registered delta-neutral funding carry (docs/plans/funding_carry_hypothesis.md).

Runs the primary configuration, the pre-registered success criteria, leave-top-5-out, margin
risk, the sensitivity grid, and the equal-risk combination with the primary momentum strategy.

  PYTHONPATH=src python scripts/run_funding_carry.py
Needs the perp 1h + funding cache (Binance USDT-M) and the spot cache from fetch_binance_spot.py.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hf_research.engine import backtest_weights, summarize
from hf_research.funding_carry import (
    SPOT_PREFIX,
    CarryConfig,
    carry_targets,
    delisted_at,
    fee_series,
    load_spot_exec,
    mask_delisted,
    signal_funding,
    spot_col,
    trailing_funding_ann,
)
from hf_research.paper_xs import PRIMARY_STRATEGY
from hf_research.xs_momentum import MomentumConfig, backtest, eligible_from_membership, load_daily_panels

TIERS = {"tiered": {10: 3.0, 30: 6.0, 60: 12.0}, "tiered_x2": {10: 6.0, 30: 12.0, 60: 24.0}}
CRITERIA = {"oos_sharpe_min": 1.5, "oos_months_positive_min": 0.70, "oos_max_dd_min": -0.10,
            "oos_sharpe_drop_top5_min": 1.0, "is_sharpe_min": 0.5}
COMBO_CRITERIA = {"sharpe_gain_min": 0.2}
GRID = {"entry_ann": [0.05, 0.10, 0.20], "lookback_days": [3, 7, 14], "max_positions": [5, 10, 20]}


def slippage_panel(membership: pd.DataFrame, index: pd.DatetimeIndex, perps: list[str], tiers: dict[int, float]) -> pd.DataFrame:
    """Per-leg slippage by the perp's monthly liquidity rank; the spot leg pays the same tier."""
    worst = max(tiers.values())

    def bps(rank: float) -> float:
        return next((tiers[c] for c in sorted(tiers) if rank <= c), worst)

    by_key = {(m, s): bps(r) for m, s, r in membership[["month", "symbol", "rank"]].itertuples(index=False)}
    months = index.strftime("%Y-%m")
    perp = pd.DataFrame({p: [by_key.get((m, p), worst) for m in months] for p in perps}, index=index)
    return pd.concat([perp, perp.rename(columns=spot_col)], axis=1)


def periods(daily: pd.DataFrame, is_start: str, oos_start: str) -> dict:
    return {"is": summarize(daily, is_start, oos_start), "oos": summarize(daily, oos_start, None),
            "sharpe_by_year": {str(y): summarize(daily, f"{y}-01-01", f"{y + 1}-01-01").get("sharpe")
                               for y in range(2021, 2027)}}


def pair_contrib(parts: dict, perps: list[str]) -> pd.DataFrame:
    net = parts["gross"] + parts["funding"] - parts["cost"]
    return pd.DataFrame({p: net[p] + net[spot_col(p)] for p in perps if p in net and spot_col(p) in net})


def margin_risk(parts: dict, perps: list[str]) -> dict:
    """Perp-leg daily loss against the margin left after funding the spot legs (equity - spot)."""
    w = parts["weight"]
    spot_cols = [c for c in w.columns if c.startswith(SPOT_PREFIX)]
    spot_exposure = w[spot_cols].sum(axis=1)
    perp_pnl = parts["gross"][[p for p in perps if p in parts["gross"]]].sum(axis=1)
    active = spot_exposure > 0
    ratio = (-perp_pnl[active]).clip(lower=0) / (1.0 - spot_exposure[active])
    held_short = w[perps] < 0
    price_ret = (parts["gross"][perps] / w[perps].where(held_short)).where(held_short)  # gross = weight * return
    worst = price_ret.max()  # worst daily price rise of a shorted perp while held
    return {"perp_leg_loss_over_free_margin_max": float(ratio.max()),
            "perp_leg_loss_over_free_margin_p99": float(ratio.quantile(0.99)),
            "worst_daily_rise_of_a_held_short": float(worst.max()),
            "worst_daily_rise_symbol": str(worst.idxmax())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", default="artifacts/research/universe_top60_v1/membership.csv")
    ap.add_argument("--spot-map", default="data/cache/binance/spot_map.json")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--is-start", default="2021-03-01")
    ap.add_argument("--oos-start", default="2024-01-01")
    ap.add_argument("--out-dir", default="artifacts/research/funding_carry_v1")
    args = ap.parse_args()
    t0 = time.perf_counter()
    per = lambda daily: periods(daily, args.is_start, args.oos_start)  # noqa: E731

    membership = pd.read_csv(args.membership)
    spot_map = json.loads(Path(args.spot_map).read_text())["mapping"]
    perps = sorted(set(membership["symbol"]))
    base = load_daily_panels(perps, cache_dir=args.cache_dir)
    idx = base["signal_close"].index
    spot_exec = load_spot_exec(spot_map, idx, args.cache_dir)
    delisted = delisted_at(perps, args.cache_dir)
    perp_exec = mask_delisted(base["exec_close"], delisted)  # frozen post-delisting candles cannot hedge
    exec_close = pd.concat([perp_exec, spot_exec], axis=1)
    has_spot = pd.DataFrame({p: spot_exec[spot_col(p)].notna() if spot_col(p) in spot_exec else False
                             for p in perps}, index=idx)
    eligible = eligible_from_membership(membership, idx, perps)
    tradable = eligible & perp_exec[perps].notna() & has_spot
    panels = {"exec_close": exec_close, "funding": base["funding"]}
    slip = {k: slippage_panel(membership, idx, perps, t) for k, t in TIERS.items()}
    daily_fund = signal_funding(perps, idx, args.cache_dir)
    print(f"panels ready: {len(idx)} days, {len(perps)} perps, {len(spot_exec.columns)} with spot "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)

    def run(cfg: CarryConfig, cost: str = "tiered", trad: pd.DataFrame = tradable, detail: bool = False):
        targets = carry_targets(trailing_funding_ann(daily_fund, cfg.lookback_days), trad, cfg)
        return backtest_weights(dict(panels, slippage_bps=slip[cost]), targets,
                                fee=fee_series(exec_close.columns, cfg), band=cfg.band, detail=detail)

    cfg = CarryConfig()
    report: dict = {"config": cfg.to_dict(), "criteria": CRITERIA,
                    "data": {"perps": len(perps), "with_spot": int(len(spot_exec.columns)), "delisted_perps": len(delisted),
                             "first_day": str(idx[0].date()), "last_day": str(idx[-1].date())}}

    daily, parts = run(cfg, detail=True)
    report["primary"] = per(daily)
    oos = daily.index >= pd.Timestamp(args.oos_start, tz="UTC")
    report["primary"]["exposure"] = {
        "avg_pairs_oos": float(daily.loc[oos, "n_positions"].mean() / 2),
        "share_days_flat_oos": float((daily.loc[oos, "n_positions"] == 0).mean()),
        "avg_spot_exposure_oos": float((daily.loc[oos, "gross_exposure"] / 2).mean())}
    report["primary_cost_x2"] = per(run(cfg, "tiered_x2"))
    report["margin_risk"] = margin_risk(parts, perps)
    print(f"primary done ({time.perf_counter() - t0:.0f}s)", flush=True)

    pc = pair_contrib(parts, perps)
    sym = pc[oos].sum().sort_values(ascending=False)
    total = float(sym.sum())
    report["symbols_oos"] = {"total_net_simple_sum": total, "n_pairs_traded": int((pc[oos] != 0).any().sum()),
                             "top10": {k: float(v) for k, v in sym.head(10).items()},
                             "bottom5": {k: float(v) for k, v in sym.tail(5).items()},
                             "top5_share_of_total": float(sym.head(5).sum() / total) if total else None}
    trad5 = tradable.copy()
    trad5.loc[:, list(sym.head(5).index)] = False
    report["drop_top5"] = per(run(cfg, trad=trad5))

    o, i = report["primary"]["oos"], report["primary"]["is"]
    report["verdict"] = {
        "oos_sharpe": o.get("sharpe", float("nan")) >= CRITERIA["oos_sharpe_min"],
        "oos_months_positive": o.get("months_positive", 0) >= CRITERIA["oos_months_positive_min"],
        "oos_max_dd": o.get("max_dd", -1) >= CRITERIA["oos_max_dd_min"],
        "oos_sharpe_drop_top5": report["drop_top5"]["oos"].get("sharpe", float("nan")) >= CRITERIA["oos_sharpe_drop_top5_min"],
        "is_sharpe": i.get("sharpe", float("nan")) >= CRITERIA["is_sharpe_min"],
    }
    report["verdict"]["pass"] = all(report["verdict"].values())

    # Equal-risk combination with the primary momentum strategy (tiered costs, top-60 universe).
    mom = backtest(dict(base, eligible=eligible, slippage_bps=slip["tiered"][perps]), MomentumConfig(**PRIMARY_STRATEGY))
    is_mask = (daily.index >= pd.Timestamp(args.is_start, tz="UTC")) & ~oos
    vol_m, vol_c = mom.loc[is_mask, "net"].std(), daily.loc[is_mask, "net"].std()
    scale_c = vol_m / vol_c  # carry scaled to momentum's in-sample volatility
    combo = mom.copy()
    combo["net"] = 0.5 * mom["net"] + 0.5 * scale_c * daily["net"]
    for c in ("gross", "funding", "cost", "turnover"):
        combo[c] = 0.5 * mom[c] + 0.5 * scale_c * daily[c]
    combo["gross_exposure"] = mom["gross_exposure"] + daily["gross_exposure"]
    report["combination"] = {"carry_scale_to_momentum_vol": float(scale_c), "momentum": per(mom),
                             "carry_scaled": per(daily.assign(net=daily["net"] * scale_c)), "combined": per(combo),
                             "oos_daily_corr": float(mom.loc[oos, "net"].corr(daily.loc[oos, "net"]))}
    cm, mm = report["combination"]["combined"]["oos"], report["combination"]["momentum"]["oos"]
    report["combination"]["verdict"] = {
        "sharpe_gain": cm["sharpe"] - mm["sharpe"] >= COMBO_CRITERIA["sharpe_gain_min"],
        "max_dd_better": cm["max_dd"] > mm["max_dd"]}
    print(f"combination done ({time.perf_counter() - t0:.0f}s)", flush=True)

    grid = []
    for entry, lb, n in itertools.product(GRID["entry_ann"], GRID["lookback_days"], GRID["max_positions"]):
        g = CarryConfig(entry_ann=entry, lookback_days=lb, max_positions=n, pair_weight=0.7 / n)
        s = per(run(g))
        grid.append({"entry_ann": entry, "lookback_days": lb, "max_positions": n,
                     "is_sharpe": s["is"].get("sharpe"), "oos_sharpe": s["oos"].get("sharpe"),
                     "oos_cagr": s["oos"].get("cagr"), "oos_max_dd": s["oos"].get("max_dd"),
                     "oos_months_positive": s["oos"].get("months_positive")})
    gdf = pd.DataFrame(grid)
    report["grid"] = {"oos_sharpe_median": float(gdf["oos_sharpe"].median()),
                      "share_oos_sharpe_ge_1": float((gdf["oos_sharpe"] >= 1.0).mean()),
                      "share_oos_sharpe_gt_0": float((gdf["oos_sharpe"] > 0).mean())}
    report["seconds"] = round(time.perf_counter() - t0, 1)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    gdf.to_csv(out / "grid.csv", index=False)
    pd.DataFrame({"carry": daily["net"], "momentum": mom["net"], "combined": combo["net"]}).to_csv(out / "daily_net.csv")
    print(json.dumps({k: report[k] for k in ("primary", "verdict")}, indent=2, default=float))
    print(f"wrote {out} in {report['seconds']}s")


if __name__ == "__main__":
    main()
