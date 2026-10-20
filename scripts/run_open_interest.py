"""Open-interest positioning signals (pre-registered in docs/plans/open_interest_hypothesis.md).

  PYTHONPATH=src python scripts/run_open_interest.py
Needs the perp 1h + funding cache and the metrics cache from fetch_binance_metrics.py.
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
from hf_research.oi_signals import load_metrics, oi_growth, rank_weights, taker_flow
from hf_research.paper_xs import PRIMARY_STRATEGY
from hf_research.xs_momentum import MomentumConfig, backtest, eligible_from_membership, load_daily_panels

TIERS = {"tiered": {10: 3.0, 30: 6.0, 60: 12.0}, "tiered_x2": {10: 6.0, 30: 12.0, 60: 24.0}}
PRIMARY = {"lookback": 7, "select_frac": 0.2, "rebalance_days": 7, "long_high": False}
CRITERIA = {"oos_sharpe_min": 0.7, "is_sharpe_min": 0.4, "oos_max_dd_min": -0.35,
            "oos_sharpe_drop_top5_min": 0.4, "bootstrap_p_max": 0.10}
COMBO_SHARPE_GAIN = 0.15
COVERAGE_MIN = 0.80
GRID = {"lookback": [3, 7, 14], "select_frac": [0.1, 0.2, 0.3], "rebalance_days": [7, 3]}
FEE, BAND = 0.0006, 0.02


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", default="artifacts/research/universe_top60_v1/membership.csv")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--oos-start", default="2024-01-01")
    ap.add_argument("--out-dir", default="artifacts/research/open_interest_v1")
    args = ap.parse_args()
    t0 = time.perf_counter()
    ts = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731

    membership = pd.read_csv(args.membership)
    perps = sorted(set(membership["symbol"]))
    panels = load_daily_panels(perps, cache_dir=args.cache_dir)
    idx = panels["signal_close"].index
    members = eligible_from_membership(membership, idx, perps)
    metrics = load_metrics(perps, idx, args.cache_dir)
    slip = {k: slippage_panel(membership, idx, perps, t) for k, t in TIERS.items()}

    # Coverage: share of the month's members with an OI value, averaged over the month's days.
    has_oi = metrics["oi"].notna() & members
    cov = (has_oi.sum(axis=1) / members.sum(axis=1).replace(0, np.nan)).groupby(idx.strftime("%Y-%m")).mean().dropna()
    ok_months = cov[(cov >= COVERAGE_MIN) & (cov.index >= "2021-03")]
    is_start = f"{ok_months.index[0]}-01"
    spans = {"is": (is_start, args.oos_start), "oos": (args.oos_start, None)}
    print(f"data ready ({time.perf_counter() - t0:.0f}s); IS starts {is_start}", flush=True)

    def per(daily: pd.DataFrame) -> dict:
        return {k: summarize(daily, s, e) for k, (s, e) in spans.items()} | {
            "sharpe_by_year": {str(y): summarize(daily, f"{y}-01-01", f"{y + 1}-01-01").get("sharpe")
                               for y in range(2021, 2027)}}

    def run(score: pd.DataFrame, *, select_frac: float, rebalance_days: int, long_high: bool, cost: str = "tiered",
            eligible: pd.DataFrame | None = None, detail: bool = False):
        el = (members if eligible is None else eligible) & score.notna() & panels["exec_close"].notna()
        w = rank_weights(score, el, select_frac=select_frac, long_high=long_high)
        return backtest_weights(dict(panels, slippage_bps=slip[cost]), w, fee=FEE, band=BAND,
                                rebalance_days=rebalance_days, detail=detail)

    prim = dict(select_frac=PRIMARY["select_frac"], rebalance_days=PRIMARY["rebalance_days"], long_high=PRIMARY["long_high"])
    score = oi_growth(metrics["oi"], PRIMARY["lookback"])
    daily, parts = run(score, detail=True, **prim)
    rep: dict = {"primary_config": PRIMARY, "criteria": CRITERIA, "is_start": is_start,
                 "coverage_by_month": {k: round(float(v), 3) for k, v in cov.items()},
                 "primary": per(daily), "primary_cost_x2": per(run(score, cost="tiered_x2", **prim))}
    oos = daily.index >= ts(args.oos_start)

    contrib = (parts["gross"] + parts["funding"] - parts["cost"])[oos].sum().sort_values(ascending=False)
    top5 = list(contrib.head(5).index)
    rep["symbols_oos"] = {"top5": {k: float(contrib[k]) for k in top5},
                          "bottom5": {k: float(v) for k, v in contrib.tail(5).items()},
                          "total": float(contrib.sum())}
    el5 = members.copy()
    el5.loc[:, top5] = False
    rep["drop_top5"] = per(run(score, eligible=el5, **prim))
    rep["bootstrap_oos_p_sharpe_le_0"] = bootstrap_p_sharpe_le_0(daily.loc[oos & (daily["gross_exposure"] > 0), "net"])
    # The opposite direction is reported only to show the sign is not an artifact; it decides nothing.
    rep["opposite_direction_not_a_finding"] = per(run(score, **(prim | {"long_high": True})))

    o, i = rep["primary"]["oos"], rep["primary"]["is"]
    rep["verdict"] = {"oos_sharpe": o["sharpe"] >= CRITERIA["oos_sharpe_min"],
                      "is_sharpe": i["sharpe"] >= CRITERIA["is_sharpe_min"],
                      "oos_max_dd": o["max_dd"] >= CRITERIA["oos_max_dd_min"],
                      "oos_sharpe_drop_top5": rep["drop_top5"]["oos"]["sharpe"] >= CRITERIA["oos_sharpe_drop_top5_min"],
                      "bootstrap": rep["bootstrap_oos_p_sharpe_le_0"] < CRITERIA["bootstrap_p_max"]}
    rep["verdict"]["pass"] = all(rep["verdict"].values())
    print(f"primary done ({time.perf_counter() - t0:.0f}s)", flush=True)

    # Equal-risk combination with the primary momentum (scale from in-sample volatility only).
    mom = backtest(dict(panels, eligible=members, slippage_bps=slip["tiered"]), MomentumConfig(**PRIMARY_STRATEGY))
    is_mask = (daily.index >= ts(is_start)) & ~oos
    scale = float(mom.loc[is_mask, "net"].std() / daily.loc[is_mask, "net"].std())
    combo = mom.copy()
    for c in ("net", "gross", "funding", "cost", "turnover"):
        combo[c] = 0.5 * mom[c] + 0.5 * scale * daily[c]
    combo["gross_exposure"] = mom["gross_exposure"] + daily["gross_exposure"]
    rep["combination"] = {"oi_scale": scale, "oos_daily_corr": float(mom.loc[oos, "net"].corr(daily.loc[oos, "net"])),
                          "momentum": per(mom), "combined": per(combo)}
    cm, mm = rep["combination"]["combined"]["oos"], rep["combination"]["momentum"]["oos"]
    rep["combination"]["verdict"] = {"sharpe_gain": cm["sharpe"] - mm["sharpe"] >= COMBO_SHARPE_GAIN,
                                     "max_dd_better": cm["max_dd"] > mm["max_dd"]}
    rep["combination"]["verdict"]["pass"] = all(rep["combination"]["verdict"].values())

    # Exploratory secondaries (decide nothing).
    ret7 = np.log(panels["signal_close"] / panels["signal_close"].shift(7))
    secondaries = {"S1_retail_positioning": (metrics["count_ls"], False),
                   "S2_taker_flow": (taker_flow(metrics["taker_ratio_log_mean"]), True),
                   "S3_oi_with_price": (score * np.sign(ret7), False)}
    rep["secondary_exploratory"] = {k: per(run(s, **(prim | {"long_high": lh}))) for k, (s, lh) in secondaries.items()}
    print(f"secondaries done ({time.perf_counter() - t0:.0f}s)", flush=True)

    grid = []
    for lb, frac, rb in itertools.product(GRID["lookback"], GRID["select_frac"], GRID["rebalance_days"]):
        s = per(run(oi_growth(metrics["oi"], lb), select_frac=frac, rebalance_days=rb, long_high=False))
        grid.append({"lookback": lb, "select_frac": frac, "rebalance_days": rb, "is_sharpe": s["is"].get("sharpe"),
                     "oos_sharpe": s["oos"].get("sharpe"), "oos_cagr": s["oos"].get("cagr"), "oos_max_dd": s["oos"].get("max_dd")})
    gdf = pd.DataFrame(grid)
    rep["grid"] = {"share_oos_sharpe_ge_0_5": float((gdf["oos_sharpe"] >= 0.5).mean()),
                   "share_oos_sharpe_gt_0": float((gdf["oos_sharpe"] > 0).mean()),
                   "oos_sharpe_median": float(gdf["oos_sharpe"].median()), "is_sharpe_median": float(gdf["is_sharpe"].median())}
    rep["seconds"] = round(time.perf_counter() - t0, 1)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    gdf.to_csv(out / "grid.csv", index=False)
    pd.DataFrame({"oi_primary": daily["net"], "momentum": mom["net"], "combined": combo["net"]}).to_csv(out / "daily_net.csv")
    print(json.dumps({"verdict": rep["verdict"], "combination": rep["combination"]["verdict"]}, indent=2))
    print(f"wrote {out} in {rep['seconds']}s")


if __name__ == "__main__":
    main()
