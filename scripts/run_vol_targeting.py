"""Volatility targeting of the primary momentum strategy and x1-x4 leverage
(pre-registered in docs/plans/vol_targeting_hypothesis.md).

  PYTHONPATH=src python scripts/run_vol_targeting.py
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
from hf_research.paper_xs import PRIMARY_STRATEGY
from hf_research.vol_target import account_path, intraday_paths, load_hourly_close, vol_scale
from hf_research.xs_momentum import MomentumConfig, eligible_from_membership, load_daily_panels, target_weights

TIERS = {10: 3.0, 30: 6.0, 60: 12.0}
PRIMARY = {"target": 0.25, "window": 60, "cap": 2.0}
CRITERIA = {"oos_sharpe_gain_min": 0.10, "is_sharpe_gain_min": 0.0}
LEVERAGE = [1, 2, 3, 4]
MAINTENANCE = 0.02
KILL_DD = -0.25
ACCEPT = {"max_dd_hourly_min": -0.40, "roll30_worst_min": -0.30}
GRID = {"window": [20, 60, 120], "cap": [1.5, 2.0, 3.0]}


def slippage_panel(membership: pd.DataFrame, index: pd.DatetimeIndex, columns, tiers: dict[int, float]) -> pd.DataFrame:
    worst = max(tiers.values())

    def bps(rank: float) -> float:
        return next((tiers[c] for c in sorted(tiers) if rank <= c), worst)

    by_key = {(m, s): bps(r) for m, s, r in membership[["month", "symbol", "rank"]].itertuples(index=False)}
    months = index.strftime("%Y-%m")
    return pd.DataFrame({c: [by_key.get((m, c), worst) for m in months] for c in columns}, index=index)


def window(daily: pd.DataFrame, start: str, end: str | None) -> pd.DataFrame:
    d = daily[daily.index >= pd.Timestamp(start, tz="UTC")]
    return d[d.index < pd.Timestamp(end, tz="UTC")] if end else d


def active(daily: pd.DataFrame) -> pd.Series:
    return daily["gross_exposure"].shift(1, fill_value=0.0).gt(0) | daily["turnover"].gt(0)


def at_vol(net: pd.Series, vol: float) -> pd.Series:
    return net * vol / net.std()


def max_dd(net: pd.Series) -> float:
    eq = (1.0 + net).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def worst_month(net: pd.Series) -> float:
    return float(((1.0 + net).groupby(net.index.strftime("%Y-%m")).prod() - 1.0).min())


def sharpe(x: np.ndarray) -> float:
    return float(x.mean() / x.std() * np.sqrt(365.0))


def paired_bootstrap(a: pd.Series, b: pd.Series, block: int = 20, reps: int = 2000, seed: int = 7) -> dict:
    """P(Sharpe(b) - Sharpe(a) <= 0) under a circular block bootstrap of the paired daily series."""
    rng = np.random.default_rng(seed)
    x, y, n = a.to_numpy(), b.to_numpy(), len(a)
    diffs = []
    for _ in range(reps):
        starts = rng.integers(0, n, size=n // block + 1)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        diffs.append(sharpe(y[idx]) - sharpe(x[idx]))
    diffs = np.array(diffs)
    return {"p_gain_le_0": float((diffs <= 0).mean()), "gain_p05": float(np.quantile(diffs, 0.05)),
            "gain_median": float(np.median(diffs))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", default="artifacts/research/universe_top60_v1/membership.csv")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--is-start", default="2021-03-01")
    ap.add_argument("--oos-start", default="2024-01-01")
    ap.add_argument("--out-dir", default="artifacts/research/vol_targeting_v1")
    args = ap.parse_args()
    t0 = time.perf_counter()
    spans = {"is": (args.is_start, args.oos_start), "oos": (args.oos_start, None)}

    membership = pd.read_csv(args.membership)
    symbols = sorted(set(membership["symbol"]))
    panels = load_daily_panels(symbols, cache_dir=args.cache_dir)
    idx = panels["signal_close"].index
    panels["eligible"] = eligible_from_membership(membership, idx, symbols)
    panels["slippage_bps"] = slippage_panel(membership, idx, symbols, TIERS)
    cfg = MomentumConfig(**PRIMARY_STRATEGY)
    base_targets = target_weights(panels["signal_close"], cfg, panels["eligible"])
    hourly = load_hourly_close(symbols, args.cache_dir)
    print(f"data ready ({time.perf_counter() - t0:.0f}s)", flush=True)

    def run(targets: pd.DataFrame, lev: float = 1.0, detail: bool = False):
        return backtest_weights(panels, targets * lev, fee=cfg.fee, slippage_bps=cfg.slippage_bps,
                                band=cfg.band * lev, rebalance_days=cfg.rebalance_days, detail=detail)

    def per(daily: pd.DataFrame) -> dict:
        return {k: summarize(daily, s, e) for k, (s, e) in spans.items()} | {
            "sharpe_by_year": {str(y): summarize(daily, f"{y}-01-01", f"{y + 1}-01-01").get("sharpe")
                               for y in range(2021, 2027)}}

    base = run(base_targets)
    base_active = active(base)

    def vt_targets(target: float, window: int, cap: float) -> tuple[pd.DataFrame, pd.Series]:
        k = vol_scale(base["net"], base_active, target=target, window=window, cap=cap)
        return base_targets.mul(k, axis=0), k

    vt_t, k = vt_targets(**PRIMARY)
    vt = run(vt_t)
    report: dict = {"momentum": PRIMARY_STRATEGY, "primary": PRIMARY, "criteria": CRITERIA,
                    "base": per(base), "vol_target": per(vt)}
    k_oos = k[(k.index >= pd.Timestamp(args.oos_start, tz="UTC")) & base_active]
    report["scale"] = {"oos_mean": float(k_oos.mean()), "oos_p05": float(k_oos.quantile(0.05)),
                       "oos_p95": float(k_oos.quantile(0.95)), "share_at_cap_oos": float((k_oos >= PRIMARY["cap"]).mean())}

    # Pre-registered criteria (x1).
    b_oos, v_oos = (window(x[active(x)], *spans["oos"])["net"] for x in (base, vt))
    common = b_oos.index.intersection(v_oos.index)
    b_oos, v_oos = b_oos[common], v_oos[common]
    v_matched = at_vol(v_oos, b_oos.std())
    (bo, bi), (vo, vi) = ((report[s]["oos"]["sharpe"], report[s]["is"]["sharpe"]) for s in ("base", "vol_target"))
    report["matched_vol_oos"] = {"base_max_dd": max_dd(b_oos), "vt_max_dd": max_dd(v_matched),
                                 "base_worst_month": worst_month(b_oos), "vt_worst_month": worst_month(v_matched)}
    m = report["matched_vol_oos"]
    report["verdict"] = {"oos_sharpe_gain": vo - bo >= CRITERIA["oos_sharpe_gain_min"],
                         "is_sharpe_not_worse": vi - bi >= CRITERIA["is_sharpe_gain_min"],
                         "oos_max_dd_better_at_matched_vol": m["vt_max_dd"] > m["base_max_dd"],
                         "oos_worst_month_better_at_matched_vol": m["vt_worst_month"] > m["base_worst_month"]}
    report["verdict"]["pass"] = all(report["verdict"].values())
    report["bootstrap_oos_sharpe_gain"] = paired_bootstrap(b_oos, v_oos)
    print(f"x1 done ({time.perf_counter() - t0:.0f}s)", flush=True)

    # Leverage x1-x4 with an hourly margin-account simulation, each period from equity 1.
    lev_rows, curves = [], {}
    for name, targets in (("base", base_targets), ("vol_target", vt_t)):
        for lev in LEVERAGE:
            daily, parts = run(targets, lev, detail=True)
            ret, notional = intraday_paths(parts["weight"], hourly)
            curves[f"{name}_x{lev}"] = daily["net"]
            for p, (s, e) in spans.items():
                rows = (daily.index >= pd.Timestamp(s, tz="UTC")) & (daily.index < pd.Timestamp(e, tz="UTC") if e else True)
                acct = account_path(daily.loc[rows, "net"], ret[rows], notional[rows], maintenance=MAINTENANCE, kill_dd=KILL_DD)
                summ = summarize(daily[rows])
                ok = (acct["liquidated"] is None and acct["max_dd_hourly"] > ACCEPT["max_dd_hourly_min"]
                      and acct["roll30"]["worst"] > ACCEPT["roll30_worst_min"])
                lev_rows.append({"strategy": name, "leverage": lev, "period": p, "cagr": summ.get("cagr"),
                                 "sharpe": summ.get("sharpe"), "acceptable": ok, **acct})
            print(f"{name} x{lev} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    report["leverage"] = lev_rows
    report["leverage_acceptable"] = {
        f"{n}_x{lev}": all(r["acceptable"] for r in lev_rows if r["strategy"] == n and r["leverage"] == lev)
        for n in ("base", "vol_target") for lev in LEVERAGE}

    grid = []
    for win, cap in itertools.product(GRID["window"], GRID["cap"]):
        s = per(run(vt_targets(PRIMARY["target"], win, cap)[0]))
        grid.append({"window": win, "cap": cap, "is_sharpe": s["is"].get("sharpe"), "oos_sharpe": s["oos"].get("sharpe"),
                     "oos_max_dd": s["oos"].get("max_dd"), "oos_cagr": s["oos"].get("cagr")})
    gdf = pd.DataFrame(grid)
    report["grid"] = {"share_oos_sharpe_gt_base": float((gdf["oos_sharpe"] > bo).mean()),
                      "oos_sharpe_median": float(gdf["oos_sharpe"].median())}
    report["seconds"] = round(time.perf_counter() - t0, 1)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    gdf.to_csv(out / "grid.csv", index=False)
    pd.DataFrame(lev_rows).drop(columns=["roll30"]).join(pd.json_normalize([r["roll30"] for r in lev_rows]).add_prefix("roll30_")) \
        .to_csv(out / "leverage.csv", index=False)
    pd.DataFrame(curves).assign(scale=k).to_csv(out / "daily_net.csv")
    print(json.dumps({k: report[k] for k in ("verdict", "leverage_acceptable")}, indent=2))
    print(f"wrote {out} in {report['seconds']}s")


if __name__ == "__main__":
    main()
