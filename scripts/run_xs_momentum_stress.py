"""Stress tests for the pre-registered cross-sectional momentum hypothesis (top-60, L30 S1 R7).

Nothing here re-selects parameters: every scenario runs the fixed primary config (plus the full
top-30 / top-60 grids for the cost scenarios, to see if the broad result survives).

Scenarios:
- costs: flat slippage 3 / 10 / 20 bps and rank-tiered slippage (tiers by monthly liquidity rank),
- ex-funding: net P&L without the funding leg,
- execution delay: fill 1h (base), 5h and 9h after the signal,
- attribution: long vs short leg, per-symbol contribution and leave-top-k-symbols-out,
- block-bootstrap confidence interval of the out-of-sample Sharpe.

Example:
  PYTHONPATH=src python scripts/run_xs_momentum_stress.py \
    --membership artifacts/research/universe_top60_v1/membership.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from hf_research.xs_momentum import MomentumConfig, backtest, eligible_from_membership, load_daily_panels, summarize

PRIMARY = dict(mode="long_short", lookback_days=30, skip_days=1, rebalance_days=7, select_frac=0.2,
               weighting="equal", band=0.02, fee=0.0006, slippage_bps=3.0)
GRID = {
    "lookback_days": [14, 30, 60],
    "skip_days": [0, 1],
    "rebalance_days": [7, 14],
    "select_frac": [0.1, 0.2, 0.3],
    "band": [0.0, 0.02, 0.05],
    "weighting": ["equal", "inv_vol"],
}
# Slippage per side in bps by monthly liquidity rank; assets outside the top 60 get the last tier.
TIERS = {"tiered": {10: 3.0, 30: 6.0, 60: 12.0}, "tiered_x2": {10: 6.0, 30: 12.0, 60: 24.0}}

_SHARED: dict | None = None


def slippage_panel(membership: pd.DataFrame, index: pd.DatetimeIndex, columns, tiers: dict[int, float]) -> pd.DataFrame:
    worst = max(tiers.values())
    def bps(rank: float) -> float:
        for cut in sorted(tiers):
            if rank <= cut:
                return tiers[cut]
        return worst
    by_key = {(m, s): bps(r) for m, s, r in membership[["month", "symbol", "rank"]].itertuples(index=False)}
    months = index.strftime("%Y-%m")
    return pd.DataFrame({c: [by_key.get((m, c), worst) for m in months] for c in columns}, index=index)


def block_bootstrap_sharpe(net: pd.Series, block: int = 20, reps: int = 2000, seed: int = 7) -> dict:
    x = net.to_numpy()
    n = len(x)
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for k in range(reps):
        starts = rng.integers(0, n - block, size=int(np.ceil(n / block)))
        s = np.concatenate([x[i:i + block] for i in starts])[:n]
        out[k] = s.mean() / s.std() * np.sqrt(365.0)
    return {"p05": float(np.quantile(out, 0.05)), "p50": float(np.median(out)), "p95": float(np.quantile(out, 0.95)),
            "prob_sharpe_le_0": float((out <= 0).mean())}


def _init(shared):
    global _SHARED
    _SHARED = shared


def _grid_job(job):
    universe, cost_name, cfg_dict = job
    panels = dict(_SHARED["base"], eligible=_SHARED["eligible"][universe])
    if cost_name in _SHARED["slip"]:
        panels["slippage_bps"] = _SHARED["slip"][cost_name]
    s = summarize(backtest(panels, MomentumConfig(**cfg_dict)), _SHARED["oos_start"], None)
    return universe, cost_name, s.get("sharpe")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", required=True)
    ap.add_argument("--is-start", default="2021-03-01")
    ap.add_argument("--oos-start", default="2024-01-01")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    t0 = time.perf_counter()
    membership = pd.read_csv(args.membership)
    symbols = sorted(set(membership["symbol"]))
    base = load_daily_panels(symbols)
    idx, cols = base["signal_close"].index, base["signal_close"].columns
    eligible = {
        "top30": eligible_from_membership(membership[membership["rank"] <= 30], idx, cols),
        "top60": eligible_from_membership(membership[membership["rank"] <= 60], idx, cols),
    }
    slip = {name: slippage_panel(membership, idx, cols, t) for name, t in TIERS.items()}
    cfg = MomentumConfig(**PRIMARY)
    panels = dict(base, eligible=eligible["top60"])

    def periods(daily: pd.DataFrame) -> dict:
        return {
            "is": summarize(daily, args.is_start, args.oos_start),
            "oos": summarize(daily, args.oos_start, None),
            "sharpe_by_year": {str(y): summarize(daily, f"{y}-01-01", f"{y + 1}-01-01").get("sharpe") for y in range(2021, 2027)},
        }

    report: dict = {"config": PRIMARY, "universe": "top60"}

    # 1) Cost scenarios on the primary config.
    costs = {}
    for name, bps in (("flat_3bp", 3.0), ("flat_10bp", 10.0), ("flat_20bp", 20.0)):
        costs[name] = periods(backtest(panels, MomentumConfig(**{**PRIMARY, "slippage_bps": bps})))
    for name in TIERS:
        costs[name] = periods(backtest(dict(panels, slippage_bps=slip[name]), cfg))
    report["costs"] = costs

    # 2) Ex-funding and attribution on the base cost case.
    daily, parts = backtest(panels, cfg, detail=True)
    ex_fund = daily.assign(net=daily["net"] - daily["funding"], funding=0.0)
    report["ex_funding"] = periods(ex_fund)

    oos = parts["weight"].index >= pd.Timestamp(args.oos_start, tz="UTC")
    w = parts["weight"]
    legs = {}
    for leg, mask in (("long", w > 0), ("short", w < 0)):
        g = parts["gross"].where(mask, 0.0).sum(axis=1)
        f = parts["funding"].where(mask, 0.0).sum(axis=1)
        legs[leg] = {}
        for tag, sel in (("is", ~oos & (w.index >= pd.Timestamp(args.is_start, tz="UTC"))), ("oos", oos)):
            legs[leg][tag] = {"gross_ann": float(g[sel].mean() * 365), "funding_ann": float(f[sel].mean() * 365),
                              "gross_plus_funding_sharpe": float((g + f)[sel].mean() / (g + f)[sel].std() * np.sqrt(365))}
    report["legs"] = legs

    sym_net = (parts["gross"] + parts["funding"] - parts["cost"])[oos].sum().sort_values(ascending=False)
    total = float(sym_net.sum())
    report["symbols_oos"] = {
        "total_net_simple_sum": total,
        "n_symbols_traded": int((parts["weight"][oos] != 0).any().sum()),
        "top10": {k: float(v) for k, v in sym_net.head(10).items()},
        "bottom10": {k: float(v) for k, v in sym_net.tail(10).items()},
        "top5_share_of_total": float(sym_net.head(5).sum() / total) if total else None,
        "top10_share_of_total": float(sym_net.head(10).sum() / total) if total else None,
    }
    loo = {}
    for k in (5, 10, 20):
        drop = set(sym_net.head(k).index)
        elig = eligible["top60"].copy()
        elig.loc[:, [c for c in elig.columns if c in drop]] = False
        loo[f"drop_top{k}_oos_winners"] = periods(backtest(dict(base, eligible=elig), cfg))
    report["leave_out"] = loo

    report["bootstrap_oos_sharpe"] = {
        "base": block_bootstrap_sharpe(daily.loc[daily.index >= pd.Timestamp(args.oos_start, tz="UTC"), "net"]),
        "ex_funding": block_bootstrap_sharpe(ex_fund.loc[ex_fund.index >= pd.Timestamp(args.oos_start, tz="UTC"), "net"]),
    }
    report["base"] = periods(daily)

    # 3) Execution delay: reload panels with a later fill bar.
    delay = {}
    for h in (4, 8):
        p = load_daily_panels(symbols, exec_hour=h)
        delay[f"fill_{h + 1}h_after_signal"] = periods(backtest(dict(p, eligible=eligible["top60"]), cfg))
    report["exec_delay"] = delay

    # 4) Full grids under each cost scenario.
    keys = list(GRID)
    cost_cases = {"taker_3bp": dict(slippage_bps=3.0), "flat_10bp": dict(slippage_bps=10.0), "tiered": {}, "tiered_x2": {}}
    jobs = [
        (u, c, dict(mode="long_short", **dict(zip(keys, v)), fee=0.0006, slippage_bps=extra.get("slippage_bps", 3.0)))
        for u in eligible for c, extra in cost_cases.items() for v in itertools.product(*GRID.values())
    ]
    shared = {"base": base, "eligible": eligible, "slip": slip, "oos_start": args.oos_start}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(shared,)) as ex:
        res = pd.DataFrame(list(ex.map(_grid_job, jobs, chunksize=4)), columns=["universe", "cost", "oos_sharpe"])
    report["grid_oos_sharpe"] = {
        f"{u}|{c}": {"median": float(g["oos_sharpe"].median()), "positive_share": float((g["oos_sharpe"] > 0).mean())}
        for (u, c), g in res.groupby(["universe", "cost"])
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir or f"artifacts/research/xs_momentum_stress_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sym_net.rename("oos_net_contribution").to_csv(out_dir / "symbol_contribution_oos.csv")
    meta = {"created_utc": stamp, "membership": args.membership, "is_start": args.is_start, "oos_start": args.oos_start,
            "tiers_bps": TIERS, "report": report, "seconds": round(time.perf_counter() - t0, 1)}
    (out_dir / "report.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))
    print(f"saved: {out_dir} ({meta['seconds']}s)")


if __name__ == "__main__":
    main()
