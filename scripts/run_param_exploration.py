"""Exploratory parameter sweep of the cross-sectional momentum (incl. frac10) and taker-flow books
(docs/plans/param_exploration.md). Decides nothing for the registered papers.

Screening rule, fixed before running: a configuration is a "substantial improvement" over the
current one only if
  1. its Sharpe beats the current configuration's by >= 0.2 in development AND out of sample,
  2. its out-of-sample worst drawdown is no more than 5 points worse,
  3. the median of its one-step grid neighbours also beats the current by >= 0.1 in both periods,
  4. with costs x2 it still beats the current configuration (at normal costs) in both periods.

  PYTHONPATH=src python scripts/run_param_exploration.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hf_research.engine import backtest_weights, summarize
from hf_research.oi_signals import load_metrics, rank_weights, taker_flow
from hf_research.paper_xs import PRIMARY_STRATEGY
from hf_research.xs_momentum import MomentumConfig, eligible_from_membership, load_daily_panels, target_weights

TIERS = {10: 3.0, 30: 6.0, 60: 12.0}
OOS_START, END = "2024-01-01", "2026-10-19"
XS_GRID = {"lookback_days": [14, 21, 30, 45, 60, 90], "skip_days": [0, 1, 3], "select_frac": [0.1, 0.15, 0.2, 0.3],
           "rebalance_days": [7, 14], "weighting": ["equal", "inv_vol"]}
XS_BANDS = [0.0, 0.01, 0.02, 0.05, 0.10]
TAKER_GRID = {"window": [3, 5, 7, 10, 14], "select_frac": [0.1, 0.15, 0.2, 0.3], "rebalance_days": [3, 7, 14],
              "band": [0.02, 0.05]}
CURRENT = {"xs_momentum": dict(lookback_days=30, skip_days=1, select_frac=0.2, rebalance_days=7, weighting="equal", band=0.02),
           "xs_frac10": dict(lookback_days=30, skip_days=1, select_frac=0.1, rebalance_days=7, weighting="equal", band=0.02),
           "taker_flow": dict(window=7, select_frac=0.2, rebalance_days=7, band=0.02)}
RULE = {"sharpe_gain_min": 0.2, "dd_worse_max": 0.05, "neighbour_gain_min": 0.1}

_S: dict = {}


def slippage_panel(membership: pd.DataFrame, index: pd.DatetimeIndex, columns, tiers: dict[int, float]) -> pd.DataFrame:
    worst = max(tiers.values())

    def bps(rank: float) -> float:
        return next((tiers[c] for c in sorted(tiers) if rank <= c), worst)

    by_key = {(m, s): bps(r) for m, s, r in membership[["month", "symbol", "rank"]].itertuples(index=False)}
    months = index.strftime("%Y-%m")
    return pd.DataFrame({c: [by_key.get((m, c), worst) for m in months] for c in columns}, index=index)


def _periods(daily: pd.DataFrame, is_start: str) -> dict:
    out = {}
    for p, (a, b) in {"is": (is_start, OOS_START), "oos": (OOS_START, END)}.items():
        s = summarize(daily, a, b)
        out.update({f"{p}_sharpe": s.get("sharpe"), f"{p}_cagr": s.get("cagr"), f"{p}_max_dd": s.get("max_dd"),
                    f"{p}_cost_ann": s.get("cost_ann"), f"{p}_turnover_ann": s.get("turnover_ann")})
    return out


def _run_job(job: tuple) -> dict:
    kind, params, cost_mult = job
    panels = dict(_S["panels"], slippage_bps=_S["slip"] * cost_mult)
    if kind == "xs":
        cfg = MomentumConfig(**{**PRIMARY_STRATEGY, **params})
        w = target_weights(panels["signal_close"], cfg, _S["members"])
        is_start = "2021-03-01"
    else:
        score = taker_flow(_S["taker_log"], window=params["window"])
        el = _S["members"] & score.notna() & panels["exec_close"].notna()
        w = rank_weights(score, el, select_frac=params["select_frac"], long_high=True)
        is_start = "2021-12-01"
    daily = backtest_weights(panels, w, fee=0.0006 * cost_mult, band=params["band"],
                             rebalance_days=params["rebalance_days"])
    return {"kind": kind, "cost_mult": cost_mult, **params, **_periods(daily, is_start)}


def neighbours(grid: dict, params: dict) -> list[dict]:
    out = []
    for k, vals in grid.items():
        i = vals.index(params[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                out.append({**params, k: vals[j]})
    return out


def screen(df: pd.DataFrame, grid: dict, keys: list[str], current: dict) -> pd.DataFrame:
    base = df[(df.cost_mult == 1.0)]
    key = lambda r: tuple(r[k] for k in keys)  # noqa: E731
    table = {key(r): r for _, r in base.iterrows()}
    cur = table[tuple(current[k] for k in keys)]
    x2 = {key(r): r for _, r in df[df.cost_mult == 2.0].iterrows()}
    rows = []
    for k, r in table.items():
        params = dict(zip(keys, k))
        nb = [table[tuple(n[kk] for kk in keys)] for n in neighbours({kk: grid[kk] for kk in keys}, params)
              if tuple(n[kk] for kk in keys) in table]
        nb_is = float(np.nanmedian([n["is_sharpe"] for n in nb])) if nb else np.nan
        nb_oos = float(np.nanmedian([n["oos_sharpe"] for n in nb])) if nb else np.nan
        g_is, g_oos = r["is_sharpe"] - cur["is_sharpe"], r["oos_sharpe"] - cur["oos_sharpe"]
        c1 = g_is >= RULE["sharpe_gain_min"] and g_oos >= RULE["sharpe_gain_min"]
        c2 = r["oos_max_dd"] >= cur["oos_max_dd"] - RULE["dd_worse_max"]
        c3 = (nb_is - cur["is_sharpe"] >= RULE["neighbour_gain_min"]) and (nb_oos - cur["oos_sharpe"] >= RULE["neighbour_gain_min"])
        c4 = k in x2 and x2[k]["is_sharpe"] > cur["is_sharpe"] and x2[k]["oos_sharpe"] > cur["oos_sharpe"]
        rows.append({**params, "is_sharpe": r["is_sharpe"], "oos_sharpe": r["oos_sharpe"], "oos_max_dd": r["oos_max_dd"],
                     "oos_cagr": r["oos_cagr"], "gain_is": g_is, "gain_oos": g_oos, "nb_is": nb_is, "nb_oos": nb_oos,
                     "x2_is": x2[k]["is_sharpe"] if k in x2 else np.nan, "x2_oos": x2[k]["oos_sharpe"] if k in x2 else np.nan,
                     "c1_gain": c1, "c2_dd": c2, "c3_neighbours": c3, "c4_costs_x2": c4, "substantial": c1 and c2 and c3 and c4})
    return pd.DataFrame(rows).sort_values(["substantial", "gain_oos"], ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", default="artifacts/research/universe_top60_v1/membership.csv")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--out-dir", default="artifacts/research/param_exploration_v1")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    t0 = time.perf_counter()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    membership = pd.read_csv(args.membership)
    perps = sorted(set(membership["symbol"]))
    panels = load_daily_panels(perps, cache_dir=args.cache_dir)
    idx = panels["signal_close"].index
    _S.update(panels=panels, members=eligible_from_membership(membership, idx, perps),
              slip=slippage_panel(membership, idx, perps, TIERS),
              taker_log=load_metrics(perps, idx, args.cache_dir)["taker_ratio_log_mean"])
    print(f"data ready ({time.perf_counter() - t0:.0f}s)", flush=True)

    xs_keys = list(XS_GRID)
    xs_jobs = [("xs", {**dict(zip(xs_keys, v)), "band": 0.02}, 1.0) for v in itertools.product(*XS_GRID.values())]
    xs_jobs += [("xs", {**CURRENT["xs_momentum"], "band": b}, 1.0) for b in XS_BANDS if b != 0.02]
    tk_keys = list(TAKER_GRID)
    tk_jobs = [("taker", dict(zip(tk_keys, v)), 1.0) for v in itertools.product(*TAKER_GRID.values())]
    jobs = xs_jobs + tk_jobs
    jobs += [(k, p, 2.0) for k, p, _ in jobs]  # costs x2 for every configuration
    with mp.get_context("fork").Pool(args.workers) as pool:
        rows = []
        for n, r in enumerate(pool.imap_unordered(_run_job, jobs, chunksize=4), 1):
            rows.append(r)
            if n % 50 == 0:
                print(f"{n}/{len(jobs)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "grid.csv", index=False)

    xs = df[(df.kind == "xs") & (df.band == 0.02)]
    tk = df[df.kind == "taker"]
    screens = {"xs_momentum": screen(xs, XS_GRID, xs_keys, CURRENT["xs_momentum"]),
               "xs_frac10": screen(xs, XS_GRID, xs_keys, CURRENT["xs_frac10"]),
               "taker_flow": screen(tk, TAKER_GRID, tk_keys, CURRENT["taker_flow"])}
    report = {"rule": RULE, "oos_start": OOS_START, "end": END, "current": CURRENT,
              "xs_band_sweep": df[(df.kind == "xs") & (df.cost_mult == 1.0) & (df.lookback_days == 30) & (df.skip_days == 1)
                                  & (df.select_frac == 0.2) & (df.rebalance_days == 7) & (df.weighting == "equal")]
                                 [["band", "is_sharpe", "oos_sharpe", "oos_max_dd", "oos_cost_ann"]].sort_values("band").to_dict("records")}
    for name, s in screens.items():
        s.to_csv(out / f"screen_{name}.csv", index=False)
        report[name] = {"n_configs": int(len(s)), "n_substantial": int(s["substantial"].sum()),
                        "share_beating_current_both": float(((s.gain_is > 0) & (s.gain_oos > 0)).mean()),
                        "substantial": s[s["substantial"]].head(10).to_dict("records"),
                        "best_oos_any": s.sort_values("gain_oos", ascending=False).head(5).to_dict("records")}
    report["seconds"] = round(time.perf_counter() - t0, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: {kk: report[k][kk] for kk in ("n_configs", "n_substantial", "share_beating_current_both")}
                      for k in screens}), flush=True)


if __name__ == "__main__":
    main()
