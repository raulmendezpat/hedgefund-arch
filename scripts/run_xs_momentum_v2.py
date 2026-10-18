"""Cross-sectional momentum v2: lower turnover and a point-in-time liquid universe.

Pre-registered primary hypothesis (fixed before looking at v2 results, from the v1 study where
the 30-day lookback ranked best in and out of sample):
  long/short, lookback 30d, skip 1d, weekly rebalance, 20% per side, equal weight,
  no-trade band 0.02, universe = point-in-time top 60 by liquidity.

Also runs a compact long/short grid on three universes (the 20 registry symbols, top 30, top 60),
selects one config per universe on the in-sample window with taker costs and reports it blind out
of sample under taker and maker cost assumptions. Maker is an optimistic bound: it assumes every
rebalance fills passively.

Example:
  PYTHONPATH=src python scripts/run_xs_momentum_v2.py \
    --membership artifacts/research/universe_top60_v1/membership.csv \
    --strategy-registry artifacts/asset_universe_ab/rt_registry_base_plus_first_crypto_discovery.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hf_research.xs_momentum import MomentumConfig, backtest, eligible_from_membership, load_daily_panels, summarize

PRIMARY = dict(mode="long_short", lookback_days=30, skip_days=1, rebalance_days=7, select_frac=0.2,
               weighting="equal", band=0.02)
PRIMARY_UNIVERSE = "top60"

GRID = {
    "lookback_days": [14, 30, 60],
    "skip_days": [0, 1],
    "rebalance_days": [7, 14],
    "select_frac": [0.1, 0.2, 0.3],
    "band": [0.0, 0.02, 0.05],
    "weighting": ["equal", "inv_vol"],
}
COSTS = {"taker": dict(fee=0.0006, slippage_bps=3.0), "maker": dict(fee=0.0002, slippage_bps=1.0)}

_PANELS: dict | None = None


def _init(panels):
    global _PANELS
    _PANELS = panels


def _run(job: tuple[str, dict]) -> tuple[str, dict, pd.DataFrame]:
    universe, cfg_dict = job
    panels = dict(_PANELS["base"])
    panels["eligible"] = _PANELS["eligible"][universe]
    return universe, cfg_dict, backtest(panels, MomentumConfig(**cfg_dict))


def _row(universe: str, cfg_dict: dict, daily: pd.DataFrame, is_start: str, oos_start: str) -> dict:
    is_, oos = summarize(daily, is_start, oos_start), summarize(daily, oos_start, None)
    out = {"universe": universe, "key": MomentumConfig(**cfg_dict).key(), **cfg_dict}
    for tag, s in (("is", is_), ("oos", oos)):
        for k in ("sharpe", "cagr", "max_dd", "months_positive", "gross_ann", "cost_ann", "funding_ann", "turnover_ann"):
            out[f"{tag}_{k}"] = s.get(k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", required=True)
    ap.add_argument("--strategy-registry", required=True)
    ap.add_argument("--is-start", default="2021-03-01")
    ap.add_argument("--oos-start", default="2024-01-01")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    t0 = time.perf_counter()
    membership = pd.read_csv(args.membership)
    registry = sorted({str(r["symbol"]) for r in json.loads(Path(args.strategy_registry).read_text()) if r.get("symbol")})
    symbols = sorted(set(membership["symbol"]) | set(registry))
    base = load_daily_panels(symbols)
    idx, cols = base["signal_close"].index, base["signal_close"].columns
    eligible = {
        "registry20": pd.DataFrame({c: c in registry for c in cols}, index=idx),
        "top30": eligible_from_membership(membership[membership["rank"] <= 30], idx, cols),
        "top60": eligible_from_membership(membership[membership["rank"] <= 60], idx, cols),
    }
    shared = {"base": base, "eligible": eligible}

    keys = list(GRID)
    jobs = [
        (u, dict(mode="long_short", **dict(zip(keys, v)), **COSTS["taker"]))
        for u in eligible
        for v in itertools.product(*GRID.values())
    ]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(shared,)) as ex:
        results = list(ex.map(_run, jobs, chunksize=4))
    grid = pd.DataFrame([_row(u, c, d, args.is_start, args.oos_start) for u, c, d in results])

    def rerun(universe: str, cfg_dict: dict) -> dict:
        panels = dict(base, eligible=eligible[universe])
        out = {}
        for cost_name, cost in COSTS.items():
            daily = backtest(panels, MomentumConfig(**{**cfg_dict, **cost}))
            out[cost_name] = {
                "is": summarize(daily, args.is_start, args.oos_start),
                "oos": summarize(daily, args.oos_start, None),
                "oos_sharpe_by_year": {
                    str(y): summarize(daily, f"{y}-01-01", f"{y + 1}-01-01").get("sharpe") for y in range(2024, 2027)
                },
                "is_sharpe_by_year": {
                    str(y): summarize(daily, f"{y}-01-01", f"{y + 1}-01-01").get("sharpe") for y in range(2021, 2024)
                },
            }
        return out

    report = {"primary": {"universe": PRIMARY_UNIVERSE, "config": PRIMARY, **rerun(PRIMARY_UNIVERSE, PRIMARY)}, "by_universe": {}}
    for u, g in grid.groupby("universe"):
        best = g.sort_values("is_sharpe", ascending=False).iloc[0]
        cfg = {"mode": "long_short", **{k: best[k] for k in keys}}
        cfg = {k: (int(v) if k in ("lookback_days", "skip_days", "rebalance_days") else v) for k, v in cfg.items()}
        report["by_universe"][u] = {
            "chosen_in_sample": MomentumConfig(**cfg).key(),
            **rerun(u, cfg),
            "grid_oos_sharpe_median": float(g["oos_sharpe"].median()),
            "grid_oos_sharpe_positive_share": float((g["oos_sharpe"] > 0).mean()),
            "grid_is_oos_rank_corr": float(g["is_sharpe"].rank().corr(g["oos_sharpe"].rank())),
            "grid_oos_gross_ann_median": float(g["oos_gross_ann"].median()),
            "grid_oos_cost_ann_median": float(g["oos_cost_ann"].median()),
        }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir or f"artifacts/research/xs_momentum_v2_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    grid.sort_values(["universe", "is_sharpe"], ascending=[True, False]).to_csv(out_dir / "grid.csv", index=False)
    meta = {
        "created_utc": stamp,
        "membership": args.membership,
        "symbols_loaded": len(symbols),
        "is_start": args.is_start,
        "oos_start": args.oos_start,
        "grid": GRID,
        "costs": COSTS,
        "universe_size_mean_by_year": {
            u: {str(y): float(e.loc[str(y)].sum(axis=1).mean()) for y in range(2021, 2027)} for u, e in eligible.items()
        },
        "report": report,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "report.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    print(json.dumps(meta["report"], indent=2, default=float))
    print(f"saved: {out_dir} ({meta['seconds']}s)")


if __name__ == "__main__":
    main()
