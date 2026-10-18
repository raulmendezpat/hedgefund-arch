"""Cross-sectional momentum study: parameter grid, in-sample selection, blind out-of-sample check.

Every config is backtested on the full period, but configs are *chosen* on the in-sample window
only; the out-of-sample window is reported for the chosen config and for the whole grid (the
share of configs that stay profitable is a better robustness read than the single winner).

Caveat: the universe is today's 20 registry symbols (chosen in 2026), so long-only results
carry survivorship bias; the long/short book is much less exposed to it.

Example:
  PYTHONPATH=src python scripts/run_xs_momentum_study.py \
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

from hf_research.xs_momentum import MomentumConfig, backtest, load_daily_panels, summarize

GRID = {
    "mode": ["long_short", "long_only"],
    "lookback_days": [3, 7, 14, 30, 60],
    "skip_days": [0, 1],
    "rebalance_days": [1, 3, 7],
    "weighting": ["equal", "inv_vol"],
    "select_frac": [0.2, 0.3],
}

_PANELS: dict | None = None


def _init(panels):
    global _PANELS
    _PANELS = panels


def _run(cfg_dict: dict) -> tuple[dict, pd.DataFrame]:
    return cfg_dict, backtest(_PANELS, MomentumConfig(**cfg_dict))


def _benchmarks(panels: dict, start: str, end: str | None) -> dict:
    exe = panels["exec_close"]
    r = (exe / exe.shift(1) - 1.0)
    btc = pd.DataFrame({"net": r["BTC/USDT:USDT"].fillna(0.0), "gross": 0.0, "funding": 0.0, "cost": 0.0,
                        "turnover": 0.0, "gross_exposure": 1.0})
    ew_ret = r.mean(axis=1, skipna=True).fillna(0.0) - panels["funding"].where(r.notna()).mean(axis=1).fillna(0.0)
    ew = btc.assign(net=ew_ret)
    return {"btc_buy_hold": summarize(btc, start, end), "equal_weight_long_all": summarize(ew, start, end)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy-registry", required=True)
    ap.add_argument("--is-start", default="2021-03-01")
    ap.add_argument("--oos-start", default="2024-01-01")
    ap.add_argument("--slippage-bps", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    t0 = time.perf_counter()
    rows = json.loads(Path(args.strategy_registry).read_text(encoding="utf-8"))
    symbols = sorted({str(r["symbol"]) for r in rows if r.get("symbol")})
    panels = load_daily_panels(symbols)

    keys = list(GRID)
    cfgs = [dict(zip(keys, v), slippage_bps=args.slippage_bps) for v in itertools.product(*GRID.values())]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(panels,)) as ex:
        results = list(ex.map(_run, cfgs, chunksize=4))

    table = []
    for cfg_dict, daily in results:
        cfg = MomentumConfig(**cfg_dict)
        is_ = summarize(daily, args.is_start, args.oos_start)
        oos = summarize(daily, args.oos_start, None)
        table.append({"key": cfg.key(), **cfg_dict,
                      **{f"is_{k}": v for k, v in is_.items() if k in ("sharpe", "cagr", "max_dd", "months_positive")},
                      **{f"oos_{k}": v for k, v in oos.items() if k in ("sharpe", "cagr", "max_dd", "months_positive")}})
    grid = pd.DataFrame(table)

    report = {"benchmarks_oos": _benchmarks(panels, args.oos_start, None),
              "benchmarks_is": _benchmarks(panels, args.is_start, args.oos_start), "by_mode": {}}
    for mode, g in grid.groupby("mode"):
        best = g.sort_values("is_sharpe", ascending=False).iloc[0]
        cfg = MomentumConfig(**{k: best[k] for k in keys}, slippage_bps=args.slippage_bps)
        daily = backtest(panels, cfg)
        by_year = {
            str(y): summarize(daily, f"{y}-01-01", f"{y + 1}-01-01").get("sharpe")
            for y in range(2021, 2027)
        }
        report["by_mode"][mode] = {
            "chosen_in_sample": cfg.key(),
            "chosen_is": summarize(daily, args.is_start, args.oos_start),
            "chosen_oos": summarize(daily, args.oos_start, None),
            "chosen_sharpe_by_year": by_year,
            "grid_oos_sharpe_median": float(g["oos_sharpe"].median()),
            "grid_oos_sharpe_positive_share": float((g["oos_sharpe"] > 0).mean()),
            "grid_is_sharpe_median": float(g["is_sharpe"].median()),
            "grid_is_oos_rank_corr": float(g["is_sharpe"].rank().corr(g["oos_sharpe"].rank())),
            "top5_is_configs_oos_sharpe": g.sort_values("is_sharpe", ascending=False).head(5)[["key", "is_sharpe", "oos_sharpe"]].to_dict("records"),
        }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir or f"artifacts/research/xs_momentum_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    grid.sort_values("is_sharpe", ascending=False).to_csv(out_dir / "grid.csv", index=False)
    meta = {"created_utc": stamp, "symbols": symbols, "is_start": args.is_start, "oos_start": args.oos_start,
            "slippage_bps": args.slippage_bps, "grid": GRID, "report": report,
            "universe_size_by_year": {
                str(y): int(panels["signal_close"].loc[f"{y}"].notna().any().sum()) for y in range(2021, 2027)
            },
            "seconds": round(time.perf_counter() - t0, 1)}
    (out_dir / "report.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    print(json.dumps(meta["report"], indent=2, default=float))
    print(f"saved: {out_dir} ({meta['seconds']}s)")


if __name__ == "__main__":
    main()
