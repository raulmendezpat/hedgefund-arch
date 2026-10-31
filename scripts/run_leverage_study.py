"""Leverage 1x-10x for each strategy, with an hourly margin-account simulation
(registered in docs/plans/leverage_1x_10x.md).

  PYTHONPATH=src python scripts/run_leverage_study.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from hf_research.engine import backtest_weights, summarize
from hf_research.oi_signals import load_metrics, rank_weights, taker_flow
from hf_research.paper_xs import PRIMARY_STRATEGY
from hf_research.tsmom import TsmomConfig, tsmom_targets
from hf_research.vol_target import account_path, intraday_paths, load_hourly_close
from hf_research.xs_momentum import MomentumConfig, eligible_from_membership, load_daily_panels, target_weights

LEVERAGE = list(range(1, 11))
MAINTENANCE, MAINTENANCE_ALT = 0.02, 0.01
KILL_DD = -0.35
ACCEPT = {"max_dd_hourly_min": -0.40, "roll30_worst_min": -0.30}
TIERS = {10: 3.0, 30: 6.0, 60: 12.0}
OOS_START, END = "2024-01-01", "2026-10-19"  # funding and taker metrics caches end here
TSMOM_ASSETS = ["BTC/USDT:USDT", "ETH/USDT:USDT"]


def slippage_panel(membership: pd.DataFrame, index: pd.DatetimeIndex, columns, tiers: dict[int, float]) -> pd.DataFrame:
    worst = max(tiers.values())

    def bps(rank: float) -> float:
        return next((tiers[c] for c in sorted(tiers) if rank <= c), worst)

    by_key = {(m, s): bps(r) for m, s, r in membership[["month", "symbol", "rank"]].itertuples(index=False)}
    months = index.strftime("%Y-%m")
    return pd.DataFrame({c: [by_key.get((m, c), worst) for m in months] for c in columns}, index=index)


def strategies(membership: pd.DataFrame, cache_dir: str) -> dict:
    """name -> dict(panels, targets, band, rebalance_days, slippage_bps, is_start)."""
    perps = sorted(set(membership["symbol"]))
    panels = load_daily_panels(perps, cache_dir=cache_dir)
    idx = panels["signal_close"].index
    members = eligible_from_membership(membership, idx, perps)
    panels["slippage_bps"] = slippage_panel(membership, idx, perps, TIERS)
    xs = MomentumConfig(**PRIMARY_STRATEGY)
    frac10 = MomentumConfig(**{**PRIMARY_STRATEGY, "select_frac": 0.1})
    score = taker_flow(load_metrics(perps, idx, cache_dir)["taker_ratio_log_mean"])
    taker_el = members & score.notna() & panels["exec_close"].notna()
    ts_cfg = TsmomConfig()
    ts_panels = load_daily_panels(TSMOM_ASSETS, cache_dir=cache_dir)
    return {
        "xs_momentum": dict(panels=panels, targets=target_weights(panels["signal_close"], xs, members),
                            band=xs.band, rebalance_days=7, is_start="2021-03-01"),
        "xs_frac10": dict(panels=panels, targets=target_weights(panels["signal_close"], frac10, members),
                          band=frac10.band, rebalance_days=7, is_start="2021-03-01"),
        "taker_flow": dict(panels=panels, targets=rank_weights(score, taker_el, select_frac=0.2, long_high=True),
                           band=0.02, rebalance_days=7, is_start="2021-12-01"),
        "tsmom": dict(panels=ts_panels, targets=tsmom_targets(ts_panels["signal_close"], ts_cfg),
                      band=ts_cfg.band, rebalance_days=1, is_start="2020-04-01", slippage_bps=ts_cfg.slippage_bps),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", default="artifacts/research/universe_top60_v1/membership.csv")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--out-dir", default="artifacts/research/leverage_v1")
    args = ap.parse_args()
    t0 = time.perf_counter()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    membership = pd.read_csv(args.membership)
    strats = strategies(membership, args.cache_dir)
    hourly = load_hourly_close(sorted(set(membership["symbol"]) | set(TSMOM_ASSETS)), args.cache_dir)
    print(f"data ready ({time.perf_counter() - t0:.0f}s)", flush=True)
    ts = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731

    rows = []
    for name, s in strats.items():
        spans = {"is": (s["is_start"], OOS_START), "oos": (OOS_START, END), "full": (s["is_start"], END)}
        for lev in LEVERAGE:
            daily, parts = backtest_weights(s["panels"], s["targets"] * lev, fee=0.0006,
                                            slippage_bps=s.get("slippage_bps", 3.0), band=s["band"] * lev,
                                            rebalance_days=s["rebalance_days"], detail=True)
            ret, notional = intraday_paths(parts["weight"], hourly)
            for p, (a, b) in spans.items():
                m = (daily.index >= ts(a)) & (daily.index < ts(b))
                acct = account_path(daily.loc[m, "net"], ret[m], notional[m], maintenance=MAINTENANCE, kill_dd=KILL_DD)
                alt = account_path(daily.loc[m, "net"], ret[m], notional[m], maintenance=MAINTENANCE_ALT, kill_dd=KILL_DD)
                summ = summarize(daily, a, b)
                ok = (acct["liquidated"] is None and acct["max_dd_hourly"] > ACCEPT["max_dd_hourly_min"]
                      and acct["roll30"]["worst"] > ACCEPT["roll30_worst_min"])
                rows.append({"strategy": name, "leverage": lev, "period": p, "cagr": summ.get("cagr"),
                             "sharpe": summ.get("sharpe"), "acceptable": ok,
                             "liquidated_at_1pct_maintenance": alt["liquidated"], **acct})
            print(f"{name} x{lev} done ({time.perf_counter() - t0:.0f}s)", flush=True)

    df = pd.json_normalize(rows)
    df.to_csv(out / "leverage.csv", index=False)
    accept = {n: {lev: bool(df[(df.strategy == n) & (df.leverage == lev) & df.period.isin(["is", "oos"])]["acceptable"].all())
                  for lev in LEVERAGE} for n in strats}
    report = {"leverage": LEVERAGE, "maintenance": MAINTENANCE, "kill_dd": KILL_DD, "accept_rule": ACCEPT,
              "oos_start": OOS_START, "end": END, "acceptable": accept,
              "max_acceptable": {n: max([lev for lev, ok in a.items() if ok], default=0) for n, a in accept.items()},
              "seconds": round(time.perf_counter() - t0, 1)}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["max_acceptable"]), flush=True)


if __name__ == "__main__":
    main()
