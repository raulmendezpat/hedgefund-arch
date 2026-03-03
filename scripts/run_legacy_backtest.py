#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hf.legacy.ltb.envelope.backtest_portfolio_regime_switch_cached_v2_modefixed_v8ml as bt


def main() -> None:
    p = argparse.ArgumentParser(description="Run legacy portfolio backtest (vendored) from the new repo.")
    p.add_argument("--start", required=True, help='e.g. "2024-09-01 00:00:00"')
    p.add_argument("--end", default=None, help='e.g. "2026-02-28 00:00:00"')
    p.add_argument("--name", default="legacy_run", help="suffix for output files")

    # default to vendored bot paths (no need to pass them every time)
    p.add_argument("--btc-bot", default=str(Path("src/hf/legacy/ltb/envelope/run_btc_trend_1h_v6_prod.py")))
    p.add_argument("--sol-bot", default=str(Path("src/hf/legacy/ltb/envelope/run_sol_bbrsi_1h_v8_prod.py")))

    # allocation-related knobs already supported by legacy backtester
    p.add_argument("--both-btc-weight", type=float, default=0.75)
    p.add_argument("--sol-min-weight-off", type=float, default=0.25)

    # optional passthrough toggles
    p.add_argument("--regime3", action="store_true")
    p.add_argument("--initial", type=float, default=1000.0)

    args, unknown = p.parse_known_args()

    argv = [
        "legacy_backtest",
        "--btc-bot", args.btc_bot,
        "--sol-bot", args.sol_bot,
        "--start", args.start,
        "--name", args.name,
        "--initial", str(args.initial),
        "--both-btc-weight", str(args.both_btc_weight),
        "--sol-min-weight-off", str(args.sol_min_weight_off),
    ]
    if args.end:
        argv += ["--end", args.end]
    if args.regime3:
        argv += ["--regime3"]

    # allow advanced legacy flags without us re-declaring them here
    argv += unknown

    sys.argv = argv
    bt.main()


if __name__ == "__main__":
    main()
