"""Paper trading runner for the cross-sectional taker-flow signal (docs/plans/taker_flow_hypothesis.md).

Same weekly cycle, universe, Bitget execution and accounting as the momentum paper
(scripts/paper_xs_momentum.py); only the target weights differ (hf_research.taker_flow).
No orders are ever sent: only public market data endpoints are used.

Subcommands:
  init    create the paper book (first rebalance = first scheduled fill after now)
  run     one hourly cycle (use --dry-run to print the plan without saving anything)
  report  equity, costs, tracking versus the logged targets replayed on Binance prices, and the
          50/50 combination with the momentum paper (--xs-paper-dir)

Example (cron, every hour at minute 15):
  15 * * * * cd /path/to/hedgefund-arch && PYTHONPATH=src .venv/bin/python scripts/paper_taker_flow.py run
"""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_xs_momentum as xs_runner  # noqa: E402

from hf_research import xs_signal  # noqa: E402
from hf_research.engine import backtest_weights  # noqa: E402
from hf_research.ledger import Store, _iso  # noqa: E402
from hf_research.liquid_universe import bitget_symbol, update_cache  # noqa: E402
from hf_research.paper_tsmom import daily_returns  # noqa: E402
from hf_research.paper_xs import PRIMARY_STRATEGY, PaperBook, PaperConfig, first_decision_day_after, scheduled_fill  # noqa: E402
from hf_research.taker_flow import SELECT_FRAC, WINDOW_DAYS, daily_log_ratio, fetch_rows, scores_for_day, target_weights  # noqa: E402
from hf_research.xs_momentum import load_daily_panels  # noqa: E402

BINANCE = xs_signal.BINANCE
NAME = "taker_flow_v1"
STRATEGY = {"signal": "taker_flow", "window_days": WINDOW_DAYS, "select_frac": SELECT_FRAC, "long_high": True,
            "band": PRIMARY_STRATEGY["band"], "rebalance": "weekly"}
MIN_COVERAGE = 0.5


def compute_targets(decision_day: pd.Timestamp, members: list[str], listed: set[str], cache_dir: str, log) -> dict:
    """Targets (Binance symbols) from the 7 UTC days ending on the decision day; RuntimeError when
    fewer than MIN_COVERAGE of the tradable members have a score (data not ready, API outage)."""
    ex = xs_signal.binance_exchange()
    tradable = [s for s in members if bitget_symbol(s, listed)]
    start = decision_day - pd.Timedelta(days=WINDOW_DAYS - 1)
    end = decision_day + pd.Timedelta(hours=23, minutes=55)
    daily = {}
    for s in tradable:
        try:
            daily[s] = daily_log_ratio(fetch_rows(ex, s, start, end))
        except Exception as exc:  # noqa: BLE001 - one symbol failing must not block the book
            log(f"skip taker {s}: {type(exc).__name__}")
    scores = scores_for_day(pd.DataFrame(daily).reindex(columns=tradable), decision_day)
    coverage = float(scores.notna().mean()) if tradable else 0.0
    if coverage < MIN_COVERAGE:
        raise RuntimeError(f"taker scores for {decision_day.date()} cover {coverage:.0%} of members")
    w = target_weights(scores)
    # Keep the Binance price cache of the traded symbols fresh for the report's model replay.
    since_ms = int((decision_day - pd.Timedelta(days=10)).value // 1_000_000)
    for s in w.index:
        try:
            update_cache(ex, s, "1h", since_ms, cache_dir, BINANCE)
        except Exception as exc:  # noqa: BLE001
            log(f"skip 1h {s}: {type(exc).__name__}")
    return {"weights": {s: float(v) for s, v in w.items()}, "eligible": tradable,
            "not_on_bitget": [s for s in members if s not in tradable], "coverage": coverage,
            "signal_close": {}, "scores": {s: float(v) for s, v in scores.dropna().items()}}


def logged_targets_replay(reb: pd.DataFrame, days: pd.Series, args) -> pd.DataFrame:
    """The weights the paper decided each week, filled at Binance exec prices with backtest costs."""
    syms = sorted({s for t in reb["targets"] for s in t})
    ex = xs_signal.binance_exchange()
    since_ms = int((days.min() - pd.Timedelta(days=2)).value // 1_000_000)
    for s in syms:
        try:
            update_cache(ex, s, "1h", since_ms, args.cache_dir, BINANCE)
        except Exception:  # noqa: BLE001 - the replay then simply lacks that symbol's last bars
            pass
    panels = load_daily_panels(syms, cache_dir=args.cache_dir, exchange=BINANCE)
    targets = pd.DataFrame(0.0, index=panels["exec_close"].index, columns=syms)
    for d, t in zip(days, reb["targets"]):
        if d in targets.index:
            targets.loc[d, list(t)] = list(t.values())
    return backtest_weights(panels, targets, fee=PRIMARY_STRATEGY["fee"], slippage_bps=PRIMARY_STRATEGY["slippage_bps"],
                            band=PRIMARY_STRATEGY["band"], rebalance_on=days)


def _summary(r: pd.Series) -> dict:
    if len(r) < 7:
        return {"days": int(len(r))}
    eq = (1 + r).cumprod()
    return {"days": int(len(r)), "cum": float(eq.iloc[-1] - 1),
            "sharpe_ann": float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else None,
            "max_dd": float((eq / eq.cummax() - 1).min())}


def report(store: Store, args) -> dict:
    out = xs_runner.report(store, args, model_fn=logged_targets_replay)
    state = store.load_state()
    paper = daily_returns(store.read("equity.jsonl"), state["config"]["capital_usdt"], state["config"]["fill_hour_utc"])
    xs_dir = Path(args.xs_paper_dir)
    if (xs_dir / "state.json").exists():
        xs_store = Store(xs_dir)
        xs_state = xs_store.load_state()
        xs = daily_returns(xs_store.read("equity.jsonl"), xs_state["config"]["capital_usdt"], xs_state["config"]["fill_hour_utc"])
        both = pd.concat([xs.rename("xs"), paper.rename("taker")], axis=1, sort=True).dropna()
        out["combination"] = {"xs_alone": _summary(both["xs"]), "taker_alone": _summary(both["taker"]),
                              "combined": _summary(0.5 * both["xs"] + 0.5 * both["taker"]),
                              "daily_corr": float(both["xs"].corr(both["taker"])) if len(both) >= 7 else None}
    out["signal_decision"] = {"rules": "docs/plans/taker_flow_hypothesis.md",
                              "paper_daily": _summary(paper[paper.index >= pd.Timestamp("2026-10-26", tz="UTC")])}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["init", "run", "report"])
    ap.add_argument("--state-dir", default=f"runtime/paper/{NAME}")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--capital", type=float, default=PaperConfig.capital_usdt)
    ap.add_argument("--xs-paper-dir", default=str(Path.home() / "Projects/hedgefund-arch-paper/runtime/paper/xs_momentum_v1"))
    ap.add_argument("--dry-run", action="store_true", help="run: print what would happen, save nothing")
    ap.add_argument("--now", default="", help="override the clock (testing only)")
    args = ap.parse_args()

    store = Store(Path(args.state_dir))
    store.root.mkdir(parents=True, exist_ok=True)

    if args.command == "init":
        if store.path("state.json").exists():
            sys.exit(f"paper book already exists at {store.root}")
        cfg = PaperConfig(name=NAME, capital_usdt=args.capital)
        now = pd.Timestamp.now(tz="UTC")
        store.save_state({"config": asdict(cfg), "strategy": STRATEGY, "start": _iso(now),
                          "book": PaperBook(equity=cfg.capital_usdt).to_dict(), "last_decision_day": None,
                          "last_run_ms": int(now.value // 1_000_000), "funding_cursor": {}})
        d = first_decision_day_after(now, cfg)
        print(f"paper book created at {store.root} with {cfg.capital_usdt:.0f} USDT; "
              f"first rebalance: decision {d.date()} -> fill {_iso(scheduled_fill(d, cfg))}")
        return

    if args.command == "report":
        print(json.dumps(report(store, args), indent=2, default=float))
        return

    lock = open(store.path(".lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another run holds the lock; exiting")
        return
    t0 = time.perf_counter()
    xs_runner.run_cycle(store, args, targets_fn=compute_targets)
    print(f"done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
