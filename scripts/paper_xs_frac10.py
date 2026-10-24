"""Paper trading runner for the 10%-per-side momentum variant (docs/plans/xs_frac10_hypothesis.md).

Same weekly cycle, universe, signal, Bitget execution and accounting as the momentum paper
(scripts/paper_xs_momentum.py); only the fraction per side differs (0.1 instead of 0.2).
No orders are ever sent: only public market data endpoints are used.

Subcommands:
  init    create the paper book (first rebalance = first scheduled fill after now)
  run     one hourly cycle (use --dry-run to print the plan without saving anything)
  report  equity, costs, tracking versus the model replay, and the comparison with the primary
          momentum paper on the same days (--xs-paper-dir)

Example (cron, every hour at minute 25):
  25 * * * * cd /path/to/hedgefund-arch && PYTHONPATH=src .venv/bin/python scripts/paper_xs_frac10.py run
"""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_xs_momentum as xs_runner  # noqa: E402

from hf_research import xs_signal  # noqa: E402
from hf_research.ledger import Store, _iso  # noqa: E402
from hf_research.paper_tsmom import daily_returns  # noqa: E402
from hf_research.paper_xs import PRIMARY_STRATEGY, PaperBook, PaperConfig, first_decision_day_after, scheduled_fill  # noqa: E402

NAME = "xs_frac10_v1"
STRATEGY = {**PRIMARY_STRATEGY, "select_frac": 0.1}
FIRST_FILL_DAY = pd.Timestamp("2026-10-26", tz="UTC")

compute_targets = partial(xs_signal.compute_targets, strategy=STRATEGY)
model_replay = partial(xs_runner._momentum_replay, strategy=STRATEGY)


def _summary(r: pd.Series) -> dict:
    if len(r) < 7:
        return {"days": int(len(r))}
    eq = (1 + r).cumprod()
    return {"days": int(len(r)), "cum": float(eq.iloc[-1] - 1),
            "sharpe_ann": float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else None,
            "max_dd": float((eq / eq.cummax() - 1).min())}


def report(store: Store, args) -> dict:
    out = xs_runner.report(store, args, model_fn=model_replay)
    state = store.load_state()
    paper = daily_returns(store.read("equity.jsonl"), state["config"]["capital_usdt"], state["config"]["fill_hour_utc"])
    paper = paper[paper.index >= FIRST_FILL_DAY]
    decision = {"rules": "docs/plans/xs_frac10_hypothesis.md", "variant": _summary(paper)}
    xs_dir = Path(args.xs_paper_dir)
    if (xs_dir / "state.json").exists():
        xs_store = Store(xs_dir)
        xs_state = xs_store.load_state()
        xs = daily_returns(xs_store.read("equity.jsonl"), xs_state["config"]["capital_usdt"], xs_state["config"]["fill_hour_utc"])
        both = pd.concat([xs.rename("primary"), paper.rename("variant")], axis=1, sort=True).dropna()
        v, p = _summary(both["variant"]), _summary(both["primary"])
        decision.update(same_days={"variant": v, "primary": p,
                                   "sharpe_minus_primary": (v["sharpe_ann"] - p["sharpe_ann"])
                                   if v.get("sharpe_ann") is not None and p.get("sharpe_ann") is not None else None,
                                   "diff_ann": float((both["variant"] - both["primary"]).mean() * 365) if len(both) else None,
                                   "daily_corr": float(both["variant"].corr(both["primary"])) if len(both) >= 7 else None})
    out["signal_decision"] = decision
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
