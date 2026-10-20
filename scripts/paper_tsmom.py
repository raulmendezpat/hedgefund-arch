"""Paper trading runner for time-series momentum on BTC/ETH (see hf_research.paper_tsmom).

Run it every hour (idempotent, takes a lock). Each run:
  1. books the Bitget funding events since the previous run,
  2. marks positions to the Bitget mid and appends an equity row,
  3. once per day, refreshes the Binance 1h cache of BTC/ETH, computes the target weights and
     fills the orders against the Bitget order book.

No orders are ever sent: only public market data endpoints are used.

Subcommands:
  init    create the paper book (first rebalance = first scheduled fill after now)
  run     one hourly cycle (use --dry-run to print the plan without saving anything)
  report  equity, costs, tracking versus the model replay, and the 50/50 combination with the
          cross-sectional momentum paper (--xs-paper-dir)

Example (cron, every hour at minute 10):
  10 * * * * cd /path/to/hedgefund-arch && PYTHONPATH=src .venv/bin/python scripts/paper_tsmom.py run
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

from hf_research import xs_signal
from hf_research.engine import backtest_weights
from hf_research.ledger import Store, _iso
from hf_research.liquid_universe import bitget_symbol, update_cache
from hf_research.paper_tsmom import ASSETS, COMBO_TSMOM_SCALE, TsmomPaperConfig, daily_returns, due_daily_decision
from hf_research.paper_xs import PaperBook, plan_weights, walk_book
from hf_research.tsmom import TsmomConfig, tsmom_targets
from hf_research.xs_momentum import load_daily_panels

BINANCE = xs_signal.BINANCE
STRATEGY = TsmomConfig()


def _bitget():
    import ccxt

    ex = ccxt.bitget({"enableRateLimit": True, "timeout": 20000, "options": {"defaultType": "swap"}})
    markets = ex.load_markets()
    listed = {k for k, v in markets.items()
              if v.get("swap") and v.get("linear") and v.get("settle") == "USDT" and v.get("active", True)}
    return ex, listed


def _mids(ex, symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    out = {}
    for s, t in ex.fetch_tickers(symbols).items():
        bid, ask, last = t.get("bid"), t.get("ask"), t.get("last")
        out[s] = (bid + ask) / 2.0 if bid and ask else float(last)
    return out


def compute_targets(decision_day: pd.Timestamp, cfg: TsmomPaperConfig, cache_dir: str, log) -> dict:
    """Target weights (Binance symbols) for the decision day; RuntimeError if a signal close is missing."""
    ex = xs_signal.binance_exchange()
    since_ms = int((decision_day - pd.Timedelta(days=cfg.history_days)).value // 1_000_000)
    for s in ASSETS:
        update_cache(ex, s, "1h", since_ms, cache_dir, BINANCE)
    sig = load_daily_panels(list(ASSETS), cache_dir=cache_dir, exchange=BINANCE)["signal_close"]
    if decision_day not in sig.index or sig.loc[decision_day].isna().any():
        raise RuntimeError(f"signal closes for {decision_day.date()} not available yet")
    w = tsmom_targets(sig, STRATEGY).loc[decision_day]
    return {"weights": {s: float(v) for s, v in w.items() if v != 0.0},
            "signal_close": {s: float(sig.loc[decision_day, s]) for s in ASSETS}}


def run_cycle(store: Store, args) -> None:
    state = store.load_state()
    cfg = TsmomPaperConfig(**state["config"])
    book = PaperBook.from_dict(state["book"])
    now = pd.Timestamp.now(tz="UTC") if not args.now else pd.Timestamp(args.now, tz="UTC")
    events, fills, funding_rows = [], [], []

    def log(msg: str) -> None:
        print(f"[{_iso(now)}] {msg}", flush=True)

    ex, listed = _bitget()
    held = sorted(book.qty)
    mids = _mids(ex, held)

    # 1) Funding events since the previous run, at the current mid (runs are hourly).
    cursor = state.get("funding_cursor", {})
    for s in held:
        since = int(cursor.get(s, state["last_run_ms"])) + 1
        try:
            hist = ex.fetch_funding_rate_history(s, since=since, limit=100)
        except Exception as exc:  # noqa: BLE001
            log(f"funding fetch failed {s}: {type(exc).__name__}")
            continue
        for h in sorted(hist, key=lambda x: x["timestamp"]):
            ts = int(h["timestamp"])
            if ts < since or ts > now.value // 1_000_000:
                continue
            px = mids.get(s, book.last_px[s])
            pnl = book.funding(s, float(h["fundingRate"]), px)
            funding_rows.append({"ts": _iso(pd.Timestamp(ts, unit="ms", tz="UTC")), "symbol": s,
                                 "rate": float(h["fundingRate"]), "qty": book.qty[s], "price": px, "pnl_usdt": pnl})
            cursor[s] = ts

    # 2) Mark to mid.
    book.mark(mids)
    gross = sum(abs(w) for w in book.weights().values()) if book.qty else 0.0

    # 3) Daily rebalance.
    last_dd = pd.Timestamp(state["last_decision_day"], tz="UTC") if state.get("last_decision_day") else None
    d, status, skipped = due_daily_decision(now, pd.Timestamp(state["start"]), last_dd, cfg)
    for k in skipped:
        events.append({"ts": _iso(now), "type": "missed_rebalance", "decision_day": str(k.date())})
        log(f"missed rebalance for {k.date()}")
    if status == "missed":
        events.append({"ts": _iso(now), "type": "missed_rebalance", "decision_day": str(d.date())})
        log(f"missed rebalance for {d.date()}")
        state["last_decision_day"] = str(d.date())
    elif status == "due":
        try:
            tgt = compute_targets(d, cfg, args.cache_dir, log)
        except Exception as exc:  # noqa: BLE001 - retried next hour until the fill window closes
            log(f"rebalance postponed: {type(exc).__name__}: {exc}")
            tgt = None
        if tgt is not None:
            to_bg = {s: bitget_symbol(s, listed) for s in ASSETS}
            targets = {to_bg[s]: w for s, w in tgt["weights"].items()}
            plan = plan_weights(targets, book.weights(), STRATEGY.band)
            equity0 = book.equity
            for s, (w_cur, w_new) in plan.items():
                ob = ex.fetch_order_book(s, limit=100)
                bid, ask = float(ob["bids"][0][0]), float(ob["asks"][0][0])
                mid = (bid + ask) / 2.0
                if w_new == 0.0:
                    dq = -book.qty.get(s, 0.0)
                else:
                    dq_raw = w_new * equity0 / mid - book.qty.get(s, 0.0)
                    try:
                        dq = float(ex.amount_to_precision(s, abs(dq_raw))) * float(np.sign(dq_raw))
                    except Exception:  # noqa: BLE001 - below the minimum amount precision
                        dq = 0.0
                    if abs(dq) * mid < cfg.min_order_usdt:
                        events.append({"ts": _iso(now), "type": "order_skipped_min_size", "symbol": s, "dq": dq_raw})
                        continue
                avg, _visible, exhausted = walk_book(ob["asks"] if dq > 0 else ob["bids"], abs(dq))
                r = book.trade(s, dq, avg, mid, cfg.taker_fee)
                fills.append({"ts": _iso(now), "decision_day": str(d.date()), "symbol": s, "w_from": w_cur, "w_to": w_new,
                              "qty": dq, "mid": mid, "avg_px": avg, "spread_bps": (ask / bid - 1) * 1e4,
                              "notional_usdt": abs(dq) * avg, "book_exhausted": exhausted, **r})
            events.append({"ts": _iso(now), "type": "rebalance", "decision_day": str(d.date()),
                           "delay_hours": (now - d - pd.Timedelta(days=1, hours=cfg.fill_hour_utc)).total_seconds() / 3600.0,
                           "equity_before": equity0, "equity_after": book.equity, "n_orders": len(plan),
                           "targets": tgt["weights"], "signal_close": tgt["signal_close"]})
            state["last_decision_day"] = str(d.date())
            gross = sum(abs(w) for w in book.weights().values())
            log(f"rebalanced {d.date()}: {len(plan)} orders, equity {equity0:.2f} -> {book.equity:.2f}")

    equity_row = {"ts": _iso(now), "equity": book.equity, "gross_exposure": gross, "n_positions": len(book.qty),
                  "weights": book.weights()}
    if args.dry_run:
        print(json.dumps({"equity": equity_row, "events": events, "fills": fills, "funding": funding_rows},
                         indent=2, default=float))
        return
    store.append("equity.jsonl", [equity_row])
    store.append("events.jsonl", events)
    store.append("fills.jsonl", fills)
    store.append("funding.jsonl", funding_rows)
    state.update(book=book.to_dict(), funding_cursor={s: v for s, v in cursor.items() if s in book.qty},
                 last_run_ms=int(now.value // 1_000_000), last_run=_iso(now))
    for s in book.qty:
        state["funding_cursor"].setdefault(s, int(now.value // 1_000_000))
    store.save_state(state)
    log(f"equity {book.equity:.2f} positions {len(book.qty)} gross {gross:.2f}")


PREREGISTERED_CHECKS = {
    "min_rebalances": 70,
    "max_missed_rebalances": 3,
    "slippage_bps_median_max": 3.0,
    "paper_vs_model_daily_corr_min": 0.9,
    "implementation_gap_ann_min": -0.05,
}


def _summary(r: pd.Series) -> dict:
    if len(r) < 7:
        return {"days": int(len(r))}
    eq = (1 + r).cumprod()
    return {"days": int(len(r)), "cum": float(eq.iloc[-1] - 1),
            "sharpe_ann": float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else None,
            "max_dd": float((eq / eq.cummax() - 1).min())}


def report(store: Store, args) -> dict:
    state = store.load_state()
    cfg = TsmomPaperConfig(**state["config"])
    eq, ev, fills, fund = (store.read(n) for n in ("equity.jsonl", "events.jsonl", "fills.jsonl", "funding.jsonl"))
    out: dict = {"name": cfg.name, "start": state["start"], "last_run": state.get("last_run"),
                 "equity": state["book"]["equity"], "capital": cfg.capital_usdt,
                 "total_return": state["book"]["equity"] / cfg.capital_usdt - 1.0}
    reb = ev[ev["type"] == "rebalance"] if len(ev) else pd.DataFrame()
    out["rebalances"] = int(len(reb))
    out["missed_rebalances"] = int((ev["type"] == "missed_rebalance").sum()) if len(ev) else 0
    paper = daily_returns(eq, cfg.capital_usdt, cfg.fill_hour_utc)
    out["paper_daily"] = _summary(paper)
    if len(fills):
        out["fills"] = {"n": int(len(fills)), "turnover_usdt": float(fills["notional_usdt"].sum()),
                        "fees_usdt": float(fills["fee_usdt"].sum()), "slippage_usdt": float(fills["slippage_usdt"].sum()),
                        "slippage_bps_median": float(fills["slippage_bps"].median()),
                        "book_exhausted": int(fills["book_exhausted"].sum())}
    if len(fund):
        out["funding_usdt"] = float(fund["pnl_usdt"].sum())

    # Model replay on Binance data with the same decision days.
    if len(reb) and len(paper) >= 2:
        days = pd.DatetimeIndex(pd.to_datetime(reb["decision_day"])).tz_localize("UTC")
        panels = load_daily_panels(list(ASSETS), cache_dir=args.cache_dir, exchange=BINANCE)
        model = backtest_weights(panels, tsmom_targets(panels["signal_close"], STRATEGY), fee=STRATEGY.fee,
                                 slippage_bps=STRATEGY.slippage_bps, band=STRATEGY.band, rebalance_on=days)
        model = model[model.index >= days.min()]["net"]
        both = pd.concat([paper.rename("paper"), model.rename("model")], axis=1, sort=True).dropna()
        if len(both) >= 2:
            out["vs_model"] = {"days": int(len(both)), "paper_cum": float((1 + both["paper"]).prod() - 1),
                               "model_cum": float((1 + both["model"]).prod() - 1),
                               "daily_corr": float(both["paper"].corr(both["model"])) if len(both) >= 5 else None,
                               "implementation_gap_ann": float((both["paper"] - both["model"]).mean() * 365)}

    # 50/50 combination with the cross-sectional momentum paper, same days.
    xs_dir = Path(args.xs_paper_dir)
    if (xs_dir / "state.json").exists():
        xs_store = Store(xs_dir)
        xs_state = xs_store.load_state()
        xs = daily_returns(xs_store.read("equity.jsonl"), xs_state["config"]["capital_usdt"],
                           xs_state["config"]["fill_hour_utc"])
        both = pd.concat([xs.rename("xs"), paper.rename("tsmom")], axis=1, sort=True).dropna()
        combined = 0.5 * both["xs"] + 0.5 * COMBO_TSMOM_SCALE * both["tsmom"]
        out["combination"] = {"xs_paper_dir": str(xs_dir), "tsmom_scale": COMBO_TSMOM_SCALE,
                              "xs_alone": _summary(both["xs"]), "tsmom_alone": _summary(both["tsmom"]),
                              "combined": _summary(combined),
                              "daily_corr": float(both["xs"].corr(both["tsmom"])) if len(both) >= 7 else None}

    c = PREREGISTERED_CHECKS
    verdict = {}
    if out["rebalances"] >= c["min_rebalances"]:
        vm = out.get("vs_model", {})
        verdict = {"missed_ok": out["missed_rebalances"] <= c["max_missed_rebalances"],
                   "slippage_ok": out.get("fills", {}).get("slippage_bps_median", 99.0) <= c["slippage_bps_median_max"],
                   "tracking_ok": (vm.get("daily_corr") or 0) >= c["paper_vs_model_daily_corr_min"],
                   "gap_ok": vm.get("implementation_gap_ann", -1) >= c["implementation_gap_ann_min"]}
    out["preregistered_checks"] = {"criteria": c, "verdict": verdict or "pending: not enough rebalances"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["init", "run", "report"])
    ap.add_argument("--state-dir", default="runtime/paper/tsmom_btc_eth_v1")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--capital", type=float, default=TsmomPaperConfig.capital_usdt)
    ap.add_argument("--xs-paper-dir", default=str(Path.home() / "Projects/hedgefund-arch-paper/runtime/paper/xs_momentum_v1"))
    ap.add_argument("--dry-run", action="store_true", help="run: print what would happen, save nothing")
    ap.add_argument("--now", default="", help="override the clock (testing only)")
    args = ap.parse_args()

    store = Store(Path(args.state_dir))
    store.root.mkdir(parents=True, exist_ok=True)

    if args.command == "init":
        if store.path("state.json").exists():
            sys.exit(f"paper book already exists at {store.root}")
        cfg = TsmomPaperConfig(capital_usdt=args.capital)
        now = pd.Timestamp.now(tz="UTC")
        state = {"config": asdict(cfg), "strategy": STRATEGY.to_dict(), "start": _iso(now),
                 "book": PaperBook(equity=cfg.capital_usdt).to_dict(), "last_decision_day": None,
                 "last_run_ms": int(now.value // 1_000_000), "funding_cursor": {}}
        store.save_state(state)
        print(f"paper book created at {store.root} with {cfg.capital_usdt:.0f} USDT; first fill: next 01:00 UTC after now")
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
    run_cycle(store, args)
    print(f"done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
