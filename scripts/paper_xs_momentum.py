"""Paper trading runner for the cross-sectional momentum hypothesis (see hf_research.paper_xs).

Run it every hour (it is idempotent and takes a lock). Each run:
  1. closes held positions whose Bitget market disappeared (at the last mark),
  2. books the Bitget funding events since the previous run,
  3. marks positions to the Bitget mid and appends an equity row,
  4. on the weekly schedule, rebuilds the universe if the month changed, refreshes the Binance
     1h cache, computes the target weights and fills the orders against the Bitget order book.

No orders are ever sent: only public market data endpoints are used.

Subcommands:
  init    create the paper book (first rebalance = first scheduled fill after now)
  run     one hourly cycle (use --dry-run to print the plan without saving anything)
  report  equity, costs, funding and tracking versus the model replay

Example (cron, every hour at minute 5):
  5 * * * * cd /path/to/hedgefund-arch && PYTHONPATH=src .venv/bin/python scripts/paper_xs_momentum.py run
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from hf_research.liquid_universe import bitget_symbol, crypto_perpetuals, monthly_membership, update_cache
from hf_research.paper_xs import (
    PRIMARY_STRATEGY,
    PaperBook,
    PaperConfig,
    due_decision_day,
    first_decision_day_after,
    plan_weights,
    scheduled_fill,
    walk_book,
)
from hf_research.xs_momentum import MomentumConfig, backtest, load_daily_panels, target_weights

BINANCE = "binanceusdm"


# ---------------------------------------------------------------- storage

class Store:
    def __init__(self, root: Path):
        self.root = root

    def path(self, name: str) -> Path:
        return self.root / name

    def load_state(self) -> dict:
        p = self.path("state.json")
        if not p.exists():
            sys.exit(f"no paper book at {self.root}; run `init` first")
        return json.loads(p.read_text())

    def save_state(self, state: dict) -> None:
        tmp = self.path("state.json.tmp")
        tmp.write_text(json.dumps(state, indent=2, default=float))
        os.replace(tmp, self.path("state.json"))

    def append(self, name: str, rows: list[dict]) -> None:
        if not rows:
            return
        with self.path(name).open("a") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")

    def read(self, name: str) -> pd.DataFrame:
        p = self.path(name)
        if not p.exists():
            return pd.DataFrame()
        return pd.DataFrame([json.loads(x) for x in p.read_text().splitlines() if x.strip()])


def _iso(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- exchanges

def _bitget():
    import ccxt

    ex = ccxt.bitget({"enableRateLimit": True, "timeout": 20000, "options": {"defaultType": "swap"}})
    markets = ex.load_markets()
    listed = {k for k, v in markets.items()
              if v.get("swap") and v.get("linear") and v.get("settle") == "USDT" and v.get("active", True)}
    return ex, listed


def _binance():
    import ccxt

    return ccxt.binanceusdm({"enableRateLimit": True, "timeout": 20000})


def _mids(ex, symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    out = {}
    for s, t in ex.fetch_tickers(symbols).items():
        bid, ask, last = t.get("bid"), t.get("ask"), t.get("last")
        out[s] = (bid + ask) / 2.0 if bid and ask else float(last)
    return out


# ---------------------------------------------------------------- universe and signal

def ensure_membership(store: Store, month: str, cfg: PaperConfig, cache_dir: str, log) -> pd.DataFrame:
    """Membership of `month`, computed from Binance daily candles strictly before the month."""
    path = store.path("membership.csv")
    have = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["month", "symbol", "rank"])
    if (have["month"] == month).any():
        return have[have["month"] == month]
    ex = _binance()
    month_start = pd.Timestamp(f"{month}-01", tz="UTC")
    since_ms = int((month_start - pd.Timedelta(days=cfg.min_history_days + cfg.window_days + 60)).value // 1_000_000)
    daily = {}
    markets = crypto_perpetuals(ex)
    for k, m in enumerate(markets, start=1):
        try:
            daily[m["symbol"]] = update_cache(ex, m["symbol"], "1d", since_ms, cache_dir, BINANCE)
        except Exception as exc:  # noqa: BLE001 - some delisted contracts reject history requests
            log(f"skip 1d {m['symbol']}: {type(exc).__name__}")
        if k % 100 == 0:
            log(f"daily candles {k}/{len(markets)}")
    m_new = monthly_membership(daily, top_n=cfg.top_n, window_days=cfg.window_days,
                               min_history_days=cfg.min_history_days, months=[month])
    if m_new.empty:
        raise RuntimeError(f"could not build membership for {month}")
    pd.concat([have, m_new], ignore_index=True).to_csv(path, index=False)
    log(f"membership {month}: {len(m_new)} symbols")
    return m_new


def compute_targets(decision_day: pd.Timestamp, members: list[str], listed: set[str], cache_dir: str, log) -> dict:
    """Target weights (Binance symbols) for the decision day plus the data used to build them."""
    ex = _binance()
    since_ms = int((decision_day - pd.Timedelta(days=120)).value // 1_000_000)
    for s in members:
        try:
            update_cache(ex, s, "1h", since_ms, cache_dir, BINANCE)
        except Exception as exc:  # noqa: BLE001
            log(f"skip 1h {s}: {type(exc).__name__}")
    tradable = [s for s in members if bitget_symbol(s, listed)]
    panels = load_daily_panels(tradable, cache_dir=cache_dir, exchange=BINANCE)
    sig = panels["signal_close"]
    if decision_day not in sig.index or sig.loc[decision_day].notna().sum() < len(tradable) // 2:
        raise RuntimeError(f"signal closes for {decision_day.date()} are not available yet")
    eligible = pd.DataFrame(True, index=sig.index, columns=sig.columns)
    w = target_weights(sig, MomentumConfig(**PRIMARY_STRATEGY), eligible).loc[decision_day]
    w = w[w != 0.0]
    return {"weights": {s: float(v) for s, v in w.items()},
            "eligible": tradable,
            "not_on_bitget": [s for s in members if s not in tradable],
            "signal_close": {s: float(sig.loc[decision_day, s]) for s in w.index}}


# ---------------------------------------------------------------- run cycle

def run_cycle(store: Store, args) -> None:
    state = store.load_state()
    cfg = PaperConfig(**state["config"])
    book = PaperBook.from_dict(state["book"])
    now = pd.Timestamp.now(tz="UTC") if not args.now else pd.Timestamp(args.now, tz="UTC")
    events, fills, funding_rows = [], [], []

    def log(msg: str) -> None:
        print(f"[{_iso(now)}] {msg}", flush=True)

    ex, listed = _bitget()

    # 1) Held markets that disappeared are closed at the last mark.
    for s in [s for s in book.qty if s not in listed]:
        q, px = book.qty[s], book.last_px[s]
        r = book.trade(s, -q, px, px, cfg.taker_fee)
        events.append({"ts": _iso(now), "type": "delisted_close", "symbol": s, "qty": -q, "price": px, **r})
        log(f"closed delisted {s}")

    held = sorted(book.qty)
    mids = _mids(ex, held)

    # 2) Funding events since the previous run, at the current mid (runs are hourly).
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

    # 3) Mark to mid.
    book.mark(mids)
    gross = sum(abs(w) for w in book.weights().values()) if book.qty else 0.0

    # 4) Weekly rebalance.
    last_dd = pd.Timestamp(state["last_decision_day"], tz="UTC") if state.get("last_decision_day") else None
    d, status = due_decision_day(now, pd.Timestamp(state["start"]), last_dd, cfg)
    rebalance = None
    if status == "missed":
        events.append({"ts": _iso(now), "type": "missed_rebalance", "decision_day": str(d.date()),
                       "scheduled_fill": _iso(scheduled_fill(d, cfg))})
        log(f"missed rebalance for {d.date()}")
        state["last_decision_day"] = str(d.date())
    elif status == "due":
        membership = ensure_membership(store, d.strftime("%Y-%m"), cfg, args.cache_dir, log)
        try:
            tgt = compute_targets(d, sorted(membership["symbol"]), listed, args.cache_dir, log)
        except RuntimeError as exc:
            log(f"rebalance postponed: {exc}")
            tgt = None
        if tgt is not None:
            to_bg = {s: bitget_symbol(s, listed) for s in tgt["weights"]}
            targets = {to_bg[s]: w for s, w in tgt["weights"].items()}
            plan = plan_weights(targets, book.weights(), PRIMARY_STRATEGY["band"])
            equity0 = book.equity
            for s, (w_cur, w_new) in plan.items():
                ob = ex.fetch_order_book(s, limit=100)
                bid, ask = float(ob["bids"][0][0]), float(ob["asks"][0][0])
                mid = (bid + ask) / 2.0
                # Target sizes use the equity marked before the first order, like the backtest.
                if w_new == 0.0:
                    dq = -book.qty.get(s, 0.0)  # close exactly
                else:
                    dq_raw = w_new * equity0 / mid - book.qty.get(s, 0.0)
                    try:
                        dq = float(ex.amount_to_precision(s, abs(dq_raw))) * float(np.sign(dq_raw))
                    except Exception:  # noqa: BLE001 - below the minimum amount precision
                        dq = 0.0
                if abs(dq) * mid < cfg.min_order_usdt and w_new != 0.0:
                    events.append({"ts": _iso(now), "type": "order_skipped_min_size", "symbol": s, "dq": dq_raw})
                    continue
                avg, visible, exhausted = walk_book(ob["asks"] if dq > 0 else ob["bids"], abs(dq))
                r = book.trade(s, dq, avg, mid, cfg.taker_fee)
                fills.append({"ts": _iso(now), "decision_day": str(d.date()), "symbol": s, "w_from": w_cur, "w_to": w_new,
                              "qty": dq, "mid": mid, "avg_px": avg, "spread_bps": (ask / bid - 1) * 1e4,
                              "notional_usdt": abs(dq) * avg, "book_exhausted": exhausted, **r})
            rebalance = {"ts": _iso(now), "type": "rebalance", "decision_day": str(d.date()),
                         "delay_hours": (now - scheduled_fill(d, cfg)).total_seconds() / 3600.0,
                         "equity_before": equity0, "equity_after": book.equity, "n_orders": len(plan),
                         "targets": tgt["weights"], "bitget_map": to_bg, "eligible": tgt["eligible"],
                         "not_on_bitget": tgt["not_on_bitget"], "signal_close": tgt["signal_close"]}
            events.append(rebalance)
            state["last_decision_day"] = str(d.date())
            gross = sum(abs(w) for w in book.weights().values())
            log(f"rebalanced {d.date()}: {len(plan)} orders, equity {equity0:.2f} -> {book.equity:.2f}")

    equity_row = {"ts": _iso(now), "equity": book.equity, "gross_exposure": gross, "n_positions": len(book.qty)}
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
    # Newly opened positions start accruing funding from now.
    for s in book.qty:
        state["funding_cursor"].setdefault(s, int(now.value // 1_000_000))
    store.save_state(state)
    log(f"equity {book.equity:.2f} positions {len(book.qty)} gross {gross:.2f}")


# ---------------------------------------------------------------- report

PREREGISTERED_CHECKS = {
    "min_rebalances": 12,
    "max_missed_rebalances": 1,
    "slippage_bps_median_max_by_tier": {"1-10": 3.0, "11-30": 6.0, "31-60": 12.0},
    "paper_vs_model_daily_corr_min": 0.9,
    "implementation_gap_ann_min": -0.05,
}


def report(store: Store, args) -> dict:
    state = store.load_state()
    eq = store.read("equity.jsonl")
    ev = store.read("events.jsonl")
    fills = store.read("fills.jsonl")
    fund = store.read("funding.jsonl")
    out: dict = {"name": state["config"]["name"], "start": state["start"], "last_run": state.get("last_run"),
                 "equity": state["book"]["equity"], "capital": state["config"]["capital_usdt"]}
    reb = ev[ev["type"] == "rebalance"] if len(ev) else pd.DataFrame()
    out["rebalances"] = int(len(reb))
    out["missed_rebalances"] = int((ev["type"] == "missed_rebalance").sum()) if len(ev) else 0
    if len(reb):
        out["fill_delay_hours_max"] = float(reb["delay_hours"].max())

    # Daily paper returns from the first mark at/after the fill hour each day.
    paper = pd.Series(dtype=float)
    if len(eq):
        e = eq.assign(ts=pd.to_datetime(eq["ts"], utc=True)).set_index("ts")["equity"]
        fill_h = state["config"]["fill_hour_utc"]
        snap = e[e.index.hour >= fill_h].groupby(e[e.index.hour >= fill_h].index.floor("D")).first()
        # The day before the first snapshot starts from the initial capital (first fill costs count).
        if len(snap):
            snap.loc[snap.index[0] - pd.Timedelta(days=1)] = float(state["config"]["capital_usdt"])
            snap = snap.sort_index()
        paper = snap.pct_change().dropna()
        # Snapshot on day t closes the model row t-1 (fill d-1 -> fill d).
        paper.index = paper.index - pd.Timedelta(days=1)
        out["total_return"] = float(state["book"]["equity"] / state["config"]["capital_usdt"] - 1.0)
        if len(paper) >= 7:
            out["paper_daily"] = {"days": int(len(paper)),
                                  "sharpe_ann": float(paper.mean() / paper.std() * np.sqrt(365)) if paper.std() > 0 else None,
                                  "max_dd": float(((1 + paper).cumprod() / (1 + paper).cumprod().cummax() - 1).min())}

    membership = pd.read_csv(store.path("membership.csv")) if store.path("membership.csv").exists() else pd.DataFrame()
    if len(fills):
        f = fills.copy()
        f["month"] = pd.to_datetime(f["decision_day"]).dt.strftime("%Y-%m")
        inv = {}
        for _, r in reb.iterrows():
            inv.update({bg: bn for bn, bg in r["bitget_map"].items()})
        f["binance_symbol"] = f["symbol"].map(lambda s: inv.get(s, s))
        if len(membership):
            f = f.merge(membership[["month", "symbol", "rank"]].rename(columns={"symbol": "binance_symbol"}),
                        on=["month", "binance_symbol"], how="left")
        else:
            f["rank"] = np.nan
        f["tier"] = pd.cut(f["rank"].fillna(99), [0, 10, 30, 60, 1000], labels=["1-10", "11-30", "31-60", "out"])
        out["fills"] = {
            "n": int(len(f)),
            "turnover_usdt": float(f["notional_usdt"].sum()),
            "fees_usdt": float(f["fee_usdt"].sum()),
            "slippage_usdt": float(f["slippage_usdt"].sum()),
            "book_exhausted": int(f["book_exhausted"].sum()),
            "slippage_bps_median_by_tier": {str(k): float(g["slippage_bps"].median())
                                            for k, g in f.groupby("tier", observed=True)},
            "slippage_bps_notional_weighted": float((f["slippage_bps"] * f["notional_usdt"]).sum() / f["notional_usdt"].sum()),
        }
    if len(fund):
        out["funding_usdt"] = float(fund["pnl_usdt"].sum())

    # Model replay: same decision days and eligible sets, Binance prices and funding, backtest costs.
    if len(reb) and len(paper) >= 2:
        days = pd.to_datetime(reb["decision_day"]).dt.tz_localize("UTC")
        syms = sorted({s for el in reb["eligible"] for s in el})
        panels = load_daily_panels(syms, cache_dir=args.cache_dir, exchange=BINANCE)
        elig = pd.DataFrame(False, index=panels["signal_close"].index, columns=panels["signal_close"].columns)
        for dday, el in zip(days, reb["eligible"]):
            if dday in elig.index:
                elig.loc[dday, [s for s in el if s in elig.columns]] = True
        start = days.min()
        model = backtest(dict(panels, eligible=elig), MomentumConfig(**PRIMARY_STRATEGY), rebalance_on=days)
        model = model[model.index >= start]["net"]
        both = pd.concat([paper.rename("paper"), model.rename("model")], axis=1, sort=True).dropna()
        if len(both) >= 2:
            gap = both["paper"] - both["model"]
            out["vs_model"] = {
                "days": int(len(both)),
                "paper_cum": float((1 + both["paper"]).prod() - 1),
                "model_cum": float((1 + both["model"]).prod() - 1),
                "daily_corr": float(both["paper"].corr(both["model"])) if len(both) >= 5 else None,
                "implementation_gap_ann": float(gap.mean() * 365),
            }

    checks = PREREGISTERED_CHECKS
    verdict = {}
    if out["rebalances"] >= checks["min_rebalances"]:
        verdict["missed_ok"] = out["missed_rebalances"] <= checks["max_missed_rebalances"]
        tiers = out.get("fills", {}).get("slippage_bps_median_by_tier", {})
        verdict["slippage_ok"] = all(tiers.get(t, 0.0) <= v for t, v in checks["slippage_bps_median_max_by_tier"].items())
        vm = out.get("vs_model", {})
        verdict["tracking_ok"] = (vm.get("daily_corr") or 0) >= checks["paper_vs_model_daily_corr_min"]
        verdict["gap_ok"] = vm.get("implementation_gap_ann", -1) >= checks["implementation_gap_ann_min"]
    out["preregistered_checks"] = {"criteria": checks, "verdict": verdict or "pending: not enough rebalances"}
    return out


# ---------------------------------------------------------------- cli

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["init", "run", "report"])
    ap.add_argument("--state-dir", default="runtime/paper/xs_momentum_v1")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--capital", type=float, default=PaperConfig.capital_usdt)
    ap.add_argument("--dry-run", action="store_true", help="run: print what would happen, save nothing")
    ap.add_argument("--now", default="", help="override the clock (testing only)")
    args = ap.parse_args()

    store = Store(Path(args.state_dir))
    store.root.mkdir(parents=True, exist_ok=True)

    if args.command == "init":
        if store.path("state.json").exists():
            sys.exit(f"paper book already exists at {store.root}")
        cfg = PaperConfig(capital_usdt=args.capital)
        now = pd.Timestamp.now(tz="UTC")
        state = {"config": asdict(cfg), "strategy": PRIMARY_STRATEGY, "start": _iso(now),
                 "book": PaperBook(equity=cfg.capital_usdt).to_dict(), "last_decision_day": None,
                 "last_run_ms": int(now.value // 1_000_000), "funding_cursor": {}}
        store.save_state(state)
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
    run_cycle(store, args)
    print(f"done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
