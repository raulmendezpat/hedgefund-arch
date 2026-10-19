"""Live runner for the cross-sectional momentum hypothesis on a dedicated Bitget (sub)account.

The account is the source of truth: every run reads equity and positions from Bitget, checks the
risk guardrails (hf_research.live_xs), and on the weekly schedule rebalances to the same targets
the paper runner computes. Run it every hour; it holds a lock and is safe to re-run.

Modes:
  --mode dry-run  (default) reads the account and logs the orders it would send. It never sends
                  orders and never changes account settings (leverage, margin or position mode).
  --mode live     sends market orders. Also requires the environment variable LIVE_TRADING=1.

Credentials (read only when a command needs the account):
  BITGET_API_KEY, BITGET_API_SECRET, BITGET_API_PASSPHRASE environment variables, or
  --secret-file pointing to a JSON file with apiKey / secret / password (optionally nested under
  "envelope"). Use a key without withdrawal permission, restricted to the host IP.

Subcommands:
  init        create the state (first rebalance = first scheduled fill after now)
  run         one hourly cycle
Each --mode keeps its own state directory (runtime/live/xs_momentum_v1/<mode>).
  status      print the state, risk and the latest events
  reset-risk  clear a kill-switch halt and restart peak/day-start equity (after a review, or
              after deposits/withdrawals); needs --confirm

Example (cron, every hour at minute 7):
  7 * * * * cd /path/to/hedgefund-arch && PYTHONPATH=src .venv/bin/python scripts/live_xs_momentum.py run --mode dry-run
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from hf_research import xs_signal
from hf_research.ledger import Store, _iso
from hf_research.liquid_universe import bitget_symbol
from hf_research.live_xs import (
    LiveConfig,
    OrderIntent,
    SanityError,
    build_orders,
    check_risk,
    flatten_orders,
    new_risk_state,
    position_drift,
)
from hf_research.paper_xs import PRIMARY_STRATEGY, due_decision_day, first_decision_day_after, scheduled_fill

PRODUCT = {"productType": "USDT-FUTURES"}


# ---------------------------------------------------------------- exchange adapter

def load_credentials(secret_file: str) -> dict:
    if secret_file:
        d = json.loads(Path(secret_file).read_text())
        d = d.get("envelope", d)
        creds = {k: d.get(k) for k in ("apiKey", "secret", "password")}
    else:
        creds = {"apiKey": os.getenv("BITGET_API_KEY"), "secret": os.getenv("BITGET_API_SECRET"),
                 "password": os.getenv("BITGET_API_PASSPHRASE")}
    if not all(creds.values()):
        sys.exit("missing Bitget credentials (BITGET_API_KEY / BITGET_API_SECRET / BITGET_API_PASSPHRASE or --secret-file)")
    return creds


class BitgetAccount:
    """The few account operations the runner needs, on top of ccxt (one-way mode, USDT-M)."""

    def __init__(self, creds: dict):
        import ccxt

        self.ex = ccxt.bitget({**creds, "enableRateLimit": True, "timeout": 20000, "options": {"defaultType": "swap"}})
        markets = self.ex.load_markets()
        self.listed = {k for k, v in markets.items()
                       if v.get("swap") and v.get("linear") and v.get("settle") == "USDT" and v.get("active", True)}

    def equity(self) -> float:
        return float(self.ex.fetch_balance(PRODUCT)["USDT"]["total"])

    def positions(self) -> tuple[dict[str, float], list[dict]]:
        qty, meta = {}, []
        for p in self.ex.fetch_positions(None, PRODUCT):
            contracts = float(p.get("contracts") or 0.0)
            if contracts == 0.0:
                continue
            size = contracts * float(p.get("contractSize") or 1.0)
            signed = size if p.get("side") == "long" else -size
            qty[p["symbol"]] = qty.get(p["symbol"], 0.0) + signed
            meta.append({"symbol": p["symbol"], "side": p.get("side"), "qty": signed, "hedged": p.get("hedged"),
                         "margin_mode": p.get("marginMode"), "entry": p.get("entryPrice"), "mark": p.get("markPrice")})
        return qty, meta

    def mids(self, symbols: list[str]) -> dict[str, float]:
        if not symbols:
            return {}
        out = {}
        for s, t in self.ex.fetch_tickers(symbols).items():
            bid, ask, last = t.get("bid"), t.get("ask"), t.get("last")
            out[s] = (bid + ask) / 2.0 if bid and ask else float(last)
        return out

    def to_precision(self, symbol: str, qty: float) -> float:
        try:
            return float(self.ex.amount_to_precision(symbol, qty))
        except Exception:  # noqa: BLE001 - below the minimum step: nothing to trade
            return 0.0

    def prepare_symbol(self, symbol: str, cfg: LiveConfig) -> list[str]:
        """Margin mode and leverage for a symbol about to be opened (only called with no position)."""
        warnings = []
        try:
            self.ex.set_margin_mode(cfg.margin_mode, symbol, {"marginCoin": "USDT"})
        except Exception as exc:  # noqa: BLE001 - often "already in this mode"
            warnings.append(f"set_margin_mode: {exc}")
        try:
            self.ex.set_leverage(cfg.exchange_leverage, symbol, {"marginMode": cfg.margin_mode, "marginCoin": "USDT"})
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"set_leverage: {exc}")
        return warnings

    def ensure_one_way(self) -> None:
        self.ex.set_position_mode(False, None, PRODUCT)

    def market_order(self, o: OrderIntent, client_oid: str, cfg: LiveConfig) -> dict:
        r = self.ex.create_order(o.symbol, "market", o.side, o.qty, None,
                                 {"reduceOnly": o.reduce_only, "marginMode": cfg.margin_mode, "clientOid": client_oid})
        return {"id": r.get("id"), "status": r.get("status"), "filled": r.get("filled"), "average": r.get("average")}


def _account(args) -> BitgetAccount:
    return BitgetAccount(load_credentials(args.secret_file))


def client_oid(decision_day: str, o: OrderIntent) -> str:
    """Deterministic per (rebalance, symbol, leg): a retried run cannot submit the same leg twice."""
    h = hashlib.sha1(f"{o.symbol}|{o.reason}".encode()).hexdigest()[:12]
    return f"xs{decision_day.replace('-', '')}{h}"


# ---------------------------------------------------------------- run cycle

def execute(acct, orders: list[OrderIntent], live: bool, tag: str, cfg: LiveConfig, held: dict[str, float],
            now: pd.Timestamp) -> list[dict]:
    rows = []
    for o in orders:
        row = {"ts": _iso(now), "tag": tag, **o.to_dict(), "client_oid": client_oid(tag, o), "sent": live}
        if live:
            try:
                if not o.reduce_only and held.get(o.symbol, 0.0) == 0.0:
                    row["prepare_warnings"] = acct.prepare_symbol(o.symbol, cfg)
                row["response"] = acct.market_order(o, row["client_oid"], cfg)
            except Exception as exc:  # noqa: BLE001 - one rejected order must not stop the others
                row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def run_cycle(store: Store, args, account_factory=_account) -> None:
    state = store.load_state()
    cfg = LiveConfig(**state["config"])
    now = pd.Timestamp.now(tz="UTC") if not args.now else pd.Timestamp(args.now, tz="UTC")
    live = args.mode == "live"
    events, order_rows = [], []

    def log(msg: str) -> None:
        print(f"[{_iso(now)}] [{args.mode}] {msg}", flush=True)

    def finish(equity: float | None, extra: dict | None = None) -> None:
        store.append("events.jsonl", events)
        store.append("orders.jsonl", order_rows)
        if equity is not None:
            store.append("equity.jsonl", [{"ts": _iso(now), "mode": args.mode, "equity": equity, **(extra or {})}])
        state["last_run"] = _iso(now)
        store.save_state(state)

    if live and os.getenv("LIVE_TRADING") != "1":
        sys.exit("--mode live requires LIVE_TRADING=1 in the environment")

    acct = account_factory(args)
    equity = acct.equity()
    positions, meta = acct.positions()

    hedged = [m["symbol"] for m in meta if m.get("hedged")]
    foreign = sorted(set(positions) - set(state["managed_symbols"]))
    if hedged or foreign:
        reason = f"hedge-mode positions {hedged}" if hedged else f"positions not opened by this runner {foreign}"
        events.append({"ts": _iso(now), "type": "abort", "reason": reason})
        log(f"ABORT: {reason}")
        finish(equity)
        sys.exit(2)

    if state.get("risk") is None:
        state["risk"] = new_risk_state(equity, now)
    status, reason = check_risk(state["risk"], equity, now, cfg)
    mids = acct.mids(sorted(positions))
    gross = sum(abs(q * mids[s]) for s, q in positions.items()) / equity if positions and equity > 0 else 0.0

    if status in ("halt", "halted"):
        if status == "halt":
            log(f"KILL SWITCH: {reason}")
            events.append({"ts": _iso(now), "type": "kill_switch", "reason": reason, "equity": equity})
            order_rows += execute(acct, flatten_orders(positions, mids), live, f"kill{now:%Y%m%d%H}", cfg, positions, now)
        else:
            log(f"halted ({reason}); run reset-risk after review")
        finish(equity, {"gross_exposure": gross, "n_positions": len(positions), "risk_status": status})
        return

    last_dd = pd.Timestamp(state["last_decision_day"], tz="UTC") if state.get("last_decision_day") else None
    d, sched = due_decision_day(now, pd.Timestamp(state["start"]), last_dd, cfg)
    if sched == "missed":
        events.append({"ts": _iso(now), "type": "missed_rebalance", "decision_day": str(d.date()),
                       "scheduled_fill": _iso(scheduled_fill(d, cfg))})
        log(f"missed rebalance for {d.date()}")
        state["last_decision_day"] = str(d.date())
    elif sched == "due":
        try:
            membership = xs_signal.ensure_membership(store.path("membership.csv"), d.strftime("%Y-%m"), cfg,
                                                     args.cache_dir, log)
            tgt = xs_signal.compute_targets(d, sorted(membership["symbol"]), acct.listed, args.cache_dir, log,
                                            min_coverage=cfg.min_signal_coverage)
            to_bg = {s: bitget_symbol(s, acct.listed) for s in tgt["weights"]}
            targets = {to_bg[s]: w for s, w in tgt["weights"].items()}
            all_mids = acct.mids(sorted(set(targets) | set(positions)))
            orders, skipped, intended = build_orders(
                targets, positions, all_mids, equity, cfg, PRIMARY_STRATEGY["band"], acct.to_precision,
                reduce_only_mode=(status == "reduce_only"))
        except (RuntimeError, SanityError) as exc:
            # Retried next hour; after max_fill_delay_hours the week becomes a missed rebalance.
            kind = "sanity_abort" if isinstance(exc, SanityError) else "rebalance_postponed"
            events.append({"ts": _iso(now), "type": kind, "decision_day": str(d.date()), "reason": str(exc)})
            log(f"{kind}: {exc}")
            finish(equity, {"gross_exposure": gross, "n_positions": len(positions), "risk_status": status})
            return

        tag = str(d.date())
        if live and not positions:
            try:
                acct.ensure_one_way()
            except Exception as exc:  # noqa: BLE001 - usually "already one-way"
                log(f"set_position_mode: {exc}")
        order_rows += execute(acct, orders, live, tag, cfg, positions, now)
        errors = [r for r in order_rows if r.get("error")]
        drift = {}
        if live:
            state["managed_symbols"] = sorted(set(state["managed_symbols"]) | {o.symbol for o in orders})
            positions_after, _ = acct.positions()
            equity_after = acct.equity()
            drift = position_drift(intended, positions_after, all_mids, equity_after, cfg)
        events.append({"ts": _iso(now), "type": "rebalance", "decision_day": tag, "risk_status": status,
                       "delay_hours": (now - scheduled_fill(d, cfg)).total_seconds() / 3600.0,
                       "equity_before": equity, "n_orders": len(orders), "n_errors": len(errors),
                       "skipped": skipped, "drift": drift, "targets": tgt["weights"], "bitget_map": to_bg,
                       "intended": intended, "eligible": tgt["eligible"], "not_on_bitget": tgt["not_on_bitget"],
                       "coverage": tgt["coverage"]})
        if not errors:
            state["last_decision_day"] = tag
        log(f"rebalance {tag}: {len(orders)} orders ({'sent' if live else 'not sent'}), {len(errors)} errors, "
            f"{len(drift)} drift breaches")

    finish(equity, {"gross_exposure": gross, "n_positions": len(positions), "risk_status": status})
    log(f"equity {equity:.2f} positions {len(positions)} gross {gross:.2f} risk {status} {reason}".rstrip())


# ---------------------------------------------------------------- cli

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["init", "run", "status", "reset-risk"])
    ap.add_argument("--state-dir", default="", help="default: runtime/live/xs_momentum_v1/<mode>")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--mode", choices=["dry-run", "live"], default="dry-run")
    ap.add_argument("--secret-file", default="")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--now", default="", help="override the clock (testing only)")
    args = ap.parse_args()

    # Each mode keeps its own state: a dry run must never mark a week as done for the live book.
    store = Store(Path(args.state_dir or f"runtime/live/xs_momentum_v1/{args.mode}"))
    store.root.mkdir(parents=True, exist_ok=True)

    if args.command == "init":
        if store.path("state.json").exists():
            sys.exit(f"live state already exists at {store.root}")
        cfg = LiveConfig()
        now = pd.Timestamp.now(tz="UTC")
        store.save_state({"config": cfg.to_dict(), "strategy": PRIMARY_STRATEGY, "start": _iso(now),
                          "managed_symbols": [], "risk": None, "last_decision_day": None})
        d = first_decision_day_after(now, cfg)
        print(f"live state created at {store.root}; first rebalance: decision {d.date()} -> fill {_iso(scheduled_fill(d, cfg))}")
        return

    if args.command == "status":
        state = store.load_state()
        ev = store.read("events.jsonl")
        print(json.dumps({k: state.get(k) for k in ("start", "last_run", "last_decision_day", "risk", "managed_symbols")},
                         indent=2, default=float))
        if len(ev):
            print(ev.tail(10).drop(columns=[c for c in ("targets", "bitget_map", "intended", "eligible") if c in ev],
                                   errors="ignore").to_string())
        return

    if args.command == "reset-risk":
        if not args.confirm:
            sys.exit("reset-risk clears a kill-switch halt; re-run with --confirm after reviewing the account")
        state = store.load_state()
        store.append("events.jsonl", [{"ts": _iso(pd.Timestamp.now(tz="UTC")), "type": "risk_reset",
                                       "previous": state.get("risk")}])
        state["risk"] = None
        store.save_state(state)
        print("risk state cleared; peak and day-start equity restart at the next run")
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
