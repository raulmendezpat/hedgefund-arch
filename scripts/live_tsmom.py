"""Live runner for time-series momentum on BTC/ETH on its own dedicated Bitget subaccount.

Same targets, schedule and band rule as the paper runner (scripts/paper_tsmom.py): decision every
day d on the Binance 23:00 UTC closes, fill on d+1 at 01:00 UTC. Same guardrails and exchange
adapter as the cross-sectional momentum live runner (scripts/live_xs_momentum.py,
hf_research.live_xs): the account is the source of truth, kill switch, daily reduce-only, sanity
limits, transfers tracked so they are not P&L, and an abort on positions it did not open.

It must run on a subaccount of its own: the momentum book also trades BTC and ETH, and each runner
aborts on the other's positions. Credentials therefore come only from --secret-file or from
$BITGET_TSMOM_ENV_FILE (default ~/.config/hedgefund/bitget_tsmom.env, chmod 600), never from the
momentum runner's file or BITGET_* environment variables, and a key equal to the momentum
runner's is refused.

Modes:
  --mode dry-run  (default) reads the account and logs the orders it would send. It never sends
                  orders and never changes account settings.
  --mode live     sends market orders. Also requires the environment variable LIVE_TRADING=1.

Subcommands:
  init        create the state (first rebalance = first scheduled fill after now)
  run         one hourly cycle
  status      print the state, risk and the latest events
  report      execution quality: slippage and fees against the backtest cost model and (with
              --paper-dir) the paper runner
  reset-risk  clear a kill-switch halt and restart peak/day-start equity; needs --confirm
Each --mode keeps its own state directory (runtime/live/tsmom_btc_eth_v1/<mode>).

Example (cron, every hour at minute 17, after the paper runner):
  17 * * * * cd /path/to/hedgefund-arch && PYTHONPATH=src .venv/bin/python scripts/live_tsmom.py run --mode dry-run
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import ccxt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_xs_momentum as xs  # noqa: E402
import paper_tsmom  # noqa: E402

from hf_research.ledger import Store, _iso  # noqa: E402
from hf_research.liquid_universe import bitget_symbol  # noqa: E402
from hf_research.live_xs import (  # noqa: E402
    LiveConfig,
    SanityError,
    build_orders,
    check_risk,
    flatten_orders,
    new_risk_state,
    position_drift,
)
from hf_research.paper_tsmom import ASSETS, TsmomPaperConfig, due_daily_decision  # noqa: E402

STRATEGY = paper_tsmom.STRATEGY
DEFAULT_ENV_FILE = "~/.config/hedgefund/bitget_tsmom.env"
OID_PREFIX = "ts"
MODEL_COST_BPS = STRATEGY.slippage_bps

# Differences from the momentum book's LiveConfig defaults:
# - each asset can be up to 1.0 of equity (TsmomConfig.max_weight), so up to 2.0 gross;
# - a fill later than 12 h is missed, as in the paper;
# - ETH trades in 0.01 steps (~22 USDT), so post-trade drift from rounding is larger at small size;
# - positions are ~0.2 of equity each, so a per-position loss alert at 1.5% would fire on any
#   ordinary 8% move.
LIVE_OVERRIDES = dict(name="tsmom_live_v1", max_symbol_weight=STRATEGY.max_weight,
                      max_gross_weight=STRATEGY.max_weight * len(ASSETS), max_fill_delay_hours=12,
                      max_position_drift=0.03, position_loss_alert=0.03)


def live_config() -> LiveConfig:
    return LiveConfig(**LIVE_OVERRIDES)


def load_credentials(secret_file: str) -> dict:
    """--secret-file (JSON), else $BITGET_TSMOM_ENV_FILE / ~/.config/hedgefund/bitget_tsmom.env.

    Refuses the momentum runner's key: two runners on one subaccount abort on each other."""
    if secret_file:
        creds = xs.load_credentials(secret_file)
    else:
        path = Path(os.getenv("BITGET_TSMOM_ENV_FILE") or DEFAULT_ENV_FILE).expanduser()
        if not path.exists():
            sys.exit(f"missing {path}: create a separate Bitget subaccount and key for this runner")
        values = xs.read_env_file(path)
        creds = {k: values.get(v) for k, v in xs.ENV_KEYS.items()}
        if not all(creds.values()):
            sys.exit(f"{path} lacks one of {', '.join(xs.ENV_KEYS.values())}")
    xs_file = Path(os.getenv("BITGET_ENV_FILE") or xs.DEFAULT_ENV_FILE).expanduser()
    if xs_file.exists() and xs.read_env_file(xs_file).get(xs.ENV_KEYS["apiKey"]) == creds["apiKey"]:
        sys.exit(f"this key is the momentum runner's ({xs_file}); the TSMOM book needs its own subaccount")
    return creds


def _account(args) -> xs.BitgetAccount:
    return xs.BitgetAccount(load_credentials(args.secret_file))


def compute_targets(d: pd.Timestamp, cache_dir: str, log) -> dict:
    """The paper runner's targets (Binance symbols), so live and paper cannot drift apart."""
    return paper_tsmom.compute_targets(d, TsmomPaperConfig(), cache_dir, log)


# ---------------------------------------------------------------- report

def report(store: Store, paper_dir: str = "") -> dict:
    fills, events = store.read("fills.jsonl"), store.read("events.jsonl")
    out: dict = {"n_fills": int(len(fills)), "model_cost_bps": MODEL_COST_BPS}
    if len(events):
        cf = events[events["type"] == "cash_flow"]
        out["cash_flows_usdt"] = float(cf["net_usdt"].sum()) if len(cf) else 0.0
        reb = events[events["type"] == "rebalance"]
        out["rebalances"] = int(len(reb))
        out["missed_rebalances"] = int((events["type"] == "missed_rebalance").sum())
        if len(reb) and "skipped" in reb:
            out["skipped_orders"] = int(sum(len(s) for s in reb["skipped"] if isinstance(s, list)))
    if not len(fills):
        return out
    out["by_symbol"] = {s: {"n": int(len(g)), "live_slippage_bps_median": float(g["slippage_bps"].median()),
                            "live_fee_bps_median": float(g["fee_bps"].median())}
                        for s, g in fills.groupby("symbol")}
    out["turnover_usdt"] = float(fills["notional_usdt"].sum())
    out["slippage_usdt"] = float(fills["slippage_usdt"].sum())
    out["fees_usdt"] = float(fills["fee_usdt"].sum())
    if paper_dir:
        pf = Store(Path(paper_dir)).read("fills.jsonl")
        if len(pf):
            for s, row in out["by_symbol"].items():
                p = pf[pf["symbol"] == s]
                row["paper_slippage_bps_median"] = float(p["slippage_bps"].median()) if len(p) else None
    return out


# ---------------------------------------------------------------- run cycle

def run_cycle(store: Store, args, account_factory=_account) -> None:
    state = store.load_state()
    cfg = LiveConfig(**state["config"])
    now = pd.Timestamp.now(tz="UTC") if not args.now else pd.Timestamp(args.now, tz="UTC")
    live = args.mode == "live"
    events, order_rows = [], []
    acct = None

    def log(msg: str) -> None:
        print(f"[{_iso(now)}] [{args.mode}] {msg}", flush=True)

    def finish(equity: float | None, extra: dict | None = None) -> None:
        if live and acct is not None:
            try:
                xs.record_fills(store, acct, order_rows, now, log)
            except Exception as exc:  # noqa: BLE001 - must never block saving state and the order log
                log(f"recording fills failed: {type(exc).__name__}: {exc}")
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
    xs.track_cash_flows(state, acct, equity, now, events, log)

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
    xs.watch_position_losses(state, positions, mids, equity, cfg, now, events, log)
    extra = {"gross_exposure": gross, "n_positions": len(positions), "risk_status": status}

    if status in ("halt", "halted"):
        if status == "halt":
            log(f"KILL SWITCH: {reason}")
            events.append({"ts": _iso(now), "type": "kill_switch", "reason": reason, "equity": equity})
        else:
            log(f"halted ({reason}); run reset-risk after review")
            if positions:  # an earlier flatten left something open: keep trying every hour
                events.append({"ts": _iso(now), "type": "kill_switch_retry", "open": sorted(positions)})
                log(f"kill switch retry: still open {sorted(positions)}")
        if positions:
            order_rows += xs.execute(acct, flatten_orders(positions, mids), live, f"kill{now:%Y%m%d%H}", cfg,
                                     positions, now, OID_PREFIX)
        finish(equity, extra)
        return

    last_dd = pd.Timestamp(state["last_decision_day"], tz="UTC") if state.get("last_decision_day") else None
    d, sched, skipped_days = due_daily_decision(now, pd.Timestamp(state["start"]), last_dd, cfg)
    for k in skipped_days:
        events.append({"ts": _iso(now), "type": "missed_rebalance", "decision_day": str(k.date())})
        log(f"missed rebalance for {k.date()}")
    if sched == "missed":
        events.append({"ts": _iso(now), "type": "missed_rebalance", "decision_day": str(d.date())})
        log(f"missed rebalance for {d.date()}")
        state["last_decision_day"] = str(d.date())
    elif sched == "due":
        if skipped_days:  # a later decision day supersedes them even if this one is postponed
            state["last_decision_day"] = str(skipped_days[-1].date())
        try:
            tgt = compute_targets(d, args.cache_dir, log)
            to_bg = {s: bitget_symbol(s, acct.listed) for s in ASSETS}
            if any(v is None for v in to_bg.values()):
                raise RuntimeError(f"not listed on Bitget: {[s for s, v in to_bg.items() if v is None]}")
            targets = {to_bg[s]: w for s, w in tgt["weights"].items()}
            all_mids = acct.mids(sorted(set(targets) | set(positions)))
            orders, skipped, intended = build_orders(
                targets, positions, all_mids, equity, cfg, STRATEGY.band, acct.to_precision,
                reduce_only_mode=(status == "reduce_only"))
        except (RuntimeError, SanityError, ccxt.BaseError) as exc:
            # Retried next hour; after max_fill_delay_hours the day becomes a missed rebalance.
            kind = "sanity_abort" if isinstance(exc, SanityError) else "rebalance_postponed"
            events.append({"ts": _iso(now), "type": kind, "decision_day": str(d.date()), "reason": str(exc)})
            log(f"{kind}: {exc}")
            finish(equity, extra)
            return

        tag = str(d.date())
        if live:
            # Persist ownership before sending (see live_xs_momentum.run_cycle).
            state["managed_symbols"] = sorted(set(state["managed_symbols"]) | {o.symbol for o in orders})
            store.save_state(state)
            if not positions:
                try:
                    acct.ensure_one_way()
                except Exception as exc:  # noqa: BLE001 - usually "already one-way"
                    log(f"set_position_mode: {exc}")
        order_rows += xs.execute(acct, orders, live, tag, cfg, positions, now, OID_PREFIX)
        errors = [r for r in order_rows if r.get("error")]
        drift = {}
        if live:
            try:
                positions_after, _ = acct.positions()
                equity_after = acct.equity()
                drift = position_drift(intended, positions_after, all_mids, equity_after, cfg)
            except Exception as exc:  # noqa: BLE001 - orders are already sent: record them regardless
                drift = None
                events.append({"ts": _iso(now), "type": "verify_failed", "decision_day": tag,
                               "reason": f"{type(exc).__name__}: {exc}"})
                log(f"post-trade verification failed: {exc}")
        events.append({"ts": _iso(now), "type": "rebalance", "decision_day": tag, "risk_status": status,
                       "delay_hours": (now - d - pd.Timedelta(days=1, hours=cfg.fill_hour_utc)).total_seconds() / 3600.0,
                       "equity_before": equity, "n_orders": len(orders), "n_errors": len(errors),
                       "skipped": skipped, "drift": drift, "targets": tgt["weights"], "bitget_map": to_bg,
                       "intended": intended, "signal_close": tgt["signal_close"]})
        if live:  # the day's loss per position is measured from the prices it was rebalanced at
            state["position_ref"] = {"equity": equity, "px": {s: all_mids[s] for s in intended}, "alerted": []}
        if not errors:
            state["last_decision_day"] = tag
        log(f"rebalance {tag}: {len(orders)} orders ({'sent' if live else 'not sent'}), {len(skipped)} skipped, "
            f"{len(errors)} errors, {'unverified' if drift is None else len(drift)} drift breaches")

    finish(equity, extra)
    log(f"equity {equity:.2f} positions {len(positions)} gross {gross:.2f} risk {status} {reason}".rstrip())


# ---------------------------------------------------------------- cli

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["init", "run", "status", "report", "reset-risk"])
    ap.add_argument("--state-dir", default="", help="default: runtime/live/tsmom_btc_eth_v1/<mode>")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--mode", choices=["dry-run", "live"], default="dry-run")
    ap.add_argument("--secret-file", default="")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--paper-dir", default="", help="report: paper state dir to compare execution costs with")
    ap.add_argument("--now", default="", help="override the clock (testing only)")
    args = ap.parse_args()

    # Each mode keeps its own state: a dry run must never mark a day as done for the live book.
    store = Store(Path(args.state_dir or f"runtime/live/tsmom_btc_eth_v1/{args.mode}"))
    store.root.mkdir(parents=True, exist_ok=True)

    if args.command == "init":
        if store.path("state.json").exists():
            sys.exit(f"live state already exists at {store.root}")
        cfg = live_config()
        now = pd.Timestamp.now(tz="UTC")
        store.save_state({"config": cfg.to_dict(), "strategy": STRATEGY.to_dict(), "schedule": "daily",
                          "start": _iso(now), "managed_symbols": [], "risk": None, "last_decision_day": None})
        fill = (now - pd.Timedelta(hours=cfg.fill_hour_utc)).ceil("D") + pd.Timedelta(hours=cfg.fill_hour_utc)
        print(f"live state created at {store.root}; first rebalance: fill {_iso(fill)}")
        return

    if args.command == "status":
        state = store.load_state()
        ev = store.read("events.jsonl")
        print(json.dumps({k: state.get(k) for k in ("start", "last_run", "last_decision_day", "risk", "managed_symbols")},
                         indent=2, default=float))
        if len(ev):
            print(ev.tail(10).drop(columns=[c for c in ("targets", "bitget_map", "intended", "signal_close") if c in ev],
                                   errors="ignore").to_string())
        return

    if args.command == "report":
        print(json.dumps(report(store, args.paper_dir), indent=2, default=float))
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
