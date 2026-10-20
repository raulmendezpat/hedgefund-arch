"""Live runner for the cross-sectional momentum hypothesis on a dedicated Bitget (sub)account.

The account is the source of truth: every run reads equity and positions from Bitget, checks the
risk guardrails (hf_research.live_xs), and on the weekly schedule rebalances to the same targets
the paper runner computes. Run it every hour; it holds a lock and is safe to re-run.

Modes:
  --mode dry-run  (default) reads the account and logs the orders it would send. It never sends
                  orders and never changes account settings (leverage, margin or position mode).
  --mode live     sends market orders. Also requires the environment variable LIVE_TRADING=1.

Credentials (read only when a command needs the account; first match wins):
  --secret-file pointing to a JSON file with apiKey / secret / password (optionally nested under
  "envelope"); else the BITGET_API_KEY, BITGET_API_SECRET, BITGET_API_PASSPHRASE environment
  variables; else those keys read as plain KEY=VALUE data from $BITGET_ENV_FILE (default
  ~/.config/hedgefund/bitget.env, must be chmod 600). Use a key without withdrawal permission.

Subcommands:
  init        create the state (first rebalance = first scheduled fill after now)
  run         one hourly cycle
Each --mode keeps its own state directory (runtime/live/xs_momentum_v1/<mode>).
  status      print the state, risk and the latest events
  report      execution quality: slippage and fees by liquidity tier and by rebalance, against
              the backtest cost model and (with --paper-dir) the paper runner
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

import ccxt
import pandas as pd

from hf_research import xs_signal
from hf_research.ledger import Store, _iso
from hf_research.liquid_universe import bitget_symbol
from hf_research.live_xs import (
    CASH_FLOW_OVERLAP_MS,
    EXTERNAL_TRANSFER_TYPES,
    LiveConfig,
    OrderIntent,
    SanityError,
    apply_cash_flow,
    build_orders,
    check_risk,
    fill_record,
    flatten_orders,
    new_cash_flows,
    new_risk_state,
    position_drift,
    position_losses,
)
from hf_research.paper_xs import PRIMARY_STRATEGY, due_decision_day, first_decision_day_after, scheduled_fill

PRODUCT = {"productType": "USDT-FUTURES"}


# ---------------------------------------------------------------- exchange adapter

ENV_KEYS = {"apiKey": "BITGET_API_KEY", "secret": "BITGET_API_SECRET", "password": "BITGET_API_PASSPHRASE"}
DEFAULT_ENV_FILE = "~/.config/hedgefund/bitget.env"


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE lines read as plain data (never executed by a shell): blank lines and # comments
    are ignored, an optional `export ` prefix and one pair of surrounding quotes are stripped.

    Refuses a file that group or others can read, since it holds trading credentials."""
    mode = path.stat().st_mode
    if mode & 0o077:
        sys.exit(f"{path} is accessible by other users (mode {oct(mode & 0o777)}); run: chmod 600 {path}")
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip().removeprefix("export ").strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        out[key] = value
    return out


def load_credentials(secret_file: str) -> dict:
    """--secret-file (JSON), else environment variables, else $BITGET_ENV_FILE / ~/.config/hedgefund/bitget.env."""
    if secret_file:
        d = json.loads(Path(secret_file).read_text())
        d = d.get("envelope", d)
        creds = {k: d.get(k) for k in ENV_KEYS}
    else:
        creds = {k: os.getenv(v) for k, v in ENV_KEYS.items()}
        env_file = Path(os.getenv("BITGET_ENV_FILE") or DEFAULT_ENV_FILE).expanduser()
        if not all(creds.values()) and env_file.exists():
            values = read_env_file(env_file)
            creds = {k: creds[k] or values.get(v) for k, v in ENV_KEYS.items()}
    if not all(creds.values()):
        sys.exit("missing Bitget credentials (BITGET_API_KEY / BITGET_API_SECRET / BITGET_API_PASSPHRASE, "
                 f"$BITGET_ENV_FILE, {DEFAULT_ENV_FILE} or --secret-file)")
    return creds


class BitgetAccount:
    """The few account operations the runner needs, on top of ccxt (one-way mode, USDT-M)."""

    def __init__(self, creds: dict):
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
            if bid and ask:
                out[s] = (bid + ask) / 2.0
            elif last:
                out[s] = float(last)
            # no quote at all: left out, so build_orders refuses to trade on a missing price
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

    def order_fill(self, order_id: str, symbol: str) -> dict | None:
        """Average price, filled quantity and fee of a sent order; None while it is still open."""
        r = self.ex.fetch_order(order_id, symbol)
        if r.get("status") not in ("closed", "canceled") or not r.get("filled") or not r.get("average"):
            return None
        fee = r.get("fee") or {}
        fee_usdt = abs(float(fee.get("cost") or 0.0)) if fee.get("currency") in (None, "USDT") else float("nan")
        return {"id": order_id, "filled": float(r["filled"]), "average": float(r["average"]), "fee_usdt": fee_usdt}

    def cash_flows(self, since_ms: int, until_ms: int) -> list[dict]:
        """Transfers in (+) and out (-) of the USDT-M futures account between two times (ms)."""
        out = []
        for business_type in EXTERNAL_TRANSFER_TYPES:
            data = self.ex.privateMixGetV2MixAccountBill({**PRODUCT, "businessType": business_type,
                                                          "startTime": since_ms, "endTime": until_ms, "limit": 100})
            bills = (data.get("data") or {}).get("bills", [])
            if len(bills) >= 100:
                raise RuntimeError(f"{business_type}: 100+ transfer bills in one window; check the account by hand")
            out += [{"id": b["billId"], "ts": int(b["cTime"]), "type": business_type, "amount": float(b["amount"])}
                    for b in bills if b.get("coin", "USDT") == "USDT"]
        return sorted(out, key=lambda f: f["ts"])


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


def track_cash_flows(state: dict, acct, equity: float, now: pd.Timestamp, events: list, log) -> None:
    """Shift the risk baselines by transfers in/out since the last run, so they are not P&L.

    The first run only records which transfers already exist (they are in the equity the risk
    state starts from). A failed query leaves the cursor alone and is retried next run."""
    now_ms = int(now.value // 1_000_000)
    cf = state.get("cash_flow")
    since = (cf["since_ms"] if cf else now_ms) - CASH_FLOW_OVERLAP_MS
    try:
        flows = acct.cash_flows(since, now_ms)
    except Exception as exc:  # noqa: BLE001 - never block trading on this; the guardrails stay conservative
        events.append({"ts": _iso(now), "type": "cash_flow_check_failed", "reason": f"{type(exc).__name__}: {exc}"})
        log(f"cash flow check failed: {exc}")
        return
    fresh, state["cash_flow"] = new_cash_flows(cf or {}, flows, now_ms)
    if cf is None or not fresh:
        return
    net = sum(f["amount"] for f in fresh)
    applied = state.get("risk") is not None
    if applied:
        apply_cash_flow(state["risk"], net)
    events.append({"ts": _iso(now), "type": "cash_flow", "net_usdt": net, "flows": fresh, "equity": equity,
                   "risk_adjusted": applied})
    log(f"cash flow {net:+.2f} USDT ({len(fresh)} transfer(s)); risk baselines {'shifted' if applied else 'not set yet'}")


def watch_position_losses(state: dict, positions: dict[str, float], mids: dict[str, float], equity: float,
                          cfg: LiveConfig, now: pd.Timestamp, events: list, log) -> None:
    """Event (once per position and rebalance) when one position has lost more than
    cfg.position_loss_alert of equity since the last rebalance. Alert only: it trades nothing."""
    ref = state.get("position_ref") or {"equity": equity, "px": {}, "alerted": []}
    ref["px"] = {s: p for s, p in ref["px"].items() if s in positions}
    for s in positions:
        if s in mids:
            ref["px"].setdefault(s, mids[s])  # held without a reference (e.g. after a restart): start now
    for s, loss in position_losses(positions, ref["px"], mids, ref["equity"], cfg.position_loss_alert).items():
        if s in ref["alerted"]:
            continue
        ref["alerted"].append(s)
        events.append({"ts": _iso(now), "type": "position_loss", "symbol": s, "loss_frac": loss,
                       "qty": positions[s], "ref_px": ref["px"][s], "px": mids[s]})
        log(f"position loss alert: {s} down {loss:.2%} of equity since the last rebalance")
    state["position_ref"] = ref


def record_fills(store: Store, acct, new_rows: list[dict], now: pd.Timestamp, log, max_age_days: int = 7) -> None:
    """Fetch execution prices for sent orders that have none recorded yet (this run's and earlier ones)."""
    done = store.read("fills.jsonl")
    done_ids = set(done["order_id"]) if len(done) else set()
    old = store.read("orders.jsonl")
    rows = (old.to_dict("records") if len(old) else []) + list(new_rows)
    cutoff = now - pd.Timedelta(days=max_age_days)
    out = []
    for r in rows:
        oid = (r.get("response") or {}).get("id") if isinstance(r.get("response"), dict) else None
        if not r.get("sent") or not oid or oid in done_ids or pd.Timestamp(r["ts"]) < cutoff:
            continue
        try:
            fill = acct.order_fill(oid, r["symbol"])
        except Exception as exc:  # noqa: BLE001 - retried next run
            log(f"fill lookup failed {r['symbol']} {oid}: {type(exc).__name__}")
            continue
        if fill:
            out.append(fill_record(r, fill))
            done_ids.add(oid)
    store.append("fills.jsonl", out)
    if out:
        log(f"recorded {len(out)} fill(s); median slippage {pd.Series([f['slippage_bps'] for f in out]).median():.1f} bps")


# ---------------------------------------------------------------- report

MODEL_COST_BPS = {"1-10": 3.0, "11-30": 6.0, "31-60": 12.0}  # the backtest's tiered cost assumption


def add_tiers(fills: pd.DataFrame, events: pd.DataFrame, membership: pd.DataFrame, day_col: str) -> pd.DataFrame:
    """Liquidity tier (rank in that month's universe) of each filled Bitget symbol."""
    f = fills.copy()
    inv = {}
    if len(events) and "bitget_map" in events:
        for m in events.loc[events["type"] == "rebalance", "bitget_map"].dropna():
            inv.update({bg: bn for bn, bg in m.items()})
    f["binance_symbol"] = f["symbol"].map(lambda s: inv.get(s, s))
    f["month"] = pd.to_datetime(f[day_col], errors="coerce").dt.strftime("%Y-%m")
    if len(membership):
        f = f.merge(membership[["month", "symbol", "rank"]].rename(columns={"symbol": "binance_symbol"}),
                    on=["month", "binance_symbol"], how="left")
    else:
        f["rank"] = float("nan")
    f["tier"] = pd.cut(f["rank"].fillna(999), [0, 10, 30, 60, 10_000], labels=[*MODEL_COST_BPS, "out"]).astype(str)
    return f


def report(store: Store, paper_dir: str = "") -> dict:
    fills, events = store.read("fills.jsonl"), store.read("events.jsonl")
    membership = pd.read_csv(store.path("membership.csv")) if store.path("membership.csv").exists() else pd.DataFrame()
    out: dict = {"n_fills": int(len(fills))}
    if len(events):
        cf = events[events["type"] == "cash_flow"]
        out["cash_flows_usdt"] = float(cf["net_usdt"].sum()) if len(cf) else 0.0
    if not len(fills):
        return out
    f = add_tiers(fills, events, membership, "tag")
    paper = pd.DataFrame()
    if paper_dir:
        pstore = Store(Path(paper_dir))
        pf = pstore.read("fills.jsonl")
        if len(pf):
            pm = pd.read_csv(pstore.path("membership.csv")) if pstore.path("membership.csv").exists() else pd.DataFrame()
            paper = add_tiers(pf, pstore.read("events.jsonl"), pm, "decision_day")
    tiers = {}
    for tier, g in f.groupby("tier"):
        row = {"n": int(len(g)), "live_slippage_bps_median": float(g["slippage_bps"].median()),
               "live_fee_bps_median": float(g["fee_bps"].median()), "model_cost_bps": MODEL_COST_BPS.get(tier)}
        if len(paper):
            p = paper[paper["tier"] == tier]
            row["paper_slippage_bps_median"] = float(p["slippage_bps"].median()) if len(p) else None
        tiers[tier] = row
    out["by_tier"] = tiers
    days = {}
    for tag, g in f.groupby("tag"):
        row = {"n": int(len(g)), "turnover_usdt": float(g["notional_usdt"].sum()),
               "slippage_usdt": float(g["slippage_usdt"].sum()), "fees_usdt": float(g["fee_usdt"].sum()),
               "slippage_bps_weighted": float(g["slippage_usdt"].sum() / g["notional_usdt"].sum() * 1e4)}
        if len(paper):
            p = paper[paper["decision_day"] == tag]
            if len(p):
                row["paper_slippage_bps_weighted"] = float(p["slippage_usdt"].sum() / p["notional_usdt"].sum() * 1e4)
        days[tag] = row
    out["by_rebalance"] = days
    return out


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
                record_fills(store, acct, order_rows, now, log)
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
    track_cash_flows(state, acct, equity, now, events, log)

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
    watch_position_losses(state, positions, mids, equity, cfg, now, events, log)

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
            order_rows += execute(acct, flatten_orders(positions, mids), live, f"kill{now:%Y%m%d%H}", cfg, positions, now)
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
        except (RuntimeError, SanityError, ccxt.BaseError) as exc:
            # Retried next hour; after max_fill_delay_hours the week becomes a missed rebalance.
            kind = "sanity_abort" if isinstance(exc, SanityError) else "rebalance_postponed"
            events.append({"ts": _iso(now), "type": kind, "decision_day": str(d.date()), "reason": str(exc)})
            log(f"{kind}: {exc}")
            finish(equity, {"gross_exposure": gross, "n_positions": len(positions), "risk_status": status})
            return

        tag = str(d.date())
        if live:
            # Persist ownership before sending: if this process dies mid-rebalance, the next run must
            # still recognise these positions as its own instead of aborting on "foreign" ones.
            state["managed_symbols"] = sorted(set(state["managed_symbols"]) | {o.symbol for o in orders})
            store.save_state(state)
            if not positions:
                try:
                    acct.ensure_one_way()
                except Exception as exc:  # noqa: BLE001 - usually "already one-way"
                    log(f"set_position_mode: {exc}")
        order_rows += execute(acct, orders, live, tag, cfg, positions, now)
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
                       "delay_hours": (now - scheduled_fill(d, cfg)).total_seconds() / 3600.0,
                       "equity_before": equity, "n_orders": len(orders), "n_errors": len(errors),
                       "skipped": skipped, "drift": drift, "targets": tgt["weights"], "bitget_map": to_bg,
                       "intended": intended, "eligible": tgt["eligible"], "not_on_bitget": tgt["not_on_bitget"],
                       "coverage": tgt["coverage"]})
        if live:  # the week's loss per position is measured from the prices it was rebalanced at
            state["position_ref"] = {"equity": equity, "px": {s: all_mids[s] for s in intended}, "alerted": []}
        if not errors:
            state["last_decision_day"] = tag
        log(f"rebalance {tag}: {len(orders)} orders ({'sent' if live else 'not sent'}), {len(errors)} errors, "
            f"{'unverified' if drift is None else len(drift)} drift breaches")

    finish(equity, {"gross_exposure": gross, "n_positions": len(positions), "risk_status": status})
    log(f"equity {equity:.2f} positions {len(positions)} gross {gross:.2f} risk {status} {reason}".rstrip())


# ---------------------------------------------------------------- cli

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["init", "run", "status", "report", "reset-risk"])
    ap.add_argument("--state-dir", default="", help="default: runtime/live/xs_momentum_v1/<mode>")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--mode", choices=["dry-run", "live"], default="dry-run")
    ap.add_argument("--secret-file", default="")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--paper-dir", default="", help="report: paper state dir to compare execution costs with")
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
