"""Hourly health check for the momentum runners (paper and live), with macOS notifications.

It only reads the runners' state, events and logs; it never touches them. Alerts:
- stale: a runner has not completed a cycle for more than --stale-hours.
- overdue: a scheduled rebalance is still pending --overdue-hours after its fill time.
- errors: new Python tracebacks in a runner's logs.
- events: notable new events (rebalance, missed rebalance, kill switch, aborts, ...).
- position loss: one position lost more than --position-loss of equity since the last rebalance
  (live runners report it as an event; for the paper book the monitor measures it itself).
Condition alerts (stale, overdue) notify when they start and when they clear, not every hour.
A daily heartbeat at --heartbeat-hour (local time) confirms the monitor itself is alive.

Usage (launchd/cron, hourly):
  PYTHONPATH=src python scripts/healthcheck_xs.py \
      --runner paper=/path/to/paper/runtime/paper/xs_momentum_v1:/path/to/paper/runtime/logs \
      --runner dry-run=/path/to/live/runtime/live/xs_momentum_v1/dry-run:/path/to/live/runtime/logs
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from hf_research.live_xs import position_losses
from hf_research.paper_xs import due_decision_day, scheduled_fill

EVENT_LEVEL = {
    "rebalance": "info",
    "missed_rebalance": "alert",
    "kill_switch": "alert",
    "kill_switch_retry": "alert",
    "abort": "alert",
    "sanity_abort": "alert",
    "verify_failed": "alert",
    "delisted_close": "info",
    "risk_reset": "info",
    "position_loss": "alert",
}


@dataclass(frozen=True)
class Runner:
    name: str
    state_dir: Path
    log_dir: Path


@dataclass(frozen=True)
class Notice:
    level: str  # "info" | "alert" | "ok"
    runner: str
    text: str


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _describe(ev: dict) -> str:
    t = ev["type"]
    if t == "rebalance":
        eq = ev.get("equity_after", ev.get("equity_before"))
        extra = f", equity {eq:,.2f}" if isinstance(eq, (int, float)) else ""
        errs = f", {ev['n_errors']} errores" if ev.get("n_errors") else ""
        return f"rebalanceo {ev.get('decision_day')}: {ev.get('n_orders')} órdenes{errs}{extra}"
    if t == "position_loss":
        return f"{ev['symbol']} pierde {ev['loss_frac']:.1%} de la cartera desde el último rebalanceo"
    detail = ev.get("reason") or ev.get("decision_day") or ev.get("symbol") or ""
    return f"{t}: {detail}".rstrip(": ")


def paper_position_losses(state: dict, ref: dict | None, threshold: float) -> tuple[list[str], dict]:
    """Paper book positions newly past the loss threshold since the last rebalance.

    The reference (prices and equity) is taken at the first check after each rebalance; the paper
    marks its book hourly, so that is the fill-time mark."""
    book = state.get("book") or {}
    qty, px = book.get("qty", {}), book.get("last_px", {})
    dd = state.get("last_decision_day")
    if ref is None or ref.get("decision_day") != dd:
        ref = {"decision_day": dd, "equity": float(book.get("equity", 0.0)), "px": dict(px), "alerted": []}
    ref["px"] = {s: p for s, p in ref["px"].items() if s in qty}
    for s in qty:
        if s in px:
            ref["px"].setdefault(s, px[s])
    texts = []
    for s, loss in position_losses(qty, ref["px"], px, ref["equity"], threshold).items():
        if s not in ref["alerted"]:
            ref["alerted"].append(s)
            texts.append(f"{s} pierde {loss:.1%} de la cartera desde el último rebalanceo")
    return texts, ref


def check_runner(r: Runner, now: pd.Timestamp, seen: dict, *, stale_hours: float, overdue_hours: float,
                 position_loss: float = 0.015) -> tuple[list[Notice], dict, dict]:
    """Returns (notices for new events and errors, current conditions {key: text}, updated cursor)."""
    notices, conditions = [], {}
    state_path = r.state_dir / "state.json"
    if not state_path.exists():
        return notices, {"missing": f"no existe {state_path}"}, seen
    state = json.loads(state_path.read_text())

    last_run = pd.Timestamp(state["last_run"]) if state.get("last_run") else pd.Timestamp(state["start"])
    age_h = (now - last_run).total_seconds() / 3600.0
    if age_h > stale_hours:
        conditions["stale"] = f"sin correr hace {age_h:.1f} h (último ciclo {last_run:%Y-%m-%d %H:%M} UTC)"

    cfg = SimpleNamespace(**state["config"])
    last_dd = pd.Timestamp(state["last_decision_day"], tz="UTC") if state.get("last_decision_day") else None
    d, status = due_decision_day(now, pd.Timestamp(state["start"]), last_dd, cfg)
    if status == "due":
        late_h = (now - scheduled_fill(d, cfg)).total_seconds() / 3600.0
        if late_h > overdue_hours:
            conditions["overdue"] = f"rebalanceo {d.date()} pendiente hace {late_h:.1f} h"

    events = _read_jsonl(r.state_dir / "events.jsonl")
    start = seen.get("events", len(events))  # first check: existing events are history, not news
    for ev in events[start:]:
        level = EVENT_LEVEL.get(ev.get("type"))
        if level:
            notices.append(Notice(level, r.name, _describe(ev)))
    tracebacks = {}
    for log in sorted(r.log_dir.glob("*.log")):
        n = log.read_text(errors="replace").count("Traceback (most recent call last)")
        if n:
            tracebacks[log.name] = n
    old_tb = seen.get("tracebacks")
    if old_tb is not None:
        new = sum(max(0, n - old_tb.get(k, 0)) for k, n in tracebacks.items())
        if new:
            notices.append(Notice("alert", r.name, f"{new} error(es) nuevo(s) de Python en los logs de {r.log_dir}"))
    cursor = {"events": len(events), "tracebacks": tracebacks}
    if "book" in state:
        texts, cursor["pos_ref"] = paper_position_losses(state, seen.get("pos_ref"), position_loss)
        notices += [Notice("alert", r.name, t) for t in texts]
    return notices, conditions, cursor


def run_check(runners: list[Runner], hc_state: dict, now: pd.Timestamp, *, stale_hours: float = 2.25,
              overdue_hours: float = 3.0, position_loss: float = 0.015, heartbeat_hour: int = 9,
              local_tz: str | None = None
              ) -> tuple[list[Notice], dict]:
    """One pass over every runner; returns (notices to send, new monitor state)."""
    out: list[Notice] = []
    new_state = {"runners": {}, "conditions": {}, "last_heartbeat": hc_state.get("last_heartbeat")}
    summary = []
    for r in runners:
        seen = hc_state.get("runners", {}).get(r.name, {})
        notices, conditions, cursor = check_runner(r, now, seen, stale_hours=stale_hours, overdue_hours=overdue_hours,
                                                   position_loss=position_loss)
        out += notices
        new_state["runners"][r.name] = cursor
        before = hc_state.get("conditions", {}).get(r.name, {})
        for k, text in conditions.items():
            if k not in before:
                out.append(Notice("alert", r.name, text))
        for k in before:
            if k not in conditions:
                out.append(Notice("ok", r.name, f"resuelto: {before[k]}"))
        new_state["conditions"][r.name] = conditions
        summary.append(f"{r.name} {'OK' if not conditions else 'con problemas'}")

    local_now = now.tz_convert(_local_tz(local_tz))
    today = str(local_now.date())
    if local_now.hour >= heartbeat_hour and hc_state.get("last_heartbeat") != today:
        out.append(Notice("info", "monitor", "chequeo diario: " + "; ".join(summary)))
        new_state["last_heartbeat"] = today
    return out, new_state


def _local_tz(name: str | None):
    if name:
        return name
    from datetime import datetime

    return datetime.now().astimezone().tzinfo


def notify(n: Notice) -> None:
    """macOS notification; alerts play a sound."""
    title = {"alert": "⚠️ Momentum", "ok": "✅ Momentum", "info": "Momentum"}[n.level]
    text = f"[{n.runner}] {n.text}".replace("\\", "\\\\").replace('"', '\\"')
    sound = ' sound name "Basso"' if n.level == "alert" else ""
    script = f'display notification "{text}" with title "{title}"{sound}'
    subprocess.run(["osascript", "-e", script], check=False, timeout=15)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", action="append", required=True, help="name=STATE_DIR:LOG_DIR (repeatable)")
    ap.add_argument("--state-file", default="runtime/healthcheck/state.json")
    ap.add_argument("--stale-hours", type=float, default=2.25)
    ap.add_argument("--overdue-hours", type=float, default=3.0)
    ap.add_argument("--position-loss", type=float, default=0.015,
                    help="alert when one position loses this share of equity since the last rebalance")
    ap.add_argument("--heartbeat-hour", type=int, default=9, help="local hour of the daily 'all OK' notice")
    ap.add_argument("--no-notify", action="store_true", help="print only")
    ap.add_argument("--now", default="", help="override the clock (testing only)")
    args = ap.parse_args()

    runners = []
    for spec in args.runner:
        name, dirs = spec.split("=", 1)
        state_dir, log_dir = dirs.split(":", 1)
        runners.append(Runner(name, Path(state_dir), Path(log_dir)))
    now = pd.Timestamp(args.now, tz="UTC") if args.now else pd.Timestamp.now(tz="UTC")
    path = Path(args.state_file)
    hc_state = json.loads(path.read_text()) if path.exists() else {}
    notices, new_state = run_check(runners, hc_state, now, stale_hours=args.stale_hours,
                                   overdue_hours=args.overdue_hours, position_loss=args.position_loss,
                                   heartbeat_hour=args.heartbeat_hour)
    for n in notices:
        print(f"[{now:%Y-%m-%dT%H:%M:%SZ}] {n.level.upper()} [{n.runner}] {n.text}", flush=True)
        if not args.no_notify:
            notify(n)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(new_state, indent=2))
    tmp.replace(path)
    if not notices:
        print(f"[{now:%Y-%m-%dT%H:%M:%SZ}] ok", flush=True)


if __name__ == "__main__":
    main()
