"""Weekly event and mapping review of the momentum book, with macOS notifications (docs/plans/xs_event_review.md).

Runs hourly and acts once per decision day, after the Sunday 23:00 UTC close is available
(Monday 00:00 UTC). It only alerts; it never changes the strategy or the orders:
- mapping: Binance vs Bitget price of every tradable member. The ratio must match the symbols'
  power-of-ten prefixes (1000PEPE vs PEPE); otherwise they are probably not the same asset.
- events: Claude with web search looks for discrete events on each target position and rates
  the risk for that position's side.

Usage (launchd/cron, hourly; needs ~/.config/hedgefund/anthropic.env):
  PYTHONPATH=src python scripts/event_review_xs.py run
  PYTHONPATH=src python scripts/event_review_xs.py run --now 2026-11-02T00:12 --no-notify   # force a past week
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from healthcheck_xs import Notice, notify  # noqa: E402

from hf_research import xs_signal  # noqa: E402
from hf_research.liquid_universe import bitget_symbol  # noqa: E402
from hf_research.paper_xs import PaperConfig  # noqa: E402

KEY_FILE = Path.home() / ".config/hedgefund/anthropic.env"
MODEL = "claude-sonnet-5-5"
EFFORT = "low"
MAX_SEARCHES = 3
# Sonnet 5.5 list prices (USD per million tokens) and web search (USD per search), 2026-10-28.
PRICE_INPUT, PRICE_OUTPUT, PRICE_SEARCH = 2.0, 10.0, 0.01
READY_AFTER = pd.Timedelta(days=1, minutes=5)  # the 23:00 bar closes at 00:00 UTC Monday
GIVE_UP_AFTER = pd.Timedelta(days=2)  # a review this late is no longer useful for the week

SYSTEM = """You review one position of a systematic crypto strategy before it is held for one week as a \
Bitget USDT-margined perpetual future. The position comes from a cross-sectional momentum rule and will \
not be changed by your answer: your job is to warn the account owner about discrete events that could \
hurt it.

Search the web for events about this token in the last 14 days or scheduled in the next 7 days:
- exchange delistings, or new listings on major exchanges (Binance, Coinbase, Upbit, Bitget, OKX, Bybit);
- hacks, exploits, bridge or protocol failures, paused deposits or withdrawals;
- token unlocks or large emissions, team/foundation/treasury sales, buybacks or burns;
- contract migrations, redenominations, ticker changes, chain swaps;
- regulatory or legal actions, depegs, airdrops, mainnet launches and upgrades.
Ignore price commentary, technical analysis, price predictions and generic market news.

Rate the risk FOR THIS POSITION'S SIDE. A long is hurt by events that can make the price fall \
(unlocks, hacks, delistings, sales); a short is hurt by events that can make it jump (new listings, \
buybacks, squeezes, takeover or ETF news). Delistings and migrations hurt both sides because the \
contract may stop trading or settle.
- high: an event likely to move the price more than 20% against the position, or to make the contract untradable;
- medium: a material event against the position with uncertain impact;
- low: minor or already priced-in events;
- none: nothing relevant found.
Only report events you found in your search results, with their URL. Never invent events or dates. \
Write summaries in Spanish, at most 25 words each."""

SCHEMA = {
    "type": "object",
    "properties": {
        "risk": {"type": "string", "enum": ["none", "low", "medium", "high"]},
        "summary": {"type": "string"},
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "type": {"type": "string", "enum": ["delisting", "listing", "hack_exploit", "unlock_emission",
                                                       "sale_buyback_burn", "migration_redenomination", "regulatory",
                                                       "depeg", "airdrop", "upgrade_launch", "other"]},
                    "direction": {"type": "string", "enum": ["against", "for", "neutral"]},
                    "summary": {"type": "string"},
                    "url": {"type": "string"},
                },
                "required": ["date", "type", "direction", "summary", "url"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["risk", "summary", "events"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------- schedule

def due_decision_day(now: pd.Timestamp, last_reviewed: str | None) -> tuple[pd.Timestamp | None, bool]:
    """(decision day to review, whether it is already past the fill), or (None, False) if nothing is due."""
    today = now.floor("D")
    d = today - pd.Timedelta(days=(today.weekday() - 6) % 7)  # most recent Sunday (today if Sunday)
    if now < d + READY_AFTER:
        d -= pd.Timedelta(days=7)
    if str(d.date()) == last_reviewed or now > d + GIVE_UP_AFTER:
        return None, False
    fill = d + pd.Timedelta(days=1, hours=PaperConfig.fill_hour_utc)
    return d, now >= fill


# ---------------------------------------------------------------- mapping

def _prefix(base: str) -> tuple[float, str]:
    m = re.match(r"^(10+)(?=[A-Z])", base)
    return (float(m.group(1)), base[m.end():]) if m else (1.0, base)


def mapping_issues(pairs: dict[str, str], bn_px: dict[str, float], bg_px: dict[str, float], tol: float = 0.02) -> list[dict]:
    """Binance/Bitget pairs whose price ratio is not the one implied by their prefixes."""
    out = []
    for bn, bg in sorted(pairs.items()):
        a, b = bn_px.get(bn), bg_px.get(bg)
        if not a or not b:
            out.append({"binance": bn, "bitget": bg, "issue": "sin precio en algún exchange"})
            continue
        expected = _prefix(bg.split("/")[0])[0] / _prefix(bn.split("/")[0])[0]
        ratio = (b / a) / expected
        if abs(ratio - 1.0) > tol:
            out.append({"binance": bn, "bitget": bg, "issue": f"precio Bitget/Binance {ratio:.3f}x lo esperado",
                        "ratio": ratio})
    return out


def _last_prices(ex, symbols: list[str]) -> dict[str, float]:
    out = {}
    for s, t in ex.fetch_tickers(symbols).items():
        bid, ask, last = t.get("bid"), t.get("ask"), t.get("last")
        px = (bid + ask) / 2.0 if bid and ask else last
        if px:
            out[s] = float(px)
    return out


# ---------------------------------------------------------------- events

def _api_key() -> str:
    for line in KEY_FILE.read_text().splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"ANTHROPIC_API_KEY not found in {KEY_FILE}")


def position_prompt(bn: str, bg: str, weight: float, today: pd.Timestamp) -> str:
    base = _prefix(bg.split("/")[0])[1]
    side = "LONG" if weight > 0 else "SHORT"
    return (f"Today is {today.date()} (UTC). Position: {side} {base} "
            f"(Bitget {bg}, Binance {bn}), {abs(weight):.1%} of equity, held for the next 7 days. "
            "Search for relevant events and rate the risk for this side.")


def parse_review(content: list) -> dict:
    texts = [b.text for b in content if getattr(b, "type", None) == "text"]
    if not texts:
        raise ValueError("no text block in the response")
    r = json.loads(texts[-1])
    if r.get("risk") not in ("none", "low", "medium", "high"):
        raise ValueError(f"bad risk {r.get('risk')!r}")
    return r


def review_position(client, bn: str, bg: str, weight: float, today: pd.Timestamp) -> dict:
    resp = client.messages.create(
        model=MODEL, max_tokens=4000, system=SYSTEM,
        messages=[{"role": "user", "content": position_prompt(bn, bg, weight, today)}],
        tools=[{"type": "web_search_20260318", "name": "web_search", "max_uses": MAX_SEARCHES}],
        thinking={"type": "adaptive"},
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
    )
    searches = getattr(getattr(resp.usage, "server_tool_use", None), "web_search_requests", 0) or 0
    cost = (resp.usage.input_tokens * PRICE_INPUT + resp.usage.output_tokens * PRICE_OUTPUT) / 1e6 + searches * PRICE_SEARCH
    return {"binance": bn, "bitget": bg, "weight": weight, "side": "long" if weight > 0 else "short",
            **parse_review(resp.content), "searches": searches, "cost_usd": round(cost, 4)}


def notices(review: dict) -> list[Notice]:
    """Alerts for doubtful mappings and for positions rated medium/high, plus one summary."""
    out = [Notice("alert", "revisión", f"mapeo dudoso {m['binance']} → {m['bitget']}: {m['issue']}")
           for m in review["mapping_issues"]]
    flagged = [p for p in review["positions"] if p.get("risk") in ("medium", "high")]
    for p in sorted(flagged, key=lambda p: p["risk"] != "high"):
        base = _prefix(p["bitget"].split("/")[0])[1]
        lado = "largo" if p["side"] == "long" else "corto"
        out.append(Notice("alert", "revisión", f"{base} {lado}: riesgo {p['risk']} — {p['summary']}"))
    failed = [p["binance"] for p in review["positions"] if "error" in p]
    when = "después del fill" if review["post_fill"] else "antes del fill"
    text = (f"revisión {review['decision_day']} ({when}): {len(review['positions'])} posiciones, "
            f"{len(flagged)} con riesgo, mapeo {'OK' if not review['mapping_issues'] else 'con problemas'}"
            f"{f', {len(failed)} sin revisar' if failed else ''}, costo {review['cost_usd']:.2f} USD")
    return out + [Notice("info", "revisión", text)]


# ---------------------------------------------------------------- run

def run(args) -> None:
    import anthropic
    import ccxt

    now = pd.Timestamp(args.now, tz="UTC") if args.now else pd.Timestamp.now(tz="UTC")
    state_dir = Path(args.state_dir)
    (state_dir / "reviews").mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    def log(msg: str) -> None:
        print(f"[{now:%Y-%m-%dT%H:%M:%SZ}] {msg}", flush=True)

    d, post_fill = due_decision_day(now, state.get("last_decision_day"))
    if d is None:
        log("nothing due")
        return
    try:
        bitget = ccxt.bitget({"enableRateLimit": True, "timeout": 20000, "options": {"defaultType": "swap"}})
        listed = {k for k, v in bitget.load_markets().items()
                  if v.get("swap") and v.get("linear") and v.get("settle") == "USDT" and v.get("active", True)}
        members = sorted(xs_signal.ensure_membership(state_dir / "membership.csv", d.strftime("%Y-%m"), PaperConfig(),
                                                     args.cache_dir, log)["symbol"])
        tgt = xs_signal.compute_targets(d, members, listed, args.cache_dir, log)

        pairs = {s: bitget_symbol(s, listed) for s in members if bitget_symbol(s, listed)}
        issues = mapping_issues(pairs, _last_prices(xs_signal.binance_exchange(), list(pairs)),
                                _last_prices(bitget, list(pairs.values())))
        log(f"decision {d.date()}: {len(tgt['weights'])} positions, {len(pairs)} mapped members, "
            f"{len(issues)} mapping issues")

        client = anthropic.Anthropic(api_key=_api_key(), max_retries=4)

        def one(item):
            bn, w = item
            try:
                return review_position(client, bn, pairs[bn], w, now)
            except Exception as exc:  # noqa: BLE001 - one failed position must not hide the others
                return {"binance": bn, "bitget": pairs[bn], "weight": w, "side": "long" if w > 0 else "short",
                        "error": f"{type(exc).__name__}: {exc}"[:300]}

        t0 = time.perf_counter()
        with ThreadPoolExecutor(args.workers) as pool:
            positions = list(pool.map(one, sorted(tgt["weights"].items())))
    except Exception as exc:  # noqa: BLE001 - notify once per decision day, retry next hour
        log(f"review failed: {type(exc).__name__}: {exc}")
        if state.get("failed_decision_day") != str(d.date()) and not args.no_notify:
            notify(Notice("alert", "revisión", f"la revisión {d.date()} falló: {type(exc).__name__}: {str(exc)[:120]}"))
        state["failed_decision_day"] = str(d.date())
        state_path.write_text(json.dumps(state, indent=2))
        raise

    review = {"decision_day": str(d.date()), "ts": f"{now:%Y-%m-%dT%H:%M:%SZ}", "post_fill": post_fill,
              "model": MODEL, "targets": tgt["weights"], "mapping_issues": issues, "positions": positions,
              "seconds": round(time.perf_counter() - t0, 1),
              "cost_usd": round(sum(p.get("cost_usd", 0.0) for p in positions), 4)}
    (state_dir / "reviews" / f"{d.date()}.json").write_text(json.dumps(review, indent=2, ensure_ascii=False))
    for n in notices(review):
        log(f"{n.level.upper()} {n.text}")
        if not args.no_notify:
            notify(n)
    state.update(last_decision_day=str(d.date()), failed_decision_day=None)
    state_path.write_text(json.dumps(state, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run"])
    ap.add_argument("--state-dir", default="runtime/event_review")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-notify", action="store_true", help="print only")
    ap.add_argument("--now", default="", help="override the clock (testing only)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
