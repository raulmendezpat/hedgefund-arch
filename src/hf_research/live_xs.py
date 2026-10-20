"""Live execution of the cross-sectional momentum hypothesis on Bitget: pure decision logic.

Same signal, universe, schedule and band rule as the paper runner (hf_research.paper_xs and
hf_research.xs_signal). What differs is that the exchange account is the source of truth:
positions and equity are read from Bitget on every run, never from a simulation.

Guardrails (all evaluated before any order is built):
- Kill switch: equity drawdown from its peak >= max_drawdown -> flatten everything and pause
  until an operator reviews it and runs reset-risk (a review pause, not a retirement).
- Daily loss: equity below the UTC day-start equity by >= max_daily_loss -> reduce-only until
  the next UTC day (positions can shrink or close, never grow or open).
- Sanity limits: a target weight above max_symbol_weight, gross exposure above gross_leverage,
  or any single order above max_order_usdt aborts the whole rebalance (it signals a bug or bad
  data; it is never silently clipped).
- Foreign positions (symbols this executor never traded) or hedge-mode positions abort the run:
  the executor must own a dedicated one-way-mode (sub)account.

Deposits and withdrawals move equity too. The runner reads the account's transfer bills every
run and shifts peak and day-start equity by the net amount (apply_cash_flow), so a transfer is
never mistaken for a profit or a loss.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

from hf_research.paper_xs import plan_weights


class SanityError(RuntimeError):
    """A limit that should never be hit by a correct run was hit: do not trade."""


@dataclass(frozen=True)
class LiveConfig:
    name: str = "xs_momentum_live_v1"
    gross_leverage: float = 1.0  # sum of |weights| times equity is the gross notional
    exchange_leverage: int = 3  # Bitget leverage setting per symbol (margin headroom, not exposure)
    margin_mode: str = "cross"
    max_symbol_weight: float = 0.10  # primary targets are ~0.04 per symbol
    max_order_usdt: float = 5_000.0
    min_order_usdt: float = 5.0
    max_drawdown: float = 0.35  # ~ the worst hourly drawdown of the x1 backtest (-26% IS, -35% OOS)
    max_daily_loss: float = 0.08
    min_signal_coverage: float = 0.9
    max_position_drift: float = 0.01  # post-trade |w_actual - w_target| reported as a breach
    position_loss_alert: float = 0.015  # alert when one position loses this share of equity since the last rebalance
    taker_fee: float = 0.0006
    decision_weekday: int = 6  # Sunday
    fill_hour_utc: int = 1
    max_fill_delay_hours: int = 24
    top_n: int = 60
    window_days: int = 30
    min_history_days: int = 60

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- risk

def new_risk_state(equity: float, now: pd.Timestamp) -> dict:
    return {"peak_equity": float(equity), "day": str(now.date()), "day_start_equity": float(equity),
            "halted": False, "halt_reason": None}


def check_risk(risk: dict, equity: float, now: pd.Timestamp, cfg: LiveConfig) -> tuple[str, str]:
    """Update the risk state in place and return (status, reason).

    status: "ok" | "reduce_only" | "halt" (newly tripped) | "halted" (tripped earlier)."""
    if risk.get("halted"):
        return "halted", str(risk.get("halt_reason"))
    risk["peak_equity"] = max(float(risk["peak_equity"]), float(equity))
    today = str(now.date())
    if risk["day"] != today:
        risk["day"] = today
        risk["day_start_equity"] = float(equity)
    dd = 1.0 - equity / risk["peak_equity"]
    if dd >= cfg.max_drawdown:
        risk["halted"] = True
        risk["halt_reason"] = f"drawdown {dd:.1%} >= {cfg.max_drawdown:.0%} (peak {risk['peak_equity']:.2f}, equity {equity:.2f})"
        return "halt", risk["halt_reason"]
    day_loss = 1.0 - equity / risk["day_start_equity"]
    if day_loss >= cfg.max_daily_loss:
        return "reduce_only", f"daily loss {day_loss:.1%} >= {cfg.max_daily_loss:.0%}"
    return "ok", ""


def position_losses(qty: dict[str, float], ref_px: dict[str, float], px: dict[str, float], ref_equity: float,
                    threshold: float) -> dict[str, float]:
    """Positions whose loss since their reference price exceeds `threshold` of the reference equity.

    Returns {symbol: loss as a positive fraction of equity}. Alert only: nothing is traded on it."""
    out = {}
    if ref_equity <= 0:
        return out
    for s, q in qty.items():
        p0, p1 = ref_px.get(s), px.get(s)
        if not q or p0 is None or p1 is None or not (math.isfinite(p0) and math.isfinite(p1)):
            continue
        loss = -q * (p1 - p0) / ref_equity
        if loss > threshold:
            out[s] = float(loss)
    return out


# ---------------------------------------------------------------- cash flows

# Bitget USDT-M futures bill types that move money in or out of the futures account. Moves
# between cross and isolated margin stay inside it and are not cash flows.
EXTERNAL_TRANSFER_TYPES = (
    "trans_from_contract", "trans_to_contract", "trans_from_exchange", "trans_to_exchange",
    "trans_from_otc", "trans_to_otc", "trans_from_strategy", "trans_to_strategy",
)
CASH_FLOW_OVERLAP_MS = 2 * 3600 * 1000  # re-query this far back: bills can appear late


def apply_cash_flow(risk: dict, amount: float) -> None:
    """Shift peak and day-start equity by a deposit (+) or withdrawal (-)."""
    risk["peak_equity"] = float(risk["peak_equity"]) + amount
    risk["day_start_equity"] = float(risk["day_start_equity"]) + amount


def new_cash_flows(cf_state: dict, flows: list[dict], now_ms: int) -> tuple[list[dict], dict]:
    """Flows not seen before, and the next cursor state.

    Each run queries [since_ms - CASH_FLOW_OVERLAP_MS, now]; ids seen inside the overlap window
    are remembered so a bill is counted exactly once even if it shows up late."""
    seen = set(cf_state.get("seen", []))
    fresh = [f for f in flows if f["id"] not in seen]
    keep_from = now_ms - CASH_FLOW_OVERLAP_MS
    return fresh, {"since_ms": now_ms, "seen": sorted(f["id"] for f in flows if f["ts"] >= keep_from)}


# ---------------------------------------------------------------- fills

def fill_record(order: dict, fill: dict) -> dict:
    """Execution quality of one sent order: slippage against the pre-trade mid (positive = cost)
    and fees, both in bps of the filled notional."""
    avg, filled, mid = float(fill["average"]), float(fill["filled"]), float(order["mid"])
    sign = 1.0 if order["side"] == "buy" else -1.0
    notional = filled * avg
    fee = float(fill.get("fee_usdt") or 0.0)
    has_mid = math.isfinite(mid) and mid > 0  # kill-switch orders may have no reference mid
    return {"ts": order["ts"], "tag": order["tag"], "symbol": order["symbol"], "side": order["side"],
            "reason": order["reason"], "order_id": fill["id"], "client_oid": order["client_oid"],
            "qty": float(order["qty"]), "filled": filled, "mid": mid, "avg_px": avg, "notional_usdt": notional,
            "slippage_bps": sign * (avg / mid - 1.0) * 1e4 if has_mid else float("nan"),
            "slippage_usdt": sign * filled * (avg - mid) if has_mid else float("nan"),
            "fee_usdt": fee, "fee_bps": fee / notional * 1e4 if notional else float("nan")}


# ---------------------------------------------------------------- orders

@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: str  # "buy" | "sell"
    qty: float  # base units, > 0, already at exchange precision
    reduce_only: bool
    mid: float
    w_from: float
    w_to: float
    reason: str

    @property
    def notional(self) -> float:
        return self.qty * self.mid

    def to_dict(self) -> dict:
        return {**asdict(self), "notional_usdt": self.notional}


def reduce_only_targets(targets: dict[str, float], current: dict[str, float]) -> dict[str, float]:
    """Clamp targets so no position grows, opens or flips: only shrink toward zero or close."""
    out = {}
    for s, cur in current.items():
        t = targets.get(s, 0.0)
        if cur == 0.0:
            continue
        if t == 0.0 or (t > 0) != (cur > 0):
            out[s] = 0.0
        else:
            out[s] = t if abs(t) < abs(cur) else cur
    return out


def build_orders(targets: dict[str, float], positions: dict[str, float], mids: dict[str, float], equity: float,
                 cfg: LiveConfig, band: float, to_precision, *, reduce_only_mode: bool = False
                 ) -> tuple[list[OrderIntent], list[dict], dict[str, float]]:
    """Orders that move the account from `positions` (signed base qty) to `targets` (weights of equity).

    Returns (orders, skipped, intended): reduce-only orders first; skipped dust orders; and the
    weights the account should hold after the orders fill (targets after the band, the
    reduce-only clamp and skipped orders), which is what post-trade verification compares to. `to_precision(symbol, qty)` rounds a
    positive base quantity to the exchange step (it may return 0 for dust)."""
    if equity <= 0 or not math.isfinite(equity):
        raise SanityError(f"non-positive equity {equity}")
    scaled = {s: w * cfg.gross_leverage for s, w in targets.items() if w != 0.0}
    for s, w in scaled.items():
        if abs(w) > cfg.max_symbol_weight + 1e-12:
            raise SanityError(f"target weight {w:.4f} for {s} exceeds max_symbol_weight {cfg.max_symbol_weight}")
    gross = sum(abs(w) for w in scaled.values())
    if gross > cfg.gross_leverage + 1e-9:
        raise SanityError(f"gross target {gross:.3f} exceeds gross_leverage {cfg.gross_leverage}")
    missing = [s for s in set(scaled) | set(positions) if not (mids.get(s) and math.isfinite(mids[s]))]
    if missing:
        raise SanityError(f"no price for {sorted(missing)}")

    current = {s: q * mids[s] / equity for s, q in positions.items() if q != 0.0}
    if reduce_only_mode:
        scaled = reduce_only_targets(scaled, current)
    plan = plan_weights(scaled, current, band)

    orders, skipped = [], []
    intended = dict(current)
    for s, (w_cur, w_new) in plan.items():
        mid, cur_q = mids[s], positions.get(s, 0.0)
        tgt_q = w_new * equity / mid
        if cur_q != 0.0 and (w_new == 0.0 or (w_new > 0) != (cur_q > 0)):
            # Close the whole position with its exact exchange quantity, then open the new side.
            orders.append(OrderIntent(s, "sell" if cur_q > 0 else "buy", abs(cur_q), True, mid, w_cur, 0.0,
                                      "close" if w_new == 0.0 else "flip_close"))
            if w_new != 0.0:
                q = float(to_precision(s, abs(tgt_q)))
                if q * mid < cfg.min_order_usdt:
                    skipped.append({"symbol": s, "reason": "below_min_order", "qty": abs(tgt_q)})
                    intended[s] = 0.0
                else:
                    orders.append(OrderIntent(s, "buy" if tgt_q > 0 else "sell", q, False, mid, 0.0, w_new, "flip_open"))
                    intended[s] = w_new
            else:
                intended[s] = 0.0
            continue
        delta = tgt_q - cur_q
        shrinking = cur_q != 0.0 and abs(tgt_q) < abs(cur_q)
        q = float(to_precision(s, abs(delta)))
        if q * mid < cfg.min_order_usdt:
            skipped.append({"symbol": s, "reason": "below_min_order", "qty": abs(delta)})
            continue
        intended[s] = w_new
        orders.append(OrderIntent(s, "buy" if delta > 0 else "sell", q, shrinking, mid, w_cur, w_new,
                                  "reduce" if shrinking else ("open" if cur_q == 0.0 else "increase")))

    for o in orders:
        if o.notional > cfg.max_order_usdt and o.reason not in ("close", "flip_close"):
            raise SanityError(f"order {o.symbol} {o.side} {o.notional:.2f} USDT exceeds max_order_usdt {cfg.max_order_usdt}")
    orders.sort(key=lambda o: (not o.reduce_only, o.symbol))
    return orders, skipped, {s: w for s, w in intended.items() if w != 0.0}


def flatten_orders(positions: dict[str, float], mids: dict[str, float]) -> list[OrderIntent]:
    """Reduce-only market orders that close every position (kill switch)."""
    return [OrderIntent(s, "sell" if q > 0 else "buy", abs(q), True, float(mids.get(s, float("nan"))),
                        float("nan"), 0.0, "kill_switch") for s, q in sorted(positions.items()) if q != 0.0]


def position_drift(intended: dict[str, float], positions: dict[str, float], mids: dict[str, float],
                   equity: float, cfg: LiveConfig) -> dict[str, float]:
    """Post-trade actual minus intended weight per symbol, only where |drift| > max_position_drift."""
    out = {}
    for s in set(intended) | set(positions):
        if s not in mids:
            continue
        actual = positions.get(s, 0.0) * mids[s] / equity
        d = actual - intended.get(s, 0.0)
        if abs(d) > cfg.max_position_drift:
            out[s] = float(d)
    return out
