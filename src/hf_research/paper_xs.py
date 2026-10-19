"""Paper trading of the pre-registered cross-sectional momentum hypothesis on Bitget.

Fixed before the paper run starts (do not tune while it runs):
- Strategy: PRIMARY (long/short, lookback 30d, skip 1d, 20% per side, equal weight, band 0.02),
  signal and universe exactly as in the backtest (Binance 1h closes at 23:00 UTC, monthly top 60
  by Binance quote volume), restricted to symbols listed on Bitget at decision time.
- Schedule: decision on Sunday's 23:00 UTC close, fill on Monday at 01:00 UTC (the backtest's
  exec bar). A fill later than max_fill_delay_hours is skipped and logged as missed.
- Execution: market orders simulated against the live Bitget order book (walking the levels),
  taker fee 0.06%. Funding: actual Bitget funding events on the held positions.

Accounting (linear USDT perpetuals, one margin balance):
- mark: equity += qty * (price - last_price)
- trade at fill price f with mid m: equity -= dq * (f - m) + |dq| * f * fee
- funding event with rate r at price p: equity -= qty * p * r (longs pay positive funding)
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import pandas as pd

PRIMARY_STRATEGY = dict(mode="long_short", lookback_days=30, skip_days=1, rebalance_days=7, select_frac=0.2,
                        weighting="equal", band=0.02, fee=0.0006, slippage_bps=3.0)


@dataclass(frozen=True)
class PaperConfig:
    name: str = "xs_momentum_v1"
    capital_usdt: float = 10_000.0
    taker_fee: float = 0.0006
    decision_weekday: int = 6  # Sunday (Monday = 0)
    fill_hour_utc: int = 1
    max_fill_delay_hours: int = 24
    top_n: int = 60
    window_days: int = 30
    min_history_days: int = 60
    min_order_usdt: float = 5.0


def scheduled_fill(decision_day: pd.Timestamp, cfg: PaperConfig) -> pd.Timestamp:
    return decision_day + pd.Timedelta(days=1, hours=cfg.fill_hour_utc)


def first_decision_day_after(start: pd.Timestamp, cfg: PaperConfig) -> pd.Timestamp:
    """First decision day whose scheduled fill is at or after `start`."""
    d = start.floor("D") - pd.Timedelta(days=(start.floor("D").weekday() - cfg.decision_weekday) % 7)
    while scheduled_fill(d, cfg) < start:
        d += pd.Timedelta(days=7)
    return d


def due_decision_day(now: pd.Timestamp, start: pd.Timestamp, last_decision_day: pd.Timestamp | None,
                     cfg: PaperConfig) -> tuple[pd.Timestamp | None, str]:
    """Latest decision day whose fill time has passed, and whether to trade it ("due") or not ("missed").

    Returns (None, "idle") when nothing is pending: before the first fill after `start`, or when
    the latest decision day was already handled."""
    today = now.floor("D")
    back = (today.weekday() - cfg.decision_weekday) % 7
    d = today - pd.Timedelta(days=back)
    if scheduled_fill(d, cfg) > now:
        d -= pd.Timedelta(days=7)
    fill = scheduled_fill(d, cfg)
    if fill < start or (last_decision_day is not None and last_decision_day >= d):
        return None, "idle"
    if now - fill > pd.Timedelta(hours=cfg.max_fill_delay_hours):
        return d, "missed"
    return d, "due"


def walk_book(levels: list, qty: float) -> tuple[float, float, bool]:
    """Average price to take `qty` base units from one side of the book.

    Returns (avg_price, visible_qty_filled, exhausted). When the visible book is too thin, the
    remainder is assumed filled at the worst visible price and exhausted=True."""
    left, cost, got = qty, 0.0, 0.0
    for px, amt, *_ in levels:
        take = min(left, float(amt))
        cost += take * float(px)
        got += take
        left -= take
        if left <= 1e-12:
            return cost / got, got, False
    if not levels:
        return float("nan"), 0.0, True
    worst = float(levels[-1][0])
    return (cost + left * worst) / qty, got, True


@dataclass
class PaperBook:
    equity: float
    qty: dict[str, float] = field(default_factory=dict)
    last_px: dict[str, float] = field(default_factory=dict)

    def mark(self, prices: dict[str, float]) -> float:
        pnl = 0.0
        for s, q in self.qty.items():
            p = prices.get(s)
            if p is None or not math.isfinite(p):
                continue
            pnl += q * (p - self.last_px[s])
            self.last_px[s] = p
        self.equity += pnl
        return pnl

    def weights(self) -> dict[str, float]:
        return {s: q * self.last_px[s] / self.equity for s, q in self.qty.items()}

    def trade(self, symbol: str, dq: float, fill_px: float, mid: float, fee_rate: float) -> dict:
        if symbol in self.qty:
            self.mark({symbol: mid})
        slip = dq * (fill_px - mid)
        fee = abs(dq) * fill_px * fee_rate
        self.equity -= slip + fee
        q = self.qty.get(symbol, 0.0) + dq
        if abs(q * mid) < 1e-6:
            self.qty.pop(symbol, None)
            self.last_px.pop(symbol, None)
        else:
            self.qty[symbol] = q
            self.last_px[symbol] = mid
        return {"slippage_usdt": slip, "fee_usdt": fee,
                "slippage_bps": (fill_px / mid - 1.0) * 10000.0 * (1 if dq > 0 else -1)}

    def funding(self, symbol: str, rate: float, price: float) -> float:
        pay = self.qty.get(symbol, 0.0) * price * rate
        self.equity -= pay
        return -pay

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PaperBook":
        return cls(equity=float(d["equity"]), qty=dict(d.get("qty", {})), last_px=dict(d.get("last_px", {})))


def plan_weights(targets: dict[str, float], current: dict[str, float], band: float) -> dict[str, tuple[float, float]]:
    """{symbol: (current_weight, new_weight)} for every symbol that must trade.

    Same rule as the backtest: a same-side resize smaller than `band` is skipped; opening,
    closing and flipping always trade."""
    out = {}
    for s in sorted(set(targets) | set(current)):
        cur, new = current.get(s, 0.0), targets.get(s, 0.0)
        if cur != 0.0 and new != 0.0 and (cur > 0) == (new > 0) and abs(new - cur) < band:
            continue
        if abs(new - cur) > 1e-12:
            out[s] = (cur, new)
    return out
