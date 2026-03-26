from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


@dataclass
class PositionState:
    symbol: str
    strategy_id: str
    side: str
    entry_ts: int
    entry_px: float
    qty: float
    entry_atr: float
    entry_adx: float = float("nan")
    entry_atrp: float = float("nan")
    entry_strength: float = 0.0
    entry_reason: str = ""
    entry_meta: dict[str, Any] = field(default_factory=dict)

    bars_held: int = 0
    cooldown_left: int = 0

    trail_stop: Optional[float] = None
    breakeven_armed: bool = False

    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

    def signed_ret(self, price: float) -> float:
        px = _f(price, self.entry_px)
        if self.entry_px == 0.0:
            return 0.0
        if str(self.side).lower() == "long":
            return float(px / self.entry_px - 1.0)
        if str(self.side).lower() == "short":
            return float(self.entry_px / max(px, 1e-12) - 1.0)
        return 0.0

    def move_atr(self, price: float, atr_now: float | None = None) -> float:
        px = _f(price, self.entry_px)
        atr = _f(atr_now, self.entry_atr)
        if atr <= 0.0:
            return 0.0
        if str(self.side).lower() == "long":
            return float((px - self.entry_px) / atr)
        if str(self.side).lower() == "short":
            return float((self.entry_px - px) / atr)
        return 0.0


@dataclass
class ExitDecision:
    action: str = "hold"  # hold | close | partial_close
    exit_reason: str = ""
    exit_price: Optional[float] = None

    tp_price: Optional[float] = None
    sl_price: Optional[float] = None

    trail_stop: Optional[float] = None
    breakeven_armed: Optional[bool] = None

    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeRecord:
    symbol: str
    strategy_id: str
    side: str
    entry_ts: int
    exit_ts: int
    entry_px: float
    exit_px: float
    qty: float
    pnl: float
    exit_reason: str
    fee: float = 0.0
    bars_held: int = 0
    entry_adx: float = float("nan")
    entry_atrp: float = float("nan")
    entry_atr: float = float("nan")
    meta: dict[str, Any] = field(default_factory=dict)
