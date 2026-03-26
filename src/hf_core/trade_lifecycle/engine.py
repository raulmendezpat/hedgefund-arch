from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .contracts import ExitDecision, PositionState, TradeRecord
from .exit_policies import ExitPolicy, build_exit_policy


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


@dataclass
class TradeLifecycleEngine:
    maker_fee: float = 0.0
    taker_fee: float = 0.0
    cooldown_after_close_bars: int = 0

    open_positions: dict[str, PositionState] = field(default_factory=dict)
    cooldown_by_symbol: dict[str, int] = field(default_factory=dict)
    trade_log: list[TradeRecord] = field(default_factory=list)

    def _position_key(self, symbol: str) -> str:
        return str(symbol)

    def get_open_position(self, symbol: str) -> Optional[PositionState]:
        return self.open_positions.get(self._position_key(symbol))

    def decrement_cooldowns(self) -> None:
        updated = {}
        for sym, left in dict(self.cooldown_by_symbol or {}).items():
            n = max(0, int(left) - 1)
            if n > 0:
                updated[sym] = n
        self.cooldown_by_symbol = updated

    def can_open(self, symbol: str) -> bool:
        if self.get_open_position(symbol) is not None:
            return False
        return int(self.cooldown_by_symbol.get(str(symbol), 0) or 0) <= 0

    def open_position(
        self,
        *,
        symbol: str,
        strategy_id: str,
        side: str,
        entry_ts: int,
        entry_px: float,
        qty: float,
        entry_atr: float,
        entry_adx: float = float("nan"),
        entry_atrp: float = float("nan"),
        entry_strength: float = 0.0,
        entry_reason: str = "",
        entry_meta: dict[str, Any] | None = None,
    ) -> PositionState:
        pos = PositionState(
            symbol=str(symbol),
            strategy_id=str(strategy_id),
            side=str(side).lower(),
            entry_ts=int(entry_ts),
            entry_px=float(entry_px),
            qty=float(qty),
            entry_atr=float(entry_atr),
            entry_adx=float(entry_adx),
            entry_atrp=float(entry_atrp),
            entry_strength=float(entry_strength),
            entry_reason=str(entry_reason),
            entry_meta=dict(entry_meta or {}),
        )
        self.open_positions[self._position_key(symbol)] = pos
        return pos

    def _realize_trade(
        self,
        *,
        position: PositionState,
        exit_ts: int,
        exit_price: float,
        exit_reason: str,
    ) -> TradeRecord:
        fee = float(abs(exit_price * position.qty) * self.taker_fee)

        if str(position.side).lower() == "long":
            pnl = float((exit_price - position.entry_px) * position.qty - fee)
        else:
            pnl = float((position.entry_px - exit_price) * position.qty - fee)

        rec = TradeRecord(
            symbol=str(position.symbol),
            strategy_id=str(position.strategy_id),
            side=str(position.side),
            entry_ts=int(position.entry_ts),
            exit_ts=int(exit_ts),
            entry_px=float(position.entry_px),
            exit_px=float(exit_price),
            qty=float(position.qty),
            pnl=float(pnl),
            exit_reason=str(exit_reason),
            fee=float(fee),
            bars_held=int(position.bars_held),
            entry_adx=float(position.entry_adx),
            entry_atrp=float(position.entry_atrp),
            entry_atr=float(position.entry_atr),
            meta=dict(position.entry_meta or {}),
        )
        self.trade_log.append(rec)
        self.open_positions.pop(self._position_key(position.symbol), None)
        cd = int(position.entry_meta.get("cooldown_after_close_bars", self.cooldown_after_close_bars) or self.cooldown_after_close_bars)
        if cd > 0:
            self.cooldown_by_symbol[str(position.symbol)] = int(cd)
        return rec

    def evaluate_exit(
        self,
        *,
        symbol: str,
        prev_bar: dict[str, Any],
        current_bar: dict[str, Any],
        exit_policy_cfg: dict[str, Any] | None,
        context: dict[str, Any] | None = None,
        exit_ts: int | None = None,
    ) -> ExitDecision:
        pos = self.get_open_position(symbol)
        if pos is None:
            return ExitDecision(action="hold", exit_reason="no_position")

        pos.bars_held += 1

        policy: ExitPolicy = build_exit_policy(exit_policy_cfg)
        decision = policy.evaluate(
            position=pos,
            prev_bar=dict(prev_bar or {}),
            current_bar=dict(current_bar or {}),
            context=dict(context or {}),
        )

        if decision.trail_stop is not None:
            pos.trail_stop = float(decision.trail_stop)
        if decision.breakeven_armed is not None:
            pos.breakeven_armed = bool(decision.breakeven_armed)

        close_now = _f(prev_bar.get("close"), pos.entry_px)
        move = float(pos.move_atr(close_now, _f(prev_bar.get("atr"), pos.entry_atr)))
        pos.max_favorable_excursion = max(float(pos.max_favorable_excursion), float(move))
        pos.max_adverse_excursion = min(float(pos.max_adverse_excursion), float(move))

        if str(decision.action).lower() == "close" and decision.exit_price is not None:
            self._realize_trade(
                position=pos,
                exit_ts=int(exit_ts if exit_ts is not None else current_bar.get("timestamp", pos.entry_ts)),
                exit_price=float(decision.exit_price),
                exit_reason=str(decision.exit_reason or "close"),
            )

        return decision

    def force_close(
        self,
        *,
        symbol: str,
        exit_ts: int,
        exit_price: float,
        exit_reason: str = "force_close",
    ) -> Optional[TradeRecord]:
        pos = self.get_open_position(symbol)
        if pos is None:
            return None
        return self._realize_trade(
            position=pos,
            exit_ts=int(exit_ts),
            exit_price=float(exit_price),
            exit_reason=str(exit_reason),
        )

    def snapshot_open_positions(self) -> list[dict[str, Any]]:
        out = []
        for pos in self.open_positions.values():
            out.append(
                {
                    "symbol": str(pos.symbol),
                    "strategy_id": str(pos.strategy_id),
                    "side": str(pos.side),
                    "entry_ts": int(pos.entry_ts),
                    "entry_px": float(pos.entry_px),
                    "qty": float(pos.qty),
                    "entry_atr": float(pos.entry_atr),
                    "entry_adx": float(pos.entry_adx),
                    "entry_atrp": float(pos.entry_atrp),
                    "bars_held": int(pos.bars_held),
                    "trail_stop": None if pos.trail_stop is None else float(pos.trail_stop),
                    "breakeven_armed": bool(pos.breakeven_armed),
                    "mfe_atr": float(pos.max_favorable_excursion),
                    "mae_atr": float(pos.max_adverse_excursion),
                }
            )
        return out
