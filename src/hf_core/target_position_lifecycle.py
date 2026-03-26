from __future__ import annotations

"""
LEGACY / DEPRECATED MODULE

This module contains an older target-position lifecycle implementation kept
temporarily for reference during the migration.

Canonical lifecycle domain going forward:
- hf_core.trade_lifecycle.contracts
- hf_core.trade_lifecycle.exit_policies
- hf_core.trade_lifecycle.engine

Do not add new behavior here unless explicitly required for migration diffing.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _s(x: Any, default: str = "") -> str:
    if x is None:
        return str(default)
    return str(x)


def _side_from_weight(w: float, eps: float = 1e-12) -> str:
    if float(w) > float(eps):
        return "long"
    if float(w) < -float(eps):
        return "short"
    return "flat"


def _sign(x: float, eps: float = 1e-12) -> int:
    if float(x) > float(eps):
        return 1
    if float(x) < -float(eps):
        return -1
    return 0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))


def infer_strategy_family(strategy_id: str) -> str:
    sid = _s(strategy_id).lower()

    if sid in {"sol_bbrsi", "sol_extreme_mr"}:
        return "mean_reversion"

    if sid in {"sol_trend_pullback"}:
        return "pullback_trend"

    if sid in {"sol_vol_breakout", "sol_vol_expansion"} or "breakout" in sid or "expansion" in sid:
        return "breakout"

    if sid in {"sol_vol_compression"} or "compression" in sid:
        return "compression"

    if sid.endswith("_trend") or sid in {"btc_trend", "btc_trend_loose"}:
        return "trend"

    return "unknown"


@dataclass
class ExitDecision:
    action: str  # hold | close
    exit_reason: str = ""
    exit_price: Optional[float] = None
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    trail_stop: Optional[float] = None
    breakeven_armed: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class LifecycleAction:
    action: str  # hold | open | increase | decrease | close | reverse | skip
    symbol: str
    strategy_id: str
    family: str
    side: str
    reason: str
    prev_weight: float
    target_weight: float
    delta_weight: float
    close_weight: float = 0.0
    open_weight: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionState:
    symbol: str
    strategy_id: str
    family: str
    side: str
    entry_ts: int
    last_ts: int
    entry_price: float
    qty: float
    current_weight: float
    target_weight: float
    bars_held: int = 0
    peak_price: Optional[float] = None
    trough_price: Optional[float] = None
    trail_stop: Optional[float] = None
    breakeven_armed: bool = False
    cooldown_left: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def signed_qty(self) -> float:
        if self.side == "long":
            return float(abs(self.qty))
        if self.side == "short":
            return float(-abs(self.qty))
        return 0.0


@dataclass
class TrendAtrDynamicExitPolicy:
    tp_atr_mult: float = 2.6
    stop_atr_mult: float = 1.5
    breakeven_activate_atr: float = 0.9
    breakeven_offset_atr: float = 0.1
    trail_activate_atr: float = 1.0
    trail_stop_atr_mult: float = 1.2
    max_hold_bars: int = 240
    use_trend_break: bool = True

    def evaluate(
        self,
        *,
        position: PositionState,
        candle_prev: dict[str, Any],
        candle_now: dict[str, Any],
        target_weight: float,
        signal_side: str,
        regime_on: bool = True,
    ) -> ExitDecision:
        if position.side not in {"long", "short"}:
            return ExitDecision(action="hold")

        atr_now = _f(candle_prev.get("atr"), float("nan"))
        close_now = _f(candle_prev.get("close"), float("nan"))
        ema_fast = _f(candle_prev.get("ema_fast"), float("nan"))
        ema_slow = _f(candle_prev.get("ema_slow"), float("nan"))

        h = _f(candle_now.get("high"), float("nan"))
        l = _f(candle_now.get("low"), float("nan"))

        if atr_now != atr_now or atr_now <= 0.0:
            return ExitDecision(action="hold", meta={"skip_reason": "atr_nan"})

        entry = float(position.entry_price)

        if position.side == "long":
            tp = entry + float(self.tp_atr_mult) * atr_now
            sl = entry - float(self.stop_atr_mult) * atr_now
            move_atr = (close_now - entry) / atr_now if atr_now > 0 else 0.0

            breakeven_armed = bool(position.breakeven_armed)
            if move_atr >= float(self.breakeven_activate_atr):
                breakeven_armed = True
                sl = max(sl, entry + float(self.breakeven_offset_atr) * atr_now)

            trail_stop = position.trail_stop
            if move_atr >= float(self.trail_activate_atr):
                dyn_trail = close_now - float(self.trail_stop_atr_mult) * atr_now
                trail_stop = dyn_trail if trail_stop is None else max(float(trail_stop), float(dyn_trail))
                sl = max(sl, float(trail_stop))

            if l <= sl <= h:
                return ExitDecision(
                    action="close",
                    exit_reason="sl",
                    exit_price=float(sl),
                    tp_price=float(tp),
                    sl_price=float(sl),
                    trail_stop=trail_stop,
                    breakeven_armed=breakeven_armed,
                )
            if l <= tp <= h:
                return ExitDecision(
                    action="close",
                    exit_reason="tp",
                    exit_price=float(tp),
                    tp_price=float(tp),
                    sl_price=float(sl),
                    trail_stop=trail_stop,
                    breakeven_armed=breakeven_armed,
                )

            if self.use_trend_break:
                trend_break = False
                if ema_fast == ema_fast and ema_slow == ema_slow:
                    trend_break = not (close_now > ema_fast and ema_fast > ema_slow)
                if trend_break:
                    return ExitDecision(
                        action="close",
                        exit_reason="trend_break",
                        exit_price=float(close_now),
                        tp_price=float(tp),
                        sl_price=float(sl),
                        trail_stop=trail_stop,
                        breakeven_armed=breakeven_armed,
                    )

        else:
            tp = entry - float(self.tp_atr_mult) * atr_now
            sl = entry + float(self.stop_atr_mult) * atr_now
            move_atr = (entry - close_now) / atr_now if atr_now > 0 else 0.0

            breakeven_armed = bool(position.breakeven_armed)
            if move_atr >= float(self.breakeven_activate_atr):
                breakeven_armed = True
                sl = min(sl, entry - float(self.breakeven_offset_atr) * atr_now)

            trail_stop = position.trail_stop
            if move_atr >= float(self.trail_activate_atr):
                dyn_trail = close_now + float(self.trail_stop_atr_mult) * atr_now
                trail_stop = dyn_trail if trail_stop is None else min(float(trail_stop), float(dyn_trail))
                sl = min(sl, float(trail_stop))

            if l <= sl <= h:
                return ExitDecision(
                    action="close",
                    exit_reason="sl",
                    exit_price=float(sl),
                    tp_price=float(tp),
                    sl_price=float(sl),
                    trail_stop=trail_stop,
                    breakeven_armed=breakeven_armed,
                )
            if l <= tp <= h:
                return ExitDecision(
                    action="close",
                    exit_reason="tp",
                    exit_price=float(tp),
                    tp_price=float(tp),
                    sl_price=float(sl),
                    trail_stop=trail_stop,
                    breakeven_armed=breakeven_armed,
                )

            if self.use_trend_break:
                trend_break = False
                if ema_fast == ema_fast and ema_slow == ema_slow:
                    trend_break = not (close_now < ema_fast and ema_fast < ema_slow)
                if trend_break:
                    return ExitDecision(
                        action="close",
                        exit_reason="trend_break",
                        exit_price=float(close_now),
                        tp_price=float(tp),
                        sl_price=float(sl),
                        trail_stop=trail_stop,
                        breakeven_armed=breakeven_armed,
                    )

        if position.bars_held >= int(self.max_hold_bars):
            return ExitDecision(
                action="close",
                exit_reason="time_stop",
                exit_price=float(close_now),
                tp_price=float(tp),
                sl_price=float(sl),
                trail_stop=position.trail_stop,
                breakeven_armed=bool(position.breakeven_armed),
            )

        if not bool(regime_on):
            return ExitDecision(
                action="close",
                exit_reason="regime_off",
                exit_price=float(close_now),
                tp_price=float(tp),
                sl_price=position.trail_stop,
                trail_stop=position.trail_stop,
                breakeven_armed=bool(position.breakeven_armed),
            )

        if _side_from_weight(target_weight) == "flat":
            return ExitDecision(
                action="close",
                exit_reason="target_flatten",
                exit_price=float(close_now),
                tp_price=None,
                sl_price=position.trail_stop,
                trail_stop=position.trail_stop,
                breakeven_armed=bool(position.breakeven_armed),
            )

        if signal_side in {"long", "short"} and signal_side != position.side:
            return ExitDecision(
                action="close",
                exit_reason="reverse_signal",
                exit_price=float(close_now),
                tp_price=None,
                sl_price=position.trail_stop,
                trail_stop=position.trail_stop,
                breakeven_armed=bool(position.breakeven_armed),
            )

        return ExitDecision(
            action="hold",
            tp_price=float(tp),
            sl_price=float(sl),
            trail_stop=position.trail_stop,
            breakeven_armed=bool(position.breakeven_armed),
        )


@dataclass
class MeanReversionExitPolicy:
    stop_atr_mult: float = 1.5
    max_hold_bars: int = 48

    def evaluate(
        self,
        *,
        position: PositionState,
        candle_prev: dict[str, Any],
        candle_now: dict[str, Any],
        target_weight: float,
        signal_side: str,
        regime_on: bool = True,
    ) -> ExitDecision:
        if position.side not in {"long", "short"}:
            return ExitDecision(action="hold")

        atr_now = _f(candle_prev.get("atr"), float("nan"))
        close_now = _f(candle_prev.get("close"), float("nan"))
        basis = _f(candle_prev.get("bb_mid"), candle_prev.get("basis", float("nan")))
        h = _f(candle_now.get("high"), float("nan"))
        l = _f(candle_now.get("low"), float("nan"))

        if atr_now != atr_now or atr_now <= 0.0:
            return ExitDecision(action="hold", meta={"skip_reason": "atr_nan"})

        entry = float(position.entry_price)

        if position.side == "long":
            tp = float(basis)
            sl = entry - float(self.stop_atr_mult) * atr_now
            if l <= sl <= h:
                return ExitDecision(action="close", exit_reason="sl", exit_price=float(sl), tp_price=float(tp), sl_price=float(sl))
            if l <= tp <= h:
                return ExitDecision(action="close", exit_reason="tp", exit_price=float(tp), tp_price=float(tp), sl_price=float(sl))
        else:
            tp = float(basis)
            sl = entry + float(self.stop_atr_mult) * atr_now
            if l <= sl <= h:
                return ExitDecision(action="close", exit_reason="sl", exit_price=float(sl), tp_price=float(tp), sl_price=float(sl))
            if l <= tp <= h:
                return ExitDecision(action="close", exit_reason="tp", exit_price=float(tp), tp_price=float(tp), sl_price=float(sl))

        if position.bars_held >= int(self.max_hold_bars):
            return ExitDecision(action="close", exit_reason="time_stop", exit_price=float(close_now), tp_price=float(tp), sl_price=float(sl))

        if not bool(regime_on):
            return ExitDecision(action="close", exit_reason="regime_off", exit_price=float(close_now), tp_price=float(tp), sl_price=float(sl))

        if _side_from_weight(target_weight) == "flat":
            return ExitDecision(action="close", exit_reason="target_flatten", exit_price=float(close_now), tp_price=float(tp), sl_price=float(sl))

        if signal_side in {"long", "short"} and signal_side != position.side:
            return ExitDecision(action="close", exit_reason="reverse_signal", exit_price=float(close_now), tp_price=float(tp), sl_price=float(sl))

        return ExitDecision(action="hold", tp_price=float(tp), sl_price=float(sl))


@dataclass
class PullbackTrendExitPolicy(TrendAtrDynamicExitPolicy):
    tp_atr_mult: float = 2.2
    stop_atr_mult: float = 1.4
    breakeven_activate_atr: float = 0.8
    breakeven_offset_atr: float = 0.05
    trail_activate_atr: float = 0.9
    trail_stop_atr_mult: float = 1.1
    max_hold_bars: int = 120
    use_trend_break: bool = True


@dataclass
class BreakoutExitPolicy(TrendAtrDynamicExitPolicy):
    tp_atr_mult: float = 2.4
    stop_atr_mult: float = 1.3
    breakeven_activate_atr: float = 1.0
    breakeven_offset_atr: float = 0.05
    trail_activate_atr: float = 1.1
    trail_stop_atr_mult: float = 1.0
    max_hold_bars: int = 96
    use_trend_break: bool = True


@dataclass
class TargetPositionLifecycleEngine:
    min_target_weight_eps: float = 1e-4
    reverse_through_flat: bool = True
    reduce_only_band: float = 0.02
    open_weight_floor: float = 1e-4

    def decide_target_action(
        self,
        *,
        symbol: str,
        strategy_id: str,
        prev_weight: float,
        target_weight: float,
        current_side: str,
        signal_side: str,
        cooldown_left: int = 0,
        meta: Optional[dict[str, Any]] = None,
    ) -> LifecycleAction:
        meta = dict(meta or {})
        family = infer_strategy_family(strategy_id)

        prev_w = _f(prev_weight, 0.0)
        tgt_w = _f(target_weight, 0.0)

        if abs(tgt_w) < float(self.min_target_weight_eps):
            tgt_w = 0.0
        if abs(prev_w) < float(self.min_target_weight_eps):
            prev_w = 0.0

        prev_side = _side_from_weight(prev_w)
        tgt_side = _side_from_weight(tgt_w)
        delta = float(tgt_w - prev_w)

        if prev_side == "flat" and tgt_side == "flat":
            return LifecycleAction(
                action="hold",
                symbol=symbol,
                strategy_id=strategy_id,
                family=family,
                side="flat",
                reason="flat_to_flat",
                prev_weight=prev_w,
                target_weight=tgt_w,
                delta_weight=delta,
                meta=meta,
            )

        if prev_side == "flat" and tgt_side in {"long", "short"}:
            if int(cooldown_left) > 0:
                return LifecycleAction(
                    action="skip",
                    symbol=symbol,
                    strategy_id=strategy_id,
                    family=family,
                    side=tgt_side,
                    reason="cooldown_after_close",
                    prev_weight=prev_w,
                    target_weight=tgt_w,
                    delta_weight=delta,
                    open_weight=abs(tgt_w),
                    meta={**meta, "cooldown_left": int(cooldown_left)},
                )

            if abs(tgt_w) < float(self.open_weight_floor):
                return LifecycleAction(
                    action="skip",
                    symbol=symbol,
                    strategy_id=strategy_id,
                    family=family,
                    side=tgt_side,
                    reason="target_below_open_floor",
                    prev_weight=prev_w,
                    target_weight=tgt_w,
                    delta_weight=delta,
                    open_weight=abs(tgt_w),
                    meta=meta,
                )

            return LifecycleAction(
                action="open",
                symbol=symbol,
                strategy_id=strategy_id,
                family=family,
                side=tgt_side,
                reason="target_open",
                prev_weight=prev_w,
                target_weight=tgt_w,
                delta_weight=delta,
                open_weight=abs(tgt_w),
                meta=meta,
            )

        if prev_side in {"long", "short"} and tgt_side == "flat":
            return LifecycleAction(
                action="close",
                symbol=symbol,
                strategy_id=strategy_id,
                family=family,
                side=prev_side,
                reason="target_flatten",
                prev_weight=prev_w,
                target_weight=tgt_w,
                delta_weight=delta,
                close_weight=abs(prev_w),
                meta=meta,
            )

        if prev_side != "flat" and tgt_side != "flat" and prev_side != tgt_side:
            return LifecycleAction(
                action="reverse" if bool(self.reverse_through_flat) else "close",
                symbol=symbol,
                strategy_id=strategy_id,
                family=family,
                side=tgt_side,
                reason="target_reverse",
                prev_weight=prev_w,
                target_weight=tgt_w,
                delta_weight=delta,
                close_weight=abs(prev_w),
                open_weight=abs(tgt_w),
                meta=meta,
            )

        abs_prev = abs(prev_w)
        abs_tgt = abs(tgt_w)
        abs_delta = abs(abs_tgt - abs_prev)

        if abs_delta <= float(self.reduce_only_band):
            return LifecycleAction(
                action="hold",
                symbol=symbol,
                strategy_id=strategy_id,
                family=family,
                side=tgt_side,
                reason="within_rebalance_band",
                prev_weight=prev_w,
                target_weight=tgt_w,
                delta_weight=delta,
                meta={**meta, "abs_delta": abs_delta},
            )

        if abs_tgt > abs_prev:
            return LifecycleAction(
                action="increase",
                symbol=symbol,
                strategy_id=strategy_id,
                family=family,
                side=tgt_side,
                reason="target_increase",
                prev_weight=prev_w,
                target_weight=tgt_w,
                delta_weight=delta,
                open_weight=max(0.0, abs_tgt - abs_prev),
                meta=meta,
            )

        return LifecycleAction(
            action="decrease",
            symbol=symbol,
            strategy_id=strategy_id,
            family=family,
            side=tgt_side,
            reason="target_decrease",
            prev_weight=prev_w,
            target_weight=tgt_w,
            delta_weight=delta,
            close_weight=max(0.0, abs_prev - abs_tgt),
            meta=meta,
        )

    def build_exit_policy(self, strategy_id: str):
        family = infer_strategy_family(strategy_id)
        if family == "mean_reversion":
            return MeanReversionExitPolicy()
        if family == "pullback_trend":
            return PullbackTrendExitPolicy()
        if family == "breakout":
            return BreakoutExitPolicy()
        return TrendAtrDynamicExitPolicy()

    def update_position_from_hold_decision(
        self,
        *,
        position: PositionState,
        exit_decision: ExitDecision,
        candle_prev: dict[str, Any],
        ts: int,
        target_weight: float,
    ) -> PositionState:
        position.last_ts = int(ts)
        position.target_weight = float(target_weight)
        position.bars_held = int(position.bars_held) + 1
        position.breakeven_armed = bool(exit_decision.breakeven_armed)

        if exit_decision.trail_stop is not None:
            position.trail_stop = float(exit_decision.trail_stop)

        close_now = _f(candle_prev.get("close"), position.entry_price)
        if position.side == "long":
            position.peak_price = close_now if position.peak_price is None else max(float(position.peak_price), float(close_now))
        elif position.side == "short":
            position.trough_price = close_now if position.trough_price is None else min(float(position.trough_price), float(close_now))

        return position
