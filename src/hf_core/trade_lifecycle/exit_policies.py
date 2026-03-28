from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import ExitDecision, PositionState


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


def _side_from_weight(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "flat"
    if v > 0:
        return "long"
    if v < 0:
        return "short"
    return "flat"


class ExitPolicy(Protocol):
    family: str

    def evaluate(
        self,
        *,
        position: PositionState,
        prev_bar: dict[str, Any],
        current_bar: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> ExitDecision:
        ...


@dataclass
class TrendAtrDynamicExitPolicy:
    family: str = "trend_atr_dynamic"
    stop_atr_mult: float = 1.5
    tp_atr_mult: float = 2.6
    breakeven_activate_atr: float = 0.9
    breakeven_offset_atr: float = 0.1
    trail_activate_atr: float = 1.0
    trail_stop_atr_mult: float = 1.2
    max_hold_bars: int = 0
    invalidate_on_trend_break: bool = False

    def evaluate(
        self,
        *,
        position: PositionState,
        prev_bar: dict[str, Any],
        current_bar: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> ExitDecision:
        context = dict(context or {})

        target_weight = _f(context.get("target_weight"), float("nan"))
        signal_side = _s(context.get("signal_side"), "").lower()
        regime_on = bool(context.get("regime_on", True))

        ctx_tp_mult = _f(context.get("tp_mult", context.get("ctx_tp_mult", 1.0)), 1.0)
        ctx_sl_mult = _f(context.get("sl_mult", context.get("ctx_sl_mult", 1.0)), 1.0)
        ctx_time_stop_bars = int(_f(context.get("time_stop_bars", context.get("ctx_time_stop_bars", self.max_hold_bars)), self.max_hold_bars))

        atr_now = _f(prev_bar.get("atr"), position.entry_atr)
        close_now = _f(prev_bar.get("close"), position.entry_px)
        high_now = _f(current_bar.get("high"), close_now)
        low_now = _f(current_bar.get("low"), close_now)

        if atr_now <= 0.0:
            return ExitDecision(action="hold", exit_reason="atr_missing")

        side = _s(position.side).lower()

        if side == "long":
            tp_atr_mult = float(self.tp_atr_mult * max(ctx_tp_mult, 0.1))
            stop_atr_mult = float(self.stop_atr_mult * max(ctx_sl_mult, 0.1))

            tp = float(position.entry_px + tp_atr_mult * atr_now)
            sl = float(position.entry_px - stop_atr_mult * atr_now)
            move_atr = float((close_now - position.entry_px) / atr_now)

            if move_atr >= float(self.breakeven_activate_atr):
                sl = max(sl, float(position.entry_px + self.breakeven_offset_atr * atr_now))

            if move_atr >= float(self.trail_activate_atr):
                sl = max(sl, float(close_now - self.trail_stop_atr_mult * atr_now))

            if bool(self.invalidate_on_trend_break):
                ema_fast = _f(prev_bar.get("ema_fast"), 0.0)
                ema_slow = _f(prev_bar.get("ema_slow"), 0.0)
                trend_break = bool(ema_fast <= ema_slow) if (ema_fast != 0.0 or ema_slow != 0.0) else False
                if trend_break:
                    return ExitDecision(
                        action="close",
                        exit_reason="trend_break",
                        exit_price=float(close_now),
                        tp_price=float(tp),
                        sl_price=float(sl),
                        trail_stop=float(sl),
                        breakeven_armed=bool(move_atr >= float(self.breakeven_activate_atr)),
                    )

            max_hold_bars = int(ctx_time_stop_bars if ctx_time_stop_bars > 0 else self.max_hold_bars)

            if max_hold_bars > 0 and int(position.bars_held) >= int(max_hold_bars):
                return ExitDecision(
                    action="close",
                    exit_reason="time_stop",
                    exit_price=float(close_now),
                    tp_price=float(tp),
                    sl_price=float(sl),
                    trail_stop=float(sl),
                    breakeven_armed=bool(move_atr >= float(self.breakeven_activate_atr)),
                )

            if low_now <= sl <= high_now:
                return ExitDecision(
                    action="close",
                    exit_reason="sl",
                    exit_price=float(sl),
                    tp_price=float(tp),
                    sl_price=float(sl),
                    trail_stop=float(sl),
                    breakeven_armed=bool(move_atr >= float(self.breakeven_activate_atr)),
                )
            if low_now <= tp <= high_now:
                return ExitDecision(
                    action="close",
                    exit_reason="tp",
                    exit_price=float(tp),
                    tp_price=float(tp),
                    sl_price=float(sl),
                    trail_stop=float(sl),
                    breakeven_armed=bool(move_atr >= float(self.breakeven_activate_atr)),
                )

            return ExitDecision(
                action="hold",
                exit_reason="hold",
                tp_price=float(tp),
                sl_price=float(sl),
                trail_stop=float(sl),
                breakeven_armed=bool(move_atr >= float(self.breakeven_activate_atr)),
            )

        if side == "short":
            tp_atr_mult = float(self.tp_atr_mult * max(ctx_tp_mult, 0.1))
            stop_atr_mult = float(self.stop_atr_mult * max(ctx_sl_mult, 0.1))

            tp = float(position.entry_px - tp_atr_mult * atr_now)
            sl = float(position.entry_px + stop_atr_mult * atr_now)
            move_atr = float((position.entry_px - close_now) / atr_now)

            if move_atr >= float(self.breakeven_activate_atr):
                sl = min(sl, float(position.entry_px - self.breakeven_offset_atr * atr_now))

            if move_atr >= float(self.trail_activate_atr):
                sl = min(sl, float(close_now + self.trail_stop_atr_mult * atr_now))

            if bool(self.invalidate_on_trend_break):
                ema_fast = _f(prev_bar.get("ema_fast"), 0.0)
                ema_slow = _f(prev_bar.get("ema_slow"), 0.0)
                trend_break = bool(ema_fast >= ema_slow) if (ema_fast != 0.0 or ema_slow != 0.0) else False
                if trend_break:
                    return ExitDecision(
                        action="close",
                        exit_reason="trend_break",
                        exit_price=float(close_now),
                        tp_price=float(tp),
                        sl_price=float(sl),
                        trail_stop=float(sl),
                        breakeven_armed=bool(move_atr >= float(self.breakeven_activate_atr)),
                    )

            max_hold_bars = int(ctx_time_stop_bars if ctx_time_stop_bars > 0 else self.max_hold_bars)

            if max_hold_bars > 0 and int(position.bars_held) >= int(max_hold_bars):
                return ExitDecision(
                    action="close",
                    exit_reason="time_stop",
                    exit_price=float(close_now),
                    tp_price=float(tp),
                    sl_price=float(sl),
                    trail_stop=float(sl),
                    breakeven_armed=bool(move_atr >= float(self.breakeven_activate_atr)),
                )

            if low_now <= sl <= high_now:
                return ExitDecision(
                    action="close",
                    exit_reason="sl",
                    exit_price=float(sl),
                    tp_price=float(tp),
                    sl_price=float(sl),
                    trail_stop=float(sl),
                    breakeven_armed=bool(move_atr >= float(self.breakeven_activate_atr)),
                )
            if low_now <= tp <= high_now:
                return ExitDecision(
                    action="close",
                    exit_reason="tp",
                    exit_price=float(tp),
                    tp_price=float(tp),
                    sl_price=float(sl),
                    trail_stop=float(sl),
                    breakeven_armed=bool(move_atr >= float(self.breakeven_activate_atr)),
                )

            return ExitDecision(
                action="hold",
                exit_reason="hold",
                tp_price=float(tp),
                sl_price=float(sl),
                trail_stop=float(sl),
                breakeven_armed=bool(move_atr >= float(self.breakeven_activate_atr)),
            )

        return ExitDecision(action="hold", exit_reason="invalid_side")


@dataclass
class MeanReversionBasisAtrExitPolicy:
    family: str = "mean_reversion_basis_atr"
    stop_atr_mult: float = 1.4
    basis_field: str = "bb_mid"
    max_hold_bars: int = 8
    use_entry_atr_for_stop: bool = True

    def evaluate(
        self,
        *,
        position: PositionState,
        prev_bar: dict[str, Any],
        current_bar: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> ExitDecision:
        context = dict(context or {})

        close_now = _f(prev_bar.get("close"), position.entry_px)
        high_now = _f(current_bar.get("high"), close_now)
        low_now = _f(current_bar.get("low"), close_now)

        basis = _f(prev_bar.get(self.basis_field), 0.0)
        atr_now = _f(prev_bar.get("atr"), position.entry_atr)
        atr_stop = float(position.entry_atr if self.use_entry_atr_for_stop else atr_now)

        if atr_stop <= 0.0:
            atr_stop = atr_now
        if atr_stop <= 0.0:
            return ExitDecision(action="hold", exit_reason="atr_missing")

        if basis <= 0.0:
            return ExitDecision(action="hold", exit_reason="basis_missing")

        side = _s(position.side).lower()

        if side == "long":
            tp = float(basis)
            sl = float(position.entry_px - self.stop_atr_mult * atr_stop)

            if self.max_hold_bars > 0 and int(position.bars_held) >= int(self.max_hold_bars):
                return ExitDecision(
                    action="close",
                    exit_reason="time_stop",
                    exit_price=float(close_now),
                    tp_price=float(tp),
                    sl_price=float(sl),
                )

            if low_now <= sl <= high_now:
                return ExitDecision(
                    action="close",
                    exit_reason="sl",
                    exit_price=float(sl),
                    tp_price=float(tp),
                    sl_price=float(sl),
                )
            if low_now <= tp <= high_now:
                return ExitDecision(
                    action="close",
                    exit_reason="tp",
                    exit_price=float(tp),
                    tp_price=float(tp),
                    sl_price=float(sl),
                )

            regime_on = bool(context.get("regime_on", True))
            target_weight = _f(context.get("target_weight"), float("nan"))
            signal_side = _s(context.get("signal_side"), "").lower()

            if not regime_on:
                return ExitDecision(
                    action="close",
                    exit_reason="regime_off",
                    exit_price=float(close_now),
                    tp_price=float(tp),
                    sl_price=float(sl),
                )

            if _side_from_weight(target_weight) == "flat":
                return ExitDecision(
                    action="close",
                    exit_reason="target_flatten",
                    exit_price=float(close_now),
                    tp_price=float(tp),
                    sl_price=float(sl),
                )

            if signal_side in {"long", "short"} and signal_side != "long":
                return ExitDecision(
                    action="close",
                    exit_reason="reverse_signal",
                    exit_price=float(close_now),
                    tp_price=float(tp),
                    sl_price=float(sl),
                )

            regime_on = bool(context.get("regime_on", True))
            target_weight = _f(context.get("target_weight"), float("nan"))
            signal_side = _s(context.get("signal_side"), "").lower()

            if not regime_on:
                return ExitDecision(
                    action="close",
                    exit_reason="regime_off",
                    exit_price=float(close_now),
                    tp_price=float(tp),
                    sl_price=float(sl),
                )

            if _side_from_weight(target_weight) == "flat":
                return ExitDecision(
                    action="close",
                    exit_reason="target_flatten",
                    exit_price=float(close_now),
                    tp_price=float(tp),
                    sl_price=float(sl),
                )

            if signal_side in {"long", "short"} and signal_side != "short":
                return ExitDecision(
                    action="close",
                    exit_reason="reverse_signal",
                    exit_price=float(close_now),
                    tp_price=float(tp),
                    sl_price=float(sl),
                )

            return ExitDecision(
                action="hold",
                exit_reason="hold",
                tp_price=float(tp),
                sl_price=float(sl),
            )

        if side == "short":
            tp = float(basis)
            sl = float(position.entry_px + self.stop_atr_mult * atr_stop)

            if self.max_hold_bars > 0 and int(position.bars_held) >= int(self.max_hold_bars):
                return ExitDecision(
                    action="close",
                    exit_reason="time_stop",
                    exit_price=float(close_now),
                    tp_price=float(tp),
                    sl_price=float(sl),
                )

            if low_now <= sl <= high_now:
                return ExitDecision(
                    action="close",
                    exit_reason="sl",
                    exit_price=float(sl),
                    tp_price=float(tp),
                    sl_price=float(sl),
                )
            if low_now <= tp <= high_now:
                return ExitDecision(
                    action="close",
                    exit_reason="tp",
                    exit_price=float(tp),
                    tp_price=float(tp),
                    sl_price=float(sl),
                )

            return ExitDecision(
                action="hold",
                exit_reason="hold",
                tp_price=float(tp),
                sl_price=float(sl),
            )

        return ExitDecision(action="hold", exit_reason="invalid_side")


def build_exit_policy(cfg: dict[str, Any] | None) -> ExitPolicy:
    raw = dict(cfg or {})
    family = _s(raw.get("family"), "").strip().lower()
    params = dict(raw.get("params", {}) or {})

    if family == "trend_atr_dynamic":
        return TrendAtrDynamicExitPolicy(
            stop_atr_mult=_f(params.get("stop_atr_mult"), 1.5),
            tp_atr_mult=_f(params.get("tp_atr_mult"), 2.6),
            breakeven_activate_atr=_f(params.get("breakeven_activate_atr"), 0.9),
            breakeven_offset_atr=_f(params.get("breakeven_offset_atr"), 0.1),
            trail_activate_atr=_f(params.get("trail_activate_atr"), 1.0),
            trail_stop_atr_mult=_f(params.get("trail_stop_atr_mult"), 1.2),
            max_hold_bars=int(_f(params.get("max_hold_bars"), 0)),
            invalidate_on_trend_break=bool(params.get("invalidate_on_trend_break", False)),
        )

    if family == "mean_reversion_basis_atr":
        return MeanReversionBasisAtrExitPolicy(
            stop_atr_mult=_f(params.get("stop_atr_mult"), 1.4),
            basis_field=_s(params.get("basis_field"), "bb_mid"),
            max_hold_bars=int(_f(params.get("max_hold_bars"), 8)),
            use_entry_atr_for_stop=bool(params.get("use_entry_atr_for_stop", True)),
        )

    return TrendAtrDynamicExitPolicy()
