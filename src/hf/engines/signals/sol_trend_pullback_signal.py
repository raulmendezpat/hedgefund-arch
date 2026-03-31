from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from hf.core.interfaces import SignalEngine
from hf.core.types import Candle, Signal


def _feat(c: Candle, key: str) -> Optional[float]:
    feats = getattr(c, "features", None)
    if not isinstance(feats, dict):
        return None

    v = feats.get(key)
    if v is None:
        return None

    try:
        v = float(v)
    except Exception:
        return None

    if v != v:  # NaN
        return None
    return v


@dataclass
class SolTrendPullbackSignalEngine(SignalEngine):
    rsi_long_min: float = 40.0
    rsi_long_max: float = 55.0
    rsi_short_min: float = 45.0
    rsi_short_max: float = 60.0

    ema_pullback_max: float = 0.015

    atrp_min: float = 0.004
    atrp_max: float = 0.050

    require_adx: bool = False
    adx_min: float = 18.0

    rsi_key: str = "rsi"
    atrp_key: str = "atrp"
    adx_key: str = "adx"
    ema_fast_key: str = "ema_fast"
    ema_slow_key: str = "ema_slow"

    only_if_symbol_contains: str = "SOL"

    def _flat(self, sym: str, reason: str, **meta) -> Signal:
        return Signal(
            symbol=sym,
            side="flat",
            strength=0.0,
            meta={"engine": "sol_trend_pullback", "reason": reason, **meta},
        )

    def generate(self, candles, print_debug: bool = False) -> Dict[str, Signal]:
        out: Dict[str, Signal] = {}
        reason_counts: Dict[str, int] = {}
        side_counts: Dict[str, int] = {"flat": 0, "long": 0, "short": 0}

        for sym, c in candles.items():
            if self.only_if_symbol_contains and self.only_if_symbol_contains not in sym:
                out[sym] = self._flat(sym, "symbol_filtered")
                continue

            rsi = _feat(c, self.rsi_key)
            atrp = _feat(c, self.atrp_key)
            adx = _feat(c, self.adx_key)
            ema_fast = _feat(c, self.ema_fast_key)
            ema_slow = _feat(c, self.ema_slow_key)
            close_v = float(getattr(c, "close", 0.0) or 0.0)

            needed = (rsi, atrp, ema_fast, ema_slow)
            if self.require_adx:
                needed = needed + (adx,)

            if None in needed:
                reason = "missing_features"
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                side_counts["flat"] += 1
                out[sym] = self._flat(sym, reason)
                continue

            trend_up = float(ema_fast) > float(ema_slow)
            trend_down = float(ema_fast) < float(ema_slow)

            atrp_out_of_range = (float(atrp) < float(self.atrp_min)) or (float(atrp) > float(self.atrp_max))
            adx_below_min = bool(self.require_adx) and (float(adx) < float(self.adx_min))

            ema_pullback_dist = abs(close_v - float(ema_fast)) / max(abs(close_v), 1e-12)
            too_far_from_ema_fast = float(ema_pullback_dist) > float(self.ema_pullback_max)

            long_rsi_ok = float(self.rsi_long_min) <= float(rsi) <= float(self.rsi_long_max)
            short_rsi_ok = float(self.rsi_short_min) <= float(rsi) <= float(self.rsi_short_max)

            if trend_up and long_rsi_ok and not too_far_from_ema_fast:
                side = "long"
                reason = "long_pullback"
            elif trend_down and short_rsi_ok and not too_far_from_ema_fast:
                side = "short"
                reason = "short_pullback"
            else:
                side = "flat"
                if too_far_from_ema_fast:
                    reason = "too_far_from_ema_fast"
                else:
                    reason = "no_setup"

            if side != "flat":
                if atrp_out_of_range:
                    side = "flat"
                    reason = "atrp_out_of_range"
                elif adx_below_min:
                    side = "flat"
                    reason = "adx_below_min"

            side_counts[side] = int(side_counts.get(side, 0)) + 1

            if side == "flat":
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                out[sym] = self._flat(
                    sym,
                    reason,
                    rsi=float(rsi),
                    atrp=float(atrp),
                    adx=None if adx is None else float(adx),
                    ema_fast=float(ema_fast),
                    ema_slow=float(ema_slow),
                    close=float(close_v),
                    ema_pullback_dist=float(ema_pullback_dist),
                    trend_up=bool(trend_up),
                    trend_down=bool(trend_down),
                    long_rsi_ok=bool(long_rsi_ok),
                    short_rsi_ok=bool(short_rsi_ok),
                    atrp_out_of_range=bool(atrp_out_of_range),
                    adx_below_min=bool(adx_below_min),
                    too_far_from_ema_fast=bool(too_far_from_ema_fast),
                    require_adx=bool(self.require_adx),
                    adx_min=float(self.adx_min),
                    rsi_long_min=float(self.rsi_long_min),
                    rsi_long_max=float(self.rsi_long_max),
                    rsi_short_min=float(self.rsi_short_min),
                    rsi_short_max=float(self.rsi_short_max),
                    ema_pullback_max=float(self.ema_pullback_max),
                    atrp_min=float(self.atrp_min),
                    atrp_max=float(self.atrp_max),
                )
                continue

            out[sym] = Signal(
                symbol=sym,
                side=side,
                strength=1.0,
                meta={
                    "engine": "sol_trend_pullback",
                    "reason": reason,
                    "rsi": float(rsi),
                    "atrp": float(atrp),
                    "adx": None if adx is None else float(adx),
                    "ema_fast": float(ema_fast),
                    "ema_slow": float(ema_slow),
                    "close": float(close_v),
                    "ema_pullback_dist": float(ema_pullback_dist),
                    "trend_up": bool(trend_up),
                    "trend_down": bool(trend_down),
                    "long_rsi_ok": bool(long_rsi_ok),
                    "short_rsi_ok": bool(short_rsi_ok),
                    "atrp_out_of_range": bool(atrp_out_of_range),
                    "adx_below_min": bool(adx_below_min),
                    "too_far_from_ema_fast": bool(too_far_from_ema_fast),
                    "require_adx": bool(self.require_adx),
                    "adx_min": float(self.adx_min),
                    "rsi_long_min": float(self.rsi_long_min),
                    "rsi_long_max": float(self.rsi_long_max),
                    "rsi_short_min": float(self.rsi_short_min),
                    "rsi_short_max": float(self.rsi_short_max),
                    "ema_pullback_max": float(self.ema_pullback_max),
                    "atrp_min": float(self.atrp_min),
                    "atrp_max": float(self.atrp_max),
                    "regime_as_metadata": True,
                },
            )

        if print_debug:
            print("[sol_trend_pullback] side_counts=", side_counts)
            print("[sol_trend_pullback] reason_counts=", reason_counts)

        return out
