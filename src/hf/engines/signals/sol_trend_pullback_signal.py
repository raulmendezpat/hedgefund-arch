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

    # research: regime/context as metadata, not hard rejection
    strength_penalty_adx: float = 0.70
    strength_penalty_atrp: float = 0.70
    strength_penalty_rsi: float = 0.85
    strength_penalty_bb_width: float = 0.85
    strength_penalty_range_expansion: float = 0.85
    strength_penalty_directional_close: float = 0.90
    strength_penalty_extension: float = 0.85
    strength_penalty_trend_alignment: float = 0.85
    strength_penalty_donchian: float = 0.85

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
                out[sym] = Signal(
                    symbol=sym,
                    side="flat",
                    strength=0.0,
                    meta={"engine": "sol_trend_pullback", "skip": "not_sol"},
                )
                continue

            rsi = _feat(c, self.rsi_key)
            atrp = _feat(c, self.atrp_key)
            adx = _feat(c, self.adx_key)
            ema_fast = _feat(c, self.ema_fast_key)
            ema_slow = _feat(c, self.ema_slow_key)
            close_v = float(getattr(c, "close", 0.0))

            needed = (rsi, atrp, ema_fast, ema_slow)
            if self.require_adx:
                needed = needed + (adx,)

            if None in needed:
                reason = "missing_features"
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                side_counts["flat"] += 1
                out[sym] = self._flat(sym, reason)
                continue

            if atrp < float(self.atrp_min) or atrp > float(self.atrp_max):
                reason = "atrp_out_of_range"
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                side_counts["flat"] += 1
                out[sym] = self._flat(sym, reason, atrp=atrp)
                continue

            if self.require_adx and float(adx) < float(self.adx_min):
                reason = "adx_below_min"
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                side_counts["flat"] += 1
                out[sym] = self._flat(sym, reason, adx=adx)
                continue

            ema_pullback_dist = abs(close_v - float(ema_fast)) / max(abs(close_v), 1e-12)
            if ema_pullback_dist > float(self.ema_pullback_max):
                reason = "too_far_from_ema_fast"
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                side_counts["flat"] += 1
                out[sym] = self._flat(
                    sym,
                    reason,
                    ema_pullback_dist=ema_pullback_dist,
                    ema_fast=ema_fast,
                    close=close_v,
                )
                continue

            trend_up = float(ema_fast) > float(ema_slow)
            trend_down = float(ema_fast) < float(ema_slow)

            if trend_up and float(self.rsi_long_min) <= float(rsi) <= float(self.rsi_long_max):
                side = "long"
                reason = "long_pullback"
            elif trend_down and float(self.rsi_short_min) <= float(rsi) <= float(self.rsi_short_max):
                side = "short"
                reason = "short_pullback"
            else:
                side = "flat"
                reason = "no_setup"

            side_counts[side] = int(side_counts.get(side, 0)) + 1

            if side == "flat":
                strength = 0.0
            else:
                trend_gap = abs(float(ema_fast) - float(ema_slow)) / max(abs(close_v), 1e-12)
                pullback_score = max(0.0, 1.0 - (ema_pullback_dist / max(float(self.ema_pullback_max), 1e-12)))
                trend_score = min(1.0, trend_gap * 20.0)

                if side == "long":
                    rsi_mid = (float(self.rsi_long_min) + float(self.rsi_long_max)) / 2.0
                    rsi_half_band = max((float(self.rsi_long_max) - float(self.rsi_long_min)) / 2.0, 1e-12)
                else:
                    rsi_mid = (float(self.rsi_short_min) + float(self.rsi_short_max)) / 2.0
                    rsi_half_band = max((float(self.rsi_short_max) - float(self.rsi_short_min)) / 2.0, 1e-12)

                rsi_score = max(0.0, 1.0 - (abs(float(rsi) - rsi_mid) / rsi_half_band))

                adx_boost = 0.0
                if adx is not None:
                    adx_boost = max(0.0, min(0.5, (float(adx) - float(self.adx_min)) / 20.0))

                strength = min(2.0, 0.75 + 0.60 * pullback_score + 0.40 * trend_score + 0.25 * rsi_score + adx_boost)

            out[sym] = Signal(
                symbol=sym,
                side=side,
                strength=float(strength),
                meta={
                    "engine": "sol_trend_pullback",
                    "reason": reason,
                    "rsi": rsi,
                    "atrp": atrp,
                    "adx": adx,
                    "ema_fast": ema_fast,
                    "ema_slow": ema_slow,
                    "ema_pullback_dist": float(ema_pullback_dist),
                    "require_adx": bool(self.require_adx),
                    "adx_min": float(self.adx_min),
                    "rsi_long_min": float(self.rsi_long_min),
                    "rsi_long_max": float(self.rsi_long_max),
                    "rsi_short_min": float(self.rsi_short_min),
                    "rsi_short_max": float(self.rsi_short_max),
                    "ema_pullback_max": float(self.ema_pullback_max),
                },
            )

        if print_debug:
            print("[sol_trend_pullback] side_counts=", side_counts)
            print("[sol_trend_pullback] reason_counts=", reason_counts)

        return out
