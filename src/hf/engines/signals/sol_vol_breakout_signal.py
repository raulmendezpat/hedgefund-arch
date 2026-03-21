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
class SolVolBreakoutSignalEngine(SignalEngine):
    breakout_lookback: int = 20
    adx_min: float = 18.0
    atrp_min: float = 0.008
    atrp_max: float = 0.080
    range_expansion_min: float = 1.10
    confirm_close_buffer: float = 0.0
    require_trend_alignment: bool = True

    adx_key: str = "adx"
    atrp_key: str = "atrp"
    donchian_high_key: str = "donchian_high"
    donchian_low_key: str = "donchian_low"
    range_expansion_key: str = "range_expansion"
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
            meta={"engine": "sol_vol_breakout", "reason": reason, **meta},
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
                    meta={"engine": "sol_vol_breakout", "skip": "not_sol"},
                )
                continue

            adx = _feat(c, self.adx_key)
            atrp = _feat(c, self.atrp_key)
            donchian_high = _feat(c, self.donchian_high_key)
            donchian_low = _feat(c, self.donchian_low_key)
            range_expansion = _feat(c, self.range_expansion_key)
            ema_fast = _feat(c, self.ema_fast_key)
            ema_slow = _feat(c, self.ema_slow_key)

            needed = (adx, atrp, donchian_high, donchian_low, range_expansion)
            if self.require_trend_alignment:
                needed = needed + (ema_fast, ema_slow)

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

            if adx < float(self.adx_min):
                reason = "adx_below_min"
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                side_counts["flat"] += 1
                out[sym] = self._flat(sym, reason, adx=adx)
                continue

            if range_expansion < float(self.range_expansion_min):
                reason = "no_vol_expansion"
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                side_counts["flat"] += 1
                out[sym] = self._flat(sym, reason, range_expansion=range_expansion)
                continue

            close_v = float(getattr(c, "close", 0.0))
            up_level = float(donchian_high) * (1.0 + float(self.confirm_close_buffer))
            dn_level = float(donchian_low) * (1.0 - float(self.confirm_close_buffer))

            if close_v > up_level:
                if self.require_trend_alignment and not (float(ema_fast) > float(ema_slow)):
                    side = "flat"
                    reason = "trend_filter_block_long"
                    breakout_ref = None
                else:
                    side = "long"
                    reason = "long_breakout"
                    breakout_ref = float(donchian_high)
            elif close_v < dn_level:
                if self.require_trend_alignment and not (float(ema_fast) < float(ema_slow)):
                    side = "flat"
                    reason = "trend_filter_block_short"
                    breakout_ref = None
                else:
                    side = "short"
                    reason = "short_breakout"
                    breakout_ref = float(donchian_low)
            else:
                side = "flat"
                reason = "no_setup"
                breakout_ref = None

            side_counts[side] = int(side_counts.get(side, 0)) + 1

            if side == "flat" or breakout_ref in (None, 0.0):
                strength = 0.0
            else:
                breakout_distance = abs(close_v - float(breakout_ref)) / max(abs(close_v), 1e-12)
                adx_boost = max(0.0, (float(adx) - float(self.adx_min)) / 25.0)
                vol_boost = max(0.0, float(range_expansion) - float(self.range_expansion_min))
                strength = min(2.0, 1.0 + breakout_distance * 20.0 + adx_boost + vol_boost)

            out[sym] = Signal(
                symbol=sym,
                side=side,
                strength=float(strength),
                meta={
                    "engine": "sol_vol_breakout",
                    "reason": reason,
                    "adx": adx,
                    "atrp": atrp,
                    "donchian_high": donchian_high,
                    "donchian_low": donchian_low,
                    "range_expansion": range_expansion,
                    "ema_fast": ema_fast,
                    "ema_slow": ema_slow,
                    "breakout_lookback": int(self.breakout_lookback),
                    "confirm_close_buffer": float(self.confirm_close_buffer),
                    "require_trend_alignment": bool(self.require_trend_alignment),
                },
            )

        if print_debug:
            print("[sol_vol_breakout] side_counts=", side_counts)
            print("[sol_vol_breakout] reason_counts=", reason_counts)

        return out
