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

    if v != v:
        return None
    return v


@dataclass
class SolVolCompressionSignalEngine(SignalEngine):
    bb_width_max: float = 0.035
    adx_max: float = 22.0

    breakout_confirm_mult: float = 1.000
    breakdown_confirm_mult: float = 1.000

    bb_width_key: str = "bb_width"
    adx_key: str = "adx"
    bb_up_key: str = "bb_up"
    bb_low_key: str = "bb_low"

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
            meta={"engine": "sol_vol_compression", "reason": reason, **meta},
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
                    meta={"engine": "sol_vol_compression", "skip": "not_sol"},
                )
                continue

            bb_width = _feat(c, self.bb_width_key)
            adx = _feat(c, self.adx_key)
            bb_up = _feat(c, self.bb_up_key)
            bb_low = _feat(c, self.bb_low_key)
            close_v = float(getattr(c, "close", 0.0))

            if None in (bb_width, adx, bb_up, bb_low):
                reason = "missing_features"
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                side_counts["flat"] += 1
                out[sym] = self._flat(sym, reason)
                continue

            if bb_width > float(self.bb_width_max):
                reason = "bb_width_too_high"
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                side_counts["flat"] += 1
                out[sym] = self._flat(sym, reason, bb_width=bb_width)
                continue

            if adx > float(self.adx_max):
                reason = "adx_too_high"
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                side_counts["flat"] += 1
                out[sym] = self._flat(sym, reason, adx=adx)
                continue

            if close_v >= float(bb_up) * float(self.breakout_confirm_mult):
                side = "long"
                reason = "compression_breakout_long"
            elif close_v <= float(bb_low) * float(self.breakdown_confirm_mult):
                side = "short"
                reason = "compression_breakout_short"
            else:
                side = "flat"
                reason = "no_breakout"

            side_counts[side] = int(side_counts.get(side, 0)) + 1
            strength = 1.0 if side != "flat" else 0.0

            if side != "flat" and bool(locals().get("atrp_low", False)):
                strength *= float(self.strength_penalty_atrp)
            if side != "flat" and bool(locals().get("adx_low", False)):
                strength *= float(self.strength_penalty_adx)
            if side != "flat" and bool(locals().get("range_expansion_low", False)):
                strength *= float(self.strength_penalty_range_expansion)

            out[sym] = Signal(
                symbol=sym,
                side=side,
                strength=strength,
                meta={
                    "engine": "sol_vol_compression",
                    "reason": reason,
                    "bb_width": bb_width,
                    "adx": adx,
                    "bb_up": bb_up,
                    "bb_low": bb_low,
                    "regime_as_metadata": True,
                    "atrp_low": bool(locals().get("atrp_low", False)),
                    "adx_low": bool(locals().get("adx_low", False)),
                    "range_expansion_low": bool(locals().get("range_expansion_low", False)),
                    "close": close_v,
                    "bb_width_max": float(self.bb_width_max),
                    "adx_max": float(self.adx_max),
                    "breakout_confirm_mult": float(self.breakout_confirm_mult),
                    "breakdown_confirm_mult": float(self.breakdown_confirm_mult),
                },
            )

        if print_debug:
            print("[sol_vol_compression] side_counts=", side_counts)
            print("[sol_vol_compression] reason_counts=", reason_counts)

        return out
