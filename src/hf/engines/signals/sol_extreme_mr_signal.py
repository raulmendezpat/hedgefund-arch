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
class SolExtremeMrSignalEngine(SignalEngine):
    rsi_long_max: float = 32.0
    rsi_short_min: float = 68.0

    atrp_min: float = 0.008
    atrp_max: float = 0.080

    adx_max: float = 22.0

    rsi_key: str = "rsi"
    atrp_key: str = "atrp"
    adx_key: str = "adx"
    bb_low_key: str = "bb_low"
    bb_up_key: str = "bb_up"
    bb_width_key: str = "bb_width"
    range_expansion_key: str = "range_expansion"

    bb_long_mult: float = 1.002
    bb_short_mult: float = 0.998
    bb_width_min: float = 0.035
    range_expansion_min: float = 1.25

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
            meta={"engine": "sol_extreme_mr", "reason": reason, **meta},
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
                    meta={"engine": "sol_extreme_mr", "skip": "not_sol"},
                )
                continue

            rsi = _feat(c, self.rsi_key)
            atrp = _feat(c, self.atrp_key)
            adx = _feat(c, self.adx_key)
            bb_low = _feat(c, self.bb_low_key)
            bb_up = _feat(c, self.bb_up_key)
            bb_width = _feat(c, self.bb_width_key)
            range_expansion = _feat(c, self.range_expansion_key)
            close_v = float(getattr(c, "close", 0.0))

            if None in (rsi, atrp, adx, bb_low, bb_up, bb_width, range_expansion):
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

            if adx > float(self.adx_max):
                reason = "adx_too_high"
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                side_counts["flat"] += 1
                out[sym] = self._flat(sym, reason, adx=adx)
                continue



            if close_v <= float(bb_low) * float(self.bb_long_mult) and float(rsi) <= float(self.rsi_long_max):
                side = "long"
                reason = "long_extreme_reversion"
            elif close_v >= float(bb_up) * float(self.bb_short_mult) and float(rsi) >= float(self.rsi_short_min):
                side = "short"
                reason = "short_extreme_reversion"
            else:
                side = "flat"
                reason = "no_setup"

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
                    "engine": "sol_extreme_mr",
                    "reason": reason,
                    "rsi": rsi,
                    "atrp": atrp,
                    "adx": adx,
                    "bb_low": bb_low,
                    "regime_as_metadata": True,
                    "atrp_low": bool(locals().get("atrp_low", False)),
                    "adx_low": bool(locals().get("adx_low", False)),
                    "range_expansion_low": bool(locals().get("range_expansion_low", False)),
                    "bb_up": bb_up,
                    "bb_width": bb_width,
                    "range_expansion": range_expansion,
                    "close": close_v,
                    "rsi_long_max": float(self.rsi_long_max),
                    "rsi_short_min": float(self.rsi_short_min),
                    "atrp_min": float(self.atrp_min),
                    "atrp_max": float(self.atrp_max),
                    "adx_max": float(self.adx_max),
                    "bb_long_mult": float(self.bb_long_mult),
                    "bb_short_mult": float(self.bb_short_mult),
                    "bb_width_min": float(self.bb_width_min),
                    "range_expansion_min": float(self.range_expansion_min),
                },
            )

        if print_debug:
            print("[sol_extreme_mr] side_counts=", side_counts)
            print("[sol_extreme_mr] reason_counts=", reason_counts)

        return out
