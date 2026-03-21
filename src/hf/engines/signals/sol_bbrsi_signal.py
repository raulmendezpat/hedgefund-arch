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
class SolBbrsiSignalEngine(SignalEngine):
    rsi_long_max: float = 36.0
    rsi_short_min: float = 64.0
    adx_hard: float = 24.0
    atrp_min: float = 0.003279
    atrp_max: float = 0.0350
    bb_width_min: float = 0.0041
    bb_width_max: float = 0.25

    adx_key: str = "adx"
    atrp_key: str = "atrp"
    rsi_key: str = "rsi"
    bb_low_key: str = "bb_low"
    bb_up_key: str = "bb_up"
    bb_mid_key: str = "bb_mid"
    bb_width_key: str = "bb_width"

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
            meta={"engine": "sol_bbrsi", "reason": reason, **meta},
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
                    meta={"engine": "sol_bbrsi", "skip": "not_sol"},
                )
                continue

            adx = _feat(c, self.adx_key)
            atrp = _feat(c, self.atrp_key)
            rsi = _feat(c, self.rsi_key)
            bb_low = _feat(c, self.bb_low_key)
            bb_up = _feat(c, self.bb_up_key)
            bb_mid = _feat(c, self.bb_mid_key)
            bb_width = _feat(c, self.bb_width_key)

            if None in (adx, atrp, rsi, bb_low, bb_up, bb_mid, bb_width):
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

            if bb_width < float(self.bb_width_min) or bb_width > float(self.bb_width_max):
                reason = "bb_width_out_of_range"
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                side_counts["flat"] += 1
                out[sym] = self._flat(sym, reason, bb_width=bb_width)
                continue

            if adx >= float(self.adx_hard):
                reason = "adx_trend_regime"
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                side_counts["flat"] += 1
                out[sym] = self._flat(sym, reason, adx=adx)
                continue

            close_v = float(getattr(c, "close", 0.0))

            if close_v <= bb_low and rsi <= float(self.rsi_long_max):
                side = "long"
                reason = "long_setup"
            elif close_v >= bb_up and rsi >= float(self.rsi_short_min):
                side = "short"
                reason = "short_setup"
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
                    "engine": "sol_bbrsi",
                    "reason": reason,
                    "adx": adx,
                    "atrp": atrp,
                    "rsi": rsi,
                    "bb_low": bb_low,
                    "regime_as_metadata": True,
                    "atrp_low": bool(locals().get("atrp_low", False)),
                    "adx_low": bool(locals().get("adx_low", False)),
                    "range_expansion_low": bool(locals().get("range_expansion_low", False)),
                    "bb_up": bb_up,
                    "bb_mid": bb_mid,
                    "bb_width": bb_width,
                },
            )

        if print_debug:
            print("[sol_bbrsi] side_counts=", side_counts)
            print("[sol_bbrsi] reason_counts=", reason_counts)

        return out
