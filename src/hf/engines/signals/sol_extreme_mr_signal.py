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
                out[sym] = self._flat(sym, "symbol_filtered")
                continue

            rsi = _feat(c, self.rsi_key)
            atrp = _feat(c, self.atrp_key)
            adx = _feat(c, self.adx_key)
            bb_low = _feat(c, self.bb_low_key)
            bb_up = _feat(c, self.bb_up_key)
            bb_width = _feat(c, self.bb_width_key)
            range_expansion = _feat(c, self.range_expansion_key)
            close_v = float(getattr(c, "close", 0.0) or 0.0)

            if None in (rsi, atrp, adx, bb_low, bb_up, bb_width, range_expansion):
                reason = "missing_features"
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                side_counts["flat"] += 1
                out[sym] = self._flat(sym, reason)
                continue

            atrp_out_of_range = (float(atrp) < float(self.atrp_min)) or (float(atrp) > float(self.atrp_max))
            adx_too_high = float(adx) > float(self.adx_max)
            bb_width_below_min = float(bb_width) < float(self.bb_width_min)
            range_expansion_below_min = float(range_expansion) < float(self.range_expansion_min)

            if close_v <= float(bb_low) * float(self.bb_long_mult) and float(rsi) <= float(self.rsi_long_max):
                side = "long"
                reason = "long_extreme_reversion"
            elif close_v >= float(bb_up) * float(self.bb_short_mult) and float(rsi) >= float(self.rsi_short_min):
                side = "short"
                reason = "short_extreme_reversion"
            else:
                side = "flat"
                reason = "no_setup"

            if side != "flat":
                if atrp_out_of_range:
                    side = "flat"
                    reason = "atrp_out_of_range"
                elif adx_too_high:
                    side = "flat"
                    reason = "adx_too_high"
                elif bb_width_below_min:
                    side = "flat"
                    reason = "bb_width_below_min"
                elif range_expansion_below_min:
                    side = "flat"
                    reason = "range_expansion_below_min"

            side_counts[side] = int(side_counts.get(side, 0)) + 1

            if side == "flat":
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                out[sym] = self._flat(
                    sym,
                    reason,
                    rsi=float(rsi),
                    atrp=float(atrp),
                    adx=float(adx),
                    bb_low=float(bb_low),
                    bb_up=float(bb_up),
                    bb_width=float(bb_width),
                    range_expansion=float(range_expansion),
                    close=float(close_v),
                    atrp_out_of_range=bool(atrp_out_of_range),
                    adx_too_high=bool(adx_too_high),
                    bb_width_below_min=bool(bb_width_below_min),
                    range_expansion_below_min=bool(range_expansion_below_min),
                    rsi_long_max=float(self.rsi_long_max),
                    rsi_short_min=float(self.rsi_short_min),
                    atrp_min=float(self.atrp_min),
                    atrp_max=float(self.atrp_max),
                    adx_max=float(self.adx_max),
                    bb_long_mult=float(self.bb_long_mult),
                    bb_short_mult=float(self.bb_short_mult),
                    bb_width_min=float(self.bb_width_min),
                    range_expansion_min=float(self.range_expansion_min),
                )
                continue

            out[sym] = Signal(
                symbol=sym,
                side=side,
                strength=1.0,
                meta={
                    "engine": "sol_extreme_mr",
                    "reason": reason,
                    "rsi": float(rsi),
                    "atrp": float(atrp),
                    "adx": float(adx),
                    "bb_low": float(bb_low),
                    "bb_up": float(bb_up),
                    "bb_width": float(bb_width),
                    "range_expansion": float(range_expansion),
                    "close": float(close_v),
                    "atrp_out_of_range": bool(atrp_out_of_range),
                    "adx_too_high": bool(adx_too_high),
                    "bb_width_below_min": bool(bb_width_below_min),
                    "range_expansion_below_min": bool(range_expansion_below_min),
                    "rsi_long_max": float(self.rsi_long_max),
                    "rsi_short_min": float(self.rsi_short_min),
                    "atrp_min": float(self.atrp_min),
                    "atrp_max": float(self.atrp_max),
                    "adx_max": float(self.adx_max),
                    "bb_long_mult": float(self.bb_long_mult),
                    "bb_short_mult": float(self.bb_short_mult),
                    "bb_width_min": float(self.bb_width_min),
                    "range_expansion_min": float(self.range_expansion_min),
                    "regime_as_metadata": True,
                },
            )

        if print_debug:
            print("[sol_extreme_mr] side_counts=", side_counts)
            print("[sol_extreme_mr] reason_counts=", reason_counts)

        return out
