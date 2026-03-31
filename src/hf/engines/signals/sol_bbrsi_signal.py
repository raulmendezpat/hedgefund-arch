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
                out[sym] = self._flat(sym, "symbol_filtered")
                continue

            adx = _feat(c, self.adx_key)
            atrp = _feat(c, self.atrp_key)
            rsi = _feat(c, self.rsi_key)
            bb_low = _feat(c, self.bb_low_key)
            bb_up = _feat(c, self.bb_up_key)
            bb_mid = _feat(c, self.bb_mid_key)
            bb_width = _feat(c, self.bb_width_key)
            close_v = float(getattr(c, "close", 0.0) or 0.0)

            if None in (adx, atrp, rsi, bb_low, bb_up, bb_mid, bb_width):
                reason = "missing_features"
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                side_counts["flat"] += 1
                out[sym] = self._flat(sym, reason)
                continue

            atrp_out_of_range = (float(atrp) < float(self.atrp_min)) or (float(atrp) > float(self.atrp_max))
            bb_width_out_of_range = (float(bb_width) < float(self.bb_width_min)) or (float(bb_width) > float(self.bb_width_max))
            adx_trend_regime = float(adx) >= float(self.adx_hard)

            if close_v <= float(bb_low) and float(rsi) <= float(self.rsi_long_max):
                side = "long"
                reason = "long_setup"
            elif close_v >= float(bb_up) and float(rsi) >= float(self.rsi_short_min):
                side = "short"
                reason = "short_setup"
            else:
                side = "flat"
                reason = "no_setup"

            if side != "flat":
                if atrp_out_of_range:
                    side = "flat"
                    reason = "atrp_out_of_range"
                elif bb_width_out_of_range:
                    side = "flat"
                    reason = "bb_width_out_of_range"
                elif adx_trend_regime:
                    side = "flat"
                    reason = "adx_trend_regime"

            side_counts[side] = int(side_counts.get(side, 0)) + 1
            if side == "flat":
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                out[sym] = self._flat(
                    sym,
                    reason,
                    adx=float(adx),
                    atrp=float(atrp),
                    rsi=float(rsi),
                    bb_low=float(bb_low),
                    bb_up=float(bb_up),
                    bb_mid=float(bb_mid),
                    bb_width=float(bb_width),
                    close=float(close_v),
                    atrp_out_of_range=bool(atrp_out_of_range),
                    bb_width_out_of_range=bool(bb_width_out_of_range),
                    adx_trend_regime=bool(adx_trend_regime),
                    rsi_long_max=float(self.rsi_long_max),
                    rsi_short_min=float(self.rsi_short_min),
                    atrp_min=float(self.atrp_min),
                    atrp_max=float(self.atrp_max),
                    bb_width_min=float(self.bb_width_min),
                    bb_width_max=float(self.bb_width_max),
                    adx_hard=float(self.adx_hard),
                )
                continue

            out[sym] = Signal(
                symbol=sym,
                side=side,
                strength=1.0,
                meta={
                    "engine": "sol_bbrsi",
                    "reason": reason,
                    "adx": float(adx),
                    "atrp": float(atrp),
                    "rsi": float(rsi),
                    "bb_low": float(bb_low),
                    "bb_up": float(bb_up),
                    "bb_mid": float(bb_mid),
                    "bb_width": float(bb_width),
                    "close": float(close_v),
                    "atrp_out_of_range": bool(atrp_out_of_range),
                    "bb_width_out_of_range": bool(bb_width_out_of_range),
                    "adx_trend_regime": bool(adx_trend_regime),
                    "rsi_long_max": float(self.rsi_long_max),
                    "rsi_short_min": float(self.rsi_short_min),
                    "atrp_min": float(self.atrp_min),
                    "atrp_max": float(self.atrp_max),
                    "bb_width_min": float(self.bb_width_min),
                    "bb_width_max": float(self.bb_width_max),
                    "adx_hard": float(self.adx_hard),
                    "regime_as_metadata": True,
                },
            )

        if print_debug:
            print("[sol_bbrsi] side_counts=", side_counts)
            print("[sol_bbrsi] reason_counts=", reason_counts)

        return out
