
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
class SolVolExpansionSignalEngine(SignalEngine):

    atrp_min: float = 0.01
    adx_min: float = 18.0

    breakout_mult: float = 1.000
    breakdown_mult: float = 1.000

    atrp_key: str = "atrp"
    adx_key: str = "adx"
    bb_up_key: str = "bb_up"
    bb_low_key: str = "bb_low"

    only_if_symbol_contains: str = "SOL"

    def _flat(self, sym: str, reason: str, **meta) -> Signal:
        return Signal(
            symbol=sym,
            side="flat",
            strength=0.0,
            meta={"engine": "sol_vol_expansion", "reason": reason, **meta},
        )

    def generate(self, candles, print_debug: bool = False) -> Dict[str, Signal]:

        out: Dict[str, Signal] = {}

        for sym, c in candles.items():

            if self.only_if_symbol_contains and self.only_if_symbol_contains not in sym:
                out[sym] = Signal(symbol=sym, side="flat", strength=0.0)
                continue

            atrp = _feat(c, self.atrp_key)
            adx = _feat(c, self.adx_key)
            bb_up = _feat(c, self.bb_up_key)
            bb_low = _feat(c, self.bb_low_key)

            close_v = float(getattr(c, "close", 0.0))

            if None in (atrp, adx, bb_up, bb_low):
                out[sym] = self._flat(sym, "missing_features")
                continue

            if atrp < float(self.atrp_min):
                out[sym] = self._flat(sym, "atrp_low")
                continue

            if adx < float(self.adx_min):
                out[sym] = self._flat(sym, "adx_low")
                continue

            if close_v >= float(bb_up) * float(self.breakout_mult):
                side = "long"
                reason = "vol_expansion_breakout_long"

            elif close_v <= float(bb_low) * float(self.breakdown_mult):
                side = "short"
                reason = "vol_expansion_breakout_short"

            else:
                side = "flat"
                reason = "no_breakout"

            strength = 1.0 if side != "flat" else 0.0

            out[sym] = Signal(
                symbol=sym,
                side=side,
                strength=strength,
                meta={
                    "engine": "sol_vol_expansion",
                    "reason": reason,
                    "atrp": atrp,
                    "adx": adx,
                    "bb_up": bb_up,
                    "bb_low": bb_low,
                },
            )

        return out
