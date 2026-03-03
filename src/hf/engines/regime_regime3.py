from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from hf.core.types import Candle, Signal, RegimeState
from hf.core.interfaces import RegimeEngine


@dataclass
class Regime3Engine(RegimeEngine):
    """
    Regime3 (Option-3) engine migrated from legacy backtester.

    SOL regime ON:
        ATR% >= sol_atrp_min  AND  ADX <= sol_adx_max

    BTC regime ON:
        ADX >= btc_adx_min  AND  slope_atr_proxy >= btc_slope_min

    slope_atr_proxy = abs(ema_fast - ema_slow) / ATR
    (same as legacy)
    """
    # thresholds (defaults chosen to be sensible; override via CLI)
    sol_atrp_min: float = 0.0030
    sol_adx_max: float = 24.0
    btc_adx_min: float = 18.0
    btc_slope_min: float = 1.5

    # indicator periods (defaults mirror legacy params usage)
    sol_adx_period: int = 14
    sol_atr_period: int = 14
    btc_adx_period: int = 14
    btc_atr_period: int = 14
    btc_ema_fast: int = 20
    btc_ema_slow: int = 200

    # precomputed feature sources (optional): if Candle carries features dict, use them
    # Expected keys: adx, atrp (SOL), adx, atr, ema_fast, ema_slow (BTC)
    use_candle_features: bool = True

    def evaluate(self, candles: Dict[str, Candle], signals: Dict[str, Signal]) -> Dict[str, RegimeState]:
        out: Dict[str, RegimeState] = {}

        for sym, c in candles.items():
            if "SOL" in sym:
                sol_on = self._sol_regime(sym, c)
                out[sym] = RegimeState(on=sol_on, reason="regime3")
            elif "BTC" in sym:
                btc_on = self._btc_regime(sym, c)
                out[sym] = RegimeState(on=btc_on, reason="regime3")
            else:
                out[sym] = RegimeState(on=False, reason="regime3_unknown_symbol")
        return out

    def _get_feat(self, c: Candle, key: str) -> Optional[float]:
        if not self.use_candle_features:
            return None
        feats = getattr(c, "features", None)
        if isinstance(feats, dict) and key in feats:
            try:
                v = float(feats[key])
                return v
            except Exception:
                return None
        return None

    def _sol_regime(self, sym: str, c: Candle) -> bool:
        adx = self._get_feat(c, "adx")
        atrp = self._get_feat(c, "atrp")
        if adx is None or atrp is None:
            # without features we cannot compute here (pipeline should provide features)
            return False
        return (atrp >= float(self.sol_atrp_min)) and (adx <= float(self.sol_adx_max))

    def _btc_regime(self, sym: str, c: Candle) -> bool:
        adx = self._get_feat(c, "adx")
        atr = self._get_feat(c, "atr")
        ema_fast = self._get_feat(c, "ema_fast")
        ema_slow = self._get_feat(c, "ema_slow")
        if adx is None or atr is None or ema_fast is None or ema_slow is None:
            return False
        if atr == 0:
            return False
        slope_atr_proxy = abs(ema_fast - ema_slow) / atr
        return (adx >= float(self.btc_adx_min)) and (slope_atr_proxy >= float(self.btc_slope_min))
