from __future__ import annotations


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1.0/float(period), adjust=False).mean()
    roll_down = down.ewm(alpha=1.0/float(period), adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

def _bbands(close: pd.Series, period: int = 20, std: float = 2.0):
    ma = close.rolling(int(period)).mean()
    sd = close.rolling(int(period)).std(ddof=0)
    upper = ma + float(std) * sd
    lower = ma - float(std) * sd
    return ma, upper, lower

from dataclasses import dataclass
from typing import Dict, Optional

from hf.core.interfaces import SignalEngine
from hf.core.types import Candle, Signal


def _feat(c: Candle, key: str) -> Optional[float]:
    feats = getattr(c, "features", None)
    if not isinstance(feats, dict):
        return None
    v = feats.get(key, None)
    if v is None:
        return None
    try:
        v = float(v)
    except Exception:
        return None
    # NaN -> missing
    if v != v:
        return None
    return v


@dataclass
class SolBbrsiSignalEngine(SignalEngine):
    """SOL BB+RSI (versión mínima, determinista).

    Solo define intención (side): long/short/flat. NO ejecuta órdenes ni TP/SL.
    Requiere features en Candle.features para SOL:
      adx, atrp, bb_low, bb_up, bb_mid, bb_width, rsi

    Reglas (core):
      - Filtros: ADX < adx_hard, ATR% en [atrp_min, atrp_max], BB_width en [bb_width_min, bb_width_max]
      - Long: close <= bb_low AND rsi <= rsi_long_max
      - Short: close >= bb_up  AND rsi >= rsi_short_min
      - Else: flat
    """

    rsi_long_max: float = 36.0
    rsi_short_min: float = 64.0
    adx_hard: float = 24.0
    atrp_min: float = 0.003279
    atrp_max: float = 0.0350
    bb_width_min: float = 0.0041
    bb_width_max: float = 0.120

    adx_key: str = "adx"
    atrp_key: str = "atrp"
    rsi_key: str = "rsi"
    bb_low_key: str = "bb_low"
    bb_up_key: str = "bb_up"
    bb_mid_key: str = "bb_mid"
    bb_width_key: str = "bb_width"

    only_if_symbol_contains: str = "SOL"

    def generate(self, candles, print_debug: bool = False) -> Dict[str, Signal]:
        out: Dict[str, Signal] = {}

        reason_counts = {}
        side_counts = {'flat': 0, 'long': 0, 'short': 0}

        for sym, c in candles.items():
            if self.only_if_symbol_contains and (self.only_if_symbol_contains not in sym):
                out[sym] = Signal(symbol=sym, side="flat", strength=0.0, meta={"engine": "sol_bbrsi", "skip": "not_sol"})
                continue

            adx = _feat(c, self.adx_key)
            atrp = _feat(c, self.atrp_key)
            rsi = _feat(c, self.rsi_key)
            bb_low = _feat(c, self.bb_low_key)
            bb_up = _feat(c, self.bb_up_key)
            bb_mid = _feat(c, self.bb_mid_key)
            bb_width = _feat(c, self.bb_width_key)

            if None in (adx, atrp, rsi, bb_low, bb_up, bb_mid, bb_width):
                reason_counts['missing_features'] = int(reason_counts.get('missing_features', 0)) + 1
out[sym] = Signal(symbol=sym, side="flat", strength=0.0, meta={"engine": "sol_bbrsi", "reason": "missing_features"})
                continue

            if atrp < float(self.atrp_min) or atrp > float(self.atrp_max):
                reason_counts['atrp_out_of_range'] = int(reason_counts.get('atrp_out_of_range', 0)) + 1
out[sym] = Signal(symbol=sym, side="flat", strength=0.0, meta={"engine": "sol_bbrsi", "reason": "atrp_out_of_range", "atrp": atrp})
                continue

            if bb_width < float(self.bb_width_min) or bb_width > float(self.bb_width_max):
                reason_counts['bb_width_out_of_range'] = int(reason_counts.get('bb_width_out_of_range', 0)) + 1
out[sym] = Signal(symbol=sym, side="flat", strength=0.0, meta={"engine": "sol_bbrsi", "reason": "bb_width_out_of_range", "bb_width": bb_width})
                continue

            if adx >= float(self.adx_hard):
                reason_counts['adx_trend_regime'] = int(reason_counts.get('adx_trend_regime', 0)) + 1
out[sym] = Signal(symbol=sym, side="flat", strength=0.0, meta={"engine": "sol_bbrsi", "reason": "adx_trend_regime", "adx": adx})
                continue

            close_v = float(getattr(c, "close", 0.0))

            side = "flat"
            if close_v <= bb_low and rsi <= float(self.rsi_long_max):
                side = "long"
            elif close_v >= bb_up and rsi >= float(self.rsi_short_min):
                side = "short"

            strength = 1.0 if side != "flat" else 0.0
            side_counts[side] = int(side_counts.get(side, 0)) + 1
out[sym] = Signal(
                symbol=sym,
                side=side,
                strength=strength,
                meta={
                    "engine": "sol_bbrsi",
                    "adx": adx,
                    "atrp": atrp,
                    "rsi": rsi,
                    "bb_low": bb_low,
                    "bb_up": bb_up,
                    "bb_mid": bb_mid,
                    "bb_width": bb_width,
                },
            )

        return out
