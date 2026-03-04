from __future__ import annotations

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
        return float(v)
    except Exception:
        return None


@dataclass
class BtcTrendSignalEngine(SignalEngine):
    """BTC TREND (primer corte) - versión mínima y determinista.

    Regla base (sin estado):
    - Si falta algún feature -> flat
    - Si ADX < adx_min -> flat
    - Si ema_fast > ema_slow -> long
    - Si ema_fast < ema_slow -> short
    - Si iguales -> flat

    Nota: esto NO intenta replicar 1:1 la ejecución legacy todavía (triggers/TP/SL),
    solo define la lógica de intención (side). En el siguiente paso lo conectamos al pipeline.
    """

    adx_min: float = 18.0
    adx_key: str = "adx"
    ema_fast_key: str = "ema_fast"
    ema_slow_key: str = "ema_slow"
    only_if_symbol_contains: str = "BTC"

    def generate(self, candles: Dict[str, Candle]) -> Dict[str, Signal]:
        out: Dict[str, Signal] = {}

        for sym, c in candles.items():
            # Solo aplicamos a BTC; el resto queda flat
            if self.only_if_symbol_contains and (self.only_if_symbol_contains not in sym):
                out[sym] = Signal(symbol=sym, side="flat", strength=0.0, meta={"engine": "btc_trend_min", "skip": "not_btc"})
                continue

            adx = _feat(c, self.adx_key)
            ema_fast = _feat(c, self.ema_fast_key)
            ema_slow = _feat(c, self.ema_slow_key)

            if adx is None or ema_fast is None or ema_slow is None:
                out[sym] = Signal(symbol=sym, side="flat", strength=0.0, meta={"engine": "btc_trend_min", "skip": "missing_features"})
                continue

            if adx < float(self.adx_min):
                out[sym] = Signal(symbol=sym, side="flat", strength=0.0, meta={"engine": "btc_trend_min", "skip": "adx_below_min", "adx": adx})
                continue

            if ema_fast > ema_slow:
                side = "long"
            elif ema_fast < ema_slow:
                side = "short"
            else:
                side = "flat"

            # strength simple (no normalización compleja aún)
            strength = 1.0 if side != "flat" else 0.0
            out[sym] = Signal(
                symbol=sym,
                side=side,
                strength=strength,
                meta={"engine": "btc_trend_min", "adx": adx, "ema_fast": ema_fast, "ema_slow": ema_slow},
            )

        return out
