from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from hf.core.interfaces import SignalEngine
from hf.core.types import Candle, Signal


@dataclass
class FlatSignalEngine(SignalEngine):
    """SignalEngine scaffold que emite 'flat' para cada símbolo.

    Mantiene el comportamiento del pipeline sin cambios mientras introducimos
    la capa SignalEngine.
    """
    note: str = "flat_placeholder"

    def generate(self, candles: Dict[str, Candle]) -> Dict[str, Signal]:
        return {
            sym: Signal(symbol=sym, side="flat", strength=0.0, meta={"note": self.note})
            for sym in candles.keys()
        }
