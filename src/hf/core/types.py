from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal

Side = Literal["long", "short", "flat"]

@dataclass(frozen=True)
class Candle:
    ts: Any  # datetime-like; se concreta en PASO 2 (pd.Timestamp)
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass(frozen=True)
class Signal:
    symbol: str
    side: Side                  # intent: long/short/flat
    strength: float = 1.0       # 0..1 (opcional)
    meta: Optional[Dict[str, Any]] = None

@dataclass(frozen=True)
class RegimeState:
    # flags por activo/estrategia (simple por ahora; se expande luego)
    on: bool
    reason: Optional[str] = None

@dataclass(frozen=True)
class Allocation:
    # pesos target por símbolo (suman 1.0 idealmente)
    weights: Dict[str, float]
    meta: Optional[Dict[str, Any]] = None
