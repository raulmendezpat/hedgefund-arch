from __future__ import annotations
from typing import Dict, Iterable, Protocol
from .types import Candle, Signal, RegimeState, Allocation

class SignalEngine(Protocol):
    def generate(self, candles: Dict[str, Candle]) -> Dict[str, Signal]:
        ...

class RegimeEngine(Protocol):
    def evaluate(self, candles: Dict[str, Candle], signals: Dict[str, Signal]) -> Dict[str, RegimeState]:
        ...

class CapitalAllocator(Protocol):
    def allocate(
        self,
        candles: Dict[str, Candle],
        signals: Dict[str, Signal],
        regimes: Dict[str, RegimeState],
        prev_allocation: Allocation | None,
    ) -> Allocation:
        ...

class PortfolioEngine(Protocol):
    def step(
        self,
        candles: Dict[str, Candle],
        signals: Dict[str, Signal],
        allocation: Allocation,
    ) -> Dict[str, float]:
        """Return metrics dict (e.g., equity_raw, equity_alloc, pnl_raw, pnl_alloc...)."""
        ...
