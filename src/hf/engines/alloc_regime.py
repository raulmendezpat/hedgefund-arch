from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from hf.core.interfaces import CapitalAllocator
from hf.core.types import Allocation, Candle, RegimeState, Signal


def _infer_btc_sol_symbols(symbols: Tuple[str, ...]) -> Tuple[Optional[str], Optional[str]]:
    """Infer BTC/SOL symbols from candle dict keys."""
    btc = None
    sol = None
    for s in symbols:
        if btc is None and "BTC" in s:
            btc = s
        if sol is None and "SOL" in s:
            sol = s
    return btc, sol


@dataclass
class RegimeAllocator(CapitalAllocator):
    """Allocator driven ONLY by per-candle regime flags.

    Rules:
      - BTC ON, SOL OFF  -> 100% BTC
      - BTC OFF, SOL ON  -> 100% SOL
      - BTC ON, SOL ON   -> both_btc_weight BTC, (1-both_btc_weight) SOL
      - BTC OFF, SOL OFF -> sticky previous allocation (if sticky_when_off and prev exists)
                            else fallback weights

    Any other symbols present in `candles` receive 0.0 weight.
    """

    both_btc_weight: float = 0.75
    sticky_when_off: bool = False
    fallback_btc_weight: float = 1.0
    fallback_sol_weight: float = 0.0
    btc_symbol: Optional[str] = None
    sol_symbol: Optional[str] = None

    def __post_init__(self) -> None:
        bw = float(self.both_btc_weight)
        if not (0.0 <= bw <= 1.0):
            raise ValueError(f"both_btc_weight must be in [0,1], got {self.both_btc_weight}")

    def allocate(
        self,
        candles: Dict[str, Candle],
        signals: Dict[str, Signal],
        regimes: Dict[str, RegimeState],
        prev_allocation: Allocation | None,
    ) -> Allocation:
        # initialize weights for all known symbols
        w: Dict[str, float] = {sym: 0.0 for sym in candles.keys()}

        symbols = tuple(candles.keys())
        btc_sym = self.btc_symbol
        sol_sym = self.sol_symbol
        if btc_sym is None or sol_sym is None:
            infer_btc, infer_sol = _infer_btc_sol_symbols(symbols)
            btc_sym = btc_sym or infer_btc
            sol_sym = sol_sym or infer_sol

        # If we can't identify both symbols, fail closed into fallback where possible.
        if btc_sym is None or sol_sym is None:
            if btc_sym is not None:
                w[btc_sym] = float(self.fallback_btc_weight)
            if sol_sym is not None:
                w[sol_sym] = float(self.fallback_sol_weight)
            return Allocation(weights=w, meta={"case": "fallback_missing_symbols"})

        btc_on = bool(regimes.get(btc_sym, RegimeState(on=False)).on)
        sol_on = bool(regimes.get(sol_sym, RegimeState(on=False)).on)

        if btc_on and (not sol_on):
            w[btc_sym] = 1.0
            w[sol_sym] = 0.0
            case = "btc_only"
        elif (not btc_on) and sol_on:
            w[btc_sym] = 0.0
            w[sol_sym] = 1.0
            case = "sol_only"
        elif btc_on and sol_on:
            bw = float(self.both_btc_weight)
            w[btc_sym] = bw
            w[sol_sym] = 1.0 - bw
            case = "both_on"
        else:
            if self.sticky_when_off and prev_allocation is not None:
                prev_w = dict(prev_allocation.weights)
                for sym in w.keys():
                    w[sym] = float(prev_w.get(sym, 0.0))
                case = "both_off_sticky"
            else:
                w[btc_sym] = float(self.fallback_btc_weight)
                w[sol_sym] = float(self.fallback_sol_weight)
                case = "both_off_fallback"

        # Normalize only if we exceed 1.0 (avoid changing intended fallbacks < 1.0)
        total = sum(float(x) for x in w.values())
        if total > 1.0 + 1e-12:
            for k in list(w.keys()):
                w[k] = float(w[k]) / total

        return Allocation(weights=w, meta={"case": case, "btc_on": btc_on, "sol_on": sol_on})
