from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from hf.core.types import Candle, Signal, RegimeState, Allocation
from hf.core.interfaces import SignalEngine, RegimeEngine, CapitalAllocator, PortfolioEngine


# ----------------------------
# Legacy wrapper utilities
# ----------------------------

LEGACY_SYMBOLS = {
    "BTC": "BTC/USDT:USDT",
    "SOL": "SOL/USDT:USDT",
}


@dataclass
class StaticRegimeEngine(RegimeEngine):
    """
    Minimal RegimeEngine to unblock the architecture integration.
    - Uses precomputed flags (you'll pass them in externally later)
    - For now default: all ON
    """
    default_on: bool = True

    def evaluate(self, candles: Dict[str, Candle], signals: Dict[str, Signal]) -> Dict[str, RegimeState]:
        return {sym: RegimeState(on=self.default_on, reason="static_default") for sym in candles.keys()}


@dataclass
class DynamicAllocator(CapitalAllocator):
    """
    Implements your base allocation rules (per-candle):
      - BTC on, SOL off -> 100% BTC
      - BTC off, SOL on -> 100% SOL
      - both on -> both_btc_weight BTC, (1-both_btc_weight) SOL
      - both off -> sticky (keep prev) or fallback to cash (0 weights)
    """
    both_btc_weight: float = 0.75
    sticky_when_both_off: bool = True
    fallback_cash_when_both_off: bool = False  # if True and no prev, return 0 weights

    def allocate(
        self,
        candles: Dict[str, Candle],
        signals: Dict[str, Signal],
        regimes: Dict[str, RegimeState],
        prev_allocation: Optional[Allocation],
    ) -> Allocation:
        btc_on = regimes.get(LEGACY_SYMBOLS["BTC"], RegimeState(False)).on
        sol_on = regimes.get(LEGACY_SYMBOLS["SOL"], RegimeState(False)).on

        if btc_on and not sol_on:
            return Allocation(weights={LEGACY_SYMBOLS["BTC"]: 1.0, LEGACY_SYMBOLS["SOL"]: 0.0}, meta={"case": "btc_only"})
        if sol_on and not btc_on:
            return Allocation(weights={LEGACY_SYMBOLS["BTC"]: 0.0, LEGACY_SYMBOLS["SOL"]: 1.0}, meta={"case": "sol_only"})
        if btc_on and sol_on:
            w_btc = float(self.both_btc_weight)
            w_sol = max(0.0, 1.0 - w_btc)
            return Allocation(weights={LEGACY_SYMBOLS["BTC"]: w_btc, LEGACY_SYMBOLS["SOL"]: w_sol}, meta={"case": "both_on"})

        # both off:
        if self.sticky_when_both_off and prev_allocation is not None:
            return Allocation(weights=dict(prev_allocation.weights), meta={"case": "both_off_sticky"})
        if self.fallback_cash_when_both_off:
            return Allocation(weights={LEGACY_SYMBOLS["BTC"]: 0.0, LEGACY_SYMBOLS["SOL"]: 0.0}, meta={"case": "both_off_cash"})
        # default: sticky if possible else 0
        return Allocation(weights={LEGACY_SYMBOLS["BTC"]: 0.0, LEGACY_SYMBOLS["SOL"]: 0.0}, meta={"case": "both_off_default"})


@dataclass
class LegacyBacktestPortfolioEngine(PortfolioEngine):
    """
    Temporary wrapper: executes legacy backtester module main().
    This is a bridge so the new repo can run the same workflow end-to-end.
    """
    def step(self, candles: Dict[str, Candle], signals: Dict[str, Signal], allocation: Allocation) -> Dict[str, float]:
        raise NotImplementedError("Legacy backtester runs as a batch job; use run_legacy_backtest().")

    def run_legacy_backtest(self) -> None:
        # Run legacy backtester as-is.
        import hf.legacy.ltb.envelope.backtest_portfolio_regime_switch_cached_v2_modefixed_v8ml as bt
        bt.main()


@dataclass
class PlaceholderSignalEngine(SignalEngine):
    """
    Minimal SignalEngine so the architecture composes.
    For now it emits 'flat' signals; we will swap to real signal extraction in PASO 2.3
    (or refactor legacy bots into pure signal generators).
    """
    def generate(self, candles: Dict[str, Candle]) -> Dict[str, Signal]:
        return {sym: Signal(symbol=sym, side="flat", strength=0.0, meta={"note": "placeholder"}) for sym in candles.keys()}
