from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from hf.core.interfaces import SignalEngine
from hf.core.types import Candle, Signal
from hf.engines.signals.btc_trend_signal import BtcTrendSignalEngine
from hf.engines.signals.sol_bbrsi_signal import SolBbrsiSignalEngine


@dataclass
class PortfolioSignalEngine(SignalEngine):
    btc_engine: BtcTrendSignalEngine = field(default_factory=BtcTrendSignalEngine)
    sol_engine: SolBbrsiSignalEngine = field(default_factory=SolBbrsiSignalEngine)

    def generate(self, candles, print_debug: bool = False) -> Dict[str, Signal]:
        out: Dict[str, Signal] = {}

        btc_signals = self.btc_engine.generate(candles)
        sol_signals = self.sol_engine.generate(candles, print_debug=print_debug)

        for sym in candles.keys():
            sig_btc: Optional[Signal] = btc_signals.get(sym)
            sig_sol: Optional[Signal] = sol_signals.get(sym)

            btc_skip = isinstance(getattr(sig_btc, "meta", None), dict) and ("skip" in (sig_btc.meta or {}))
            sol_skip = isinstance(getattr(sig_sol, "meta", None), dict) and ("skip" in (sig_sol.meta or {}))

            if sig_btc is not None and not btc_skip:
                out[sym] = sig_btc
                continue

            if sig_sol is not None and not sol_skip:
                out[sym] = sig_sol
                continue

            if sig_btc is not None:
                out[sym] = sig_btc
                continue

            if sig_sol is not None:
                out[sym] = sig_sol
                continue

            out[sym] = Signal(
                symbol=sym,
                side="flat",
                strength=0.0,
                meta={"engine": "portfolio", "skip": "not_supported"},
            )

        return out
