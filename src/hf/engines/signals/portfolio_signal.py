from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json

from hf.core.interfaces import SignalEngine
from hf.core.types import Candle, Signal
from hf.engines.signals.btc_trend_signal import BtcTrendSignalEngine
from hf.engines.signals.sol_bbrsi_signal import SolBbrsiSignalEngine
from hf.engines.signals.sol_vol_breakout_signal import SolVolBreakoutSignalEngine
from hf.engines.opportunity_book import RegistryOpportunityBook, to_signal_dict


def _default_registry() -> List[dict]:
    return [
        {
            "strategy_id": "btc_trend",
            "symbol": "BTC/USDT:USDT",
            "engine": "btc_trend_signal",
            "enabled": True,
            "params": {},
        },
        {
            "strategy_id": "sol_bbrsi",
            "symbol": "SOL/USDT:USDT",
            "engine": "sol_bbrsi_signal",
            "enabled": True,
            "params": {},
        },
        {
            "strategy_id": "sol_vol_breakout",
            "symbol": "SOL/USDT:USDT",
            "engine": "sol_vol_breakout_signal",
            "enabled": True,
            "params": {},
        },
    ]


@dataclass
class RegistryPortfolioSignalEngine(SignalEngine):
    registry_path: str = "artifacts/strategy_registry.json"
    engine_factories: Dict[str, Callable[[dict], SignalEngine]] = field(default_factory=dict)
    strict_symbol_match: bool = False
    selection_mode: str = "best_per_symbol"

    def __post_init__(self) -> None:
        if not self.engine_factories:
            self.engine_factories = {
                "btc_trend_signal": lambda cfg: BtcTrendSignalEngine(**dict(cfg.get("params", {}) or {})),
                "sol_bbrsi_signal": lambda cfg: SolBbrsiSignalEngine(**dict(cfg.get("params", {}) or {})),
                "sol_vol_breakout_signal": lambda cfg: SolVolBreakoutSignalEngine(**dict(cfg.get("params", {}) or {})),
            }

    def _load_registry(self) -> List[dict]:
        p = Path(self.registry_path)
        if not p.exists():
            return _default_registry()

        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Strategy registry must be a list, got: {type(data).__name__}")
        return data

    def _select_symbol_candles(self, candles: Dict[str, Candle], symbol: str) -> Dict[str, Candle]:
        if symbol in candles:
            return {symbol: candles[symbol]}

        if self.strict_symbol_match:
            return {}

        out = {}
        wanted = str(symbol).upper()
        for sym, candle in candles.items():
            if str(sym).upper() == wanted:
                out[sym] = candle
        return out

    def _decorate_signal(self, *, cfg: dict, signal: Signal) -> Signal:
        meta = dict(getattr(signal, "meta", {}) or {})
        return Signal(
            symbol=signal.symbol,
            side=signal.side,
            strength=float(getattr(signal, "strength", 0.0) or 0.0),
            meta={
                **meta,
                "strategy_id": cfg.get("strategy_id"),
                "registry_symbol": cfg.get("symbol"),
                "engine": cfg.get("engine"),
            },
        )

    def _candidate_rank(self, sig: Optional[Signal]) -> tuple:
        if sig is None:
            return (0, 0, 0.0)

        meta = dict(getattr(sig, "meta", {}) or {})
        has_skip = 1 if "skip" not in meta else 0
        not_flat = 1 if getattr(sig, "side", "flat") != "flat" else 0
        strength = abs(float(getattr(sig, "strength", 0.0) or 0.0))
        return (has_skip, not_flat, strength)

    def generate(self, candles: Dict[str, Candle], print_debug: bool = False) -> Dict[str, Signal]:
        book = RegistryOpportunityBook(
            registry_path=self.registry_path,
            engine_factories=self.engine_factories,
            strict_symbol_match=self.strict_symbol_match,
        )
        opportunities = book.generate(candles, ts=None, print_debug=print_debug)
        self.last_opportunities = list(opportunities or [])

        compatible_selection_mode = self.selection_mode
        if str(self.selection_mode or "").strip() == "all":
            compatible_selection_mode = "best_per_symbol"

        return to_signal_dict(
            opportunities,
            symbols=list(candles.keys()),
            selection_mode=compatible_selection_mode,
        )


@dataclass
class PortfolioSignalEngine(RegistryPortfolioSignalEngine):
    btc_engine: Optional[BtcTrendSignalEngine] = None
    sol_engine: Optional[SolBbrsiSignalEngine] = None
    sol_vol_breakout_engine: Optional[SolVolBreakoutSignalEngine] = None

    def __post_init__(self) -> None:
        btc_engine = self.btc_engine or BtcTrendSignalEngine()
        sol_engine = self.sol_engine or SolBbrsiSignalEngine()
        sol_vol_breakout_engine = self.sol_vol_breakout_engine or SolVolBreakoutSignalEngine()

        self.engine_factories = {
            "btc_trend_signal": lambda cfg: btc_engine,
            "sol_bbrsi_signal": lambda cfg: sol_engine,
            "sol_vol_breakout_signal": lambda cfg: sol_vol_breakout_engine,
        }

        if not getattr(self, "registry_path", None):
            self.registry_path = "artifacts/strategy_registry.json"
