from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional
import json

from hf.core.interfaces import SignalEngine
from hf.core.types import Candle, Signal
from hf.engines.signals.btc_trend_signal import BtcTrendSignalEngine
from hf.engines.signals.sol_bbrsi_signal import SolBbrsiSignalEngine
from hf.engines.signals.sol_vol_breakout_signal import SolVolBreakoutSignalEngine
from hf.engines.signals.sol_trend_pullback_signal import SolTrendPullbackSignalEngine
from hf.engines.signals.sol_extreme_mr_signal import SolExtremeMrSignalEngine
from hf.engines.signals.sol_vol_compression_signal import SolVolCompressionSignalEngine
from hf.engines.signals.sol_vol_expansion_signal import SolVolExpansionSignalEngine
from hf.engines.signals.link_trend_signal import LinkTrendSignalEngine
from hf.engines.signals.aave_trend_signal import AaveTrendSignalEngine
from hf.engines.signals.bnb_trend_signal import BnbTrendSignalEngine
from hf.engines.signals.eth_trend_signal import EthTrendSignalEngine
from hf.engines.signals.xrp_trend_signal import XrpTrendSignalEngine
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
                "sol_trend_pullback_signal": lambda cfg: SolTrendPullbackSignalEngine(**dict(cfg.get("params", {}) or {})),
                "sol_extreme_mr_signal": lambda cfg: SolExtremeMrSignalEngine(**dict(cfg.get("params", {}) or {})),
                "sol_vol_compression_signal": lambda cfg: SolVolCompressionSignalEngine(**dict(cfg.get("params", {}) or {})),
                "sol_vol_expansion_signal": lambda cfg: SolVolExpansionSignalEngine(**dict(cfg.get("params", {}) or {})),
                "link_trend_signal": lambda cfg: LinkTrendSignalEngine(**dict(cfg.get("params", {}) or {})),
                "aave_trend_signal": lambda cfg: AaveTrendSignalEngine(**dict(cfg.get("params", {}) or {})),
                "bnb_trend_signal": lambda cfg: BnbTrendSignalEngine(**dict(cfg.get("params", {}) or {})),
                "eth_trend_signal": lambda cfg: EthTrendSignalEngine(**dict(cfg.get("params", {}) or {})),
                "xrp_trend_signal": lambda cfg: XrpTrendSignalEngine(**dict(cfg.get("params", {}) or {})),
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
    sol_trend_pullback_engine: Optional[SolTrendPullbackSignalEngine] = None
    sol_extreme_mr_engine: Optional[SolExtremeMrSignalEngine] = None
    sol_vol_compression_engine: Optional[SolVolCompressionSignalEngine] = None
    sol_vol_expansion_engine: Optional[SolVolExpansionSignalEngine] = None

    def __post_init__(self) -> None:
        btc_engine = self.btc_engine or BtcTrendSignalEngine()
        sol_engine = self.sol_engine or SolBbrsiSignalEngine()
        sol_vol_breakout_engine = self.sol_vol_breakout_engine or SolVolBreakoutSignalEngine()
        sol_trend_pullback_engine = self.sol_trend_pullback_engine or SolTrendPullbackSignalEngine()
        sol_extreme_mr_engine = self.sol_extreme_mr_engine or SolExtremeMrSignalEngine()
        sol_vol_compression_engine = self.sol_vol_compression_engine or SolVolCompressionSignalEngine()
        sol_vol_expansion_engine = self.sol_vol_expansion_engine or SolVolExpansionSignalEngine()

        self.engine_factories = {
            "btc_trend_signal": lambda cfg: btc_engine,
            "sol_bbrsi_signal": lambda cfg: sol_engine,
            "sol_vol_breakout_signal": lambda cfg: sol_vol_breakout_engine,
            "sol_trend_pullback_signal": lambda cfg: sol_trend_pullback_engine,
            "sol_extreme_mr_signal": lambda cfg: sol_extreme_mr_engine,
            "sol_vol_compression_signal": lambda cfg: sol_vol_compression_engine,
            "sol_vol_expansion_signal": lambda cfg: sol_vol_expansion_engine,
        }

        if not getattr(self, "registry_path", None):
            self.registry_path = "artifacts/strategy_registry.json"
