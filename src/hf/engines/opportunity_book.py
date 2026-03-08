from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional
import json

from hf.core.interfaces import SignalEngine
from hf.core.types import Candle, Signal
from hf.core.opportunity import Opportunity
from hf.engines.signals.btc_trend_signal import BtcTrendSignalEngine
from hf.engines.signals.sol_bbrsi_signal import SolBbrsiSignalEngine


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
    ]


@dataclass
class RegistryOpportunityBook:
    registry_path: str = "artifacts/strategy_registry.json"
    engine_factories: Dict[str, Callable[[dict], SignalEngine]] = field(default_factory=dict)
    strict_symbol_match: bool = False

    def __post_init__(self) -> None:
        if not self.engine_factories:
            self.engine_factories = {
                "btc_trend_signal": lambda cfg: BtcTrendSignalEngine(**dict(cfg.get("params", {}) or {})),
                "sol_bbrsi_signal": lambda cfg: SolBbrsiSignalEngine(**dict(cfg.get("params", {}) or {})),
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

    def generate(self, candles: Dict[str, Candle], ts: Optional[int] = None, print_debug: bool = False) -> List[Opportunity]:
        registry = self._load_registry()
        opportunities: List[Opportunity] = []

        for cfg in registry:
            if not bool(cfg.get("enabled", True)):
                continue

            engine_name = str(cfg.get("engine", "") or "").strip()
            strategy_id = str(cfg.get("strategy_id", "") or "").strip()
            target_symbol = str(cfg.get("symbol", "") or "").strip()

            if not engine_name or not strategy_id or not target_symbol:
                continue
            if engine_name not in self.engine_factories:
                raise ValueError(f"Unknown signal engine in registry: {engine_name}")

            sub_candles = self._select_symbol_candles(candles, target_symbol)
            if not sub_candles:
                continue

            engine = self.engine_factories[engine_name](cfg)
            try:
                generated = engine.generate(sub_candles, print_debug=print_debug)  # type: ignore[arg-type]
            except TypeError:
                generated = engine.generate(sub_candles)

            for _, sig in (generated or {}).items():
                decorated = self._decorate_signal(cfg=cfg, signal=sig)
                opportunities.append(
                    Opportunity.from_signal(
                        decorated,
                        strategy_id=strategy_id,
                        timestamp=ts,
                    )
                )

        return opportunities


def _opportunity_rank(opp: Opportunity) -> tuple:
    active = 1 if opp.is_active() else 0
    strength = abs(float(getattr(opp, "strength", 0.0) or 0.0))
    return (active, strength)


def select_best_opportunities_per_symbol(opportunities: List[Opportunity]) -> List[Opportunity]:
    best_by_symbol: Dict[str, Opportunity] = {}

    for opp in opportunities or []:
        sym = str(opp.symbol)
        prev = best_by_symbol.get(sym)
        if prev is None or _opportunity_rank(opp) > _opportunity_rank(prev):
            best_by_symbol[sym] = opp

    return list(best_by_symbol.values())


def to_signal_dict(opportunities: List[Opportunity], symbols: Optional[List[str]] = None) -> Dict[str, Signal]:
    selected = select_best_opportunities_per_symbol(opportunities)
    out: Dict[str, Signal] = {}

    for opp in selected:
        out[str(opp.symbol)] = Signal(
            symbol=opp.symbol,
            side=opp.side,
            strength=float(getattr(opp, "strength", 0.0) or 0.0),
            meta=dict(getattr(opp, "meta", {}) or {}),
        )

    for sym in (symbols or []):
        if sym not in out:
            out[sym] = Signal(
                symbol=sym,
                side="flat",
                strength=0.0,
                meta={"engine": "opportunity_adapter", "skip": "no_opportunity"},
            )

    return out
