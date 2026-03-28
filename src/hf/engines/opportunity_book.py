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
from hf.engines.signals.trx_trend_signal import TrxTrendSignalEngine
from hf.engines.signals.btc_short_trend_signal import BtcShortTrendSignalEngine
from hf.engines.signals.avax_trend_signal import AvaxTrendSignalEngine
from hf.engines.signals.dot_trend_signal import DotTrendSignalEngine


def _default_registry() -> List[dict]:
    return [
        {
            "strategy_id": "btc_trend",
            "symbol": "BTC/USDT:USDT",
            "engine": "btc_trend_signal",
            "enabled": True,
            "base_weight": 1.0,
            "params": {},
        },
        {
            "strategy_id": "sol_bbrsi",
            "symbol": "SOL/USDT:USDT",
            "engine": "sol_bbrsi_signal",
            "enabled": True,
            "base_weight": 1.0,
            "params": {},
        },
        {
            "strategy_id": "sol_vol_breakout",
            "symbol": "SOL/USDT:USDT",
            "engine": "sol_vol_breakout_signal",
            "enabled": True,
            "base_weight": 1.0,
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
                "sol_vol_breakout_signal": lambda cfg: SolVolBreakoutSignalEngine(**dict(cfg.get("params", {}) or {})),
                "sol_trend_pullback_signal": lambda cfg: SolTrendPullbackSignalEngine(
                    rsi_long_min=float((cfg.get("params", {}) or {}).get("rsi_long_min", 40.0)),
                    rsi_long_max=float((cfg.get("params", {}) or {}).get("rsi_long_max", 55.0)),
                    rsi_short_min=float((cfg.get("params", {}) or {}).get("rsi_short_min", 45.0)),
                    rsi_short_max=float((cfg.get("params", {}) or {}).get("rsi_short_max", 60.0)),
                    ema_pullback_max=float((cfg.get("params", {}) or {}).get("ema_pullback_max", 0.015)),
                    atrp_min=float((cfg.get("params", {}) or {}).get("atrp_min", 0.004)),
                    atrp_max=float((cfg.get("params", {}) or {}).get("atrp_max", 0.050)),
                    require_adx=bool((cfg.get("params", {}) or {}).get("require_adx", False)),
                    adx_min=float((cfg.get("params", {}) or {}).get("adx_min", 18.0)),
                ),
                "sol_extreme_mr_signal": lambda cfg: SolExtremeMrSignalEngine(**dict(cfg.get("params", {}) or {})),
                "sol_vol_compression_signal": lambda cfg: SolVolCompressionSignalEngine(**dict(cfg.get("params", {}) or {})),
                "sol_vol_expansion_signal": lambda cfg: SolVolExpansionSignalEngine(**dict(cfg.get("params", {}) or {})),
                "link_trend_signal": lambda cfg: LinkTrendSignalEngine(**dict(cfg.get("params", {}) or {})),
                "aave_trend_signal": lambda cfg: AaveTrendSignalEngine(**dict(cfg.get("params", {}) or {})),
                "bnb_trend_signal": lambda cfg: BnbTrendSignalEngine(**dict(cfg.get("params", {}) or {})),
                "eth_trend_signal": lambda cfg: EthTrendSignalEngine(**dict(cfg.get("params", {}) or {})),
                "xrp_trend_signal": lambda cfg: XrpTrendSignalEngine(**dict(cfg.get("params", {}) or {})),
            "trx_trend_signal": lambda cfg: TrxTrendSignalEngine(**dict(cfg.get("params", {}) or {})),
            "btc_short_trend_signal": lambda cfg: BtcShortTrendSignalEngine(**dict(cfg.get("params", {}) or {})),
            "avax_trend_signal": lambda cfg: AvaxTrendSignalEngine(**dict(cfg.get("params", {}) or {})),
            "dot_trend_signal": lambda cfg: DotTrendSignalEngine(**dict(cfg.get("params", {}) or {})),
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
        try:
            base_weight = float(cfg.get("base_weight", 1.0) or 1.0)
        except (TypeError, ValueError):
            base_weight = 1.0

        return Signal(
            symbol=signal.symbol,
            side=signal.side,
            strength=float(getattr(signal, "strength", 0.0) or 0.0),
            meta={
                **meta,
                "strategy_id": cfg.get("strategy_id"),
                "registry_symbol": cfg.get("symbol"),
                "engine": cfg.get("engine"),
                "base_weight": base_weight,
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
                generated = engine.generate(sub_candles, print_debug=print_debug)
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


def compute_competitive_score(opp: Opportunity) -> float:
    active_flag = 1.0 if opp.is_active() else 0.0
    strength = abs(float(getattr(opp, "strength", 0.0) or 0.0))
    meta = dict(getattr(opp, "meta", {}) or {})

    try:
        base_weight = float(meta.get("base_weight", 1.0) or 1.0)
    except (TypeError, ValueError):
        base_weight = 1.0

    return active_flag * strength * base_weight


def compute_post_ml_competitive_score(opp: Opportunity) -> float:
    base_score = float(compute_competitive_score(opp))
    meta = dict(getattr(opp, "meta", {}) or {})

    try:
        p_win_factor = float(meta.get("p_win", 1.0) or 1.0)
    except (TypeError, ValueError):
        p_win_factor = 1.0
    if p_win_factor <= 0.0:
        p_win_factor = 1.0

    try:
        size_factor = float(meta.get("ml_position_size_mult", 1.0) or 1.0)
    except (TypeError, ValueError):
        size_factor = 1.0
    if size_factor <= 0.0:
        size_factor = 1.0

    size_factor = max(0.50, min(1.50, size_factor))

    return float(base_score * p_win_factor * size_factor)


def select_competitive_opportunities(opportunities: List[Opportunity]) -> List[Opportunity]:
    best_by_symbol: Dict[str, Opportunity] = {}

    for opp in opportunities or []:
        sym = str(opp.symbol)

        _meta = dict(getattr(opp, "meta", {}) or {})
        _meta["competitive_score"] = float(compute_competitive_score(opp))
        _meta["post_ml_competitive_score"] = float(compute_post_ml_competitive_score(opp))
        opp.meta = _meta

        prev = best_by_symbol.get(sym)

        opp_score = float(_meta.get("post_ml_competitive_score", 0.0) or 0.0)
        prev_score = compute_post_ml_competitive_score(prev) if prev is not None else float("-inf")

        if prev is None:
            best_by_symbol[sym] = opp
            continue

        if opp_score > prev_score:
            best_by_symbol[sym] = opp
            continue

        if opp_score == prev_score:
            opp_strength = abs(float(getattr(opp, "strength", 0.0) or 0.0))
            prev_strength = abs(float(getattr(prev, "strength", 0.0) or 0.0))

            if opp_strength > prev_strength:
                best_by_symbol[sym] = opp
                continue

    return list(best_by_symbol.values())


def select_opportunities(opportunities: List[Opportunity], mode: str = "best_per_symbol") -> List[Opportunity]:
    mode = str(mode or "best_per_symbol").strip()

    if mode == "best_per_symbol":
        return select_best_opportunities_per_symbol(opportunities)

    if mode == "competitive":
        return select_competitive_opportunities(opportunities)

    if mode == "all":
        return list(opportunities or [])

    raise ValueError(f"Unknown opportunity selection mode: {mode}")


def to_signal_dict(
    opportunities: List[Opportunity],
    symbols: Optional[List[str]] = None,
    selection_mode: str = "best_per_symbol",
) -> Dict[str, Signal]:
    if str(selection_mode or "").strip() == "all":
        raise ValueError(
            "selection_mode='all' is not compatible with to_signal_dict(), "
            "because multiple opportunities may share the same symbol and "
            "dict[symbol] -> Signal cannot represent them without overwriting. "
            "Use 'best_per_symbol' for legacy pipeline compatibility, or "
            "implement a competitive selector/allocator for multi-opportunity execution."
        )

    selected = select_opportunities(opportunities, mode=selection_mode)
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
