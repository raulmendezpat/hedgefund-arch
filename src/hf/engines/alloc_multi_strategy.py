from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from hf.core.opportunity import Opportunity
from hf.core.types import Allocation, Candle
from hf.engines.opportunity_book import (
    compute_competitive_score,
    compute_post_ml_competitive_score,
)


@dataclass
class MultiStrategyAllocator:
    score_power: float = 1.0
    min_score: float = 1e-12
    symbol_score_agg: str = "sum"
    normalize_total: bool = True
    switch_hysteresis: float = 0.10
    min_switch_bars: int = 6
    rebalance_deadband: float = 0.10
    weight_blend_alpha: float = 0.40

    def _safe_score(self, opp: Opportunity) -> float:
        meta = dict(getattr(opp, "meta", {}) or {})

        post_ml_candidates = [
            meta.get("post_ml_competitive_score", None),
            meta.get("post_ml_score", None),
        ]

        try:
            post_ml_candidates.append(compute_post_ml_competitive_score(opp))
        except Exception:
            pass

        for raw in post_ml_candidates:
            try:
                score = float(raw or 0.0)
            except Exception:
                score = 0.0
            if score >= self.min_score:
                return score

        base_candidates = [
            meta.get("competitive_score", None),
        ]

        try:
            base_candidates.append(compute_competitive_score(opp))
        except Exception:
            pass

        for raw in base_candidates:
            try:
                score = float(raw or 0.0)
            except Exception:
                score = 0.0
            if score >= self.min_score:
                return score

        return 0.0

    def _symbol_budget(self, symbol_scores: Dict[str, float]) -> Dict[str, float]:
        active = {k: float(v) for k, v in symbol_scores.items() if float(v) > 0}

        if not active:
            return {}

        total = sum(active.values())

        if total <= 0:
            return {}

        return {k: v / total for k, v in active.items()}

    def allocate_from_opportunities(
        self,
        *,
        candles: Dict[str, Candle],
        opportunities: List[Opportunity],
        prev_allocation: Optional[Allocation] = None,
    ) -> Allocation:

        weights_by_symbol: Dict[str, float] = {sym: 0.0 for sym in candles.keys()}
        strategy_weights: Dict[str, float] = {}

        opps_by_symbol: Dict[str, List[Opportunity]] = {}
        symbol_scores: Dict[str, float] = {}

        for opp in opportunities or []:

            symbol = str(getattr(opp, "symbol", "") or "")

            if symbol not in weights_by_symbol:
                continue

            score = self._safe_score(opp)

            if score <= 0:
                continue

            opps_by_symbol.setdefault(symbol, []).append(opp)

            if self.symbol_score_agg == "max":
                symbol_scores[symbol] = max(symbol_scores.get(symbol, 0.0), score)
            else:
                symbol_scores[symbol] = symbol_scores.get(symbol, 0.0) + score

        hysteresis_meta = {
            "applied": False,
            "prev_symbol": None,
            "best_symbol": None,
            "best_score": 0.0,
            "prev_score": 0.0,
            "switch_threshold": 0.0,
            "kept_symbol": None,
            "prev_dominant_symbol": None,
            "dominant_symbol": None,
            "dominant_age_bars": 0,
            "min_switch_bars": int(self.min_switch_bars),
            "hold_applied": False,
            "deadband_applied": False,
            "deadband_symbols": [],
            "blend_applied": False,
            "weight_blend_alpha": float(self.weight_blend_alpha),
        }

        if self.switch_hysteresis > 0 and prev_allocation is not None and symbol_scores:
            prev_weights = {
                str(k): float(v or 0.0)
                for k, v in dict(getattr(prev_allocation, "weights", {}) or {}).items()
            }
            prev_active = {k: v for k, v in prev_weights.items() if v > 0.0}

            if prev_active:
                prev_symbol = max(prev_active.items(), key=lambda kv: kv[1])[0]
                best_symbol, best_score = max(symbol_scores.items(), key=lambda kv: kv[1])
                prev_score = float(symbol_scores.get(prev_symbol, 0.0) or 0.0)

                hysteresis_meta.update({
                    "prev_symbol": prev_symbol,
                    "best_symbol": best_symbol,
                    "best_score": float(best_score),
                    "prev_score": float(prev_score),
                })

                if (
                    prev_symbol in symbol_scores
                    and best_symbol != prev_symbol
                    and prev_score > 0.0
                ):
                    switch_threshold = prev_score * (1.0 + float(self.switch_hysteresis))
                    hysteresis_meta["switch_threshold"] = float(switch_threshold)

                    if float(best_score) < float(switch_threshold):
                        symbol_scores[prev_symbol] = max(
                            float(symbol_scores.get(prev_symbol, 0.0) or 0.0),
                            float(switch_threshold),
                        )
                        hysteresis_meta["applied"] = True
                        hysteresis_meta["kept_symbol"] = prev_symbol

        if prev_allocation is not None and symbol_scores:
            prev_meta = dict(getattr(prev_allocation, "meta", {}) or {})
            prev_hmeta = dict(prev_meta.get("allocator_hysteresis", {}) or {})

            prev_dom_symbol = prev_hmeta.get("dominant_symbol", None)
            prev_dom_age = int(prev_hmeta.get("dominant_age_bars", 0) or 0)

            current_best_symbol = None
            current_best_score = 0.0
            if symbol_scores:
                current_best_symbol, current_best_score = max(symbol_scores.items(), key=lambda kv: kv[1])

            if current_best_symbol is not None:
                if prev_dom_symbol == current_best_symbol:
                    dominant_symbol = current_best_symbol
                    dominant_age_bars = prev_dom_age + 1
                else:
                    dominant_symbol = current_best_symbol
                    dominant_age_bars = 1

                hysteresis_meta["prev_dominant_symbol"] = prev_dom_symbol
                hysteresis_meta["dominant_symbol"] = dominant_symbol
                hysteresis_meta["dominant_age_bars"] = int(dominant_age_bars)

                if (
                    self.min_switch_bars > 0
                    and prev_dom_symbol
                    and current_best_symbol != prev_dom_symbol
                    and prev_dom_age < int(self.min_switch_bars)
                    and prev_dom_symbol in symbol_scores
                ):
                    keep_score = max(
                        float(symbol_scores.get(prev_dom_symbol, 0.0) or 0.0),
                        float(current_best_score or 0.0) + self.min_score,
                    )
                    symbol_scores[prev_dom_symbol] = keep_score
                    hysteresis_meta["hold_applied"] = True
                    hysteresis_meta["kept_symbol"] = prev_dom_symbol
                    hysteresis_meta["dominant_symbol"] = prev_dom_symbol
                    hysteresis_meta["dominant_age_bars"] = int(prev_dom_age + 1)

        symbol_budget = self._symbol_budget(symbol_scores)

        if not symbol_budget:

            if prev_allocation is not None:
                return Allocation(
                    weights=dict(prev_allocation.weights),
                    meta={"case": "multi_strategy_empty_sticky"},
                )

            return Allocation(
                weights=weights_by_symbol,
                meta={"case": "multi_strategy_empty"},
            )

        for symbol, opps in opps_by_symbol.items():

            powered = []

            for opp in opps:
                score = self._safe_score(opp)
                adj = score ** self.score_power

                if adj > 0:
                    powered.append((opp, adj))

            denom = sum(v for _, v in powered)

            if denom <= 0:
                continue

            symbol_cap = float(symbol_budget.get(symbol, 0.0))

            for opp, adj in powered:

                local_w = adj / denom
                final_w = symbol_cap * local_w

                strategy_id = str(getattr(opp, "strategy_id", "") or "unknown_strategy")
                key = f"{symbol}::{strategy_id}"

                strategy_weights[key] = final_w
                weights_by_symbol[symbol] = weights_by_symbol.get(symbol, 0.0) + final_w

        if prev_allocation is not None and self.weight_blend_alpha > 0:
            prev_weights = {
                str(k): float(v or 0.0)
                for k, v in dict(getattr(prev_allocation, "weights", {}) or {}).items()
            }

            alpha = float(self.weight_blend_alpha)
            for symbol in list(weights_by_symbol.keys()):
                target_w = float(weights_by_symbol.get(symbol, 0.0) or 0.0)
                prev_w = float(prev_weights.get(symbol, 0.0) or 0.0)
                weights_by_symbol[symbol] = alpha * target_w + (1.0 - alpha) * prev_w

            hysteresis_meta["blend_applied"] = True

        if prev_allocation is not None and self.rebalance_deadband > 0:
            prev_weights = {
                str(k): float(v or 0.0)
                for k, v in dict(getattr(prev_allocation, "weights", {}) or {}).items()
            }

            deadband_symbols = []

            for symbol in list(weights_by_symbol.keys()):
                new_w = float(weights_by_symbol.get(symbol, 0.0) or 0.0)
                prev_w = float(prev_weights.get(symbol, 0.0) or 0.0)

                if abs(new_w - prev_w) < float(self.rebalance_deadband):
                    weights_by_symbol[symbol] = prev_w
                    deadband_symbols.append(symbol)

            if deadband_symbols:
                hysteresis_meta["deadband_applied"] = True
                hysteresis_meta["deadband_symbols"] = deadband_symbols

        total = sum(weights_by_symbol.values())

        if self.normalize_total and total > 1:
            weights_by_symbol = {k: v / total for k, v in weights_by_symbol.items()}
            strategy_weights = {k: v / total for k, v in strategy_weights.items()}

        return Allocation(
            weights=weights_by_symbol,
            meta={
                "case": "multi_strategy",
                "strategy_weights": strategy_weights,
                "symbol_budget": symbol_budget,
                "allocator_hysteresis": hysteresis_meta,
            },
        )
