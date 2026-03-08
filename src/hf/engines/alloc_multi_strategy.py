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
            },
        )
