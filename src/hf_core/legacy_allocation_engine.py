from __future__ import annotations

from dataclasses import dataclass

from hf.core.types import Allocation, Candle
from hf.engines.alloc_multi_strategy import MultiStrategyAllocator

from hf_core.contracts import OpportunityCandidate
from hf_core.legacy_opportunity_adapter import LegacyOpportunityAdapter
from hf_core.legacy_allocation_postprocessor import LegacyAllocationPostprocessor
from hf_core.legacy_allocation_regime_scaler import LegacyAllocationRegimeScaler
from hf_core.legacy_portfolio_risk_overlay import LegacyPortfolioRiskOverlay
from hf_core.legacy_strategy_side_weight_overlay import LegacyStrategySideWeightOverlay
from hf_core.legacy_allocation_step_guardrail import LegacyAllocationStepGuardrail


def _weights_snapshot(allocation: Allocation | None) -> dict[str, float]:
    if allocation is None:
        return {}
    return {
        str(k): float(v or 0.0)
        for k, v in dict(getattr(allocation, "weights", {}) or {}).items()
    }


def _gross_exposure(weights: dict[str, float]) -> float:
    return float(sum(abs(float(v or 0.0)) for v in dict(weights or {}).values()))


@dataclass
class LegacyAllocationEngine:
    opportunity_adapter: LegacyOpportunityAdapter
    allocator: MultiStrategyAllocator
    regime_scaler: LegacyAllocationRegimeScaler
    portfolio_risk_overlay: LegacyPortfolioRiskOverlay
    strategy_side_weight_overlay: LegacyStrategySideWeightOverlay
    postprocessor: LegacyAllocationPostprocessor
    step_guardrail: LegacyAllocationStepGuardrail
    config_snapshot: dict | None = None

    def allocate(
        self,
        *,
        candles: dict[str, Candle],
        candidates: list[OpportunityCandidate],
        prev_allocation: Allocation | None = None,
        portfolio_context: dict | None = None,
    ) -> Allocation:
        legacy_opps = self.opportunity_adapter.to_legacy_opportunities(candidates)

        alloc_core = self.allocator.allocate_from_opportunities(
            candles=candles,
            opportunities=legacy_opps,
            prev_allocation=prev_allocation,
        )
        alloc_regime = self.regime_scaler.apply(
            allocation=alloc_core,
            portfolio_context=portfolio_context,
        )
        alloc_risk = self.portfolio_risk_overlay.apply(
            allocation=alloc_regime,
            portfolio_context=portfolio_context,
        )
        alloc_side = self.strategy_side_weight_overlay.apply(
            allocation=alloc_risk,
        )
        alloc_post = self.postprocessor.apply(
            allocation=alloc_side,
            prev_allocation=prev_allocation,
        )
        alloc = self.step_guardrail.apply(
            allocation=alloc_post,
            prev_allocation=prev_allocation,
        )

        _stage_weights = {
            "allocator_core": _weights_snapshot(alloc_core),
            "after_regime_scaler": _weights_snapshot(alloc_regime),
            "after_portfolio_risk_overlay": _weights_snapshot(alloc_risk),
            "after_strategy_side_overlay": _weights_snapshot(alloc_side),
            "after_postprocessor": _weights_snapshot(alloc_post),
            "after_step_guardrail": _weights_snapshot(alloc),
        }
        _stage_gross = {
            k: _gross_exposure(v)
            for k, v in _stage_weights.items()
        }

        meta = dict(getattr(alloc, "meta", {}) or {})
        meta["legacy_engine_applied"] = True
        meta["legacy_opportunity_count"] = int(len(legacy_opps))
        meta["legacy_allocation_config"] = dict(self.config_snapshot or {})
        meta["legacy_stage_weights"] = _stage_weights
        meta["legacy_stage_gross_exposure"] = _stage_gross
        meta["legacy_stage_order"] = [
            "allocator_core",
            "after_regime_scaler",
            "after_portfolio_risk_overlay",
            "after_strategy_side_overlay",
            "after_postprocessor",
            "after_step_guardrail",
        ]
        meta["legacy_opportunities"] = [
            {
                "symbol": str(getattr(o, "symbol", "") or ""),
                "strategy_id": str(getattr(o, "strategy_id", "") or ""),
                "side": str(getattr(o, "side", "") or ""),
                "strength": float(getattr(o, "strength", 0.0) or 0.0),
                "timestamp": int(getattr(o, "timestamp", 0) or 0),
                "meta": dict(getattr(o, "meta", {}) or {}),
            }
            for o in list(legacy_opps or [])
        ]
        return Allocation(
            weights=dict(getattr(alloc, "weights", {}) or {}),
            meta=meta,
        )

    def to_legacy_opportunities(
        self,
        candidates: list[OpportunityCandidate],
    ):
        return self.opportunity_adapter.to_legacy_opportunities(candidates)
