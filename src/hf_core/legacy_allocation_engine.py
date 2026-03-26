from __future__ import annotations

from dataclasses import dataclass
from hf.core.types import Allocation, Candle
from hf.engines.alloc_multi_strategy import MultiStrategyAllocator

from hf_core.contracts import OpportunityCandidate
from hf_core.opportunity_adapter import OpportunityAdapter
from hf_core.allocation_postprocessor import AllocationPostprocessor
from hf_core.portfolio_regime_scaler import PortfolioRegimeScaler
from hf_core.portfolio_risk_overlay import PortfolioRiskOverlay
from hf_core.strategy_side_weight_overlay import StrategySideWeightOverlay
from hf_core.allocation_step_guardrail import AllocationStepGuardrail
from hf_core.competition import (
    opp_score,
    build_pre_allocator_trace,
    apply_competition_mode,
)
from hf_core.allocation_pipeline import run_allocation_stages


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
    opportunity_adapter: OpportunityAdapter
    allocator: MultiStrategyAllocator
    regime_scaler: PortfolioRegimeScaler
    portfolio_risk_overlay: PortfolioRiskOverlay
    strategy_side_weight_overlay: StrategySideWeightOverlay
    postprocessor: AllocationPostprocessor
    step_guardrail: AllocationStepGuardrail
    config_snapshot: dict | None = None

    def allocate(
        self,
        *,
        candles: dict[str, Candle],
        candidates: list[OpportunityCandidate],
        prev_allocation: Allocation | None = None,
        portfolio_context: dict | None = None,
    ) -> Allocation:
        legacy_opps = self.opportunity_adapter.to_opportunities(candidates)
        legacy_pre_allocator_trace = build_pre_allocator_trace(legacy_opps)
        legacy_opps_filtered, legacy_competition_summary = apply_competition_mode(
            legacy_opps,
            config_snapshot=self.config_snapshot,
        )

        allocator_prev_allocation = None
        if prev_allocation is not None:
            prev_weights_signed = dict(getattr(prev_allocation, "weights", {}) or {})
            allocator_prev_allocation = Allocation(
                weights={
                    str(k): float(abs(float(v or 0.0)))
                    for k, v in prev_weights_signed.items()
                },
                meta={
                    **dict(getattr(prev_allocation, "meta", {}) or {}),
                    "allocator_prev_weights_source": "abs_signed_prev_allocation",
                },
            )

        alloc_core = self.allocator.allocate_from_opportunities(
            candles=candles,
            opportunities=legacy_opps_filtered,
            prev_allocation=allocator_prev_allocation,
        )

        alloc_core_meta = dict(getattr(alloc_core, "meta", {}) or {})
        alloc_core_meta["legacy_pre_allocator_trace"] = dict(legacy_pre_allocator_trace or {})
        alloc_core_meta["legacy_competition_summary"] = dict(legacy_competition_summary or {})
        alloc_core_meta["legacy_opportunities_pre_competition"] = [
            {
                "symbol": str(getattr(o, "symbol", "") or ""),
                "strategy_id": str(getattr(o, "strategy_id", "") or ""),
                "side": str(getattr(o, "side", "") or ""),
                "strength": float(getattr(o, "strength", 0.0) or 0.0),
                "timestamp": int(getattr(o, "timestamp", 0) or 0),
                "meta": dict(getattr(o, "meta", {}) or {}),
            }
            for o in list(legacy_opps_filtered or [])
        ]
        alloc_core_meta["legacy_opportunities"] = [
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
        alloc_core = Allocation(
            weights=dict(getattr(alloc_core, "weights", {}) or {}),
            meta=alloc_core_meta,
        )

        _stages = run_allocation_stages(
            alloc_core=alloc_core,
            regime_scaler=self.regime_scaler,
            portfolio_risk_overlay=self.portfolio_risk_overlay,
            strategy_side_weight_overlay=self.strategy_side_weight_overlay,
            postprocessor=self.postprocessor,
            step_guardrail=self.step_guardrail,
            prev_allocation=prev_allocation,
            portfolio_context=portfolio_context,
        )

        alloc_regime = _stages["alloc_regime"]
        alloc_risk = _stages["alloc_risk"]

        alloc_risk_meta = dict(getattr(alloc_risk, "meta", {}) or {})
        alloc_risk_meta["legacy_pre_allocator_trace"] = dict(legacy_pre_allocator_trace or {})
        alloc_risk_meta["legacy_competition_summary"] = dict(legacy_competition_summary or {})
        alloc_risk_meta["legacy_opportunities_pre_competition"] = [
            {
                "symbol": str(getattr(o, "symbol", "") or ""),
                "strategy_id": str(getattr(o, "strategy_id", "") or ""),
                "side": str(getattr(o, "side", "") or ""),
                "strength": float(getattr(o, "strength", 0.0) or 0.0),
                "timestamp": int(getattr(o, "timestamp", 0) or 0),
                "meta": dict(getattr(o, "meta", {}) or {}),
            }
            for o in list(legacy_opps_filtered or [])
        ]
        alloc_risk_meta["legacy_opportunities"] = [
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
        alloc_risk = Allocation(
            weights=dict(getattr(alloc_risk, "weights", {}) or {}),
            meta=alloc_risk_meta,
        )

        alloc_side = _stages["alloc_side"]
        alloc_post = _stages["alloc_post"]
        alloc = _stages["alloc_final"]
        _stage_weights = dict(_stages["stage_weights"] or {})
        _stage_gross = dict(_stages["stage_gross"] or {})

        meta = dict(getattr(alloc, "meta", {}) or {})
        meta["legacy_engine_applied"] = True
        meta["legacy_opportunity_count"] = int(len(legacy_opps))
        meta["legacy_allocation_config"] = dict(self.config_snapshot or {})
        meta["legacy_pre_allocator_trace"] = dict(legacy_pre_allocator_trace or {})
        meta["legacy_competition_summary"] = dict(legacy_competition_summary or {})
        meta["legacy_opportunities_pre_competition"] = [
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
        meta["legacy_stage_weights"] = _stage_weights
        meta["legacy_stage_gross_exposure"] = _stage_gross
        meta["legacy_allocator_prev_allocation_mode"] = "abs_signed_prev_allocation"
        if prev_allocation is not None:
            meta["legacy_prev_weights_signed"] = {
                str(k): float(v or 0.0)
                for k, v in dict(getattr(prev_allocation, "weights", {}) or {}).items()
            }
        if allocator_prev_allocation is not None:
            meta["legacy_prev_weights_abs_for_allocator"] = {
                str(k): float(v or 0.0)
                for k, v in dict(getattr(allocator_prev_allocation, "weights", {}) or {}).items()
            }
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
