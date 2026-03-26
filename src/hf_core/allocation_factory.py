from __future__ import annotations

from hf.engines.alloc_multi_strategy import MultiStrategyAllocator

from hf_core.allocation_config import AllocationConfig, allocation_config_to_dict
from hf_core.allocation_engine import AllocationEngine
from hf_core.opportunity_adapter import OpportunityAdapter
from hf_core.allocation_postprocessor import AllocationPostprocessor
from hf_core.portfolio_regime_scaler import PortfolioRegimeScaler
from hf_core.allocation_step_guardrail import AllocationStepGuardrail
from hf_core.strategy_side_weight_overlay import StrategySideWeightOverlay
from hf_core.strategy_side_weight_rules import parse_strategy_side_post_ml_weight_rules
from hf_core.portfolio_risk_overlay import PortfolioRiskOverlay


def build_allocation_engine(
    *,
    config: AllocationConfig,
) -> AllocationEngine:
    allocator = MultiStrategyAllocator(
        score_power=float(config.score_power),
        min_score=float(config.min_score),
        symbol_score_agg=str(config.symbol_score_agg),
        normalize_total=bool(config.normalize_total),
        switch_hysteresis=float(config.switch_hysteresis),
        min_switch_bars=int(config.min_switch_bars),
        rebalance_deadband=float(config.rebalance_deadband),
        weight_blend_alpha=float(config.weight_blend_alpha),
        symbol_cap=float(config.symbol_cap),
        target_exposure=float(config.target_exposure),
    )

    adapter = OpportunityAdapter(
        base_weight_projection=str(config.base_weight_projection),
    )

    regime_scaler = PortfolioRegimeScaler(
        defensive_scale=float(config.defensive_scale),
        defensive_conviction_k=float(config.defensive_conviction_k),
        aggressive_scale=float(config.aggressive_scale),
    )

    portfolio_risk_overlay = PortfolioRiskOverlay(
        breadth_high_risk=int(config.breadth_high_risk),
        pwin_high_risk=float(config.pwin_high_risk),
        high_risk_scale=float(config.high_risk_scale),
    )

    strategy_side_weight_overlay = StrategySideWeightOverlay(
        rules=parse_strategy_side_post_ml_weight_rules(config.strategy_side_post_ml_weight_rules),
    )

    postprocessor = AllocationPostprocessor(
        smoothing_alpha=float(config.smoothing_alpha),
        smoothing_snap_eps=float(config.smoothing_snap_eps),
    )

    step_guardrail = AllocationStepGuardrail(
        max_step_per_bar=float(config.max_step_per_bar),
    )

    return AllocationEngine(
        opportunity_adapter=adapter,
        allocator=allocator,
        regime_scaler=regime_scaler,
        portfolio_risk_overlay=portfolio_risk_overlay,
        strategy_side_weight_overlay=strategy_side_weight_overlay,
        postprocessor=postprocessor,
        step_guardrail=step_guardrail,
        config_snapshot=allocation_config_to_dict(config),
    )
