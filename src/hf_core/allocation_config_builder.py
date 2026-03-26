from __future__ import annotations

from hf_core.allocation_config import AllocationConfig


def build_allocation_config_from_args(args) -> AllocationConfig:
    return AllocationConfig(
        score_power=float(args.legacy_score_power),
        min_score=float(args.legacy_min_score),
        symbol_score_agg=str(args.legacy_symbol_score_agg),
        normalize_total=bool(args.legacy_normalize_total),
        switch_hysteresis=float(args.legacy_switch_hysteresis),
        min_switch_bars=int(args.legacy_min_switch_bars),
        rebalance_deadband=float(args.legacy_rebalance_deadband),
        weight_blend_alpha=float(args.legacy_weight_blend_alpha),
        symbol_cap=float(args.symbol_cap),
        target_exposure=float(args.target_exposure),
        base_weight_projection="raw",
        smoothing_alpha=float(args.legacy_allocator_smoothing_alpha),
        smoothing_snap_eps=float(args.legacy_allocator_smoothing_snap_eps),
        defensive_scale=float(args.legacy_portfolio_regime_defensive_scale),
        defensive_conviction_k=float(args.legacy_portfolio_regime_defensive_conviction_k),
        aggressive_scale=float(args.legacy_portfolio_regime_aggressive_scale),
        breadth_high_risk=int(args.legacy_portfolio_breadth_high_risk),
        pwin_high_risk=float(args.legacy_portfolio_pwin_high_risk),
        high_risk_scale=float(args.legacy_portfolio_high_risk_scale),
        strategy_side_post_ml_weight_rules=str(args.legacy_strategy_side_post_ml_weight_rules or ""),
        max_step_per_bar=float(args.legacy_allocator_max_step_per_bar),
        legacy_competition_mode=str(getattr(args, "legacy_competition_mode", "off")),
    )
