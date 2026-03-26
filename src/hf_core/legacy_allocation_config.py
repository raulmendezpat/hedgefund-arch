from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegacyAllocationConfig:
    score_power: float = 1.0
    min_score: float = 1e-12
    symbol_score_agg: str = "sum"
    normalize_total: bool = False
    switch_hysteresis: float = 0.10
    min_switch_bars: int = 6
    rebalance_deadband: float = 0.03
    weight_blend_alpha: float = 0.55
    symbol_cap: float = 0.50
    target_exposure: float = 0.07
    base_weight_projection: str = "raw"
    smoothing_alpha: float = 0.0
    smoothing_snap_eps: float = 0.0
    defensive_scale: float = 1.0
    defensive_conviction_k: float = 0.0
    aggressive_scale: float = 1.0
    breadth_high_risk: int = 0
    pwin_high_risk: float = 0.0
    high_risk_scale: float = 1.0
    strategy_side_post_ml_weight_rules: str = ""
    max_step_per_bar: float = 1.0
    legacy_competition_mode: str = "off"

def legacy_allocation_config_to_dict(config: LegacyAllocationConfig) -> dict:
    return {
        "score_power": float(config.score_power),
        "min_score": float(config.min_score),
        "symbol_score_agg": str(config.symbol_score_agg),
        "normalize_total": bool(config.normalize_total),
        "switch_hysteresis": float(config.switch_hysteresis),
        "min_switch_bars": int(config.min_switch_bars),
        "rebalance_deadband": float(config.rebalance_deadband),
        "weight_blend_alpha": float(config.weight_blend_alpha),
        "symbol_cap": float(config.symbol_cap),
        "target_exposure": float(config.target_exposure),
        "base_weight_projection": str(config.base_weight_projection),
        "smoothing_alpha": float(config.smoothing_alpha),
        "smoothing_snap_eps": float(config.smoothing_snap_eps),
        "defensive_scale": float(config.defensive_scale),
        "defensive_conviction_k": float(config.defensive_conviction_k),
        "aggressive_scale": float(config.aggressive_scale),
        "breadth_high_risk": int(config.breadth_high_risk),
        "pwin_high_risk": float(config.pwin_high_risk),
        "high_risk_scale": float(config.high_risk_scale),
        "strategy_side_post_ml_weight_rules": str(config.strategy_side_post_ml_weight_rules),
        "max_step_per_bar": float(config.max_step_per_bar),
        "legacy_competition_mode": str(config.legacy_competition_mode),
    }

