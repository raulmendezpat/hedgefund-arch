from __future__ import annotations

from dataclasses import dataclass

from hf_core.legacy_allocation_config import legacy_allocation_config_to_dict


@dataclass(frozen=True)
class AllocationConfig:
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


def allocation_config_to_dict(config) -> dict:
    return legacy_allocation_config_to_dict(config)
