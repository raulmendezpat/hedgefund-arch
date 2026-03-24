from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .pipeline import PolicyPipeline
from .rules import (
    FlatRejectRule,
    StrategySideDisableRule,
    VeryWeakContextRejectRule,
    ExtremeConfidenceRejectRule,
    RegimePenaltyTagRule,
    AlphaConfidenceTagRule,
    AlphaEdgeTagRule,
    AlphaProbabilityAdjustTagRule,
    AlphaRescoreFromTagsRule,
    AlphaKillRule,
    BandSizingRule,
    RegimeAwareSizePenaltyRule,
    SideBiasSizeRule,
    AlphaSizeBlendRule,
    MinSizeFloorRule,
    MaxSizeClampRule,
)


@dataclass
class PolicyFactory:
    profile: str = "default"
    config: dict[str, Any] = field(default_factory=dict)

    def build(self) -> PolicyPipeline:
        cfg = dict(self.config or {})
        disabled_pairs = set(str(x) for x in list(cfg.get("disabled_strategy_sides", []) or []))

        common_tail = [
            RegimePenaltyTagRule(),
            AlphaConfidenceTagRule(
                conf_overconfidence_w=float(cfg.get("alpha_conf_overconfidence_w", 1.75)),
                conf_shrink_w=float(cfg.get("alpha_conf_shrink_w", 1.25)),
                conf_min=float(cfg.get("alpha_conf_min", 0.05)),
                conf_max=float(cfg.get("alpha_conf_max", 1.0)),
            ),
            AlphaEdgeTagRule(
                win_proxy=float(cfg.get("alpha_win_proxy", 0.015)),
                loss_proxy=float(cfg.get("alpha_loss_proxy", 0.010)),
            ),
            AlphaProbabilityAdjustTagRule(
                p0=float(cfg.get("alpha_p0", 0.52)),
                kp=float(cfg.get("alpha_kp", 10.0)),
            ),
            AlphaRescoreFromTagsRule(
                raw_score_weight=float(cfg.get("alpha_raw_score_weight", 0.35)),
                alpha_score_weight=float(cfg.get("alpha_alpha_score_weight", 0.65)),
                score_ref=float(cfg.get("alpha_score_ref", 0.00008)),
            ),
            AlphaKillRule(
                kill_pwin_floor=float(cfg.get("alpha_kill_pwin_floor", 0.47)),
                kill_conf_floor=float(cfg.get("alpha_kill_conf_floor", 0.32)),
                require_positive_expected_return=bool(cfg.get("alpha_require_positive_expected_return", True)),
                apply_only_in_defensive_longs=bool(cfg.get("alpha_apply_only_in_defensive_longs", True)),
            ),
            BandSizingRule(
                small_size=float(cfg.get("small_size", 0.35)),
                normal_size=float(cfg.get("normal_size", 0.60)),
                high_conviction_size=float(cfg.get("high_conviction_size", 1.00)),
            ),
            RegimeAwareSizePenaltyRule(
                penalty_adx=float(cfg.get("penalty_adx", 0.85)),
                penalty_ema_gap=float(cfg.get("penalty_ema_gap", 0.90)),
                penalty_atrp=float(cfg.get("penalty_atrp", 0.90)),
                penalty_range_expansion=float(cfg.get("penalty_range_expansion", 0.92)),
            ),
            SideBiasSizeRule(
                long_size_bias=float(cfg.get("long_size_bias", 1.0)),
                short_size_bias=float(cfg.get("short_size_bias", 1.0)),
            ),
            AlphaSizeBlendRule(
                score_ratio_floor=float(cfg.get("alpha_size_score_ratio_floor", 0.0)),
                score_ratio_cap=float(cfg.get("alpha_size_score_ratio_cap", 4.0)),
                size_min=float(cfg.get("alpha_size_min", 0.25)),
                size_max=float(cfg.get("alpha_size_max", 1.35)),
                base_add=float(cfg.get("alpha_size_base_add", 0.35)),
                linear_w=float(cfg.get("alpha_size_linear_w", 0.45)),
                sqrt_w=float(cfg.get("alpha_size_sqrt_w", 0.20)),
            ),
            MinSizeFloorRule(min_size=float(cfg.get("min_size", 0.10))),
            MaxSizeClampRule(max_size=float(cfg.get("max_size_mult", 1.0))),
        ]

        if self.profile == "open":
            return PolicyPipeline(
                rules=[
                    FlatRejectRule(),
                    StrategySideDisableRule(disabled_pairs=disabled_pairs),
                    ExtremeConfidenceRejectRule(
                        extreme_pwin_reject_threshold=float(cfg.get("extreme_pwin_reject_threshold", 9.9)),
                        extreme_score_reject_threshold=float(cfg.get("extreme_score_reject_threshold", 9.9)),
                    ),
                    *common_tail,
                ]
            )

        return PolicyPipeline(
            rules=[
                FlatRejectRule(),
                StrategySideDisableRule(disabled_pairs=disabled_pairs),
                VeryWeakContextRejectRule(p_floor=float(cfg.get("p_floor", 0.45))),
                ExtremeConfidenceRejectRule(
                    extreme_pwin_reject_threshold=float(cfg.get("extreme_pwin_reject_threshold", 9.9)),
                    extreme_score_reject_threshold=float(cfg.get("extreme_score_reject_threshold", 9.9)),
                ),
                *common_tail,
            ]
        )
