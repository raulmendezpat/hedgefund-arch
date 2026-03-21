from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .pipeline import PolicyPipeline
from .rules import (
    FlatRejectRule,
    VeryWeakContextRejectRule,
    RegimePenaltyTagRule,
    BandSizingRule,
    RegimeAwareSizePenaltyRule,
    MinSizeFloorRule,
)


@dataclass
class PolicyFactory:
    profile: str = "default"
    config: dict[str, Any] = field(default_factory=dict)

    def build(self) -> PolicyPipeline:
        cfg = dict(self.config or {})

        if self.profile == "open":
            return PolicyPipeline(
                rules=[
                    FlatRejectRule(),
                    RegimePenaltyTagRule(),
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
                    MinSizeFloorRule(min_size=float(cfg.get("min_size", 0.10))),
                ]
            )

        return PolicyPipeline(
            rules=[
                FlatRejectRule(),
                VeryWeakContextRejectRule(p_floor=float(cfg.get("p_floor", 0.45))),
                RegimePenaltyTagRule(),
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
                MinSizeFloorRule(min_size=float(cfg.get("min_size", 0.10))),
            ]
        )
