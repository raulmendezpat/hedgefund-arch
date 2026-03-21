from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import PolicyRule, PolicyState


@dataclass
class FlatRejectRule(PolicyRule):
    def apply(self, state: PolicyState) -> PolicyState:
        if str(state.side).lower() == "flat":
            state.accept = False
            state.size_mult = 0.0
            state.band = "reject"
            state.reason = "flat_signal"
        return state


@dataclass
class StrategySideDisableRule(PolicyRule):
    disabled_pairs: set[str] = field(default_factory=set)

    def apply(self, state: PolicyState) -> PolicyState:
        if not state.accept:
            return state

        key = f"{str(state.strategy_id)}|{str(state.side).lower()}"
        if key in set(self.disabled_pairs or set()):
            state.accept = False
            state.size_mult = 0.0
            state.band = "reject"
            state.reason = "disabled_strategy_side"
        return state


@dataclass
class VeryWeakContextRejectRule(PolicyRule):
    p_floor: float = 0.45

    def apply(self, state: PolicyState) -> PolicyState:
        if not state.accept:
            return state
        if state.p_win < float(self.p_floor) and state.expected_return < 0.0 and state.score <= 0.0:
            state.accept = False
            state.size_mult = 0.0
            state.band = "reject"
            state.reason = "very_weak_context"
        return state


@dataclass
class RegimePenaltyTagRule(PolicyRule):
    def apply(self, state: PolicyState) -> PolicyState:
        mm = dict(state.model_meta or {})
        state.tags["adx_below_min"] = bool(mm.get("adx_below_min", False))
        state.tags["ema_gap_below_min"] = bool(mm.get("ema_gap_below_min", False))
        state.tags["atrp_low"] = bool(mm.get("atrp_low", False))
        state.tags["adx_low"] = bool(mm.get("adx_low", False))
        state.tags["range_expansion_low"] = bool(mm.get("range_expansion_low", False))
        return state


@dataclass
class BandSizingRule(PolicyRule):
    small_size: float = 0.35
    normal_size: float = 0.60
    high_conviction_size: float = 1.00

    def apply(self, state: PolicyState) -> PolicyState:
        if not state.accept:
            return state

        band = "small"
        size = float(self.small_size)

        if state.p_win >= 0.55 or state.expected_return >= 0.001:
            band = "normal"
            size = float(self.normal_size)

        # High conviction intentionally disabled for now until ranking is recalibrated.
        # Strong-looking candidates were empirically underperforming.
        if state.p_win >= 0.60 or state.expected_return >= 0.003 or state.score >= 0.0003:
            band = "normal"
            size = float(self.normal_size)

        state.band = band
        state.size_mult = size
        if state.reason == "init":
            state.reason = "accepted"
        return state


@dataclass
class RegimeAwareSizePenaltyRule(PolicyRule):
    penalty_adx: float = 0.85
    penalty_ema_gap: float = 0.90
    penalty_atrp: float = 0.90
    penalty_range_expansion: float = 0.92

    def apply(self, state: PolicyState) -> PolicyState:
        if not state.accept:
            return state

        if bool(state.tags.get("adx_below_min", False)) or bool(state.tags.get("adx_low", False)):
            state.size_mult *= float(self.penalty_adx)

        if bool(state.tags.get("ema_gap_below_min", False)):
            state.size_mult *= float(self.penalty_ema_gap)

        if bool(state.tags.get("atrp_low", False)):
            state.size_mult *= float(self.penalty_atrp)

        if bool(state.tags.get("range_expansion_low", False)):
            state.size_mult *= float(self.penalty_range_expansion)

        # Temporary directional penalty until long-side alpha is recalibrated.
        if str(state.side).lower() == "long":
            state.size_mult *= 0.50

        return state


@dataclass
class SideBiasSizeRule(PolicyRule):
    long_size_bias: float = 1.0
    short_size_bias: float = 1.0

    def apply(self, state: PolicyState) -> PolicyState:
        if not state.accept:
            return state

        side = str(state.side).lower()
        if side == "long":
            state.size_mult *= float(self.long_size_bias)
        elif side == "short":
            state.size_mult *= float(self.short_size_bias)

        return state


@dataclass
class MinSizeFloorRule(PolicyRule):
    min_size: float = 0.10

    def apply(self, state: PolicyState) -> PolicyState:
        if not state.accept:
            return state
        state.size_mult = max(float(self.min_size), float(state.size_mult))
        return state


@dataclass
class MaxSizeClampRule(PolicyRule):
    max_size: float = 1.0

    def apply(self, state: PolicyState) -> PolicyState:
        if not state.accept:
            return state
        state.size_mult = min(float(self.max_size), float(state.size_mult))
        return state
