from __future__ import annotations

from dataclasses import dataclass, field
from math import exp

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
class ExtremeConfidenceRejectRule(PolicyRule):
    extreme_pwin_reject_threshold: float = 9.9
    extreme_score_reject_threshold: float = 9.9

    def apply(self, state: PolicyState) -> PolicyState:
        if not state.accept:
            return state

        if (
            state.p_win >= float(self.extreme_pwin_reject_threshold)
            or state.score >= float(self.extreme_score_reject_threshold)
        ):
            state.accept = False
            state.size_mult = 0.0
            state.band = "reject"
            state.reason = "extreme_confidence_reject"
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


def _clip(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = exp(-float(x))
        return 1.0 / (1.0 + z)
    z = exp(float(x))
    return z / (1.0 + z)


@dataclass
class AlphaConfidenceTagRule(PolicyRule):
    conf_overconfidence_w: float = 1.75
    conf_shrink_w: float = 1.25
    conf_min: float = 0.05
    conf_max: float = 1.0

    def apply(self, state: PolicyState) -> PolicyState:
        mm = dict(state.model_meta or {})
        overconf = float(mm.get("overconfidence_penalty", 0.0) or 0.0)
        shrink_weight = float(mm.get("shrink_weight", 0.0) or 0.0)

        conf = 1.0 / (
            1.0
            + float(self.conf_overconfidence_w) * max(0.0, overconf)
            + float(self.conf_shrink_w) * max(0.0, shrink_weight)
        )
        conf = _clip(conf, float(self.conf_min), float(self.conf_max))
        state.tags["alpha_conf"] = float(conf)
        return state


@dataclass
class AlphaEdgeTagRule(PolicyRule):
    win_proxy: float = 0.015
    loss_proxy: float = 0.010

    def apply(self, state: PolicyState) -> PolicyState:
        p_final = float(state.p_win)
        win_proxy = max(1e-9, float(self.win_proxy))
        loss_proxy = max(1e-9, float(self.loss_proxy))

        mu = float(p_final * win_proxy - (1.0 - p_final) * loss_proxy)
        b = float(win_proxy / loss_proxy)
        kelly = float(p_final - (1.0 - p_final) / max(b, 1e-9))

        state.tags["alpha_mu"] = float(mu)
        state.tags["alpha_b"] = float(b)
        state.tags["alpha_kelly"] = float(kelly)
        return state


@dataclass
class AlphaProbabilityAdjustTagRule(PolicyRule):
    p0: float = 0.52
    kp: float = 10.0

    def apply(self, state: PolicyState) -> PolicyState:
        p_adj = float(_sigmoid(float(self.kp) * (float(state.p_win) - float(self.p0))))
        state.tags["alpha_p_adj"] = float(p_adj)
        return state


@dataclass
class AlphaRescoreFromTagsRule(PolicyRule):
    raw_score_weight: float = 0.35
    alpha_score_weight: float = 0.65
    score_ref: float = 0.00008

    def apply(self, state: PolicyState) -> PolicyState:
        if not state.accept:
            return state

        raw_score = max(0.0, float(state.score))
        conf = float(state.tags.get("alpha_conf", 1.0) or 1.0)
        mu = float(state.tags.get("alpha_mu", 0.0) or 0.0)
        kelly = float(state.tags.get("alpha_kelly", 0.0) or 0.0)
        p_adj = float(state.tags.get("alpha_p_adj", 0.0) or 0.0)

        alpha_core = float(max(0.0, mu) * max(0.0, kelly) * conf * p_adj)
        rescored = float(
            float(self.raw_score_weight) * raw_score
            + float(self.alpha_score_weight) * alpha_core
        )

        score_ref = max(1e-9, float(self.score_ref))
        score_ratio = _clip(rescored / score_ref, 0.0, 4.0)

        state.tags["raw_score_pre_alpha"] = float(raw_score)
        state.tags["alpha_core_score"] = float(alpha_core)
        state.tags["alpha_score_ratio"] = float(score_ratio)
        state.score = float(rescored)

        if state.reason == "init":
            state.reason = "alpha_rescored"

        return state


@dataclass
class AlphaKillRule(PolicyRule):
    kill_pwin_floor: float = 0.47
    kill_conf_floor: float = 0.32
    require_positive_expected_return: bool = True
    apply_only_in_defensive_longs: bool = True

    def apply(self, state: PolicyState) -> PolicyState:
        if not state.accept:
            return state

        mm = dict(state.model_meta or {})
        regime = str(mm.get("regime", "unknown") or "unknown").lower()
        side = str(state.side).lower()

        apply_gate = True
        if bool(self.apply_only_in_defensive_longs):
            apply_gate = (regime == "defensive" and side == "long")

        if not apply_gate:
            state.tags["alpha_gate_applied"] = False
            return state

        p_final = float(state.p_win)
        conf = float(state.tags.get("alpha_conf", 1.0) or 1.0)
        exp_ret = float(state.expected_return)

        kill = False
        if bool(self.require_positive_expected_return) and exp_ret <= 0.0:
            kill = True
        if p_final < float(self.kill_pwin_floor):
            kill = True
        if conf < float(self.kill_conf_floor):
            kill = True

        state.tags["alpha_gate_applied"] = True
        state.tags["alpha_gate_regime"] = str(regime)
        state.tags["alpha_gate_side"] = str(side)
        state.tags["alpha_kill_candidate"] = bool(kill)

        if kill:
            state.accept = False
            state.size_mult = 0.0
            state.band = "reject"
            state.reason = "alpha_kill"

        return state


@dataclass
class AlphaSizeBlendRule(PolicyRule):
    score_ratio_floor: float = 0.0
    score_ratio_cap: float = 4.0
    size_min: float = 0.25
    size_max: float = 1.35
    base_add: float = 0.35
    linear_w: float = 0.45
    sqrt_w: float = 0.20

    def apply(self, state: PolicyState) -> PolicyState:
        if not state.accept:
            return state

        score_ratio = float(state.tags.get("alpha_score_ratio", 0.0) or 0.0)
        score_ratio = _clip(
            score_ratio,
            float(self.score_ratio_floor),
            float(self.score_ratio_cap),
        )

        alpha_mult = float(
            _clip(
                float(self.base_add)
                + float(self.linear_w) * score_ratio
                + float(self.sqrt_w) * (score_ratio ** 0.5),
                float(self.size_min),
                float(self.size_max),
            )
        )

        state.tags["alpha_size_mult"] = float(alpha_mult)
        state.size_mult *= alpha_mult
        return state

