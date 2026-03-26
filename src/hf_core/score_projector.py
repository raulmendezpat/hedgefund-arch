from __future__ import annotations

from hf_core.contracts import OpportunityCandidate


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


class ScoreProjector:
    """
    Proyecta score-style metadata sobre OpportunityCandidate ya seleccionado,
    para que AllocationBridge y allocator downstream reciban metadata compatible
    con el flujo histórico de run_portfolio.
    """

    def __init__(
        self,
        *,
        use_policy_scaled_base_weight: bool = False,
        inject_post_ml_competitive_score: bool = True,
    ):
        self.use_policy_scaled_base_weight = bool(use_policy_scaled_base_weight)
        self.inject_post_ml_competitive_score = bool(inject_post_ml_competitive_score)

    def _resolve_base_weight(self, candidate: OpportunityCandidate, meta: dict) -> float:
        raw_base_weight = _safe_float(meta.get("base_weight", candidate.base_weight), candidate.base_weight)
        policy_size_mult = _safe_float(meta.get("policy_size_mult", 1.0), 1.0)

        if self.use_policy_scaled_base_weight:
            return float(raw_base_weight * policy_size_mult)
        return float(raw_base_weight)

    def _is_active(self, candidate: OpportunityCandidate) -> bool:
        side = str(getattr(candidate, "side", "flat") or "flat").lower()
        strength = abs(_safe_float(getattr(candidate, "signal_strength", 0.0), 0.0))
        return side != "flat" and strength > 0.0

    def enrich_candidate(self, candidate: OpportunityCandidate) -> OpportunityCandidate:
        meta = dict(getattr(candidate, "signal_meta", {}) or {})

        strength = abs(_safe_float(getattr(candidate, "signal_strength", 0.0), 0.0))
        active_flag = 1.0 if self._is_active(candidate) else 0.0
        base_weight_for_score = self._resolve_base_weight(candidate, meta)

        competitive_score = float(active_flag * strength * base_weight_for_score)

        p_win = _safe_float(meta.get("p_win", meta.get("ml_p_win", 1.0)), 1.0)
        if p_win <= 0.0:
            p_win = 1.0

        ml_position_size_mult = _safe_float(meta.get("ml_position_size_mult", 1.0), 1.0)
        if ml_position_size_mult <= 0.0:
            ml_position_size_mult = 1.0

        size_factor_clamped = max(0.50, min(1.50, float(ml_position_size_mult)))
        post_ml_score = float(competitive_score * p_win * size_factor_clamped)
        post_ml_competitive_score = float(post_ml_score)

        meta["competitive_score"] = float(competitive_score)
        meta["post_ml_score"] = float(post_ml_score)
        if self.inject_post_ml_competitive_score:
            meta["post_ml_competitive_score"] = float(post_ml_competitive_score)

        meta["legacy_score_projected"] = True
        meta["legacy_score_base_weight"] = float(base_weight_for_score)
        meta["legacy_score_strength"] = float(strength)
        meta["legacy_score_active_flag"] = float(active_flag)

        candidate.signal_meta = meta
        return candidate

    def enrich_many(self, candidates: list[OpportunityCandidate]) -> list[OpportunityCandidate]:
        out = []
        for c in candidates or []:
            out.append(self.enrich_candidate(c))
        return out
