from __future__ import annotations

from hf_core.contracts import OpportunityCandidate
from hf.engines.opportunity_book import compute_competitive_score, compute_post_ml_competitive_score


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

    def enrich_candidate_for_selection(
        self,
        *,
        candidate: OpportunityCandidate,
        score,
        decision,
        pwin_mode: str,
        p_win: float,
        p_win_ml: float,
        p_win_math_v1: float,
        p_win_math_v2: float,
        p_win_math_v3: float,
        p_win_hybrid_v1: float,
        expected_return: float,
        portfolio_context: dict | None = None,
        opportunity=None,
    ) -> OpportunityCandidate:
        meta = dict(getattr(candidate, "signal_meta", {}) or {})

        meta["strategy_id"] = str(getattr(candidate, "strategy_id", "") or "")
        meta["side"] = str(getattr(candidate, "side", "flat") or "flat")
        meta["p_win"] = float(p_win)
        meta["p_win_ml"] = float(p_win_ml)
        meta["p_win_math_v1"] = float(p_win_math_v1)
        meta["p_win_math_v2"] = float(p_win_math_v2)
        meta["p_win_math_v3"] = float(p_win_math_v3)
        meta["p_win_hybrid_v1"] = float(p_win_hybrid_v1)
        meta["p_win_mode"] = str(pwin_mode)
        meta["expected_return"] = float(expected_return)
        meta["score"] = float(getattr(score, "score", 0.0) or 0.0)

        accept_flag = bool(getattr(decision, "accept", False))
        meta["policy_score"] = float(getattr(decision, "policy_score", getattr(score, "score", 0.0)) or 0.0)
        meta["policy_band"] = str(getattr(decision, "band", "") or "")
        meta["policy_reason"] = str(getattr(decision, "reason", "") or "")
        meta["policy_size_mult"] = float(getattr(decision, "size_mult", 0.0) or 0.0)
        meta["accept"] = bool(accept_flag)

        competitive_score = float(meta.get("meta_competitive_score", meta.get("competitive_score", 0.0)) or 0.0)
        post_ml_score = float(meta.get("meta_post_ml_score", meta.get("post_ml_score", 0.0)) or 0.0)

        if opportunity is not None:
            try:
                opportunity.meta = dict(getattr(opportunity, "meta", {}) or {})
                opportunity.meta.update(meta)
                competitive_score = float(compute_competitive_score(opportunity))
            except Exception:
                pass

            try:
                opportunity.meta = dict(getattr(opportunity, "meta", {}) or {})
                opportunity.meta.update(meta)
                opportunity.meta["competitive_score"] = float(competitive_score)
                opportunity.meta["post_ml_score"] = float(competitive_score) * float(meta.get("p_win", 0.0) or 0.0)
                post_ml_score = float(compute_post_ml_competitive_score(opportunity))
            except Exception:
                pass

        if accept_flag and competitive_score <= 0.0:
            fallback_strength = abs(float(getattr(candidate, "signal_strength", 0.0) or 0.0))
            fallback_base_weight = float(
                getattr(candidate, "base_weight", meta.get("base_weight", 1.0)) or meta.get("base_weight", 1.0) or 1.0
            )
            competitive_score = float(fallback_strength * fallback_base_weight)

        if accept_flag and post_ml_score <= 0.0 and competitive_score > 0.0:
            fallback_pwin = max(0.0, float(meta.get("p_win", 0.0) or 0.0))
            fallback_size_mult = float(getattr(decision, "size_mult", meta.get("policy_size_mult", 1.0)) or 1.0)
            fallback_size_mult = max(0.50, min(1.50, fallback_size_mult))
            post_ml_score = float(competitive_score * fallback_pwin * fallback_size_mult)

        meta["competitive_score"] = float(competitive_score)
        meta["post_ml_score"] = float(post_ml_score)
        meta["post_ml_competitive_score"] = float(post_ml_score)
        meta["meta_competitive_score"] = float(competitive_score)
        meta["meta_post_ml_score"] = float(post_ml_score)
        meta["meta_post_ml_competitive_score"] = float(post_ml_score)

        if bool(meta.get("accept", False)):
            score_floor = 1.0e-4
            policy_floor = 1.0e-4
            if float(meta.get("score", 0.0) or 0.0) < score_floor:
                meta["score"] = float(score_floor)
            if float(meta.get("policy_score", 0.0) or 0.0) < policy_floor:
                meta["policy_score"] = float(policy_floor)

        portfolio_context = dict(portfolio_context or {})
        meta["portfolio_regime"] = portfolio_context.get("portfolio_regime")
        meta["portfolio_breadth"] = portfolio_context.get("portfolio_breadth")
        meta["portfolio_avg_pwin"] = portfolio_context.get("portfolio_avg_pwin")
        meta["portfolio_avg_atrp"] = portfolio_context.get("portfolio_avg_atrp")
        meta["portfolio_avg_strength"] = portfolio_context.get("portfolio_avg_strength")
        meta["portfolio_conviction"] = portfolio_context.get("portfolio_conviction")

        candidate.signal_meta = meta
        return candidate


    def enrich_many(self, candidates: list[OpportunityCandidate]) -> list[OpportunityCandidate]:
        out = []
        for c in candidates or []:
            out.append(self.enrich_candidate(c))
        return out
