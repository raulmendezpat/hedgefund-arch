from __future__ import annotations

from dataclasses import dataclass, field

from hf.core.opportunity import Opportunity

from hf_core.contracts import OpportunityCandidate
from hf_core.opportunity_score_enricher import OpportunityScoreEnricher


@dataclass
class OpportunityAdapter:
    base_weight_projection: str = "policy_scaled"
    score_enricher: OpportunityScoreEnricher = field(default_factory=OpportunityScoreEnricher)

    def _effective_base_weight(self, candidate: OpportunityCandidate, meta: dict) -> float:
        candidate_base_weight = float(getattr(candidate, "base_weight", 0.0) or 0.0)
        policy_size_mult = float(meta.get("policy_size_mult", 1.0) or 1.0)

        if self.base_weight_projection == "raw":
            return float(candidate_base_weight)

        return float(candidate_base_weight * policy_size_mult)

    def to_opportunity(self, candidate: OpportunityCandidate) -> Opportunity:
        meta = dict(getattr(candidate, "signal_meta", {}) or {})

        symbol = str(getattr(candidate, "symbol", "") or "")
        side = str(getattr(candidate, "side", "flat") or "flat")
        strategy_id = str(getattr(candidate, "strategy_id", "") or "")
        timestamp = int(getattr(candidate, "ts", 0) or 0)
        signal_strength = float(getattr(candidate, "signal_strength", 0.0) or 0.0)
        candidate_base_weight = float(getattr(candidate, "base_weight", 0.0) or 0.0)
        policy_size_mult = float(meta.get("policy_size_mult", 1.0) or 1.0)
        ml_position_size_mult = float(meta.get("ml_position_size_mult", 1.0) or 1.0)
        p_win = float(meta.get("p_win", meta.get("ml_p_win", 0.0)) or 0.0)
        expected_return = float(meta.get("expected_return", 0.0) or 0.0)

        effective_base_weight = self._effective_base_weight(candidate, meta)

        out_meta = dict(meta)
        out_meta["base_weight"] = float(effective_base_weight)
        out_meta["signal_strength"] = float(signal_strength)
        out_meta["policy_size_mult"] = float(policy_size_mult)
        out_meta["ml_position_size_mult"] = float(ml_position_size_mult)
        out_meta["p_win"] = float(p_win)
        out_meta["expected_return"] = float(expected_return)
        out_meta["adapter_base_weight_projection"] = str(self.base_weight_projection)
        out_meta["adapter_candidate_base_weight"] = float(candidate_base_weight)
        out_meta["adapter_effective_base_weight"] = float(effective_base_weight)
        out_meta["legacy_adapter_base_weight_projection"] = str(self.base_weight_projection)
        out_meta["legacy_adapter_candidate_base_weight"] = float(candidate_base_weight)
        out_meta["legacy_adapter_effective_base_weight"] = float(effective_base_weight)

        opp = Opportunity(
            symbol=symbol,
            side=side,
            strength=float(signal_strength),
            strategy_id=strategy_id,
            timestamp=timestamp,
            meta=out_meta,
        )

        return self.score_enricher.enrich(opp)

    def to_opportunities(self, candidates: list[OpportunityCandidate]) -> list[Opportunity]:
        return [self.to_opportunity(c) for c in list(candidates or [])]
