from __future__ import annotations

from dataclasses import dataclass

from hf.core.opportunity import Opportunity
from hf.engines.opportunity_book import compute_competitive_score, compute_post_ml_competitive_score


@dataclass
class OpportunityScoreEnricher:
    def enrich(self, opportunity: Opportunity) -> Opportunity:
        meta = dict(getattr(opportunity, "meta", {}) or {})
        p_win = float(meta.get("p_win", meta.get("ml_p_win", 0.0)) or 0.0)

        competitive_score = float(
            meta.get("competitive_score", meta.get("meta_competitive_score", 0.0)) or 0.0
        )
        if competitive_score <= 0.0:
            competitive_score = float(compute_competitive_score(opportunity))

        post_ml_competitive_score = float(
            meta.get(
                "post_ml_competitive_score",
                meta.get("meta_post_ml_competitive_score", 0.0),
            ) or 0.0
        )
        if post_ml_competitive_score <= 0.0:
            post_ml_competitive_score = float(compute_post_ml_competitive_score(opportunity))

        post_ml_score = float(
            meta.get("post_ml_score", meta.get("meta_post_ml_score", 0.0)) or 0.0
        )
        if post_ml_score <= 0.0:
            post_ml_score = float(post_ml_competitive_score or (competitive_score * max(float(p_win), 0.0)))

        meta["competitive_score"] = float(competitive_score)
        meta["post_ml_score"] = float(post_ml_score)
        meta["post_ml_competitive_score"] = float(post_ml_competitive_score)
        meta["meta_competitive_score"] = float(competitive_score)
        meta["meta_post_ml_score"] = float(post_ml_score)
        meta["meta_post_ml_competitive_score"] = float(post_ml_competitive_score)

        opportunity.meta = meta
        return opportunity

    def enrich_many(self, opportunities: list[Opportunity]) -> list[Opportunity]:
        return [self.enrich(o) for o in list(opportunities or [])]
