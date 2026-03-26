from __future__ import annotations

from hf_core.opportunity_adapter import OpportunityAdapter


class LegacyOpportunityAdapter(OpportunityAdapter):
    """Backward-compatible alias during migration."""

    def to_legacy_opportunity(self, candidate):
        return self.to_opportunity(candidate)

    def to_legacy_opportunities(self, candidates):
        return self.to_opportunities(candidates)
