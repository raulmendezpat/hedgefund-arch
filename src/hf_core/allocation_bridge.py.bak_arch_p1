from __future__ import annotations

from hf_core.contracts import OpportunityCandidate
from hf_core.policy import PolicyDecision


class AllocationBridge:
    def apply(
        self,
        *,
        candidates: list[OpportunityCandidate],
        decisions: list[PolicyDecision],
    ) -> list[OpportunityCandidate]:
        if len(candidates) != len(decisions):
            raise ValueError(f"Length mismatch: candidates={len(candidates)} decisions={len(decisions)}")

        out: list[OpportunityCandidate] = []

        for c, d in zip(candidates, decisions):
            if not bool(d.accept):
                continue

            meta = dict(c.signal_meta or {})
            meta["policy_score"] = float(d.policy_score)
            meta["policy_band"] = str(d.band)
            meta["policy_reason"] = str(d.reason)
            meta["policy_size_mult"] = float(d.size_mult)
            meta["policy_accept"] = bool(d.accept)

            out.append(
                OpportunityCandidate(
                    ts=int(c.ts),
                    symbol=str(c.symbol),
                    strategy_id=str(c.strategy_id),
                    side=str(c.side),
                    signal_strength=float(c.signal_strength),
                    base_weight=float(c.base_weight) * float(d.size_mult),
                    signal_meta=meta,
                )
            )

        return out
