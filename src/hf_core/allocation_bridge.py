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

    def to_allocator_inputs(
        self,
        *,
        candidates: list[OpportunityCandidate],
        decisions: list[PolicyDecision],
    ) -> list[dict]:
        bridged = self.apply(candidates=candidates, decisions=decisions)

        out: list[dict] = []
        for c in bridged:
            meta = dict(c.signal_meta or {})

            policy_score = float(meta.get("policy_score", meta.get("score", 0.0)) or 0.0)
            p_win = float(meta.get("p_win", meta.get("ml_p_win", 0.5)) or 0.5)
            expected_return = float(meta.get("expected_return", 0.0) or 0.0)

            out.append(
                {
                    "symbol": str(c.symbol),
                    "side": str(c.side),
                    "score": float(policy_score),
                    "p_win": float(p_win),
                    "expected_return": float(expected_return),
                    "base_weight": float(c.base_weight),
                    "meta": {
                        **meta,
                        "score": float(policy_score),
                        "policy_score": float(policy_score),
                        "p_win": float(p_win),
                        "expected_return": float(expected_return),
                    },
                }
            )

        return out
