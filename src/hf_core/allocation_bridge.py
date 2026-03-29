from __future__ import annotations

from hf_core.contracts import OpportunityCandidate
from hf_core.policy import PolicyDecision


class AllocationBridge:
    def __init__(
        self,
        *,
        score_projection: str = "legacy_post_ml_first",
        base_weight_projection: str = "raw",
    ):
        self.score_projection = str(score_projection or "legacy_post_ml_first")
        self.base_weight_projection = str(base_weight_projection or "raw")

    def _project_score(self, meta: dict) -> float:
        policy_score = float(meta.get("policy_score", meta.get("score", 0.0)) or 0.0)
        competitive_score = float(meta.get("competitive_score", meta.get("meta_competitive_score", 0.0)) or 0.0)
        post_ml_score = float(meta.get("post_ml_score", meta.get("meta_post_ml_score", 0.0)) or 0.0)
        post_ml_competitive_score = float(
            meta.get("post_ml_competitive_score", meta.get("meta_post_ml_competitive_score", 0.0)) or 0.0
        )
        p_win = float(meta.get("p_win", meta.get("ml_p_win", 0.5)) or 0.5)
        expected_return = float(meta.get("expected_return", 0.0) or 0.0)

        if self.score_projection == "policy_only":
            return float(policy_score)
        if self.score_projection == "competitive_only":
            return float(competitive_score)
        if self.score_projection == "post_ml_only":
            return float(post_ml_score)
        if self.score_projection == "pwin_expected_return":
            return float(max(0.0, p_win - 0.5) * max(0.0, expected_return))
        if self.score_projection == "frozen_allocation_score":
            return float(meta.get("allocation_score", meta.get("bridge_projected_score", 0.0)) or 0.0)

        if post_ml_competitive_score > 0.0:
            return float(post_ml_competitive_score)
        if post_ml_score > 0.0:
            return float(post_ml_score)
        if competitive_score > 0.0:
            return float(competitive_score)
        return float(policy_score)

    def _project_base_weight(self, candidate_base_weight: float, size_mult: float) -> float:
        base = float(candidate_base_weight or 0.0)
        mult = float(size_mult or 0.0)

        if self.base_weight_projection == "raw":
            return float(base)

        return float(base * mult)

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
            meta["competitive_score"] = float(meta.get("competitive_score", meta.get("meta_competitive_score", 0.0)) or 0.0)
            meta["post_ml_score"] = float(meta.get("post_ml_score", meta.get("meta_post_ml_score", 0.0)) or 0.0)
            meta["bridge_projected_score"] = float(self._project_score(meta))
            meta["bridge_score_projection"] = str(self.score_projection)
            meta["bridge_base_weight_projection"] = str(self.base_weight_projection)

            out.append(
                OpportunityCandidate(
                    ts=int(c.ts),
                    symbol=str(c.symbol),
                    strategy_id=str(c.strategy_id),
                    side=str(c.side),
                    signal_strength=float(c.signal_strength),
                    base_weight=self._project_base_weight(float(c.base_weight), float(d.size_mult)),
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
            competitive_score = float(meta.get("competitive_score", meta.get("meta_competitive_score", 0.0)) or 0.0)
            post_ml_score = float(meta.get("post_ml_score", meta.get("meta_post_ml_score", 0.0)) or 0.0)
            post_ml_competitive_score = float(
                meta.get("post_ml_competitive_score", meta.get("meta_post_ml_competitive_score", 0.0)) or 0.0
            )
            p_win = float(meta.get("p_win", meta.get("ml_p_win", 0.5)) or 0.5)
            expected_return = float(meta.get("expected_return", 0.0) or 0.0)
            policy_size_mult = float(meta.get("policy_size_mult", 0.0) or 0.0)
            ml_position_size_mult = float(meta.get("ml_position_size_mult", 0.0) or 0.0)
            signal_strength = float(getattr(c, "signal_strength", 0.0) or 0.0)
            projected_score = float(self._project_score(meta))

            trace_candidate_id = str(
                meta.get(
                    "trace_candidate_id",
                    f"{int(getattr(c, 'ts', 0) or 0)}|{str(c.symbol)}|{str(c.strategy_id)}|{str(c.side)}"
                )
            )
            alloc_input_id = f"{trace_candidate_id}|alloc"

            out.append(
                {
                    "symbol": str(c.symbol),
                    "strategy_id": str(c.strategy_id),
                    "side": str(c.side),
                    "score": float(projected_score),
                    "p_win": float(p_win),
                    "expected_return": float(expected_return),
                    "base_weight": float(c.base_weight),
                    "signal_strength": float(signal_strength),
                    "trace_candidate_id": trace_candidate_id,
                    "alloc_input_id": alloc_input_id,
                    "meta": {
                        **meta,
                        "score": float(projected_score),
                        "policy_score": float(policy_score),
                        "competitive_score": float(competitive_score),
                        "post_ml_score": float(post_ml_score),
                        "post_ml_competitive_score": float(post_ml_competitive_score),
                        "p_win": float(p_win),
                        "expected_return": float(expected_return),
                        "policy_size_mult": float(policy_size_mult),
                        "ml_position_size_mult": float(ml_position_size_mult),
                        "signal_strength": float(signal_strength),
                        "bridge_projected_score": float(projected_score),
                        "bridge_score_projection": str(self.score_projection),
                        "bridge_base_weight_projection": str(self.base_weight_projection),
                        "trace_candidate_id": str(trace_candidate_id),
                        "alloc_input_id": str(alloc_input_id),
                    },
                }
            )

        return out
