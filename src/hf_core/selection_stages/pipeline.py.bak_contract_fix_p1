from __future__ import annotations

from hf_core.contracts import OpportunityCandidate
from hf_core.policy import PolicyDecision

from .contracts import SelectionContext, SelectionRow
from .trace import SelectionTraceWriter


class SelectionPipeline:
    def run_candidates_only(
        self,
        *,
        candidates: list[OpportunityCandidate],
    ) -> tuple[list[OpportunityCandidate], dict]:
        rows = []
        for i, c in enumerate(candidates or []):
            meta_signal = dict(getattr(c, "signal_meta", {}) or {})
            meta_plain = dict(getattr(c, "meta", {}) or {})
            meta_opp = dict(getattr(c, "opportunity_meta", {}) or {})

            meta = {}
            meta.update(meta_signal)
            meta.update(meta_plain)
            meta.update(meta_opp)

            rows.append(
                SelectionRow(
                    idx=int(i),
                    ts=int(getattr(c, "ts", 0) or 0),
                    symbol=str(getattr(c, "symbol", "") or ""),
                    strategy_id=str(getattr(c, "strategy_id", "") or meta.get("strategy_id", "")),
                    side=str(getattr(c, "side", "flat") or meta.get("side", "flat") or "flat").lower(),
                    signal_strength=float(getattr(c, "signal_strength", meta.get("strength", 0.0)) or 0.0),
                    base_weight=float(getattr(c, "base_weight", meta.get("base_weight", 0.0)) or 0.0),
                    p_win=float(meta.get("p_win", meta.get("ml_p_win", 0.0)) or 0.0),
                    post_ml_score=float(meta.get("post_ml_score", 0.0) or 0.0),
                    competitive_score=float(meta.get("competitive_score", 0.0) or 0.0),
                    policy_score=float(meta.get("policy_score", meta.get("score", 0.0)) or 0.0),
                    policy_band=str(meta.get("policy_band", meta.get("band", "")) or ""),
                    policy_reason=str(meta.get("policy_reason", meta.get("reason", "")) or ""),
                    policy_size_mult=float(meta.get("policy_size_mult", meta.get("size_mult", 0.0)) or 0.0),
                    accept_in=bool(meta.get("accept", True)),
                    meta=meta,
                )
            )

        ctx = SelectionContext(rows=rows, selected_idx=[r.idx for r in rows])

        for stage in self.stages:
            ctx = stage.apply(ctx)

        if self.trace_path:
            SelectionTraceWriter(self.trace_path).append(ctx.trace_rows)

        keep = set(ctx.selected_idx)
        out_candidates = [c for i, c in enumerate(candidates or []) if i in keep]
        return out_candidates, dict(ctx.meta or {})

    def __init__(self, *, stages: list, trace_path: str | None = None):
        self.stages = list(stages or [])
        self.trace_path = trace_path

    def run(
        self,
        *,
        candidates: list[OpportunityCandidate],
        decisions: list[PolicyDecision],
    ) -> tuple[list[OpportunityCandidate], list[PolicyDecision], dict]:
        rows = []
        for i, (c, d) in enumerate(zip(candidates or [], decisions or [])):
            meta = dict(getattr(c, "signal_meta", {}) or {})
            rows.append(
                SelectionRow(
                    idx=int(i),
                    ts=int(getattr(c, "ts", 0) or 0),
                    symbol=str(getattr(c, "symbol", "") or ""),
                    strategy_id=str(getattr(c, "strategy_id", "") or ""),
                    side=str(getattr(c, "side", "flat") or "flat").lower(),
                    signal_strength=float(getattr(c, "signal_strength", 0.0) or 0.0),
                    base_weight=float(getattr(c, "base_weight", 0.0) or 0.0),
                    p_win=float(meta.get("p_win", 0.0) or 0.0),
                    post_ml_score=float(meta.get("post_ml_score", 0.0) or 0.0),
                    competitive_score=float(meta.get("competitive_score", 0.0) or 0.0),
                    policy_score=float(getattr(d, "policy_score", 0.0) or 0.0),
                    policy_band=str(getattr(d, "band", "") or ""),
                    policy_reason=str(getattr(d, "reason", "") or ""),
                    policy_size_mult=float(getattr(d, "size_mult", 0.0) or 0.0),
                    accept_in=bool(getattr(d, "accept", False)),
                    meta=meta,
                )
            )

        ctx = SelectionContext(rows=rows, selected_idx=[r.idx for r in rows])

        for stage in self.stages:
            ctx = stage.apply(ctx)

        if self.trace_path:
            SelectionTraceWriter(self.trace_path).append(ctx.trace_rows)

        keep = set(ctx.selected_idx)
        out_candidates = [c for i, c in enumerate(candidates or []) if i in keep]
        out_decisions = [d for i, d in enumerate(decisions or []) if i in keep]

        return out_candidates, out_decisions, dict(ctx.meta or {})
