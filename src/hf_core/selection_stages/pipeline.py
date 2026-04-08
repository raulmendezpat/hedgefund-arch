from __future__ import annotations

from hf_core.contracts import OpportunityCandidate
from hf_core.policy import PolicyDecision

from .contracts import SelectionContext, SelectionRow
from .trace import SelectionTraceWriter


def _build_merged_meta(candidate) -> tuple[dict, dict, dict, dict]:
    meta_signal = dict(getattr(candidate, "signal_meta", {}) or {})
    meta_plain = dict(getattr(candidate, "meta", {}) or {})
    meta_opp = dict(getattr(candidate, "opportunity_meta", {}) or {})

    merged = {}
    merged.update(meta_plain)
    merged.update(meta_opp)
    merged.update(meta_signal)

    return meta_signal, meta_plain, meta_opp, merged


class SelectionPipeline:
    def __init__(self, *, stages: list, trace_path: str | None = None):
        self.stages = list(stages or [])
        self.trace_path = trace_path

    def run_candidates_only(
        self,
        *,
        candidates: list[OpportunityCandidate],
    ) -> tuple[list[OpportunityCandidate], dict]:
        rows = []
        for i, c in enumerate(candidates or []):
            _meta_signal, _meta_plain, _meta_opp, meta = _build_merged_meta(c)

            rows.append(
                SelectionRow(
                    idx=int(i),
                    ts=int(getattr(c, "ts", 0) or 0),
                    symbol=str(getattr(c, "symbol", "") or ""),
                    strategy_id=str(getattr(c, "strategy_id", "") or meta.get("strategy_id", "")),
                    side=str(getattr(c, "side", "flat") or meta.get("side", "flat") or "flat").lower(),
                    signal_strength=float(getattr(c, "signal_strength", meta.get("strength", 0.0)) or 0.0),
                    base_weight=float(getattr(c, "base_weight", meta.get("base_weight", 0.0)) or 0.0),
                    p_win=float(meta.get("p_win", meta.get("ml_p_win", meta.get("meta_p_win", 0.0))) or 0.0),
                    expected_return=float(meta.get("expected_return", 0.0) or 0.0),
                    post_ml_score=float(
                        meta.get(
                            "post_ml_score",
                            meta.get(
                                "post_ml_competitive_score",
                                meta.get(
                                    "meta_post_ml_score",
                                    meta.get("meta_post_ml_competitive_score", 0.0),
                                ),
                            ),
                        ) or 0.0
                    ),
                    competitive_score=float(
                        meta.get("competitive_score", meta.get("meta_competitive_score", 0.0)) or 0.0
                    ),
                    policy_score=float(meta.get("policy_score", meta.get("score", 0.0)) or 0.0),
                    policy_band=str(meta.get("policy_band", meta.get("band", "")) or ""),
                    policy_reason=str(meta.get("policy_reason", meta.get("reason", "")) or ""),
                    policy_size_mult=float(meta.get("policy_size_mult", meta.get("size_mult", 0.0)) or 0.0),
                    accept_in=bool(meta.get("accept", True)),
                    selection_meta={},
                    meta=meta,
                )
            )

        ctx = SelectionContext(rows=rows, selected_idx=[r.idx for r in rows])

        for stage in self.stages:
            ctx = stage.apply(ctx)

        if self.trace_path:
            SelectionTraceWriter(self.trace_path).append(ctx.trace_rows)

        keep = set(ctx.selected_idx)
        row_map = {int(r.idx): r for r in ctx.rows}
        out_candidates = []
        for i, c in enumerate(candidates or []):
            if i not in keep:
                continue
            row = row_map.get(int(i))
            if row is not None:
                signal_meta = dict(getattr(c, "signal_meta", {}) or {})
                row_meta = dict(getattr(row, "meta", {}) or {})

                merged_signal_meta = {}
                merged_signal_meta.update(row_meta)
                merged_signal_meta.update(signal_meta)

                merged_signal_meta["selection_meta"] = dict(getattr(row, "selection_meta", {}) or {})
                merged_signal_meta["selection_passed_stages"] = sorted(
                    [
                        k
                        for k, v in dict(getattr(row, "selection_meta", {}) or {}).items()
                        if isinstance(v, dict) and bool(v.get("kept", v.get("pass", False)))
                    ]
                )
                c.signal_meta = merged_signal_meta
            out_candidates.append(c)
        return out_candidates, dict(ctx.meta or {})

    def run(
        self,
        *,
        candidates: list[OpportunityCandidate],
        decisions: list[PolicyDecision],
    ) -> tuple[list[OpportunityCandidate], list[PolicyDecision], dict]:
        rows = []
        for i, (c, d) in enumerate(zip(candidates or [], decisions or [])):
            _meta_signal, _meta_plain, _meta_opp, meta = _build_merged_meta(c)

            rows.append(
                SelectionRow(
                    idx=int(i),
                    ts=int(getattr(c, "ts", 0) or 0),
                    symbol=str(getattr(c, "symbol", "") or ""),
                    strategy_id=str(getattr(c, "strategy_id", "") or meta.get("strategy_id", "")),
                    side=str(getattr(c, "side", "flat") or meta.get("side", "flat") or "flat").lower(),
                    signal_strength=float(getattr(c, "signal_strength", meta.get("strength", 0.0)) or 0.0),
                    base_weight=float(getattr(c, "base_weight", meta.get("base_weight", 0.0)) or 0.0),
                    p_win=float(meta.get("p_win", meta.get("ml_p_win", meta.get("meta_p_win", 0.0))) or 0.0),
                    expected_return=float(meta.get("expected_return", 0.0) or 0.0),
                    post_ml_score=float(
                        meta.get(
                            "post_ml_score",
                            meta.get(
                                "post_ml_competitive_score",
                                meta.get(
                                    "meta_post_ml_score",
                                    meta.get("meta_post_ml_competitive_score", 0.0),
                                ),
                            ),
                        ) or 0.0
                    ),
                    competitive_score=float(
                        meta.get("competitive_score", meta.get("meta_competitive_score", 0.0)) or 0.0
                    ),
                    policy_score=float(
                        getattr(d, "policy_score", meta.get("policy_score", meta.get("score", 0.0))) or 0.0
                    ),
                    policy_band=str(getattr(d, "band", meta.get("policy_band", meta.get("band", ""))) or ""),
                    policy_reason=str(getattr(d, "reason", meta.get("policy_reason", meta.get("reason", ""))) or ""),
                    policy_size_mult=float(
                        getattr(d, "size_mult", meta.get("policy_size_mult", meta.get("size_mult", 0.0))) or 0.0
                    ),
                    accept_in=bool(getattr(d, "accept", meta.get("accept", False))),
                    selection_meta={},
                    meta=meta,
                )
            )

        ctx = SelectionContext(rows=rows, selected_idx=[r.idx for r in rows])

        for stage in self.stages:
            ctx = stage.apply(ctx)

        if self.trace_path:
            SelectionTraceWriter(self.trace_path).append(ctx.trace_rows)

        keep = set(ctx.selected_idx)
        row_map = {int(r.idx): r for r in ctx.rows}
        out_candidates = []
        out_decisions = []
        for i, (c, d) in enumerate(zip(candidates or [], decisions or [])):
            if i not in keep:
                continue
            row = row_map.get(int(i))
            if row is not None:
                signal_meta = dict(getattr(c, "signal_meta", {}) or {})
                row_meta = dict(getattr(row, "meta", {}) or {})

                merged_signal_meta = {}
                merged_signal_meta.update(row_meta)
                merged_signal_meta.update(signal_meta)

                merged_signal_meta["selection_meta"] = dict(getattr(row, "selection_meta", {}) or {})
                merged_signal_meta["selection_passed_stages"] = sorted(
                    [
                        k
                        for k, v in dict(getattr(row, "selection_meta", {}) or {}).items()
                        if isinstance(v, dict) and bool(v.get("kept", v.get("pass", False)))
                    ]
                )
                c.signal_meta = merged_signal_meta
            out_candidates.append(c)
            out_decisions.append(d)

        return out_candidates, out_decisions, dict(ctx.meta or {})
