from __future__ import annotations

from hf_core.allocation_engine.contracts import AllocationCandidate, AllocationResult
from hf_core.allocation_engine.score_strategy import ScoreStrategy
from hf_core.allocation_engine.weight_strategy import WeightStrategy
from hf_core.allocation_engine.constraints import apply_symbol_cap, apply_target_exposure


class SnapshotAllocatorEngine:
    def __init__(
        self,
        *,
        score_strategy: ScoreStrategy,
        weight_strategy: WeightStrategy,
        target_exposure: float = 0.07,
        symbol_cap: float = 0.25,
    ):
        self.score_strategy = score_strategy
        self.weight_strategy = weight_strategy
        self.target_exposure = float(target_exposure)
        self.symbol_cap = float(symbol_cap)

    def allocate(self, candidates: list[AllocationCandidate]) -> AllocationResult:
        if not candidates:
            return AllocationResult(weights={}, meta={"n_candidates": 0, "n_selected": 0})

        raw_scores = {}
        sides = {}
        selected_meta = {}

        for c in candidates:
            s = float(self.score_strategy.score(c))
            if s <= 0.0:
                continue
            raw_scores[c.symbol] = max(float(raw_scores.get(c.symbol, 0.0)), s)
            sides[c.symbol] = str(c.side)
            selected_meta[c.symbol] = {
                "p_win": float(c.p_win),
                "expected_return": float(c.expected_return),
                "score_input": float(c.score),
                "score_final": float(s),
                "side": str(c.side),
            }

        base_weights = self.weight_strategy.transform(raw_scores)
        capped = apply_symbol_cap(base_weights, self.symbol_cap)
        final_weights = apply_target_exposure(capped, self.target_exposure, sides)

        return AllocationResult(
            weights=final_weights,
            meta={
                "n_candidates": len(candidates),
                "n_selected": len(base_weights),
                "raw_scores": raw_scores,
                "base_weights": base_weights,
                "capped_weights": capped,
                "selected_meta": selected_meta,
                "engine": "snapshot_allocator_engine",
            },
        )
