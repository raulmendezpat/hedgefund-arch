from __future__ import annotations

from dataclasses import dataclass

from hf.core.types import Allocation, Candle

from hf_core.allocator import Allocator
from hf_core.contracts import OpportunityCandidate
from hf_core.allocation_engine import AllocationEngine


@dataclass
class ResearchAllocationRouter:
    snapshot_allocator: Allocator
    legacy_allocation_engine: AllocationEngine

    def allocate(
        self,
        *,
        mode: str,
        candles: dict[str, Candle],
        selected_candidates: list[OpportunityCandidate],
        alloc_inputs: list[dict],
        prev_allocation: Allocation | None = None,
        portfolio_context: dict | None = None,
    ) -> Allocation:
        if str(mode) == "legacy_multi_strategy":
            alloc = self.legacy_allocation_engine.allocate(
                candles=candles,
                candidates=selected_candidates,
                prev_allocation=prev_allocation,
                portfolio_context=portfolio_context,
            )
            meta = dict(getattr(alloc, "meta", {}) or {})
            meta["allocation_mode"] = "legacy_multi_strategy"
            return Allocation(
                weights=dict(getattr(alloc, "weights", {}) or {}),
                meta=meta,
            )

        alloc = self.snapshot_allocator.allocate(candidates=alloc_inputs)
        meta = dict(getattr(alloc, "meta", {}) or {})
        meta["allocation_mode"] = "snapshot"
        return Allocation(
            weights=dict(getattr(alloc, "weights", {}) or {}),
            meta=meta,
        )
