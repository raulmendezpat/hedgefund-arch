from __future__ import annotations

from dataclasses import dataclass

from hf.core.types import Allocation, Candle

from hf_core.allocator import Allocator
from hf_core.contracts import OpportunityCandidate
from hf_core.allocation_engine import AllocationEngine
from hf_core.production_like_allocation_postprocess import ProductionLikeAllocationPostProcessor


@dataclass
class ResearchAllocationRouter:
    snapshot_allocator: Allocator
    legacy_allocation_engine: AllocationEngine
    production_like_postprocessor: ProductionLikeAllocationPostProcessor | None = None

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

        if str(mode) == "production_like_snapshot":
            if self.production_like_postprocessor is not None:
                alloc = self.production_like_postprocessor.apply(
                    allocation=Allocation(
                        weights=dict(getattr(alloc, "weights", {}) or {}),
                        meta=dict(getattr(alloc, "meta", {}) or {}),
                    ),
                    selected_candidates=selected_candidates,
                    prev_allocation=prev_allocation,
                    portfolio_context=portfolio_context,
                )
                meta = dict(getattr(alloc, "meta", {}) or {})
                meta["allocation_mode"] = "production_like_snapshot"
                return Allocation(
                    weights=dict(getattr(alloc, "weights", {}) or {}),
                    meta=meta,
                )

        meta = dict(getattr(alloc, "meta", {}) or {})
        meta["allocation_mode"] = "snapshot"
        return Allocation(
            weights=dict(getattr(alloc, "weights", {}) or {}),
            meta=meta,
        )
