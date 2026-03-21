from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hf_core.contracts import OpportunityCandidate
from hf_core.allocator_parts import AllocatorContext, AllocatorFactory


@dataclass
class Allocation:
    weights: dict[str, float]
    meta: dict[str, Any] = field(default_factory=dict)


class Allocator:
    def __init__(
        self,
        *,
        target_exposure: float = 1.0,
        symbol_cap: float = 0.35,
        profile: str = "symbol_net",
    ):
        self.target_exposure = float(target_exposure)
        self.symbol_cap = float(symbol_cap)
        self.profile = str(profile)

    def allocate(
        self,
        *,
        candidates: list[OpportunityCandidate],
    ) -> Allocation:
        if not candidates:
            return Allocation(
                weights={},
                meta={
                    "case": "no_inputs",
                    "n_candidates": 0,
                    "target_exposure": self.target_exposure,
                    "symbol_cap": self.symbol_cap,
                    "allocator_profile": self.profile,
                },
            )

        pipeline = AllocatorFactory(profile=self.profile).build(
            target_exposure=float(self.target_exposure),
            symbol_cap=float(self.symbol_cap),
        )

        ctx = AllocatorContext(candidates=list(candidates or []))
        ctx = pipeline.run(ctx)

        return Allocation(
            weights=dict(ctx.symbol_weights or {}),
            meta={
                "case": "allocator_factory_pipeline",
                "n_candidates": len(candidates),
                "n_symbols": len(dict(ctx.symbol_weights or {})),
                "target_exposure": self.target_exposure,
                "symbol_cap": self.symbol_cap,
                "allocator_profile": self.profile,
                **dict(ctx.meta or {}),
            },
        )
