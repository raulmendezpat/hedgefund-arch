from __future__ import annotations

from dataclasses import dataclass

from .pipeline import AllocatorPipeline
from .steps import (
    AggregateBySymbolStep,
    CapWeightsStep,
    CollectDeployableStep,
    NormalizeToTargetExposureStep,
)


@dataclass
class AllocatorFactory:
    profile: str = "symbol_net"

    def build(
        self,
        *,
        target_exposure: float,
        symbol_cap: float,
    ) -> AllocatorPipeline:
        profile = str(self.profile or "symbol_net").lower()

        if profile == "symbol_net":
            return AllocatorPipeline(
                steps=[
                    CollectDeployableStep(),
                    AggregateBySymbolStep(enable_symbol_netting=True),
                    NormalizeToTargetExposureStep(target_exposure=float(target_exposure)),
                    CapWeightsStep(symbol_cap=float(symbol_cap)),
                ]
            )

        if profile == "symbol_gross":
            return AllocatorPipeline(
                steps=[
                    CollectDeployableStep(),
                    AggregateBySymbolStep(enable_symbol_netting=False),
                    NormalizeToTargetExposureStep(target_exposure=float(target_exposure)),
                    CapWeightsStep(symbol_cap=float(symbol_cap)),
                ]
            )

        raise ValueError(f"Unknown allocator profile: {profile}")
