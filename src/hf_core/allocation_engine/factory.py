from __future__ import annotations

from hf_core.allocation_engine.engine import SnapshotAllocatorEngine
from hf_core.allocation_engine.score_strategy import PWinExpectedReturnScoreStrategy
from hf_core.allocation_engine.weight_strategy import SoftmaxWeightStrategy


def build_snapshot_allocator(
    *,
    target_exposure: float = 0.07,
    symbol_cap: float = 0.25,
    min_pwin: float = 0.57,
    temperature: float = 0.05,
    use_expected_return: bool = True,
):
    return SnapshotAllocatorEngine(
        score_strategy=PWinExpectedReturnScoreStrategy(
            min_pwin=min_pwin,
            use_expected_return=use_expected_return,
            pwin_power=1.0,
        ),
        weight_strategy=SoftmaxWeightStrategy(
            temperature=temperature,
        ),
        target_exposure=target_exposure,
        symbol_cap=symbol_cap,
    )
