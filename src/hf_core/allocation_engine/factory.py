from __future__ import annotations

from hf_core.allocation_engine.engine import SnapshotAllocatorEngine
from hf_core.allocation_engine.score_strategy import PWinExpectedReturnScoreStrategy, PolicyFirstScoreStrategy
from hf_core.allocation_engine.weight_strategy import EliteTopWeightStrategy


def build_snapshot_allocator(
    *,
    target_exposure: float = 0.07,
    symbol_cap: float = 0.25,
    min_pwin: float = 0.55,
    temperature: float = 0.05,
    use_expected_return: bool = True,
    score_mode: str = "policy_first",
    min_policy_score: float = 0.0,
    p0: float = 0.50,
    pwin_scale: float = 2.0,
    er_scale: float = 500.0,
):
    score_mode = str(score_mode or "policy_first").lower()

    if score_mode == "policy_first":
        score_strategy = PolicyFirstScoreStrategy(
            min_policy_score=min_policy_score,
            min_pwin=min_pwin,
            p0=p0,
            pwin_scale=pwin_scale,
            use_expected_return=use_expected_return,
            er_scale=er_scale,
        )
    else:
        score_strategy = PWinExpectedReturnScoreStrategy(
            min_pwin=min_pwin,
            use_expected_return=use_expected_return,
            pwin_power=1.0,
        )

    return SnapshotAllocatorEngine(
        score_strategy=score_strategy,
        weight_strategy=EliteTopWeightStrategy(
            top_share=0.90,
        ),
        target_exposure=target_exposure,
        symbol_cap=symbol_cap,
    )
