from __future__ import annotations

from hf.core.types import Allocation


def weights_snapshot(allocation: Allocation | None) -> dict[str, float]:
    if allocation is None:
        return {}
    return {
        str(k): float(v or 0.0)
        for k, v in dict(getattr(allocation, "weights", {}) or {}).items()
    }


def gross_exposure(weights: dict[str, float]) -> float:
    return float(sum(abs(float(v or 0.0)) for v in dict(weights or {}).values()))


def run_allocation_stages(
    *,
    alloc_core: Allocation,
    regime_scaler,
    portfolio_risk_overlay,
    strategy_side_weight_overlay,
    postprocessor,
    step_guardrail,
    prev_allocation=None,
    portfolio_context: dict | None = None,
):
    alloc_regime = regime_scaler.apply(
        allocation=alloc_core,
        portfolio_context=portfolio_context,
    )
    alloc_risk = portfolio_risk_overlay.apply(
        allocation=alloc_regime,
        portfolio_context=portfolio_context,
    )
    alloc_side = strategy_side_weight_overlay.apply(
        allocation=alloc_risk,
        prev_allocation=prev_allocation,
    )
    alloc_post = postprocessor.apply(
        allocation=alloc_side,
        prev_allocation=prev_allocation,
    )
    alloc_final = step_guardrail.apply(
        allocation=alloc_post,
        prev_allocation=prev_allocation,
    )

    stage_weights = {
        "allocator_core": weights_snapshot(alloc_core),
        "after_regime_scaler": weights_snapshot(alloc_regime),
        "after_portfolio_risk_overlay": weights_snapshot(alloc_risk),
        "after_strategy_side_overlay": weights_snapshot(alloc_side),
        "after_postprocessor": weights_snapshot(alloc_post),
        "after_step_guardrail": weights_snapshot(alloc_final),
    }
    stage_gross = {
        k: gross_exposure(v)
        for k, v in stage_weights.items()
    }

    return {
        "alloc_regime": alloc_regime,
        "alloc_risk": alloc_risk,
        "alloc_side": alloc_side,
        "alloc_post": alloc_post,
        "alloc_final": alloc_final,
        "stage_weights": stage_weights,
        "stage_gross": stage_gross,
    }
