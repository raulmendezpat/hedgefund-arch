from __future__ import annotations

from dataclasses import dataclass

from hf.core.types import Allocation


@dataclass
class AllocationStepGuardrail:
    max_step_per_bar: float = 1.0

    def apply(
        self,
        *,
        allocation: Allocation,
        prev_allocation: Allocation | None = None,
    ) -> Allocation:
        if prev_allocation is None:
            return allocation

        step_cap = float(self.max_step_per_bar)
        if step_cap >= 1.0:
            return allocation

        prev_weights = dict(getattr(prev_allocation, "weights", {}) or {})
        curr_weights = dict(getattr(allocation, "weights", {}) or {})
        keys = set(prev_weights.keys()) | set(curr_weights.keys())

        clamped = {}
        for k in keys:
            prev_w = float(prev_weights.get(k, 0.0) or 0.0)
            curr_w = float(curr_weights.get(k, 0.0) or 0.0)
            delta = curr_w - prev_w
            if delta > step_cap:
                curr_w = prev_w + step_cap
            elif delta < -step_cap:
                curr_w = prev_w - step_cap
            clamped[str(k)] = float(curr_w)

        out_meta = dict(getattr(allocation, "meta", {}) or {})
        out_meta["max_step_guardrail_applied"] = True
        out_meta["max_step_per_bar"] = float(step_cap)

        return Allocation(
            weights=clamped,
            meta=out_meta,
        )
