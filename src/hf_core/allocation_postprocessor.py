from __future__ import annotations

from dataclasses import dataclass

from hf.core.types import Allocation


@dataclass
class AllocationPostprocessor:
    smoothing_alpha: float = 0.0
    smoothing_snap_eps: float = 0.0

    def apply(
        self,
        *,
        allocation: Allocation,
        prev_allocation: Allocation | None = None,
    ) -> Allocation:
        if prev_allocation is None:
            return allocation

        alloc_meta = dict(getattr(allocation, "meta", {}) or {})
        alloc_case = str(alloc_meta.get("case", "") or "")

        if alloc_case != "multi_strategy":
            return allocation

        alpha = float(self.smoothing_alpha)
        snap_eps = float(self.smoothing_snap_eps)

        if alpha <= 0.0:
            return allocation

        prev_weights = dict(getattr(prev_allocation, "weights", {}) or {})
        curr_weights = dict(getattr(allocation, "weights", {}) or {})
        keys = set(prev_weights.keys()) | set(curr_weights.keys())

        smoothed_weights = {}
        for k in keys:
            prev_w = float(prev_weights.get(k, 0.0) or 0.0)
            curr_w = float(curr_weights.get(k, 0.0) or 0.0)
            w = prev_w + alpha * (curr_w - prev_w)
            if abs(w) < snap_eps:
                w = 0.0
            smoothed_weights[str(k)] = float(w)

        out_meta = dict(alloc_meta)
        out_meta["postprocess_smoothing_applied"] = True
        out_meta["postprocess_smoothing_alpha"] = float(alpha)
        out_meta["postprocess_smoothing_snap_eps"] = float(snap_eps)

        return Allocation(
            weights=smoothed_weights,
            meta=out_meta,
        )
