from __future__ import annotations

from dataclasses import dataclass

from hf.core.types import Allocation
from hf_core.production_like_allocation_ml_sizer import ProductionLikeAllocationMlSizer
from hf_core.production_like_allocation_cluster_controls import ProductionLikeAllocationClusterControls


@dataclass
class ProductionLikeAllocationPostProcessor:
    smoothing_alpha: float = 0.0
    smoothing_snap_eps: float = 0.0
    max_step_per_bar: float = 1.0
    apply_signal_gating: bool = True
    ml_sizer: ProductionLikeAllocationMlSizer | None = None
    cluster_controls: ProductionLikeAllocationClusterControls | None = None

    def _selected_side_by_symbol(self, selected_candidates) -> dict[str, str]:
        out: dict[str, str] = {}
        for c in list(selected_candidates or []):
            sym = str(getattr(c, "symbol", "") or "")
            side = str(getattr(c, "side", "flat") or "flat").lower()
            if not sym or side not in {"long", "short"}:
                continue
            if sym not in out:
                out[sym] = side
        return out

    def apply(
        self,
        *,
        allocation: Allocation,
        selected_candidates,
        prev_allocation: Allocation | None = None,
    ) -> Allocation:
        raw_weights = {
            str(k): float(v or 0.0)
            for k, v in dict(getattr(allocation, "weights", {}) or {}).items()
        }

        alloc_after_ml = allocation
        if self.ml_sizer is not None:
            alloc_after_ml = self.ml_sizer.apply(
                allocation=Allocation(
                    weights=dict(raw_weights),
                    meta=dict(getattr(allocation, "meta", {}) or {}),
                ),
                selected_candidates=selected_candidates,
            )

        after_ml_weights = {
            str(k): float(v or 0.0)
            for k, v in dict(getattr(alloc_after_ml, "weights", {}) or {}).items()
        }

        # --- smoothing ---
        smoothed = dict(after_ml_weights)
        if prev_allocation is not None and float(self.smoothing_alpha) > 0.0:
            prev_w = {
                str(k): float(v or 0.0)
                for k, v in dict(getattr(prev_allocation, "weights", {}) or {}).items()
            }
            keys = set(prev_w.keys()) | set(smoothed.keys())
            alpha = float(self.smoothing_alpha)
            snap_eps = float(self.smoothing_snap_eps)
            tmp = {}
            for k in keys:
                pw = float(prev_w.get(k, 0.0) or 0.0)
                cw = float(smoothed.get(k, 0.0) or 0.0)
                w = pw + alpha * (cw - pw)
                if abs(w) < snap_eps:
                    w = 0.0
                tmp[k] = float(w)
            smoothed = tmp

        # --- max-step guardrail ---
        if prev_allocation is not None and float(self.max_step_per_bar) < 1.0:
            step_cap = max(0.0, float(self.max_step_per_bar))
            prev_w = {
                str(k): float(v or 0.0)
                for k, v in dict(getattr(prev_allocation, "weights", {}) or {}).items()
            }
            keys = set(prev_w.keys()) | set(smoothed.keys())
            tmp = {}
            for k in keys:
                pw = float(prev_w.get(k, 0.0) or 0.0)
                cw = float(smoothed.get(k, 0.0) or 0.0)
                delta = cw - pw
                if delta > step_cap:
                    cw = pw + step_cap
                elif delta < -step_cap:
                    cw = pw - step_cap
                tmp[k] = float(cw)
            smoothed = tmp

        after_smoothing_weights = dict(smoothed)

        # --- signal gating from selected candidates ---
        gated = dict(after_smoothing_weights)
        signal_gated = {}
        if bool(self.apply_signal_gating):
            selected_side = self._selected_side_by_symbol(selected_candidates)
            for sym, w in list(gated.items()):
                side = str(selected_side.get(sym, "flat") or "flat").lower()
                if side == "flat":
                    if float(w) != 0.0:
                        gated[sym] = 0.0
                        signal_gated[sym] = "flat"
                    continue
                if side == "long" and float(w) < 0.0:
                    gated[sym] = 0.0
                    signal_gated[sym] = "side_mismatch_long"
                elif side == "short" and float(w) > 0.0:
                    gated[sym] = 0.0
                    signal_gated[sym] = "side_mismatch_short"

        after_signal_gating_weights = dict(gated)

        alloc_after_cluster_controls = Allocation(
            weights=dict(after_signal_gating_weights),
            meta=dict(getattr(alloc_after_ml, "meta", {}) or {}),
        )
        if self.cluster_controls is not None:
            alloc_after_cluster_controls = self.cluster_controls.apply(
                allocation=alloc_after_cluster_controls,
            )

        final_weights = {
            str(k): float(v or 0.0)
            for k, v in dict(getattr(alloc_after_cluster_controls, "weights", {}) or {}).items()
        }

        meta = dict(getattr(alloc_after_cluster_controls, "meta", {}) or {})
        meta.update(
            {
                "allocation_mode": "production_like_snapshot",
                "pipeline_weight_order": "raw->ml->smoothing->signal_gating->cluster_controls",
                "raw_allocator_weights": dict(raw_weights),
                "after_ml_position_sizing_weights": dict(after_ml_weights),
                "after_smoothing_weights": dict(after_smoothing_weights),
                "after_signal_gating_weights": dict(after_signal_gating_weights),
                "after_cluster_controls_weights": dict(final_weights),
                "signal_gate_applied": bool(self.apply_signal_gating),
                "signal_gated": dict(signal_gated),
                "prodlike_smoothing_alpha": float(self.smoothing_alpha),
                "prodlike_smoothing_snap_eps": float(self.smoothing_snap_eps),
                "prodlike_max_step_per_bar": float(self.max_step_per_bar),
            }
        )

        return Allocation(weights=final_weights, meta=meta)
