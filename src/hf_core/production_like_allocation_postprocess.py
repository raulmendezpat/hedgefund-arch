from __future__ import annotations

from dataclasses import dataclass

from hf.core.types import Allocation
from hf_core.portfolio_regime_scaler import PortfolioRegimeScaler
from hf_core.portfolio_risk_overlay import PortfolioRiskOverlay
from hf_core.strategy_side_weight_overlay import StrategySideWeightOverlay
from hf_core.portfolio_atrp_risk_scaler import PortfolioAtrpRiskScaler
from hf_core.production_like_allocation_ml_sizer import ProductionLikeAllocationMlSizer
from hf_core.production_like_allocation_cluster_controls import ProductionLikeAllocationClusterControls


@dataclass
class ProductionLikeAllocationPostProcessor:
    smoothing_alpha: float = 0.0
    smoothing_snap_eps: float = 0.0
    max_step_per_bar: float = 1.0
    apply_signal_gating: bool = True
    regime_scaler: PortfolioRegimeScaler | None = None
    risk_overlay: PortfolioRiskOverlay | None = None
    strategy_side_weight_overlay: StrategySideWeightOverlay | None = None
    atrp_risk_scaler: PortfolioAtrpRiskScaler | None = None
    ml_sizer: ProductionLikeAllocationMlSizer | None = None
    cluster_controls: ProductionLikeAllocationClusterControls | None = None
    execution_symbol_cap: float = 0.0

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
        portfolio_context: dict | None = None,
    ) -> Allocation:
        raw_weights = {
            str(k): float(v or 0.0)
            for k, v in dict(getattr(allocation, "weights", {}) or {}).items()
        }

        alloc_after_regime = Allocation(
            weights=dict(raw_weights),
            meta=dict(getattr(allocation, "meta", {}) or {}),
        )
        if self.regime_scaler is not None:
            alloc_after_regime = self.regime_scaler.apply(
                allocation=alloc_after_regime,
                portfolio_context=portfolio_context,
            )

        after_regime_weights = {
            str(k): float(v or 0.0)
            for k, v in dict(getattr(alloc_after_regime, "weights", {}) or {}).items()
        }

        alloc_after_risk = alloc_after_regime
        if self.risk_overlay is not None:
            alloc_after_risk = self.risk_overlay.apply(
                allocation=Allocation(
                    weights=dict(after_regime_weights),
                    meta=dict(getattr(alloc_after_regime, "meta", {}) or {}),
                ),
                portfolio_context=portfolio_context,
            )

        alloc_after_risk_meta = dict(getattr(alloc_after_risk, "meta", {}) or {})
        alloc_after_risk_meta["legacy_opportunities"] = [
            {
                "symbol": str(getattr(c, "symbol", "") or ""),
                "strategy_id": str(getattr(c, "strategy_id", "") or ""),
                "side": str(getattr(c, "side", "") or ""),
                "strength": float(getattr(c, "signal_strength", getattr(c, "strength", 0.0)) or 0.0),
                "timestamp": int(getattr(c, "ts", getattr(c, "timestamp", 0)) or 0),
                "meta": dict(getattr(c, "signal_meta", getattr(c, "meta", {})) or {}),
            }
            for c in list(selected_candidates or [])
        ]
        alloc_after_risk = Allocation(
            weights=dict(getattr(alloc_after_risk, "weights", {}) or {}),
            meta=alloc_after_risk_meta,
        )

        after_risk_weights = {
            str(k): float(v or 0.0)
            for k, v in dict(getattr(alloc_after_risk, "weights", {}) or {}).items()
        }

        alloc_after_side = alloc_after_risk
        if self.strategy_side_weight_overlay is not None:
            alloc_after_side = self.strategy_side_weight_overlay.apply(
                allocation=Allocation(
                    weights=dict(after_risk_weights),
                    meta=dict(getattr(alloc_after_risk, "meta", {}) or {}),
                ),
                prev_allocation=prev_allocation,
            )

        after_side_overlay_weights = {
            str(k): float(v or 0.0)
            for k, v in dict(getattr(alloc_after_side, "weights", {}) or {}).items()
        }

        alloc_after_ml = alloc_after_side
        if self.ml_sizer is not None:
            alloc_after_ml = self.ml_sizer.apply(
                allocation=Allocation(
                    weights=dict(after_side_overlay_weights),
                    meta=dict(getattr(alloc_after_side, "meta", {}) or {}),
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

        alloc_after_atrp_risk = Allocation(
            weights=dict(after_smoothing_weights),
            meta=dict(getattr(alloc_after_ml, "meta", {}) or {}),
        )
        if self.atrp_risk_scaler is not None:
            alloc_after_atrp_risk = self.atrp_risk_scaler.apply(
                allocation=alloc_after_atrp_risk,
                selected_candidates=selected_candidates,
            )

        after_atrp_risk_weights = {
            str(k): float(v or 0.0)
            for k, v in dict(getattr(alloc_after_atrp_risk, "weights", {}) or {}).items()
        }

        # --- signal gating from selected candidates ---
        gated = dict(after_atrp_risk_weights)
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

        execution_symbol_cap = float(self.execution_symbol_cap or 0.0)
        if execution_symbol_cap > 0.0:
            performance_weights = {
                str(k): max(-execution_symbol_cap, min(execution_symbol_cap, float(v or 0.0)))
                for k, v in dict(final_weights or {}).items()
            }
        else:
            performance_weights = dict(final_weights)

        meta = dict(getattr(alloc_after_cluster_controls, "meta", {}) or {})
        meta.update(
            {
                "allocation_mode": "production_like_snapshot",
                "pipeline_weight_order": "raw->regime_scaler->risk_overlay->strategy_side_overlay->ml->smoothing->atrp_risk_scale->signal_gating->cluster_controls",
                "raw_allocator_weights": dict(raw_weights),
                "after_regime_scaler_weights": dict(after_regime_weights),
                "after_risk_overlay_weights": dict(after_risk_weights),
                "after_strategy_side_overlay_weights": dict(after_side_overlay_weights),
                "after_ml_position_sizing_weights": dict(after_ml_weights),
                "after_smoothing_weights": dict(after_smoothing_weights),
                "after_atrp_risk_scale_weights": dict(after_atrp_risk_weights),
                "after_signal_gating_weights": dict(after_signal_gating_weights),
                "after_cluster_controls_weights": dict(final_weights),
                "performance_weights_by_symbol": dict(performance_weights),
                "execution_symbol_cap": float(execution_symbol_cap),
                "signal_gate_applied": bool(self.apply_signal_gating),
                "signal_gated": dict(signal_gated),
                "prodlike_regime_scaler_applied": bool(self.regime_scaler is not None),
                "prodlike_risk_overlay_applied": bool(self.risk_overlay is not None),
                "prodlike_strategy_side_overlay_applied": bool(self.strategy_side_weight_overlay is not None),
                "prodlike_atrp_risk_scaler_applied": bool(self.atrp_risk_scaler is not None),
                "prodlike_smoothing_alpha": float(self.smoothing_alpha),
                "prodlike_smoothing_snap_eps": float(self.smoothing_snap_eps),
                "prodlike_max_step_per_bar": float(self.max_step_per_bar),
            }
        )

        return Allocation(weights=final_weights, meta=meta)
