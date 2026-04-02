from __future__ import annotations

from dataclasses import dataclass

from hf.core.types import Allocation


@dataclass
class PortfolioAtrpRiskScaler:
    atrp_low: float = 0.0
    atrp_high: float = 0.0
    floor: float = 1.0

    def apply(
        self,
        *,
        allocation: Allocation,
        selected_candidates,
    ) -> Allocation:
        low = float(self.atrp_low)
        high = float(self.atrp_high)
        floor = max(0.0, min(1.0, float(self.floor)))

        weighted_sum = 0.0
        weight_sum = 0.0
        for c in list(selected_candidates or []):
            meta = dict(getattr(c, "signal_meta", getattr(c, "meta", {})) or {})
            try:
                atrp = float(meta.get("atrp", 0.0) or 0.0)
            except Exception:
                atrp = 0.0
            try:
                score_w = float(
                    meta.get("post_ml_score", meta.get("meta_post_ml_score", 0.0)) or 0.0
                )
            except Exception:
                score_w = 0.0

            if atrp > 0.0 and score_w > 0.0:
                weighted_sum += float(atrp) * float(score_w)
                weight_sum += float(score_w)

        portfolio_atrp = float(weighted_sum / weight_sum) if weight_sum > 0.0 else 0.0

        if high <= low:
            risk_mult = 1.0 if portfolio_atrp <= low else floor
        elif portfolio_atrp <= low:
            risk_mult = 1.0
        elif portfolio_atrp >= high:
            risk_mult = floor
        else:
            span = high - low
            x = (portfolio_atrp - low) / span
            risk_mult = 1.0 - x * (1.0 - floor)

        weights = {
            str(k): float(v or 0.0) * float(risk_mult)
            for k, v in dict(getattr(allocation, "weights", {}) or {}).items()
        }

        meta = dict(getattr(allocation, "meta", {}) or {})
        meta["portfolio_atrp_risk_scale_applied"] = True
        meta["portfolio_atrp_risk_scale_mult"] = float(risk_mult)
        meta["portfolio_atrp_risk_scale_avg_atrp"] = float(portfolio_atrp)
        meta["portfolio_atrp_risk_scale_low"] = float(low)
        meta["portfolio_atrp_risk_scale_high"] = float(high)
        meta["portfolio_atrp_risk_scale_floor"] = float(floor)
        meta["portfolio_atrp_risk_scale_mode"] = "weighted_selected_atrp"

        return Allocation(weights=weights, meta=meta)
