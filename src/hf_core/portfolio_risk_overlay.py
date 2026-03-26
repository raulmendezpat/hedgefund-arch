from __future__ import annotations

from dataclasses import dataclass

from hf.core.types import Allocation


@dataclass
class PortfolioRiskOverlay:
    breadth_high_risk: int = 0
    pwin_high_risk: float = 0.0
    high_risk_scale: float = 1.0

    def apply(
        self,
        *,
        allocation: Allocation,
        portfolio_context: dict | None = None,
    ) -> Allocation:
        ctx = dict(portfolio_context or {})

        breadth = float(ctx.get("portfolio_breadth", 0.0) or 0.0)
        avg_pwin = float(ctx.get("portfolio_avg_pwin", 0.0) or 0.0)

        breadth_thr = int(self.breadth_high_risk)
        pwin_thr = float(self.pwin_high_risk)
        scale = float(self.high_risk_scale)

        high_risk = False
        if breadth_thr > 0 and pwin_thr > 0.0:
            high_risk = bool(breadth >= breadth_thr and avg_pwin <= pwin_thr)

        if (not high_risk) or abs(scale - 1.0) < 1e-12:
            out_meta = dict(getattr(allocation, "meta", {}) or {})
            out_meta["portfolio_high_risk"] = bool(high_risk)
            out_meta["portfolio_high_risk_breadth_threshold"] = int(breadth_thr)
            out_meta["portfolio_high_risk_pwin_threshold"] = float(pwin_thr)
            out_meta["portfolio_high_risk_scale"] = float(scale)
            return Allocation(
                weights=dict(getattr(allocation, "weights", {}) or {}),
                meta=out_meta,
            )

        weights = {
            str(k): float(v or 0.0) * float(scale)
            for k, v in dict(getattr(allocation, "weights", {}) or {}).items()
        }

        out_meta = dict(getattr(allocation, "meta", {}) or {})
        out_meta["portfolio_high_risk"] = True
        out_meta["portfolio_high_risk_breadth_threshold"] = int(breadth_thr)
        out_meta["portfolio_high_risk_pwin_threshold"] = float(pwin_thr)
        out_meta["portfolio_high_risk_scale"] = float(scale)

        return Allocation(
            weights=weights,
            meta=out_meta,
        )
