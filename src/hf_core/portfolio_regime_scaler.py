from __future__ import annotations

from dataclasses import dataclass

from hf.core.types import Allocation


@dataclass
class PortfolioRegimeScaler:
    defensive_scale: float = 1.0
    defensive_conviction_k: float = 0.0
    aggressive_scale: float = 1.0

    def _portfolio_scale(self, portfolio_context: dict) -> tuple[float, float, str]:
        regime = str(portfolio_context.get("portfolio_regime", "") or "")
        breadth = float(portfolio_context.get("portfolio_breadth", 0.0) or 0.0)
        avg_pwin = float(portfolio_context.get("portfolio_avg_pwin", 0.0) or 0.0)
        avg_strength = float(portfolio_context.get("portfolio_avg_strength", 0.0) or 0.0)

        try:
            breadth_score = min(1.0, float(breadth) / 5.0)
            pwin_score = max(0.0, min(1.0, float(avg_pwin)))
            strength_score = max(0.0, min(1.0, float(avg_strength)))
            scaler_conviction = float(breadth_score * pwin_score * strength_score)
        except Exception:
            breadth_score = 0.0
            pwin_score = 0.0
            strength_score = 0.0
            scaler_conviction = 0.0

        scale = 0.95 + 0.40 * float(scaler_conviction)

        if regime == "defensive":
            def_base = float(self.defensive_scale)
            def_k = float(self.defensive_conviction_k)
            conv = max(0.0, min(1.0, float(scaler_conviction)))
            dynamic_mult = 1.0 + def_k * (conv - 0.5)
            dynamic_mult = max(0.70, min(1.30, dynamic_mult))
            scale *= float(max(0.0, def_base * dynamic_mult))
        elif regime == "aggressive":
            scale *= float(self.aggressive_scale)

        return (
            float(scale),
            float(scaler_conviction),
            regime,
            float(breadth_score),
            float(pwin_score),
            float(strength_score),
        )

    def apply(
        self,
        *,
        allocation: Allocation,
        portfolio_context: dict | None = None,
    ) -> Allocation:
        ctx = dict(portfolio_context or {})
        upstream_conviction = float(ctx.get("portfolio_conviction", 0.0) or 0.0)
        scale, scaler_conviction, regime, breadth_score, pwin_score, strength_score = self._portfolio_scale(ctx)

        weights = {
            str(k): float(v or 0.0) * float(scale)
            for k, v in dict(getattr(allocation, "weights", {}) or {}).items()
        }

        out_meta = dict(getattr(allocation, "meta", {}) or {})
        out_meta["portfolio_regime"] = str(regime)
        out_meta["portfolio_conviction_upstream"] = float(upstream_conviction)
        out_meta["portfolio_regime_scaler_conviction"] = float(scaler_conviction)
        out_meta["portfolio_conviction"] = float(scaler_conviction)
        out_meta["portfolio_regime_scale"] = float(scale)
        out_meta["portfolio_regime_scaler_applied"] = True
        out_meta["portfolio_context_seen_by_regime_scaler"] = dict(ctx)
        out_meta["portfolio_breadth"] = float(ctx.get("portfolio_breadth", 0.0) or 0.0)
        out_meta["portfolio_avg_pwin"] = float(ctx.get("portfolio_avg_pwin", 0.0) or 0.0)
        out_meta["portfolio_avg_strength"] = float(ctx.get("portfolio_avg_strength", 0.0) or 0.0)
        out_meta["portfolio_regime_scaler_breadth_score"] = float(breadth_score)
        out_meta["portfolio_regime_scaler_pwin_score"] = float(pwin_score)
        out_meta["portfolio_regime_scaler_strength_score"] = float(strength_score)
        out_meta["portfolio_regime_defensive_scale"] = float(self.defensive_scale)
        out_meta["portfolio_regime_defensive_conviction_k"] = float(self.defensive_conviction_k)
        out_meta["portfolio_regime_aggressive_scale"] = float(self.aggressive_scale)

        return Allocation(
            weights=weights,
            meta=out_meta,
        )
