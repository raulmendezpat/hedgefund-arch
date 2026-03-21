from __future__ import annotations

from typing import Any

from hf_core.contracts import FeatureRow, OpportunityCandidate


class FeatureBuilder:
    def build_feature_row(
        self,
        *,
        candidate: OpportunityCandidate,
        portfolio_context: dict[str, Any] | None = None,
    ) -> FeatureRow:
        ctx = dict(portfolio_context or {})
        meta = dict(candidate.signal_meta or {})

        values: dict[str, float | int | str] = {
            "symbol": str(candidate.symbol).replace("/USDT:USDT", ""),
            "strategy_id": str(candidate.strategy_id),
            "side": str(candidate.side),
            "signal_strength": float(candidate.signal_strength),
            "base_weight": float(candidate.base_weight),
            "abs_weight": abs(float(candidate.base_weight)),
            "signed_weight": float(candidate.base_weight if candidate.side == "long" else -candidate.base_weight),
            "portfolio_regime": str(ctx.get("portfolio_regime", "unknown")),
            "portfolio_breadth": float(ctx.get("portfolio_breadth", 0.0) or 0.0),
            "portfolio_avg_pwin": float(ctx.get("portfolio_avg_pwin", 0.0) or 0.0),
            "portfolio_avg_atrp": float(ctx.get("portfolio_avg_atrp", 0.0) or 0.0),
            "portfolio_avg_strength": float(ctx.get("portfolio_avg_strength", 0.0) or 0.0),
            "portfolio_conviction": float(ctx.get("portfolio_conviction", 0.0) or 0.0),
            "portfolio_regime_scale_applied": float(ctx.get("portfolio_regime_scale_applied", 1.0) or 1.0),
            "meta_p_win": float(meta.get("p_win", 0.0) or 0.0),
            "meta_competitive_score": float(meta.get("competitive_score", 0.0) or 0.0),
            "meta_post_ml_score": float(meta.get("post_ml_score", 0.0) or 0.0),
            "meta_adx": float(meta.get("adx", 0.0) or 0.0),
            "meta_atrp": float(meta.get("atrp", 0.0) or 0.0),
            "meta_rsi": float(meta.get("rsi", 0.0) or 0.0),
        }

        return FeatureRow(
            ts=int(candidate.ts),
            symbol=str(candidate.symbol),
            strategy_id=str(candidate.strategy_id),
            side=str(candidate.side),
            values=values,
        )

    def build_feature_rows(
        self,
        *,
        candidates: list[OpportunityCandidate],
        portfolio_context: dict[str, Any] | None = None,
    ) -> list[FeatureRow]:
        return [
            self.build_feature_row(candidate=c, portfolio_context=portfolio_context)
            for c in list(candidates or [])
        ]
