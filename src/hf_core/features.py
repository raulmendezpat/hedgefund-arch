from __future__ import annotations

from typing import Any

from hf_core.contracts import FeatureRow, OpportunityCandidate


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _b(x: Any) -> float:
    return 1.0 if bool(x) else 0.0


class FeatureBuilder:
    def build_feature_row(
        self,
        *,
        candidate: OpportunityCandidate,
        portfolio_context: dict[str, Any] | None = None,
    ) -> FeatureRow:
        ctx = dict(portfolio_context or {})
        meta = dict(candidate.signal_meta or {})

        side = str(candidate.side or "flat").lower()
        signed_weight = 0.0
        if side == "long":
            signed_weight = float(candidate.base_weight)
        elif side == "short":
            signed_weight = -float(candidate.base_weight)

        values: dict[str, float | int | str] = {
            "symbol": str(candidate.symbol).replace("/USDT:USDT", ""),
            "strategy_id": str(candidate.strategy_id),
            "side": side,
            "signal_strength": _f(candidate.signal_strength),
            "base_weight": _f(candidate.base_weight),
            "abs_weight": abs(_f(candidate.base_weight)),
            "signed_weight": signed_weight,

            # portfolio context
            "portfolio_regime": str(ctx.get("portfolio_regime", "unknown")),
            "portfolio_breadth": _f(ctx.get("portfolio_breadth", 0.0)),
            "portfolio_avg_pwin": _f(ctx.get("portfolio_avg_pwin", 0.0)),
            "portfolio_avg_atrp": _f(ctx.get("portfolio_avg_atrp", 0.0)),
            "portfolio_avg_strength": _f(ctx.get("portfolio_avg_strength", 0.0)),
            "portfolio_conviction": _f(ctx.get("portfolio_conviction", 0.0)),
            "portfolio_regime_scale_applied": _f(ctx.get("portfolio_regime_scale_applied", 1.0), 1.0),

            # raw signal/meta
            "meta_p_win": _f(meta.get("p_win", 0.0)),
            "meta_competitive_score": _f(meta.get("competitive_score", 0.0)),
            "meta_post_ml_score": _f(meta.get("post_ml_score", 0.0)),
            "meta_adx": _f(meta.get("adx", 0.0)),
            "meta_atrp": _f(meta.get("atrp", 0.0)),
            "meta_rsi": _f(meta.get("rsi", 0.0)),
            "meta_bb_width": _f(meta.get("bb_width", 0.0)),
            "meta_range_expansion": _f(meta.get("range_expansion", 0.0)),
            "meta_ema_gap_pct": _f(meta.get("ema_gap_pct", 0.0)),

            "ctx_close_vs_ema_slow": _f(meta.get("ctx_close_vs_ema_slow", 0.0)),
            "ctx_ema_fast_vs_ema_slow": _f(meta.get("ctx_ema_fast_vs_ema_slow", 0.0)),
            "ctx_ret_24h": _f(meta.get("ctx_ret_24h", 0.0)),
            "ctx_ret_7d": _f(meta.get("ctx_ret_7d", 0.0)),
            "ctx_ret_30d": _f(meta.get("ctx_ret_30d", 0.0)),
            "ctx_slope_ema_fast_24h": _f(meta.get("ctx_slope_ema_fast_24h", 0.0)),
            "ctx_adx_pct": _f(meta.get("ctx_adx_pct", 0.5)),
            "ctx_atrp_pct": _f(meta.get("ctx_atrp_pct", 0.5)),
            "ctx_rsi_pct": _f(meta.get("ctx_rsi_pct", 0.5)),
            "ctx_bb_width_pct": _f(meta.get("ctx_bb_width_pct", 0.5)),
            "ctx_range_expansion_pct": _f(meta.get("ctx_range_expansion_pct", 0.5)),
            "ctx_backdrop": str(meta.get("ctx_backdrop", "neutral")),
            "ctx_trend_alignment_score": _f(meta.get("ctx_trend_alignment_score", 0.0)),
            "ctx_side_backdrop_alignment": _f(meta.get("ctx_side_backdrop_alignment", 0.0)),
            "ctx_expected_holding_bars": _f(meta.get("ctx_expected_holding_bars", 12.0)),
            "ctx_tp_mult": _f(meta.get("ctx_tp_mult", 1.0), 1.0),
            "ctx_sl_mult": _f(meta.get("ctx_sl_mult", 1.0), 1.0),
            "ctx_time_stop_bars": _f(meta.get("ctx_time_stop_bars", 12.0), 12.0),

            # regime / quality flags from signal layer
            "flag_adx_below_min": _b(meta.get("adx_below_min", False)),
            "flag_ema_gap_below_min": _b(meta.get("ema_gap_below_min", False)),
            "flag_atrp_low": _b(meta.get("atrp_low", False)),
            "flag_adx_low": _b(meta.get("adx_low", False)),
            "flag_range_expansion_low": _b(meta.get("range_expansion_low", False)),
            "flag_long_non_directional_bar": _b(meta.get("long_non_directional_bar", False)),
            "flag_short_non_directional_bar": _b(meta.get("short_non_directional_bar", False)),
            "flag_long_trend_misaligned": _b(meta.get("long_trend_misaligned", False)),
            "flag_short_trend_misaligned": _b(meta.get("short_trend_misaligned", False)),
            "flag_long_no_donchian_break": _b(meta.get("long_no_donchian_break", False)),
            "flag_short_no_donchian_break": _b(meta.get("short_no_donchian_break", False)),
            "flag_long_rsi_exhausted": _b(meta.get("long_rsi_exhausted", False)),
            "flag_short_rsi_exhausted": _b(meta.get("short_rsi_exhausted", False)),
            "flag_long_overextended": _b(meta.get("long_overextended", False)),
            "flag_short_overextended": _b(meta.get("short_overextended", False)),
            "flag_regime_as_metadata": _b(meta.get("regime_as_metadata", False)),
        }

        return FeatureRow(
            ts=int(candidate.ts),
            symbol=str(candidate.symbol),
            strategy_id=str(candidate.strategy_id),
            side=side,
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
