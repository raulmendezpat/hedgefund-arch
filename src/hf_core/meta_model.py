from __future__ import annotations

from hf_core.contracts import FeatureRow, MetaScore


class MetaModel:
    def __init__(
        self,
        *,
        pwin_floor: float = 0.35,
        pwin_cap: float = 0.80,
        expected_return_floor: float = 0.0,
    ):
        self.pwin_floor = float(pwin_floor)
        self.pwin_cap = float(pwin_cap)
        self.expected_return_floor = float(expected_return_floor)

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(x)))

    def _global_pwin(self, v: dict) -> float:
        base = float(v.get("meta_p_win", 0.50) or 0.50)
        sig = float(v.get("signal_strength", 0.0) or 0.0)
        adx = float(v.get("meta_adx", 0.0) or 0.0)
        ema_gap = float(v.get("meta_ema_gap_pct", 0.0) or 0.0)
        side_align = float(v.get("ctx_side_backdrop_alignment", 0.0) or 0.0)

        p = base
        p += 0.030 * self._clip(sig, -1.0, 1.0)
        p += 0.010 * self._clip((adx - 18.0) / 20.0, -1.0, 1.0)
        p += 0.015 * self._clip(ema_gap / 0.01, 0.0, 1.0)
        p += 0.025 * self._clip(side_align, -1.0, 1.0)

        return self._clip(p, self.pwin_floor, self.pwin_cap)

    def _context_bonus(self, v: dict) -> tuple[float, dict]:
        strategy_id = str(v.get("strategy_id", "") or "")
        side = str(v.get("side", "flat") or "flat").lower()
        backdrop = str(v.get("ctx_backdrop", "neutral") or "neutral").lower()

        adx_pct = float(v.get("ctx_adx_pct", 0.5) or 0.5)
        atrp_pct = float(v.get("ctx_atrp_pct", 0.5) or 0.5)
        rsi_pct = float(v.get("ctx_rsi_pct", 0.5) or 0.5)
        range_exp_pct = float(v.get("ctx_range_expansion_pct", 0.5) or 0.5)
        trend_align = float(v.get("ctx_trend_alignment_score", 0.0) or 0.0)
        side_align = float(v.get("ctx_side_backdrop_alignment", 0.0) or 0.0)
        ret_7d = float(v.get("ctx_ret_7d", 0.0) or 0.0)
        ret_30d = float(v.get("ctx_ret_30d", 0.0) or 0.0)
        slope_ema = float(v.get("ctx_slope_ema_fast_24h", 0.0) or 0.0)

        bonus = 0.0

        strategy_bonus_map = {
            "dot_trend": 0.016,
            "xrp_trend": 0.012,
            "trx_trend": 0.010,
            "eth_trend": 0.008,
            "btc_trend": 0.006,
            "btc_trend_loose": 0.002,
            "link_trend": 0.004,
            "aave_trend": 0.003,
            "avax_trend": 0.004,
            "bnb_trend": 0.005,
        }
        bonus += float(strategy_bonus_map.get(strategy_id, 0.0))

        bonus += 0.020 * self._clip(side_align, -1.0, 1.0)
        bonus += 0.010 * self._clip(trend_align, -1.0, 1.0)
        bonus += 0.010 * self._clip((adx_pct - 0.5) / 0.5, -1.0, 1.0)
        bonus += 0.010 * self._clip((range_exp_pct - 0.5) / 0.5, -1.0, 1.0)
        bonus += 0.006 * self._clip(ret_7d / 0.08, -1.0, 1.0)
        bonus += 0.008 * self._clip(ret_30d / 0.20, -1.0, 1.0)
        bonus += 0.006 * self._clip(slope_ema / 0.03, -1.0, 1.0)

        if side == "long":
            bonus += 0.006 * self._clip((0.65 - rsi_pct) / 0.65, -1.0, 1.0)
            if backdrop == "bullish":
                bonus += 0.008
            elif backdrop == "bearish":
                bonus -= 0.010
        elif side == "short":
            bonus += 0.006 * self._clip((rsi_pct - 0.35) / 0.65, -1.0, 1.0)
            if backdrop == "bearish":
                bonus += 0.008
            elif backdrop == "bullish":
                bonus -= 0.010
        else:
            bonus -= 0.020

        penalty = 0.0
        penalty += 0.020 * float(v.get("flag_adx_below_min", 0.0) or 0.0)
        penalty += 0.012 * float(v.get("flag_ema_gap_below_min", 0.0) or 0.0)
        penalty += 0.010 * float(v.get("flag_atrp_low", 0.0) or 0.0)
        penalty += 0.010 * float(v.get("flag_adx_low", 0.0) or 0.0)
        penalty += 0.008 * float(v.get("flag_range_expansion_low", 0.0) or 0.0)
        penalty += 0.006 * float(v.get("flag_long_non_directional_bar", 0.0) or 0.0)
        penalty += 0.006 * float(v.get("flag_short_non_directional_bar", 0.0) or 0.0)
        penalty += 0.006 * float(v.get("flag_long_trend_misaligned", 0.0) or 0.0)
        penalty += 0.006 * float(v.get("flag_short_trend_misaligned", 0.0) or 0.0)
        penalty += 0.006 * float(v.get("flag_long_no_donchian_break", 0.0) or 0.0)
        penalty += 0.006 * float(v.get("flag_short_no_donchian_break", 0.0) or 0.0)
        penalty += 0.006 * float(v.get("flag_long_rsi_exhausted", 0.0) or 0.0)
        penalty += 0.006 * float(v.get("flag_short_rsi_exhausted", 0.0) or 0.0)
        penalty += 0.006 * float(v.get("flag_long_overextended", 0.0) or 0.0)
        penalty += 0.006 * float(v.get("flag_short_overextended", 0.0) or 0.0)

        bonus -= penalty

        return (
            float(bonus),
            {
                "strategy_bonus": float(strategy_bonus_map.get(strategy_id, 0.0)),
                "backdrop": backdrop,
                "trend_alignment_score": float(trend_align),
                "side_backdrop_alignment": float(side_align),
                "adx_pct": float(adx_pct),
                "atrp_pct": float(atrp_pct),
                "rsi_pct": float(rsi_pct),
                "range_expansion_pct": float(range_exp_pct),
                "ret_7d": float(ret_7d),
                "ret_30d": float(ret_30d),
                "slope_ema_fast_24h": float(slope_ema),
                "flag_penalty": float(penalty),
            },
        )

    def _expected_return(self, v: dict, p_win: float) -> float:
        post_ml = float(v.get("meta_post_ml_score", 0.0) or 0.0)
        comp = float(v.get("meta_competitive_score", 0.0) or 0.0)
        sig = float(v.get("signal_strength", 0.0) or 0.0)
        side_align = float(v.get("ctx_side_backdrop_alignment", 0.0) or 0.0)
        trend_align = float(v.get("ctx_trend_alignment_score", 0.0) or 0.0)
        adx_pct = float(v.get("ctx_adx_pct", 0.5) or 0.5)
        atrp_pct = float(v.get("ctx_atrp_pct", 0.5) or 0.5)
        tp_mult = float(v.get("ctx_tp_mult", 1.0) or 1.0)
        sl_mult = float(v.get("ctx_sl_mult", 1.0) or 1.0)

        er = max(post_ml, comp)
        er += 0.0005 * self._clip(sig, -1.0, 1.0)
        er += 0.0018 * max(0.0, p_win - 0.50)
        er += 0.0008 * self._clip(side_align, -1.0, 1.0)
        er += 0.0005 * self._clip(trend_align, -1.0, 1.0)
        er += 0.0005 * self._clip((adx_pct - 0.5) / 0.5, -1.0, 1.0)
        er += 0.0004 * self._clip((atrp_pct - 0.5) / 0.5, -1.0, 1.0)
        er += 0.0003 * self._clip(tp_mult - sl_mult, -1.0, 1.0)

        er -= 0.0005 * float(v.get("flag_adx_below_min", 0.0) or 0.0)
        er -= 0.0004 * float(v.get("flag_ema_gap_below_min", 0.0) or 0.0)
        er -= 0.0003 * float(v.get("flag_atrp_low", 0.0) or 0.0)
        er -= 0.0003 * float(v.get("flag_range_expansion_low", 0.0) or 0.0)

        return max(self.expected_return_floor, float(er))

    def predict_one(self, feature_row: FeatureRow) -> MetaScore:
        v = dict(feature_row.values or {})

        p_global = self._global_pwin(v)
        ctx_bonus, ctx_meta = self._context_bonus(v)
        p_final = self._clip(p_global + ctx_bonus, self.pwin_floor, self.pwin_cap)

        expected_return = self._expected_return(v, p_final)
        edge = max(0.0, p_final - 0.50)
        score = edge * max(0.0, expected_return)

        return MetaScore(
            ts=int(feature_row.ts),
            symbol=str(feature_row.symbol),
            strategy_id=str(feature_row.strategy_id),
            side=str(feature_row.side),
            p_win=float(p_final),
            expected_return=float(expected_return),
            score=float(score),
            model_meta={
                "model_family": "contextual_v3_asset_backdrop",
                "p_win_global": float(p_global),
                "context_bonus": float(ctx_bonus),
                **ctx_meta,
                "adx_below_min": bool(v.get("flag_adx_below_min", 0.0)),
                "ema_gap_below_min": bool(v.get("flag_ema_gap_below_min", 0.0)),
                "atrp_low": bool(v.get("flag_atrp_low", 0.0)),
                "adx_low": bool(v.get("flag_adx_low", 0.0)),
                "range_expansion_low": bool(v.get("flag_range_expansion_low", 0.0)),
                "expected_holding_bars": int(v.get("ctx_expected_holding_bars", 12.0) or 12.0),
                "tp_mult": float(v.get("ctx_tp_mult", 1.0) or 1.0),
                "sl_mult": float(v.get("ctx_sl_mult", 1.0) or 1.0),
                "time_stop_bars": int(v.get("ctx_time_stop_bars", 12.0) or 12.0),
                "exit_profile": str(v.get("ctx_exit_profile", "normal") or "normal"),
            },
        )

    def predict_many(self, feature_rows: list[FeatureRow]) -> list[MetaScore]:
        return [self.predict_one(fr) for fr in list(feature_rows or [])]
