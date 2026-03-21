from __future__ import annotations

from hf_core.contracts import FeatureRow, MetaScore


class MetaModel:
    def __init__(
        self,
        *,
        pwin_floor: float = 0.35,
        pwin_cap: float = 0.75,
        expected_return_floor: float = 0.0,
    ):
        self.pwin_floor = float(pwin_floor)
        self.pwin_cap = float(pwin_cap)
        self.expected_return_floor = float(expected_return_floor)

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(x)))

    @staticmethod
    def _sgn_side_bonus(side: str) -> float:
        side = str(side or "flat").lower()
        if side == "long":
            return 0.004
        if side == "short":
            return 0.002
        return -0.020

    def _global_pwin(self, v: dict) -> float:
        base = float(v.get("meta_p_win", 0.50) or 0.50)
        sig = float(v.get("signal_strength", 0.0) or 0.0)
        adx = float(v.get("meta_adx", 0.0) or 0.0)
        ema_gap = float(v.get("meta_ema_gap_pct", 0.0) or 0.0)

        p = base
        p += 0.030 * self._clip(sig, -1.0, 1.0)
        p += 0.010 * self._clip((adx - 18.0) / 20.0, -1.0, 1.0)
        p += 0.015 * self._clip(ema_gap / 0.01, 0.0, 1.0)
        p += self._sgn_side_bonus(str(v.get("side", "flat")))

        return self._clip(p, self.pwin_floor, self.pwin_cap)

    def _context_bonus(self, v: dict) -> tuple[float, dict]:
        strategy_id = str(v.get("strategy_id", "") or "")
        side = str(v.get("side", "flat") or "flat").lower()

        bonus = 0.0

        strategy_bonus_map = {
            "dot_trend": 0.018,
            "xrp_trend": 0.014,
            "trx_trend": 0.010,
            "eth_trend": 0.006,
            "btc_trend": 0.004,
            "btc_trend_loose": -0.004,
            "link_trend": -0.003,
            "aave_trend": -0.004,
            "avax_trend": -0.003,
            "bnb_trend": -0.001,
        }
        bonus += float(strategy_bonus_map.get(strategy_id, 0.0))

        adx = float(v.get("meta_adx", 0.0) or 0.0)
        atrp = float(v.get("meta_atrp", 0.0) or 0.0)
        rsi = float(v.get("meta_rsi", 0.0) or 0.0)
        bb_width = float(v.get("meta_bb_width", 0.0) or 0.0)
        range_exp = float(v.get("meta_range_expansion", 0.0) or 0.0)
        ema_gap = float(v.get("meta_ema_gap_pct", 0.0) or 0.0)
        sig = float(v.get("signal_strength", 0.0) or 0.0)

        bonus += 0.010 * self._clip((adx - 20.0) / 20.0, -1.0, 1.0)
        bonus += 0.010 * self._clip((atrp - 0.008) / 0.020, -1.0, 1.0)
        bonus += 0.008 * self._clip(range_exp / 1.5, 0.0, 1.0)
        bonus += 0.012 * self._clip(ema_gap / 0.01, 0.0, 1.0)
        bonus += 0.008 * self._clip(sig, -1.0, 1.0)

        if side == "long":
            bonus += 0.004 * self._clip((60.0 - rsi) / 30.0, -1.0, 1.0)
        elif side == "short":
            bonus += 0.004 * self._clip((rsi - 40.0) / 30.0, -1.0, 1.0)

        # penalize weak regime flags
        penalty = 0.0
        penalty += 0.018 * float(v.get("flag_adx_below_min", 0.0) or 0.0)
        penalty += 0.012 * float(v.get("flag_ema_gap_below_min", 0.0) or 0.0)
        penalty += 0.010 * float(v.get("flag_atrp_low", 0.0) or 0.0)
        penalty += 0.010 * float(v.get("flag_adx_low", 0.0) or 0.0)
        penalty += 0.008 * float(v.get("flag_range_expansion_low", 0.0) or 0.0)
        penalty += 0.008 * float(v.get("flag_long_non_directional_bar", 0.0) or 0.0)
        penalty += 0.008 * float(v.get("flag_short_non_directional_bar", 0.0) or 0.0)
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
                "adx": float(adx),
                "atrp": float(atrp),
                "rsi": float(rsi),
                "bb_width": float(bb_width),
                "range_expansion": float(range_exp),
                "ema_gap_pct": float(ema_gap),
                "flag_penalty": float(penalty),
            },
        )

    def _expected_return(self, v: dict, p_win: float) -> float:
        post_ml = float(v.get("meta_post_ml_score", 0.0) or 0.0)
        comp = float(v.get("meta_competitive_score", 0.0) or 0.0)
        sig = float(v.get("signal_strength", 0.0) or 0.0)
        adx = float(v.get("meta_adx", 0.0) or 0.0)
        atrp = float(v.get("meta_atrp", 0.0) or 0.0)
        ema_gap = float(v.get("meta_ema_gap_pct", 0.0) or 0.0)

        er = max(post_ml, comp)
        er += 0.0005 * self._clip(sig, -1.0, 1.0)
        er += 0.0015 * max(0.0, p_win - 0.50)
        er += 0.0006 * self._clip((adx - 18.0) / 20.0, -1.0, 1.0)
        er += 0.0006 * self._clip((atrp - 0.008) / 0.020, -1.0, 1.0)
        er += 0.0008 * self._clip(ema_gap / 0.01, 0.0, 1.0)

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
                "model_family": "contextual_v2",
                "p_win_global": float(p_global),
                "context_bonus": float(ctx_bonus),
                **ctx_meta,
                "adx_below_min": bool(v.get("flag_adx_below_min", 0.0)),
                "ema_gap_below_min": bool(v.get("flag_ema_gap_below_min", 0.0)),
                "atrp_low": bool(v.get("flag_atrp_low", 0.0)),
                "adx_low": bool(v.get("flag_adx_low", 0.0)),
                "range_expansion_low": bool(v.get("flag_range_expansion_low", 0.0)),
            },
        )

    def predict_many(self, feature_rows: list[FeatureRow]) -> list[MetaScore]:
        return [self.predict_one(fr) for fr in list(feature_rows or [])]
