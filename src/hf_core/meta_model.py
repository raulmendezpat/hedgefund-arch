from __future__ import annotations

from hf_core.contracts import FeatureRow, MetaScore


class MetaModel:
    def __init__(
        self,
        *,
        pwin_floor: float = 0.46,
        pwin_cap: float = 0.88,
        context_shrink_max: float = 0.85,
        expected_return_floor: float = 0.0,
    ):
        self.pwin_floor = float(pwin_floor)
        self.pwin_cap = float(pwin_cap)
        self.context_shrink_max = float(context_shrink_max)
        self.expected_return_floor = float(expected_return_floor)

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(x)))

    
    def _global_pwin(self, v: dict) -> float:
        p = float(v.get("meta_p_win", 0.50) or 0.50)
        sig = float(v.get("signal_strength", 0.0) or 0.0)
        p += 0.012 * self._clip(sig, -1.0, 1.0)
        return self._clip(p, self.pwin_floor, self.pwin_cap)

    def _strategy_side_bias(self, strategy_id: str, side: str) -> float:
        bias_map = {
            "eth_trend|short": 0.055,
            "avax_trend|short": 0.055,
            "xrp_trend|short": 0.050,
            "aave_trend|short": 0.050,
            "dot_trend|short": 0.050,
            "link_trend|short": 0.040,
            "btc_trend|short": 0.028,
            "btc_trend_loose|short": 0.028,
            "bnb_trend|short": 0.022,
            "trx_trend|short": 0.015,
            "trx_trend|long": 0.020,

            "bnb_trend|long": -0.020,
            "eth_trend|long": -0.035,
            "btc_trend|long": -0.035,
            "btc_trend_loose|long": -0.035,
        }
        return float(bias_map.get(f"{strategy_id}|{side}", 0.0))

    def _context_bonus(self, v: dict) -> tuple[float, float, dict]:
        strategy_id = str(v.get("strategy_id", "") or "")
        side = str(v.get("side", "flat") or "flat").lower()
        regime = str(v.get("portfolio_regime", "unknown") or "unknown").lower()

        conviction = float(v.get("portfolio_conviction", 0.0) or 0.0)
        breadth = float(v.get("portfolio_breadth", 0.0) or 0.0)
        avg_pwin = float(v.get("portfolio_avg_pwin", 0.50) or 0.50)
        avg_strength = float(v.get("portfolio_avg_strength", 0.0) or 0.0)
        avg_atrp = float(v.get("portfolio_avg_atrp", 0.0) or 0.0)

        bonus = 0.0

        strategy_side_bias = self._strategy_side_bias(strategy_id, side)
        bonus += strategy_side_bias

        if regime == "defensive":
            bonus += 0.045 * self._clip(conviction, 0.0, 1.0)
            bonus += 0.030 * self._clip(avg_pwin - 0.50, -1.0, 1.0)
            bonus -= 0.025 * self._clip(max(0.0, breadth - 4.0) / 5.0, 0.0, 1.0)
        elif regime == "normal":
            bonus += 0.030 * self._clip(avg_strength, 0.0, 1.0)
            bonus += 0.020 * self._clip(avg_pwin - 0.50, -1.0, 1.0)

        # algo de penalización por ruido excesivo
        bonus -= 0.015 * self._clip(avg_atrp / 0.03, 0.0, 1.0)

        context_conf = 0.0
        context_conf += 0.40 * self._clip(conviction, 0.0, 1.0)
        context_conf += 0.25 * self._clip(avg_strength, 0.0, 1.0)
        context_conf += 0.20 * self._clip((avg_pwin - 0.50) / 0.20, 0.0, 1.0)
        context_conf += 0.15 * self._clip(min(breadth, 8.0) / 8.0, 0.0, 1.0)

        raw_meta_p = float(v.get("meta_p_win", 0.50) or 0.50)
        overconf = self._clip((raw_meta_p - 0.78) / 0.10, 0.0, 1.0)
        context_conf *= (1.0 - 0.35 * overconf)

        shrink_weight = self.context_shrink_max * self._clip(context_conf, 0.0, 1.0)

        return (
            float(bonus),
            float(shrink_weight),
            {
                "strategy_side_bias": float(strategy_side_bias),
                "regime": regime,
                "conviction": float(conviction),
                "avg_pwin": float(avg_pwin),
                "avg_strength": float(avg_strength),
                "avg_atrp": float(avg_atrp),
                "overconfidence_penalty": float(overconf),
                "shrink_weight": float(shrink_weight),
            },
        )

    def _expected_return(self, v: dict, p_win: float) -> float:
        post_ml = float(v.get("meta_post_ml_score", 0.0) or 0.0)
        comp = float(v.get("meta_competitive_score", 0.0) or 0.0)
        sig = float(v.get("signal_strength", 0.0) or 0.0)
        side = str(v.get("side", "flat") or "flat").lower()

        base = 0.55 * max(0.0, post_ml) + 0.45 * max(0.0, comp)
        base = min(base, 0.0040)

        er = base
        er += 0.00045 * self._clip(sig, -1.0, 1.0)
        er += 0.0060 * max(0.0, p_win - 0.50)

        if p_win >= 0.84:
            er *= 0.72
        elif p_win >= 0.78:
            er *= 0.85

        if side == "long":
            er *= 0.95

        return max(self.expected_return_floor, float(er))

    def predict_one(self, feature_row: FeatureRow) -> MetaScore:
        v = dict(feature_row.values or {})

        p_global = self._global_pwin(v)
        ctx_bonus, shrink_w, ctx_meta = self._context_bonus(v)

        p_context_raw = self._clip(p_global + ctx_bonus, self.pwin_floor, self.pwin_cap)
        p_final = self._clip(
            ((1.0 - shrink_w) * p_global) + (shrink_w * p_context_raw),
            self.pwin_floor,
            self.pwin_cap,
        )

        expected_return = self._expected_return(v, p_final)
        edge = max(0.0, p_final - 0.50)

        # menos colapso cerca de 0.50, pero sigue siendo conservador
        score = (edge ** 0.90) * max(0.0, expected_return)

        if p_final >= 0.84:
            score *= 0.80
        elif p_final >= 0.78:
            score *= 0.90

        return MetaScore(
            ts=int(feature_row.ts),
            symbol=str(feature_row.symbol),
            strategy_id=str(feature_row.strategy_id),
            side=str(feature_row.side),
            p_win=float(p_final),
            expected_return=float(expected_return),
            score=float(score),
            model_meta={
                "model_family": "contextual_bootstrap_recalibrated_v2",
                "p_win_global": float(p_global),
                "p_win_context_raw": float(p_context_raw),
                "context_bonus": float(ctx_bonus),
                "shrink_weight": float(shrink_w),
                **ctx_meta,
            },
        )

    def predict_many(self, feature_rows: list[FeatureRow]) -> list[MetaScore]:
        return [self.predict_one(fr) for fr in list(feature_rows or [])]
