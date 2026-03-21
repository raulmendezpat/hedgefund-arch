from __future__ import annotations

from hf_core.contracts import FeatureRow, MetaScore


class MetaModel:
    def __init__(
        self,
        *,
        pwin_floor: float = 0.35,
        pwin_cap: float = 0.75,
        context_shrink_max: float = 0.35,
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
        p += 0.03 * self._clip(sig, -1.0, 1.0)
        return self._clip(p, self.pwin_floor, self.pwin_cap)

    def _context_bonus(self, v: dict) -> tuple[float, float, dict]:
        strategy_id = str(v.get("strategy_id", "") or "")
        side = str(v.get("side", "flat") or "flat").lower()
        regime = str(v.get("portfolio_regime", "unknown") or "unknown").lower()

        conviction = float(v.get("portfolio_conviction", 0.0) or 0.0)
        breadth = float(v.get("portfolio_breadth", 0.0) or 0.0)
        avg_pwin = float(v.get("portfolio_avg_pwin", 0.0) or 0.0)
        avg_strength = float(v.get("portfolio_avg_strength", 0.0) or 0.0)

        bonus = 0.0
        strategy_bonus_map = {
            "dot_trend": 0.025,
            "xrp_trend": 0.018,
            "trx_trend": 0.015,
            "eth_trend": 0.006,
            "btc_trend": 0.000,
            "btc_trend_loose": -0.010,
            "link_trend": -0.008,
            "aave_trend": -0.010,
            "avax_trend": -0.008,
            "bnb_trend": -0.004,
        }
        bonus += float(strategy_bonus_map.get(strategy_id, 0.0))

        if side == "long":
            bonus += 0.004
        elif side == "short":
            bonus -= 0.002

        if regime == "defensive":
            bonus += 0.010 * self._clip(conviction, 0.0, 1.0)
            bonus -= 0.008 * self._clip(max(0.0, breadth - 3.0) / 4.0, 0.0, 1.0)
        elif regime == "normal":
            bonus += 0.004 * self._clip(avg_strength, 0.0, 1.0)

        bonus += 0.015 * self._clip(avg_pwin - 0.50, -1.0, 1.0)

        context_conf = 0.0
        context_conf += 0.40 * self._clip(conviction, 0.0, 1.0)
        context_conf += 0.30 * self._clip(avg_strength, 0.0, 1.0)
        context_conf += 0.30 * self._clip(avg_pwin, 0.0, 1.0)

        shrink_weight = self.context_shrink_max * self._clip(context_conf, 0.0, 1.0)

        return (
            float(bonus),
            float(shrink_weight),
            {
                "strategy_bonus": float(strategy_bonus_map.get(strategy_id, 0.0)),
                "regime": regime,
                "conviction": float(conviction),
                "avg_pwin": float(avg_pwin),
                "avg_strength": float(avg_strength),
                "shrink_weight": float(shrink_weight),
            },
        )

    def _expected_return(self, v: dict, p_win: float) -> float:
        post_ml = float(v.get("meta_post_ml_score", 0.0) or 0.0)
        comp = float(v.get("meta_competitive_score", 0.0) or 0.0)
        sig = float(v.get("signal_strength", 0.0) or 0.0)

        er = max(post_ml, comp)
        er += 0.0005 * self._clip(sig, -1.0, 1.0)
        er += 0.0020 * max(0.0, p_win - 0.50)
        return max(self.expected_return_floor, float(er))

    def predict_one(self, feature_row: FeatureRow) -> MetaScore:
        v = dict(feature_row.values or {})

        p_global = self._global_pwin(v)
        ctx_bonus, shrink_w, ctx_meta = self._context_bonus(v)

        p_context_raw = self._clip(p_global + ctx_bonus, self.pwin_floor, self.pwin_cap)
        p_final = self._clip(((1.0 - shrink_w) * p_global) + (shrink_w * p_context_raw), self.pwin_floor, self.pwin_cap)

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
                "model_family": "contextual_bootstrap",
                "p_win_global": float(p_global),
                "p_win_context_raw": float(p_context_raw),
                "context_bonus": float(ctx_bonus),
                "shrink_weight": float(shrink_w),
                **ctx_meta,
            },
        )

    def predict_many(self, feature_rows: list[FeatureRow]) -> list[MetaScore]:
        return [self.predict_one(fr) for fr in list(feature_rows or [])]
