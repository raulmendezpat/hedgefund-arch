from __future__ import annotations

from hf_core.contracts import FeatureRow, MetaScore


class MetaModel:
    def __init__(
        self,
        *,
        pwin_floor: float = 0.46,
        pwin_cap: float = 0.62,
        context_shrink_max: float = 0.22,
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

        # Menos agresivo que antes
        p += 0.012 * self._clip(sig, -1.0, 1.0)

        return self._clip(p, self.pwin_floor, self.pwin_cap)

    def _strategy_side_bias(self, strategy_id: str, side: str) -> float:
        # Biases tácticos basados en attribution reciente.
        bias_map = {
            "eth_trend|short": 0.010,
            "avax_trend|short": 0.010,
            "xrp_trend|short": 0.009,
            "aave_trend|short": 0.009,
            "dot_trend|short": 0.010,
            "link_trend|short": 0.007,
            "btc_trend|short": 0.004,
            "btc_trend_loose|short": 0.004,
            "bnb_trend|short": 0.003,
            "trx_trend|short": 0.002,
            "trx_trend|long": 0.004,

            # Longs no bloqueados pero aún débiles
            "bnb_trend|long": -0.004,
            "eth_trend|long": -0.006,
            "btc_trend|long": -0.006,
            "btc_trend_loose|long": -0.006,
        }
        return float(bias_map.get(f"{strategy_id}|{side}", 0.0))

    def _context_bonus(self, v: dict) -> tuple[float, float, dict]:
        strategy_id = str(v.get("strategy_id", "") or "")
        side = str(v.get("side", "flat") or "flat").lower()
        regime = str(v.get("portfolio_regime", "unknown") or "unknown").lower()

        conviction = float(v.get("portfolio_conviction", 0.0) or 0.0)
        breadth = float(v.get("portfolio_breadth", 0.0) or 0.0)
        avg_pwin = float(v.get("portfolio_avg_pwin", 0.0) or 0.0)
        avg_strength = float(v.get("portfolio_avg_strength", 0.0) or 0.0)

        bonus = 0.0

        # Sesgo táctico por strategy+side
        strategy_side_bias = self._strategy_side_bias(strategy_id, side)
        bonus += strategy_side_bias

        # Contexto portfolio suavizado
        if regime == "defensive":
            bonus += 0.004 * self._clip(conviction, 0.0, 1.0)
            bonus -= 0.004 * self._clip(max(0.0, breadth - 3.0) / 4.0, 0.0, 1.0)
        elif regime == "normal":
            bonus += 0.002 * self._clip(avg_strength, 0.0, 1.0)

        bonus += 0.006 * self._clip(avg_pwin - 0.50, -1.0, 1.0)

        # shrink más bajo
        context_conf = 0.0
        context_conf += 0.35 * self._clip(conviction, 0.0, 1.0)
        context_conf += 0.25 * self._clip(avg_strength, 0.0, 1.0)
        context_conf += 0.20 * self._clip(avg_pwin, 0.0, 1.0)

        # penalizar sobreconfianza contextual
        raw_meta_p = float(v.get("meta_p_win", 0.50) or 0.50)
        overconf = self._clip((raw_meta_p - 0.58) / 0.08, 0.0, 1.0)
        context_conf *= (1.0 - 0.55 * overconf)

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
                "overconfidence_penalty": float(overconf),
                "shrink_weight": float(shrink_weight),
            },
        )

    def _expected_return(self, v: dict, p_win: float) -> float:
        post_ml = float(v.get("meta_post_ml_score", 0.0) or 0.0)
        comp = float(v.get("meta_competitive_score", 0.0) or 0.0)
        sig = float(v.get("signal_strength", 0.0) or 0.0)
        side = str(v.get("side", "flat") or "flat").lower()

        # Más conservador y con saturación
        base = 0.55 * max(0.0, post_ml) + 0.45 * max(0.0, comp)
        base = min(base, 0.0015)

        er = base
        er += 0.00018 * self._clip(sig, -1.0, 1.0)
        er += 0.0009 * max(0.0, p_win - 0.50)

        # penalización de cola alta
        if p_win >= 0.595:
            er *= 0.55
        elif p_win >= 0.585:
            er *= 0.72

        # leve sesgo contra longs todavía
        if side == "long":
            er *= 0.92

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

        # score menos explosivo
        score = (edge ** 1.15) * max(0.0, expected_return)

        # castigo adicional a extremos
        if p_final >= 0.595:
            score *= 0.45
        elif p_final >= 0.585:
            score *= 0.70

        return MetaScore(
            ts=int(feature_row.ts),
            symbol=str(feature_row.symbol),
            strategy_id=str(feature_row.strategy_id),
            side=str(feature_row.side),
            p_win=float(p_final),
            expected_return=float(expected_return),
            score=float(score),
            model_meta={
                "model_family": "contextual_bootstrap_recalibrated",
                "p_win_global": float(p_global),
                "p_win_context_raw": float(p_context_raw),
                "context_bonus": float(ctx_bonus),
                "shrink_weight": float(shrink_w),
                **ctx_meta,
            },
        )

    def predict_many(self, feature_rows: list[FeatureRow]) -> list[MetaScore]:
        return [self.predict_one(fr) for fr in list(feature_rows or [])]
