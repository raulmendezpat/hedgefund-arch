from __future__ import annotations

from dataclasses import dataclass


def _f(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))


@dataclass(frozen=True)
class MathPWinV2Config:
    p_floor: float = 0.45
    p_cap: float = 0.88


class MathPWinV2:
    """
    Calibrador puramente matemático / heurístico.
    No usa ML.
    Parte de math_v1 y corrige con reglas empíricas observadas en los trades.
    """

    def __init__(self, cfg: MathPWinV2Config | None = None):
        self.cfg = cfg or MathPWinV2Config()

    def _strategy_side_bias(self, strategy_id: str, side: str) -> float:
        key = f"{str(strategy_id or '').lower()}|{str(side or '').lower()}"
        bias = {
            "btc_trend|short": 0.010,
            "btc_trend_loose|short": -0.006,
            "link_trend|short": 0.004,
            "aave_trend|short": -0.018,
            "xrp_trend|short": -0.010,
            "bnb_trend|short": -0.008,
            "sol_trend_pullback|long": -0.004,
            "eth_trend|short": 0.006,
        }
        return float(bias.get(key, 0.0))

    def predict_from_meta(self, meta: dict) -> float:
        m = dict(meta or {})

        # Base segura:
        # - preferir p_win_math_v1 si existe
        # - NO reutilizar p_win porque ya puede venir sobrescrito / colapsado
        p0 = _f(m.get("p_win_math_v1", 0.5), 0.5)

        strategy_id = str(m.get("strategy_id", "") or "").lower()
        side = str(m.get("side", "flat") or "flat").lower()

        adx = _f(m.get("adx", 0.0), 0.0)
        atrp = _f(m.get("atrp", 0.0), 0.0)
        rsi = _f(m.get("rsi", 50.0), 50.0)
        signal_strength = _f(m.get("signal_strength", 0.0), 0.0)
        policy_score = _f(m.get("policy_score", 0.0), 0.0)
        expected_return = _f(m.get("expected_return", 0.0), 0.0)
        conviction = _f(m.get("portfolio_conviction", 0.0), 0.0)

        p = float(p0)

        # 1) sesgo por estrategia|lado
        p += self._strategy_side_bias(strategy_id, side)

        # 2) ADX: BTC short y LINK short tienden a funcionar mejor con ADX alto
        if strategy_id in {"btc_trend", "btc_trend_loose", "link_trend"} and side == "short":
            if adx >= 35.0:
                p += 0.015
            elif adx >= 28.0:
                p += 0.008
            elif adx < 20.0:
                p -= 0.010

        # 3) ATRP: en varias estrategias perder sube cuando ATRP está alto
        if atrp >= 0.016:
            p -= 0.018
        elif atrp >= 0.0135:
            p -= 0.010
        elif atrp <= 0.009:
            p += 0.006

        # 4) policy_score: señal pequeña pero útil
        if policy_score >= 0.00021:
            p += 0.008
        elif policy_score >= 0.00019:
            p += 0.004
        elif policy_score <= 0.00015:
            p -= 0.006

        # 5) expected_return: usarlo solo como ajuste suave
        if expected_return >= 0.00072:
            p += 0.008
        elif expected_return >= 0.00069:
            p += 0.004
        elif expected_return <= 0.00058:
            p -= 0.008

        # 6) conviction del portafolio
        if conviction >= 0.65:
            p += 0.006
        elif conviction <= 0.56:
            p -= 0.006

        # 7) reglas específicas por estrategia observadas en los análisis
        if strategy_id == "aave_trend" and side == "short":
            if atrp >= 0.014:
                p -= 0.012
            if adx >= 33.0:
                p += 0.006

        if strategy_id == "sol_trend_pullback" and side == "long":
            if atrp >= 0.014:
                p -= 0.010
            if 20.0 <= adx <= 25.0:
                p += 0.006
            if adx >= 28.0:
                p -= 0.004

        if strategy_id == "xrp_trend" and side == "short":
            if adx <= 18.5:
                p += 0.006
            elif adx >= 27.0:
                p -= 0.010

        # 8) reglas útiles para longs SOL
        if strategy_id == "sol_bbrsi" and side == "long":
            if rsi <= 32.0:
                p += 0.035
            elif rsi <= 38.0:
                p += 0.022
            elif rsi >= 55.0:
                p -= 0.012

            if 16.0 <= adx <= 28.0:
                p += 0.010
            elif adx >= 38.0:
                p -= 0.010

            if 0.007 <= atrp <= 0.0135:
                p += 0.010

            if signal_strength >= 1.0:
                p += 0.012
            elif signal_strength >= 0.7:
                p += 0.006

        if strategy_id == "sol_trend_pullback" and side == "long":
            if 45.0 <= rsi <= 52.0:
                p += 0.028
            elif 42.0 <= rsi < 45.0:
                p += 0.014
            elif rsi >= 58.0:
                p -= 0.010

            if 18.0 <= adx <= 27.0:
                p += 0.012
            elif adx >= 34.0:
                p -= 0.010

            if 0.009 <= atrp <= 0.0135:
                p += 0.010
            elif atrp >= 0.016:
                p -= 0.012

            if signal_strength >= 1.4:
                p += 0.016
            elif signal_strength >= 1.0:
                p += 0.010
            elif signal_strength >= 0.75:
                p += 0.004

            if conviction >= 0.62:
                p += 0.008

        return _clip(p, self.cfg.p_floor, self.cfg.p_cap)
