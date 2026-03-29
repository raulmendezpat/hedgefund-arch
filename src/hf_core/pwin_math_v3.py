from __future__ import annotations


class MathPWinV3:
    """
    p_win matemático v3:
    - no desactiva ningún activo
    - usa calibración por strategy_id|side
    - score por bandas favorables de ADX/ATRP
    - ajuste suave por expected_return / policy_score / portfolio_conviction
    - fallback global si no existe regla específica
    """

    def __init__(self) -> None:
        self.global_cfg = {
            "floor": 0.45,
            "cap": 0.67,
            "base": 0.505,
            "adx_center": 28.0,
            "adx_halfwidth": 18.0,
            "atrp_center": 0.010,
            "atrp_halfwidth": 0.008,
            "w_adx": 0.060,
            "w_atrp": 0.055,
            "w_er": 0.035,
            "w_policy": 0.025,
            "w_conviction": 0.020,
            "w_strength": 0.018,
            "high_atrp_penalty_start": 0.016,
            "high_atrp_penalty_scale": 0.035,
        }

        self.by_key = {
            # Trend shorts: mantener todos los assets
            "btc_trend|short": {
                "floor": 0.47, "cap": 0.66, "base": 0.525,
                "adx_center": 36.0, "adx_halfwidth": 14.0,
                "atrp_center": 0.0082, "atrp_halfwidth": 0.0035,
                "w_adx": 0.075, "w_atrp": 0.050, "w_er": 0.040,
                "w_policy": 0.028, "w_conviction": 0.018, "w_strength": 0.012,
                "high_atrp_penalty_start": 0.0105, "high_atrp_penalty_scale": 0.040,
            },
            "btc_trend_loose|short": {
                "floor": 0.46, "cap": 0.64, "base": 0.515,
                "adx_center": 28.0, "adx_halfwidth": 16.0,
                "atrp_center": 0.0085, "atrp_halfwidth": 0.0040,
                "w_adx": 0.060, "w_atrp": 0.045, "w_er": 0.038,
                "w_policy": 0.022, "w_conviction": 0.015, "w_strength": 0.010,
                "high_atrp_penalty_start": 0.0115, "high_atrp_penalty_scale": 0.035,
            },
            "link_trend|short": {
                "floor": 0.46, "cap": 0.64, "base": 0.522,
                "adx_center": 38.0, "adx_halfwidth": 16.0,
                "atrp_center": 0.0115, "atrp_halfwidth": 0.0048,
                "w_adx": 0.070, "w_atrp": 0.040, "w_er": 0.038,
                "w_policy": 0.026, "w_conviction": 0.016, "w_strength": 0.010,
                "high_atrp_penalty_start": 0.0140, "high_atrp_penalty_scale": 0.032,
            },
            "bnb_trend|short": {
                "floor": 0.46, "cap": 0.63, "base": 0.518,
                "adx_center": 25.0, "adx_halfwidth": 15.0,
                "atrp_center": 0.0078, "atrp_halfwidth": 0.0030,
                "w_adx": 0.050, "w_atrp": 0.050, "w_er": 0.034,
                "w_policy": 0.020, "w_conviction": 0.014, "w_strength": 0.010,
                "high_atrp_penalty_start": 0.0098, "high_atrp_penalty_scale": 0.038,
            },
            "xrp_trend|short": {
                "floor": 0.46, "cap": 0.63, "base": 0.520,
                "adx_center": 20.0, "adx_halfwidth": 10.0,
                "atrp_center": 0.0112, "atrp_halfwidth": 0.0038,
                "w_adx": 0.045, "w_atrp": 0.040, "w_er": 0.034,
                "w_policy": 0.024, "w_conviction": 0.014, "w_strength": 0.010,
                "high_atrp_penalty_start": 0.0138, "high_atrp_penalty_scale": 0.032,
            },
            "eth_trend|short": {
                "floor": 0.46, "cap": 0.63, "base": 0.518,
                "adx_center": 24.0, "adx_halfwidth": 14.0,
                "atrp_center": 0.0085, "atrp_halfwidth": 0.0030,
                "w_adx": 0.050, "w_atrp": 0.045, "w_er": 0.030,
                "w_policy": 0.020, "w_conviction": 0.012, "w_strength": 0.010,
                "high_atrp_penalty_start": 0.0105, "high_atrp_penalty_scale": 0.032,
            },
            "aave_trend|short": {
                "floor": 0.45, "cap": 0.62, "base": 0.514,
                "adx_center": 34.0, "adx_halfwidth": 16.0,
                "atrp_center": 0.0090, "atrp_halfwidth": 0.0040,
                "w_adx": 0.060, "w_atrp": 0.060, "w_er": 0.032,
                "w_policy": 0.020, "w_conviction": 0.014, "w_strength": 0.010,
                "high_atrp_penalty_start": 0.0120, "high_atrp_penalty_scale": 0.055,
            },

            # Pullback long
            "sol_trend_pullback|long": {
                "floor": 0.46, "cap": 0.60, "base": 0.505,
                "adx_center": 23.0, "adx_halfwidth": 9.0,
                "atrp_center": 0.0110, "atrp_halfwidth": 0.0030,
                "w_adx": 0.045, "w_atrp": 0.055, "w_er": 0.025,
                "w_policy": 0.018, "w_conviction": 0.010, "w_strength": 0.012,
                "high_atrp_penalty_start": 0.0135, "high_atrp_penalty_scale": 0.045,
            },
        }

    @staticmethod
    def _f(x, default: float = 0.0) -> float:
        try:
            if x is None:
                return float(default)
            return float(x)
        except Exception:
            return float(default)

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        x = float(x)
        if x < lo:
            return float(lo)
        if x > hi:
            return float(hi)
        return float(x)

    @staticmethod
    def _band_score(value: float, center: float, halfwidth: float) -> float:
        hw = max(float(halfwidth), 1e-9)
        dist = abs(float(value) - float(center)) / hw
        score = 1.0 - dist
        if score < -1.0:
            return -1.0
        if score > 1.0:
            return 1.0
        return float(score)

    def _cfg_for(self, strategy_id: str, side: str) -> dict:
        key = f"{strategy_id}|{side}"
        cfg = dict(self.global_cfg)
        cfg.update(self.by_key.get(key, {}))
        return cfg

    def predict_from_meta(self, meta: dict) -> float:
        m = dict(meta or {})

        strategy_id = str(m.get("strategy_id", "") or "")
        side = str(m.get("side", "flat") or "flat").lower()
        cfg = self._cfg_for(strategy_id, side)

        adx = self._f(m.get("adx"), 0.0)
        atrp = self._f(m.get("atrp"), 0.0)
        expected_return = self._f(m.get("expected_return"), 0.0)
        policy_score = self._f(m.get("policy_score"), 0.0)
        conviction = self._f(m.get("portfolio_conviction"), 0.0)
        signal_strength = self._f(m.get("signal_strength"), 0.0)

        p = float(cfg["base"])

        adx_band = self._band_score(adx, cfg["adx_center"], cfg["adx_halfwidth"])
        atrp_band = self._band_score(atrp, cfg["atrp_center"], cfg["atrp_halfwidth"])

        p += float(cfg["w_adx"]) * adx_band
        p += float(cfg["w_atrp"]) * atrp_band

        er_norm = self._clip(expected_return / 0.0010, -1.0, 1.0)
        pol_norm = self._clip(policy_score / 0.00025, -1.0, 1.0)
        conv_norm = self._clip((conviction - 0.50) / 0.20, -1.0, 1.0)
        str_norm = self._clip((signal_strength - 1.0) / 0.75, -1.0, 1.0)

        p += float(cfg["w_er"]) * er_norm
        p += float(cfg["w_policy"]) * pol_norm
        p += float(cfg["w_conviction"]) * conv_norm
        p += float(cfg["w_strength"]) * str_norm

        high_atrp_start = float(cfg["high_atrp_penalty_start"])
        if atrp > high_atrp_start:
            excess = (atrp - high_atrp_start) / max(high_atrp_start, 1e-9)
            p -= float(cfg["high_atrp_penalty_scale"]) * self._clip(excess, 0.0, 1.0)

        if side == "flat":
            p = min(p, 0.50)

        return self._clip(p, float(cfg["floor"]), float(cfg["cap"]))
