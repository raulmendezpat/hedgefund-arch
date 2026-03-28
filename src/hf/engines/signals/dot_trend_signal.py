from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from hf.core.types import Candle, Signal


def _feat(c: Candle, key: str, default=None):
    try:
        if c.features is None:
            return default
        v = c.features.get(key, default)
        return default if v is None else v
    except Exception:
        return default


@dataclass
class DotTrendSignalEngine:
    """
    DOT trend engine especializado.

    Diseño:
    - evita reutilizar el engine genérico de BTC
    - exige estructura de tendencia más clara
    - controla ATRP para evitar zonas demasiado muertas o demasiado explosivas
    - deja trazabilidad explícita en signal.meta

    Timestamp lógico: 2026-03-28
    """

    adx_min: float = 26.0
    atrp_min: float = 0.0045
    atrp_max: float = 0.0300

    min_ema_sep_atr: float = 0.18
    min_ema_slope_atr: float = 0.030

    breakout_buffer_atr: float = 0.18

    pullback_max_atr: float = 0.65
    require_pullback_for_long: bool = False
    require_pullback_for_short: bool = False

    use_longs: bool = True
    use_shorts: bool = True

    def _build_flat(self, sym: str, meta: dict) -> Signal:
        return Signal(symbol=sym, side="flat", strength=0.0, meta=meta)

    def generate(self, candles: Dict[str, Candle]) -> Dict[str, Signal]:
        out: Dict[str, Signal] = {}

        for sym, c in candles.items():
            close = float(getattr(c, "close", 0.0) or 0.0)

            ema_fast = _feat(c, "ema_fast")
            ema_slow = _feat(c, "ema_slow")
            adx = _feat(c, "adx")
            atr = _feat(c, "atr")
            atrp = _feat(c, "atrp")

            ret_4h = _feat(c, "ret_4h_lag", 0.0)
            ret_12h = _feat(c, "ret_12h_lag", 0.0)
            ret_24h = _feat(c, "ret_24h_lag", 0.0)

            ema_gap = _feat(c, "ema_gap_fast_slow", 0.0)
            dist_close_ema_fast = _feat(c, "dist_close_ema_fast", 0.0)
            breakout_up = _feat(c, "breakout_distance_up", 0.0)
            breakout_down = _feat(c, "breakout_distance_down", 0.0)
            rolling_vol_24h = _feat(c, "rolling_vol_24h", 0.0)

            meta = {
                "engine": "dot_trend_signal",
                "strategy_family": "trend",
                "symbol": sym,
                "close": float(close),
                "ema_fast": float(ema_fast or 0.0),
                "ema_slow": float(ema_slow or 0.0),
                "adx": float(adx or 0.0),
                "atr": float(atr or 0.0),
                "atrp": float(atrp or 0.0),
                "ema_gap_fast_slow": float(ema_gap or 0.0),
                "dist_close_ema_fast": float(dist_close_ema_fast or 0.0),
                "ret_4h_lag": float(ret_4h or 0.0),
                "ret_12h_lag": float(ret_12h or 0.0),
                "ret_24h_lag": float(ret_24h or 0.0),
                "breakout_distance_up": float(breakout_up or 0.0),
                "breakout_distance_down": float(breakout_down or 0.0),
                "rolling_vol_24h": float(rolling_vol_24h or 0.0),
                "signal_notes": [],
            }

            if None in (ema_fast, ema_slow, adx, atr, atrp) or float(atr or 0.0) <= 0.0:
                meta["signal_notes"].append("missing_core_features")
                out[sym] = self._build_flat(sym, meta)
                continue

            ema_fast = float(ema_fast)
            ema_slow = float(ema_slow)
            adx = float(adx)
            atr = float(atr)
            atrp = float(atrp)
            breakout_up = float(breakout_up or 0.0)
            breakout_down = float(breakout_down or 0.0)
            dist_close_ema_fast = float(dist_close_ema_fast or 0.0)

            ema_sep_atr = abs(ema_fast - ema_slow) / atr if atr > 0 else 0.0
            ema_slope_atr = abs(float(ret_4h or 0.0)) / atrp if atrp > 0 else 0.0

            meta["ema_sep_atr"] = float(ema_sep_atr)
            meta["ema_slope_atr_proxy"] = float(ema_slope_atr)

            regime_ok = (
                adx >= float(self.adx_min)
                and atrp >= float(self.atrp_min)
                and atrp <= float(self.atrp_max)
                and ema_sep_atr >= float(self.min_ema_sep_atr)
                and ema_slope_atr >= float(self.min_ema_slope_atr)
            )

            if not regime_ok:
                if adx < float(self.adx_min):
                    meta["adx_below_min"] = True
                    meta["signal_notes"].append("adx_below_min")
                if atrp < float(self.atrp_min):
                    meta["atrp_low"] = True
                    meta["signal_notes"].append("atrp_low")
                if atrp > float(self.atrp_max):
                    meta["atrp_high"] = True
                    meta["signal_notes"].append("atrp_high")
                if ema_sep_atr < float(self.min_ema_sep_atr):
                    meta["ema_gap_below_min"] = True
                    meta["signal_notes"].append("ema_sep_atr_below_min")
                if ema_slope_atr < float(self.min_ema_slope_atr):
                    meta["signal_notes"].append("ema_slope_atr_below_min")
                out[sym] = self._build_flat(sym, meta)
                continue

            long_bias = (
                self.use_longs
                and ema_fast > ema_slow
                and float(ret_12h or 0.0) > -0.015
                and breakout_up <= float(self.breakout_buffer_atr) * atrp
            )

            short_bias = (
                self.use_shorts
                and ema_fast < ema_slow
                and float(ret_12h or 0.0) < 0.015
                and breakout_down >= -float(self.breakout_buffer_atr) * atrp
            )

            if self.require_pullback_for_long:
                long_bias = long_bias and abs(dist_close_ema_fast) <= float(self.pullback_max_atr) * atrp

            if self.require_pullback_for_short:
                short_bias = short_bias and abs(dist_close_ema_fast) <= float(self.pullback_max_atr) * atrp

            if long_bias and not short_bias:
                strength = min(
                    1.0,
                    max(
                        0.0,
                        0.30
                        + 0.25 * min(adx / max(self.adx_min, 1e-9), 2.0)
                        + 0.25 * min(ema_sep_atr / max(self.min_ema_sep_atr, 1e-9), 2.0)
                        + 0.20 * min(ema_slope_atr / max(self.min_ema_slope_atr, 1e-9), 2.0),
                    ),
                )
                meta["signal_notes"].append("dot_specialized_long")
                out[sym] = Signal(symbol=sym, side="long", strength=float(strength), meta=meta)
                continue

            if short_bias and not long_bias:
                strength = min(
                    1.0,
                    max(
                        0.0,
                        0.30
                        + 0.25 * min(adx / max(self.adx_min, 1e-9), 2.0)
                        + 0.25 * min(ema_sep_atr / max(self.min_ema_sep_atr, 1e-9), 2.0)
                        + 0.20 * min(ema_slope_atr / max(self.min_ema_slope_atr, 1e-9), 2.0),
                    ),
                )
                meta["signal_notes"].append("dot_specialized_short")
                out[sym] = Signal(symbol=sym, side="short", strength=float(strength), meta=meta)
                continue

            meta["signal_notes"].append("no_clean_dot_trend_setup")
            out[sym] = self._build_flat(sym, meta)

        return out
