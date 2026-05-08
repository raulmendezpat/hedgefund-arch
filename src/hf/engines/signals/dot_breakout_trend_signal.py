from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from hf.core.types import Candle, Signal


def _num(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _feat(candle: Candle, key: str, default: float | None = None) -> float | None:
    """
    Match the signal-engine pattern used by existing engines.

    Runtime Candle exposes base OHLCV attributes plus a features dict. Most
    trend/context indicators live under candle.features, not as top-level attrs.
    """
    features = getattr(candle, "features", None) or {}
    if key in features:
        return _num(features.get(key), 0.0)
    if hasattr(candle, key):
        return _num(getattr(candle, key), 0.0)
    return default


@dataclass
class DotBreakoutTrendSignalEngine:
    """
    Experimental DOT-specific long breakout/momentum engine.

    Stateless by design. It consumes precomputed runtime features from
    Candle.features, following the same contract as DotTrendSignalEngine.

    Output:
    generate(candles) -> Dict[str, Signal]
    """

    adx_min: float = 8.0
    atrp_min: float = 0.0005
    atrp_max: float = 0.12

    min_ema_gap: float = -0.006
    min_ret_4h: float = -0.030
    min_ret_12h: float = -0.060

    breakout_up_max: float = 0.050
    pullback_abs_max: float = 0.120
    rolling_vol_24h_min: float = 0.0

    use_longs: bool = True
    use_shorts: bool = False

    strength_base: float = 1.0
    only_if_symbol_contains: str = "DOT"
    emit_flat: bool = True

    def _flat(self, sym: str, reason: str, meta: dict) -> Signal:
        meta = dict(meta)
        meta["reason"] = reason
        meta.setdefault("engine", "dot_breakout_trend_signal")
        return Signal(symbol=sym, side="flat", strength=0.0, meta=meta)

    def generate(self, candles: Dict[str, Candle]) -> Dict[str, Signal]:
        out: Dict[str, Signal] = {}

        for sym, c in candles.items():
            sym_s = str(sym)

            if self.only_if_symbol_contains:
                if self.only_if_symbol_contains.upper() not in sym_s.upper():
                    if self.emit_flat:
                        out[sym_s] = self._flat(sym_s, "symbol_filtered", {"symbol": sym_s})
                    continue

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
                "engine": "dot_breakout_trend_signal",
                "strategy_family": "trend",
                "symbol": sym_s,
                "close": float(close),
                "ema_fast": float(ema_fast or 0.0),
                "ema_slow": float(ema_slow or 0.0),
                "adx": float(adx or 0.0),
                "atr": float(atr or 0.0),
                "atrp": float(atrp or 0.0),
                "ret_4h_lag": float(ret_4h or 0.0),
                "ret_12h_lag": float(ret_12h or 0.0),
                "ret_24h_lag": float(ret_24h or 0.0),
                "ema_gap_fast_slow": float(ema_gap or 0.0),
                "dist_close_ema_fast": float(dist_close_ema_fast or 0.0),
                "breakout_distance_up": float(breakout_up or 0.0),
                "breakout_distance_down": float(breakout_down or 0.0),
                "rolling_vol_24h": float(rolling_vol_24h or 0.0),
                "dedicated_engine_name": "dot_breakout_trend_signal",
                "dedicated_engine_allowed_sides": "long",
                "signal_notes": [],
            }

            if not self.use_longs:
                if self.emit_flat:
                    out[sym_s] = self._flat(sym_s, "longs_disabled", meta)
                continue

            if None in (ema_fast, ema_slow, adx, atr, atrp):
                meta["signal_notes"].append("missing_core_features")
                if self.emit_flat:
                    out[sym_s] = self._flat(sym_s, "missing_core_features", meta)
                continue

            ema_fast = float(ema_fast)
            ema_slow = float(ema_slow)
            adx = float(adx)
            atr = float(atr)
            atrp = float(atrp)
            ret_4h = float(ret_4h or 0.0)
            ret_12h = float(ret_12h or 0.0)
            ema_gap = float(ema_gap or 0.0)
            dist_close_ema_fast = float(dist_close_ema_fast or 0.0)
            breakout_up = float(breakout_up or 0.0)
            rolling_vol_24h = float(rolling_vol_24h or 0.0)

            reasons = []

            if close <= 0.0:
                reasons.append("bad_close")
            if adx < float(self.adx_min):
                reasons.append("adx_below_min")
            if atrp < float(self.atrp_min):
                reasons.append("atrp_low")
            if atrp > float(self.atrp_max):
                reasons.append("atrp_high")
            if ema_gap < float(self.min_ema_gap):
                reasons.append("ema_gap_below_min")
            if ret_4h < float(self.min_ret_4h):
                reasons.append("ret_4h_below_min")
            if ret_12h < float(self.min_ret_12h):
                reasons.append("ret_12h_below_min")
            if breakout_up > float(self.breakout_up_max):
                reasons.append("too_extended_above_breakout")
            if abs(dist_close_ema_fast) > float(self.pullback_abs_max):
                reasons.append("too_far_from_ema_fast")
            if rolling_vol_24h < float(self.rolling_vol_24h_min):
                reasons.append("rolling_vol_24h_below_min")

            # Long bias is intentionally looser than DotTrendSignalEngine.
            # It accepts early or recovering trend setups, not only clean breakouts.
            trend_or_recovery = (
                ema_fast >= ema_slow
                or ema_gap >= float(self.min_ema_gap)
                or ret_4h > 0.0
                or ret_12h > 0.0
            )

            if not trend_or_recovery:
                reasons.append("no_trend_or_recovery_bias")

            if reasons:
                meta["signal_notes"].extend(reasons)
                if self.emit_flat:
                    out[sym_s] = self._flat(sym_s, "|".join(reasons), meta)
                continue

            strength = float(self.strength_base)
            strength += max(0.0, min(1.0, adx / max(float(self.adx_min), 1e-9))) * 0.20
            strength += max(0.0, min(1.0, atrp / max(float(self.atrp_max), 1e-9))) * 0.15
            strength += max(0.0, min(1.0, (ema_gap - float(self.min_ema_gap)) / 0.02)) * 0.20
            strength += max(0.0, min(1.0, (ret_4h - float(self.min_ret_4h)) / 0.08)) * 0.15
            strength += max(0.0, min(1.0, (ret_12h - float(self.min_ret_12h)) / 0.12)) * 0.15

            meta["signal_notes"].append("dot_breakout_lax_long")
            meta["dot_breakout_strength"] = float(strength)

            out[sym_s] = Signal(
                symbol=sym_s,
                side="long",
                strength=float(max(0.0, strength)),
                meta=meta,
            )

        return out
