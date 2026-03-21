from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from hf.core.interfaces import SignalEngine
from hf.core.types import Candle, Signal


def _feat(c: Candle, key: str) -> Optional[float]:
    feats = getattr(c, "features", None)
    if not isinstance(feats, dict):
        return None
    v = feats.get(key, None)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


@dataclass
class BtcTrendSignalEngine(SignalEngine):
    """BTC trend con strength escalonado opcional para pyramiding simple.

    Regla base:
    - Si falta algún feature -> flat
    - Si ADX < adx_min -> flat
    - Si ema_fast > ema_slow -> long
    - Si ema_fast < ema_slow -> short
    - Si iguales -> flat

    Modo opcional:
    - use_strength_tiers=True
    - escala strength según ADX y gap relativo entre EMAs
    - esto permite aumentar target weight de forma neta y compatible con live reconcile
    """

    adx_min: float = 18.0

    # Optional pyramiding / strength tiers (disabled by default)
    use_strength_tiers: bool = False
    strength_base: float = 1.0
    strength_step_1: float = 1.25
    strength_step_2: float = 1.50
    adx_tier_1: float = 25.0
    adx_tier_2: float = 35.0
    ema_gap_tier_1: float = 0.002
    ema_gap_tier_2: float = 0.004

    # Optional trend-quality guard: require minimum EMA separation.
    require_ema_gap_min: bool = False
    ema_gap_min: float = 0.0015

    # Research mode: do not hard-filter weak trend context at signal phase.
    # Instead, emit side + weaker strength and expose context in meta.
    soft_adx_gate: bool = False
    soft_ema_gap_gate: bool = False
    strength_penalty_adx: float = 0.60
    strength_penalty_ema_gap: float = 0.80

    adx_key: str = "adx"
    ema_fast_key: str = "ema_fast"
    ema_slow_key: str = "ema_slow"
    only_if_symbol_contains: str = ""

    def generate(self, candles: Dict[str, Candle]) -> Dict[str, Signal]:
        out: Dict[str, Signal] = {}

        for sym, c in candles.items():
            if self.only_if_symbol_contains and (self.only_if_symbol_contains not in sym):
                out[sym] = Signal(
                    symbol=sym,
                    side="flat",
                    strength=0.0,
                    meta={"engine": "btc_trend_min", "reason": "symbol_filtered"},
                )
                continue

            adx = _feat(c, self.adx_key)
            ema_fast = _feat(c, self.ema_fast_key)
            ema_slow = _feat(c, self.ema_slow_key)

            if adx is None or ema_fast is None or ema_slow is None:
                out[sym] = Signal(symbol=sym, side="flat", strength=0.0, meta={"engine": "btc_trend_min", "reason": "missing_features"})
                continue

            _adx_below_min = float(adx) < float(self.adx_min)

            if ema_fast > ema_slow:
                side = "long"
            elif ema_fast < ema_slow:
                side = "short"
            else:
                side = "flat"

            strength = float(self.strength_base) if side != "flat" else 0.0
            ema_gap_pct = abs(float(ema_fast) / max(abs(float(ema_slow)), 1e-12) - 1.0)
            _ema_gap_below_min = bool(self.require_ema_gap_min) and float(ema_gap_pct) < float(self.ema_gap_min)

            if _adx_below_min and not bool(self.soft_adx_gate):
                out[sym] = Signal(
                    symbol=sym,
                    side="flat",
                    strength=0.0,
                    meta={"engine": "btc_trend_min", "reason": "adx_below_min", "adx": adx},
                )
                continue

            if side != "flat" and _adx_below_min and bool(self.soft_adx_gate):
                strength *= float(self.strength_penalty_adx)

            if side != "flat" and _ema_gap_below_min and not bool(self.soft_ema_gap_gate):
                out[sym] = Signal(
                    symbol=sym,
                    side="flat",
                    strength=0.0,
                    meta={
                        "engine": "btc_trend_min",
                        "reason": "ema_gap_below_min",
                        "adx": adx,
                        "ema_fast": ema_fast,
                        "ema_slow": ema_slow,
                        "ema_gap_pct": float(ema_gap_pct),
                        "ema_gap_min": float(self.ema_gap_min),
                    },
                )
                continue

            if side != "flat" and _ema_gap_below_min and bool(self.soft_ema_gap_gate):
                strength *= float(self.strength_penalty_ema_gap)

            if side != "flat" and bool(self.use_strength_tiers):
                if float(adx) >= float(self.adx_tier_2) and float(ema_gap_pct) >= float(self.ema_gap_tier_2):
                    strength = float(self.strength_step_2)
                elif float(adx) >= float(self.adx_tier_1) and float(ema_gap_pct) >= float(self.ema_gap_tier_1):
                    strength = float(self.strength_step_1)

            out[sym] = Signal(
                symbol=sym,
                side=side,
                strength=float(max(0.0, strength)),
                meta={
                    "engine": "btc_trend_min",
                    "adx": adx,
                    "ema_fast": ema_fast,
                    "ema_slow": ema_slow,
                    "ema_gap_pct": float(ema_gap_pct),
                    "use_strength_tiers": bool(self.use_strength_tiers),
                    "strength_base": float(self.strength_base),
                    "strength_step_1": float(self.strength_step_1),
                    "strength_step_2": float(self.strength_step_2),
                    "soft_adx_gate": bool(self.soft_adx_gate),
                    "soft_ema_gap_gate": bool(self.soft_ema_gap_gate),
                    "adx_below_min": bool(_adx_below_min),
                    "ema_gap_below_min": bool(_ema_gap_below_min),
                    "strength_penalty_adx": float(self.strength_penalty_adx),
                    "strength_penalty_ema_gap": float(self.strength_penalty_ema_gap),
                },
            )

        return out
