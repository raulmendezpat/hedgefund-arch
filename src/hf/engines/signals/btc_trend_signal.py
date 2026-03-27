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
        v = float(v)
    except Exception:
        return None
    if v != v:  # NaN
        return None
    return v


@dataclass
class BtcTrendSignalEngine(SignalEngine):
    # Core thresholds
    adx_min: float = 18.0
    require_ema_gap_min: bool = False
    ema_gap_min: float = 0.0015

    # Optional per-side overrides
    long_adx_min: Optional[float] = None
    short_adx_min: Optional[float] = None
    long_require_ema_gap_min: Optional[bool] = None
    short_require_ema_gap_min: Optional[bool] = None
    long_ema_gap_min: Optional[float] = None
    short_ema_gap_min: Optional[float] = None

    # Gate semantics
    # soft_* = False  -> hard reject to flat
    # soft_* = True   -> allow signal but penalize strength
    soft_adx_gate: bool = False
    soft_ema_gap_gate: bool = False

    # Optional per-side soft gate overrides
    long_soft_adx_gate: Optional[bool] = None
    short_soft_adx_gate: Optional[bool] = None
    long_soft_ema_gap_gate: Optional[bool] = None
    short_soft_ema_gap_gate: Optional[bool] = None

    # Strength model
    strength_base: float = 1.0
    strength_penalty_adx: float = 0.60
    strength_penalty_ema_gap: float = 0.80

    # Optional strength tiers
    use_strength_tiers: bool = False
    strength_step_1: float = 1.25
    strength_step_2: float = 1.50
    adx_tier_1: float = 25.0
    adx_tier_2: float = 35.0
    ema_gap_tier_1: float = 0.002
    ema_gap_tier_2: float = 0.004

    # Feature keys
    adx_key: str = "adx"
    ema_fast_key: str = "ema_fast"
    ema_slow_key: str = "ema_slow"

    # Optional symbol scoping
    only_if_symbol_contains: str = ""

    def _flat(self, sym: str, reason: str, **meta) -> Signal:
        return Signal(
            symbol=sym,
            side="flat",
            strength=0.0,
            meta={"engine": "btc_trend_min", "reason": reason, **meta},
        )

    @staticmethod
    def _resolve_side_value(
        side: str,
        default,
        long_value,
        short_value,
    ):
        if side == "long" and long_value is not None:
            return long_value
        if side == "short" and short_value is not None:
            return short_value
        return default

    def generate(self, candles: Dict[str, Candle]) -> Dict[str, Signal]:
        out: Dict[str, Signal] = {}

        for sym, c in candles.items():
            if self.only_if_symbol_contains and self.only_if_symbol_contains not in sym:
                out[sym] = self._flat(sym, "symbol_filtered")
                continue

            adx = _feat(c, self.adx_key)
            ema_fast = _feat(c, self.ema_fast_key)
            ema_slow = _feat(c, self.ema_slow_key)

            if adx is None or ema_fast is None or ema_slow is None:
                out[sym] = self._flat(sym, "missing_features")
                continue

            if ema_fast > ema_slow:
                side = "long"
            elif ema_fast < ema_slow:
                side = "short"
            else:
                out[sym] = self._flat(
                    sym,
                    "ema_equal",
                    adx=float(adx),
                    ema_fast=float(ema_fast),
                    ema_slow=float(ema_slow),
                )
                continue

            resolved_adx_min = float(
                self._resolve_side_value(
                    side=side,
                    default=self.adx_min,
                    long_value=self.long_adx_min,
                    short_value=self.short_adx_min,
                )
            )
            resolved_require_ema_gap_min = bool(
                self._resolve_side_value(
                    side=side,
                    default=self.require_ema_gap_min,
                    long_value=self.long_require_ema_gap_min,
                    short_value=self.short_require_ema_gap_min,
                )
            )
            resolved_ema_gap_min = float(
                self._resolve_side_value(
                    side=side,
                    default=self.ema_gap_min,
                    long_value=self.long_ema_gap_min,
                    short_value=self.short_ema_gap_min,
                )
            )
            resolved_soft_adx_gate = bool(
                self._resolve_side_value(
                    side=side,
                    default=self.soft_adx_gate,
                    long_value=self.long_soft_adx_gate,
                    short_value=self.short_soft_adx_gate,
                )
            )
            resolved_soft_ema_gap_gate = bool(
                self._resolve_side_value(
                    side=side,
                    default=self.soft_ema_gap_gate,
                    long_value=self.long_soft_ema_gap_gate,
                    short_value=self.short_soft_ema_gap_gate,
                )
            )

            ema_gap_pct = abs(float(ema_fast) / max(abs(float(ema_slow)), 1e-12) - 1.0)
            adx_below_min = float(adx) < resolved_adx_min
            ema_gap_below_min = resolved_require_ema_gap_min and (float(ema_gap_pct) < resolved_ema_gap_min)

            # Hard gates when configured
            if adx_below_min and not resolved_soft_adx_gate:
                out[sym] = self._flat(
                    sym,
                    "adx_below_min",
                    side=side,
                    adx=float(adx),
                    adx_min=resolved_adx_min,
                    ema_fast=float(ema_fast),
                    ema_slow=float(ema_slow),
                    ema_gap_pct=float(ema_gap_pct),
                    soft_adx_gate=resolved_soft_adx_gate,
                    soft_ema_gap_gate=resolved_soft_ema_gap_gate,
                )
                continue

            if ema_gap_below_min and not resolved_soft_ema_gap_gate:
                out[sym] = self._flat(
                    sym,
                    "ema_gap_below_min",
                    side=side,
                    adx=float(adx),
                    adx_min=resolved_adx_min,
                    ema_fast=float(ema_fast),
                    ema_slow=float(ema_slow),
                    ema_gap_pct=float(ema_gap_pct),
                    ema_gap_min=resolved_ema_gap_min,
                    require_ema_gap_min=resolved_require_ema_gap_min,
                    soft_adx_gate=resolved_soft_adx_gate,
                    soft_ema_gap_gate=resolved_soft_ema_gap_gate,
                )
                continue

            strength = float(self.strength_base)

            # Optional tiering only after hard gates have passed
            if bool(self.use_strength_tiers):
                if float(adx) >= float(self.adx_tier_2) and float(ema_gap_pct) >= float(self.ema_gap_tier_2):
                    strength = float(self.strength_step_2)
                elif float(adx) >= float(self.adx_tier_1) and float(ema_gap_pct) >= float(self.ema_gap_tier_1):
                    strength = float(self.strength_step_1)

            # Soft penalties only when configured to allow weak context through
            if adx_below_min and resolved_soft_adx_gate:
                strength *= float(self.strength_penalty_adx)

            if ema_gap_below_min and resolved_soft_ema_gap_gate:
                strength *= float(self.strength_penalty_ema_gap)

            out[sym] = Signal(
                symbol=sym,
                side=side,
                strength=float(max(0.0, strength)),
                meta={
                    "engine": "btc_trend_min",
                    "reason": "trend_signal",
                    "adx": float(adx),
                    "adx_min": resolved_adx_min,
                    "ema_fast": float(ema_fast),
                    "ema_slow": float(ema_slow),
                    "ema_gap_pct": float(ema_gap_pct),
                    "require_ema_gap_min": resolved_require_ema_gap_min,
                    "ema_gap_min": resolved_ema_gap_min,
                    "adx_below_min": bool(adx_below_min),
                    "ema_gap_below_min": bool(ema_gap_below_min),
                    "soft_adx_gate": resolved_soft_adx_gate,
                    "soft_ema_gap_gate": resolved_soft_ema_gap_gate,
                    "use_strength_tiers": bool(self.use_strength_tiers),
                    "strength_base": float(self.strength_base),
                    "strength_step_1": float(self.strength_step_1),
                    "strength_step_2": float(self.strength_step_2),
                    "strength_penalty_adx": float(self.strength_penalty_adx),
                    "strength_penalty_ema_gap": float(self.strength_penalty_ema_gap),
                    "long_adx_min": self.long_adx_min,
                    "short_adx_min": self.short_adx_min,
                    "long_require_ema_gap_min": self.long_require_ema_gap_min,
                    "short_require_ema_gap_min": self.short_require_ema_gap_min,
                    "long_ema_gap_min": self.long_ema_gap_min,
                    "short_ema_gap_min": self.short_ema_gap_min,
                    "regime_as_metadata": True,
                },
            )

        return out
