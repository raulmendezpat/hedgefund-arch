
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

from hf.core.interfaces import SignalEngine
from hf.core.types import Candle, Signal


def _feat(c: Candle, key: str) -> Optional[float]:
    feats = getattr(c, "features", None)
    if not isinstance(feats, dict):
        return None

    v = feats.get(key)
    if v is None:
        return None

    try:
        v = float(v)
    except Exception:
        return None

    if v != v:
        return None
    return v


@dataclass
class SolVolExpansionSignalEngine(SignalEngine):

    atrp_min: float = 0.01
    adx_min: float = 18.0

    breakout_mult: float = 1.000
    breakdown_mult: float = 1.000

    # Optional robustness guards. All disabled by default to preserve current behavior.
    use_rsi_exhaustion_guard: bool = False
    rsi_long_max: float = 72.0
    rsi_short_min: float = 28.0

    require_directional_close: bool = False

    use_extension_guard: bool = False
    max_breakout_extension_pct: float = 0.02
    max_breakdown_extension_pct: float = 0.02

    # Optional confirmed-expansion guards.
    require_trend_alignment: bool = False
    require_donchian_break: bool = False
    min_range_expansion: float = 0.0
    breakout_confirm_buffer: float = 0.002

    atrp_key: str = "atrp"
    adx_key: str = "adx"
    rsi_key: str = "rsi"
    bb_up_key: str = "bb_up"
    bb_low_key: str = "bb_low"
    ema_fast_key: str = "ema_fast"
    ema_slow_key: str = "ema_slow"
    donchian_high_key: str = "donchian_high"
    donchian_low_key: str = "donchian_low"
    range_expansion_key: str = "range_expansion"

    only_if_symbol_contains: str = "SOL"

    def _flat(self, sym: str, reason: str, **meta) -> Signal:
        return Signal(
            symbol=sym,
            side="flat",
            strength=0.0,
            meta={"engine": "sol_vol_expansion", "reason": reason, **meta},
        )

    def generate(self, candles, print_debug: bool = False) -> Dict[str, Signal]:

        out: Dict[str, Signal] = {}

        for sym, c in candles.items():

            if self.only_if_symbol_contains and self.only_if_symbol_contains not in sym:
                out[sym] = Signal(symbol=sym, side="flat", strength=0.0)
                continue

            atrp = _feat(c, self.atrp_key)
            adx = _feat(c, self.adx_key)
            rsi = _feat(c, self.rsi_key)
            bb_up = _feat(c, self.bb_up_key)
            bb_low = _feat(c, self.bb_low_key)
            ema_fast = _feat(c, self.ema_fast_key)
            ema_slow = _feat(c, self.ema_slow_key)
            donchian_high = _feat(c, self.donchian_high_key)
            donchian_low = _feat(c, self.donchian_low_key)
            range_expansion = _feat(c, self.range_expansion_key)

            open_v = float(getattr(c, "open", 0.0))
            close_v = float(getattr(c, "close", 0.0))

            if None in (atrp, adx, bb_up, bb_low):
                out[sym] = self._flat(sym, "missing_features")
                continue

            if float(self.min_range_expansion) > 0.0:
                if range_expansion is None:
                    out[sym] = self._flat(sym, "missing_range_expansion")
                    continue
                if float(range_expansion) < float(self.min_range_expansion):
                    out[sym] = self._flat(sym, "range_expansion_low", range_expansion=range_expansion)
                    continue

            atrp_low = atrp < float(self.atrp_min)

            adx_low = adx < float(self.adx_min)

            if close_v >= float(bb_up) * float(self.breakout_mult):
                long_non_directional_bar = bool(self.require_directional_close) and close_v <= open_v
                missing_trend_alignment_long = bool(self.require_trend_alignment) and (ema_fast is None or ema_slow is None)
                long_trend_misaligned = (
                    bool(self.require_trend_alignment)
                    and (ema_fast is not None)
                    and (ema_slow is not None)
                    and not (float(ema_fast) > float(ema_slow) and close_v > float(ema_fast))
                )

                missing_donchian_long = bool(self.require_donchian_break) and (donchian_high is None)
                long_no_donchian_break = (
                    bool(self.require_donchian_break)
                    and (donchian_high is not None)
                    and (close_v <= float(donchian_high) * (1.0 + float(self.breakout_confirm_buffer)))
                )
                missing_rsi_long_guard = bool(self.use_rsi_exhaustion_guard) and (rsi is None)
                long_rsi_exhausted = bool(self.use_rsi_exhaustion_guard) and (rsi is not None) and (float(rsi) > float(self.rsi_long_max))
                long_extension_pct = max(0.0, (close_v / max(float(bb_up), 1e-12)) - 1.0)
                long_overextended = bool(self.use_extension_guard) and (long_extension_pct > float(self.max_breakout_extension_pct))

                side = "long"
                reason = "vol_expansion_breakout_long"

            elif close_v <= float(bb_low) * float(self.breakdown_mult):
                short_non_directional_bar = bool(self.require_directional_close) and close_v >= open_v
                missing_trend_alignment_short = bool(self.require_trend_alignment) and (ema_fast is None or ema_slow is None)
                short_trend_misaligned = (
                    bool(self.require_trend_alignment)
                    and (ema_fast is not None)
                    and (ema_slow is not None)
                    and not (float(ema_fast) < float(ema_slow) and close_v < float(ema_fast))
                )

                missing_donchian_short = bool(self.require_donchian_break) and (donchian_low is None)
                short_no_donchian_break = (
                    bool(self.require_donchian_break)
                    and (donchian_low is not None)
                    and (close_v >= float(donchian_low) * (1.0 - float(self.breakout_confirm_buffer)))
                )
                missing_rsi_short_guard = bool(self.use_rsi_exhaustion_guard) and (rsi is None)
                short_rsi_exhausted = bool(self.use_rsi_exhaustion_guard) and (rsi is not None) and (float(rsi) < float(self.rsi_short_min))
                short_extension_pct = max(0.0, 1.0 - (close_v / max(float(bb_low), 1e-12)))
                short_overextended = bool(self.use_extension_guard) and (short_extension_pct > float(self.max_breakdown_extension_pct))

                side = "short"
                reason = "vol_expansion_breakout_short"

            else:
                side = "flat"
                reason = "no_breakout"

            strength = 1.0 if side != "flat" else 0.0

            # soft regime penalties (no hard rejection)
            if side != "flat":
                if 'atrp_low' in locals() and atrp_low:
                    strength *= 0.6
                if 'adx_low' in locals() and adx_low:
                    strength *= 0.7

            out[sym] = Signal(
                symbol=sym,
                side=side,
                strength=strength,
                meta={
                    "engine": "sol_vol_expansion",
                    "reason": reason,
                    "open": open_v,
                    "close": close_v,
                    "atrp": atrp,
                    "adx": adx,
                    "rsi": rsi,
                    "bb_up": bb_up,
                    "bb_low": bb_low,
                    "ema_fast": ema_fast,
                    "ema_slow": ema_slow,
                    "donchian_high": donchian_high,
                    "donchian_low": donchian_low,
                    "range_expansion": range_expansion,
                    "atrp_low": bool(locals().get("atrp_low", False)),
                    "adx_low": bool(locals().get("adx_low", False)),
                    "long_non_directional_bar": bool(locals().get("long_non_directional_bar", False)),
                    "short_non_directional_bar": bool(locals().get("short_non_directional_bar", False)),
                    "missing_trend_alignment_long": bool(locals().get("missing_trend_alignment_long", False)),
                    "missing_trend_alignment_short": bool(locals().get("missing_trend_alignment_short", False)),
                    "long_trend_misaligned": bool(locals().get("long_trend_misaligned", False)),
                    "short_trend_misaligned": bool(locals().get("short_trend_misaligned", False)),
                    "missing_donchian_long": bool(locals().get("missing_donchian_long", False)),
                    "missing_donchian_short": bool(locals().get("missing_donchian_short", False)),
                    "long_no_donchian_break": bool(locals().get("long_no_donchian_break", False)),
                    "short_no_donchian_break": bool(locals().get("short_no_donchian_break", False)),
                    "missing_rsi_long_guard": bool(locals().get("missing_rsi_long_guard", False)),
                    "missing_rsi_short_guard": bool(locals().get("missing_rsi_short_guard", False)),
                    "long_rsi_exhausted": bool(locals().get("long_rsi_exhausted", False)),
                    "short_rsi_exhausted": bool(locals().get("short_rsi_exhausted", False)),
                    "long_extension_pct": float(locals().get("long_extension_pct", 0.0)),
                    "short_extension_pct": float(locals().get("short_extension_pct", 0.0)),
                    "long_overextended": bool(locals().get("long_overextended", False)),
                    "short_overextended": bool(locals().get("short_overextended", False)),
                    "regime_as_metadata": True,
                },
            )

        return out
