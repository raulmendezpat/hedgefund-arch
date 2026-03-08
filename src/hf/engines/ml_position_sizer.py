from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from hf.core.types import Allocation, Candle, Signal
from hf.engines.ml_filter import (
    build_feature_row,
    predict_proba,
    select_model_for_signal,
)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


@dataclass
class MlPositionSizingEngine:
    """Convert p_win into a position multiplier in [min_mult, max_mult].

    Default sizing rule:
        multiplier = clip(scale * (2 * p_win - 1), min_mult, max_mult)

    Notes:
    - p_win=0.50 -> multiplier=0.0 (unless min_mult > 0)
    - p_win=1.00 -> multiplier=scale (clipped by max_mult)
    - If p_win is missing, the engine can infer it from the ML model/registry.
    - Allocation weights are multiplied by this factor; remaining capital stays as cash.
    """

    scale: float = 1.0
    min_mult: float = 0.0
    max_mult: float = 1.0
    use_abs_formula: bool = False
    mode: str = "linear_edge"
    base_size: float = 0.25
    pwin_threshold: float = 0.55

    def size_from_pwin(self, p_win: float) -> float:
        p = _clamp(float(p_win), 0.0, 1.0)

        if self.mode == "calibrated":
            edge = max(0.0, p - float(self.pwin_threshold))
            raw = float(self.base_size) + float(self.scale) * edge
            return _clamp(raw, float(self.min_mult), float(self.max_mult))

        raw = float(self.scale) * (2.0 * p - 1.0)
        if self.use_abs_formula:
            raw = abs(raw)
        return _clamp(raw, float(self.min_mult), float(self.max_mult))

    def apply_to_signals(
        self,
        *,
        candles: Dict[str, Candle],
        signals: Dict[str, Signal],
        model: Any = None,
        model_registry: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Signal], Dict[str, float]]:
        out: Dict[str, Signal] = {}
        sized: Dict[str, float] = {}
        registry = dict(model_registry or {})

        for sym, sig in (signals or {}).items():
            if sig is None:
                out[sym] = sig
                continue

            meta = dict(getattr(sig, "meta", {}) or {})
            side = getattr(sig, "side", "flat")

            if side == "flat" or "skip" in meta:
                out[sym] = sig
                continue

            p_win = meta.get("p_win", None)
            if p_win is None:
                candle = candles.get(sym)
                if candle is not None:
                    feats = build_feature_row(sym, candle, sig)
                    chosen_model = select_model_for_signal(model, registry, sym, side)
                    if registry and chosen_model is None:
                        out[sym] = sig
                        continue
                    p_win = predict_proba(chosen_model, feats)
                else:
                    out[sym] = sig
                    continue

            mult = self.size_from_pwin(float(p_win))
            sized[sym] = float(mult)

            out[sym] = Signal(
                symbol=sym,
                side=side,
                strength=float(getattr(sig, "strength", 1.0) or 0.0) * float(mult),
                meta={
                    **meta,
                    "p_win": float(p_win),
                    "ml_position_size_mult": float(mult),
                    "ml_position_size_scale": float(self.scale),
                },
            )

        return out, sized

    def apply_to_allocation(
        self,
        *,
        allocation: Allocation,
        signals: Dict[str, Signal],
    ) -> Allocation:
        alloc_meta = dict(getattr(allocation, "meta", {}) or {})
        alloc_case = str(alloc_meta.get("case", "") or "")
        allowed_cases = {"btc_only", "sol_only", "both_on"}

        base_w = dict(getattr(allocation, "weights", {}) or {})
        out_w: Dict[str, float] = {}
        applied: Dict[str, float] = {}

        if alloc_case not in allowed_cases:
            return Allocation(
                weights={k: float(v) for k, v in base_w.items()},
                meta={
                    **alloc_meta,
                    "ml_position_sizing_applied": False,
                    "ml_position_sizing_skipped_case": alloc_case,
                    "ml_position_size_mults": {},
                },
            )

        for sym, w in base_w.items():
            sig = (signals or {}).get(sym)
            meta = dict(getattr(sig, "meta", {}) or {}) if sig is not None else {}
            mult = meta.get("ml_position_size_mult", None)
            if mult is None:
                out_w[sym] = float(w)
                continue
            out_w[sym] = float(w) * float(mult)
            applied[sym] = float(mult)

        return Allocation(
            weights=out_w,
            meta={
                **alloc_meta,
                "ml_position_sizing_applied": bool(applied),
                "ml_position_size_mults": applied,
            },
        )
