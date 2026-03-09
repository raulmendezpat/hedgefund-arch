from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    """Convert p_win into a position multiplier.

    Supported modes:
    - linear_edge:     multiplier = clip(scale * (2 * p_win - 1), min_mult, max_mult)
    - calibrated:      multiplier = clip(base_size + scale * max(0, p_win - threshold), ...)
    - artifact_map:    multiplier obtained from JSON bins artifact

    Notes:
    - If p_win is missing, the engine can infer it from ML model/registry.
    - Allocation weights are multiplied by this factor; remaining capital stays as cash.
    """

    scale: float = 1.0
    min_mult: float = 0.0
    max_mult: float = 1.0
    use_abs_formula: bool = False
    mode: str = "linear_edge"
    base_size: float = 0.25
    pwin_threshold: float = 0.55
    artifact_path: str = "artifacts/ml_position_size_map_v1.json"
    artifact_bins: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if str(self.mode) == "artifact_map":
            self._load_artifact_map()

    def _load_artifact_map(self) -> None:
        path = Path(self.artifact_path)
        if not path.exists():
            self.artifact_bins = []
            return

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self.artifact_bins = []
            return

        bins = payload.get("bins", []) if isinstance(payload, dict) else []
        if not isinstance(bins, list):
            self.artifact_bins = []
            return

        clean: List[Dict[str, Any]] = []
        for row in bins:
            if not isinstance(row, dict):
                continue
            try:
                clean.append(
                    {
                        "min": float(row.get("min", 0.0)),
                        "max": float(row.get("max", 1.0)),
                        "size": float(row.get("size", 0.0)),
                    }
                )
            except Exception:
                continue

        clean = sorted(clean, key=lambda x: (float(x["min"]), float(x["max"])))
        self.artifact_bins = clean

    def _size_from_artifact_map(self, p_win: float) -> float:
        p = _clamp(float(p_win), 0.0, 1.0)

        for row in self.artifact_bins:
            lo = float(row["min"])
            hi = float(row["max"])
            size = float(row["size"])

            is_last = abs(hi - 1.0) <= 1e-12
            if (lo <= p < hi) or (is_last and lo <= p <= hi):
                return _clamp(size, 0.0, float(self.max_mult))

        return 0.0

    def size_from_pwin(self, p_win: float) -> float:
        p = _clamp(float(p_win), 0.0, 1.0)

        if self.mode == "artifact_map" and self.artifact_bins:
            return self._size_from_artifact_map(p)

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
                    "ml_position_size_mode": str(self.mode),
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
