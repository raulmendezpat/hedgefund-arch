from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


@dataclass
class SubPositionPlan:
    symbol: str
    strategy_id: str
    side: str
    total_target_weight: float
    slices: List[float]
    mode: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubPositionPlanner:
    """
    Planifica una oportunidad como múltiples subposiciones.

    Primer alcance:
    - mode='equal': divide el target_weight en partes iguales.
    - No ejecuta trades ni altera señales; solo genera el plan.
    """

    slices: int = 3
    mode: str = "equal"
    min_slice_weight: float = 0.0
    max_slices: int = 20

    def _normalize_slices(self, weights: List[float], total_target_weight: float) -> List[float]:
        total = sum(float(w) for w in weights)
        if total <= 0:
            return [0.0 for _ in weights]
        scale = float(total_target_weight) / float(total)
        out = [float(w) * scale for w in weights]
        return out

    def _build_equal_slices(self, total_target_weight: float, slices: int) -> List[float]:
        if slices <= 0:
            return []
        per_slice = float(total_target_weight) / float(slices)
        return [per_slice for _ in range(slices)]

    def plan(
        self,
        *,
        symbol: str,
        strategy_id: str,
        side: str,
        total_target_weight: float,
        opportunity_meta: Optional[Dict[str, Any]] = None,
    ) -> SubPositionPlan:
        target = max(0.0, float(total_target_weight))
        requested_slices = int(self.slices)
        requested_slices = max(1, min(requested_slices, int(self.max_slices)))

        if target <= 0.0:
            final_slices: List[float] = []
        else:
            if self.mode == "equal":
                raw = self._build_equal_slices(target, requested_slices)
            else:
                raise ValueError(f"Unsupported subposition planning mode: {self.mode}")

            min_slice = max(0.0, float(self.min_slice_weight))
            filtered = [float(w) for w in raw if float(w) >= min_slice]

            if not filtered:
                filtered = [target]

            final_slices = self._normalize_slices(filtered, target)

        return SubPositionPlan(
            symbol=str(symbol),
            strategy_id=str(strategy_id),
            side=str(side),
            total_target_weight=float(target),
            slices=[float(x) for x in final_slices],
            mode=str(self.mode),
            meta=dict(opportunity_meta or {}),
        )
