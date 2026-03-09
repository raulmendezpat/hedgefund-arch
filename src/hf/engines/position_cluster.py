from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class PlannedSubPosition:
    index: int
    weight: float
    entry_ts: Optional[int] = None
    entry_price: Optional[float] = None
    status: str = "planned"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PositionCluster:
    cluster_id: str
    symbol: str
    strategy_id: str
    side: str
    target_weight: float
    subpositions: tuple[PlannedSubPosition, ...] = field(default_factory=tuple)
    entry_schedule: tuple[Any, ...] = field(default_factory=tuple)
    exit_schedule: tuple[Any, ...] = field(default_factory=tuple)
    risk_limits: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def planned_weight(self) -> float:
        return float(sum(float(x.weight) for x in self.subpositions))

    @property
    def subposition_count(self) -> int:
        return int(len(self.subpositions))


class PositionClusterBuilder:
    def build_from_weights(
        self,
        *,
        cluster_id: str,
        symbol: str,
        strategy_id: str,
        side: str,
        target_weight: float,
        weights: list[float],
        meta: Optional[dict[str, Any]] = None,
    ) -> PositionCluster:
        subpositions = tuple(
            PlannedSubPosition(
                index=i,
                weight=float(w),
            )
            for i, w in enumerate(weights)
            if float(w) > 0.0
        )

        return PositionCluster(
            cluster_id=str(cluster_id),
            symbol=str(symbol),
            strategy_id=str(strategy_id),
            side=str(side),
            target_weight=float(target_weight),
            subpositions=subpositions,
            meta=dict(meta or {}),
        )
