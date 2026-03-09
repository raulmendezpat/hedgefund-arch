from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from hf.engines.position_cluster import PositionCluster


@dataclass(frozen=True)
class ExecutionSlice:
    cluster_id: str
    subposition_index: int
    symbol: str
    side: str
    target_weight: float
    execution_mode: str = "market"
    order_type: str = "market"
    time_in_force: str = "GTC"
    limit_price: Optional[float] = None
    trigger_price: Optional[float] = None
    time_offset_bars: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionPlan:
    cluster_id: str
    symbol: str
    strategy_id: str
    side: str
    total_target_weight: float
    slices: tuple[ExecutionSlice, ...] = field(default_factory=tuple)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def slice_count(self) -> int:
        return int(len(self.slices))

    @property
    def planned_weight(self) -> float:
        return float(sum(float(x.target_weight) for x in self.slices))


class ExecutionPlanner:
    def build_plan(
        self,
        *,
        cluster: PositionCluster,
        execution_mode: str = "market",
        default_order_type: str = "market",
        time_in_force: str = "GTC",
        ladder_limit_offsets: Optional[list[float]] = None,
        time_sliced_offsets: Optional[list[int]] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> ExecutionPlan:
        mode = str(execution_mode or "market")

        if mode == "ladder_limit":
            offsets = list(ladder_limit_offsets or [])
            slices = []
            for i, sp in enumerate(cluster.subpositions):
                if float(sp.weight) <= 0.0:
                    continue
                limit_offset = float(offsets[i]) if i < len(offsets) else 0.0
                slices.append(
                    ExecutionSlice(
                        cluster_id=str(cluster.cluster_id),
                        subposition_index=int(sp.index),
                        symbol=str(cluster.symbol),
                        side=str(cluster.side),
                        target_weight=float(sp.weight),
                        execution_mode=mode,
                        order_type="limit",
                        time_in_force=str(time_in_force),
                        limit_price=None,
                        trigger_price=None,
                        time_offset_bars=0,
                        meta={**dict(sp.meta), "limit_offset_pct": limit_offset},
                    )
                )
            slices = tuple(slices)

        elif mode == "time_sliced":
            offsets = list(time_sliced_offsets or [])
            slices = []
            for i, sp in enumerate(cluster.subpositions):
                if float(sp.weight) <= 0.0:
                    continue
                bar_offset = int(offsets[i]) if i < len(offsets) else i
                slices.append(
                    ExecutionSlice(
                        cluster_id=str(cluster.cluster_id),
                        subposition_index=int(sp.index),
                        symbol=str(cluster.symbol),
                        side=str(cluster.side),
                        target_weight=float(sp.weight),
                        execution_mode=mode,
                        order_type=str(default_order_type),
                        time_in_force=str(time_in_force),
                        limit_price=None,
                        trigger_price=None,
                        time_offset_bars=int(bar_offset),
                        meta=dict(sp.meta),
                    )
                )
            slices = tuple(slices)

        else:
            slices = tuple(
                ExecutionSlice(
                    cluster_id=str(cluster.cluster_id),
                    subposition_index=int(sp.index),
                    symbol=str(cluster.symbol),
                    side=str(cluster.side),
                    target_weight=float(sp.weight),
                    execution_mode="market",
                    order_type=str(default_order_type),
                    time_in_force=str(time_in_force),
                    meta=dict(sp.meta),
                )
                for sp in cluster.subpositions
                if float(sp.weight) > 0.0
            )
            mode = "market"

        return ExecutionPlan(
            cluster_id=str(cluster.cluster_id),
            symbol=str(cluster.symbol),
            strategy_id=str(cluster.strategy_id),
            side=str(cluster.side),
            total_target_weight=float(cluster.target_weight),
            slices=slices,
            meta={**dict(meta or {}), "execution_mode": mode},
        )
