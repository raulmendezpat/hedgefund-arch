from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hf.engines.position_cluster import PositionCluster, PlannedSubPosition


@dataclass(frozen=True)
class ClusterRiskDecision:
    approved: bool
    adjusted_target_weight: float
    adjusted_subpositions: tuple[PlannedSubPosition, ...] = field(default_factory=tuple)
    reasons: tuple[str, ...] = field(default_factory=tuple)
    meta: dict[str, Any] = field(default_factory=dict)


class ClusterRiskEngine:
    def __init__(
        self,
        *,
        max_cluster_weight: float = 1.0,
        max_subpositions: int = 10,
        allow_zero_weight_clusters: bool = True,
    ) -> None:
        self.max_cluster_weight = float(max_cluster_weight)
        self.max_subpositions = int(max_subpositions)
        self.allow_zero_weight_clusters = bool(allow_zero_weight_clusters)

    def evaluate(self, cluster: PositionCluster) -> ClusterRiskDecision:
        reasons: list[str] = []

        capped_target = min(float(cluster.target_weight), float(self.max_cluster_weight))
        if capped_target < float(cluster.target_weight):
            reasons.append("target_weight_capped")

        subs = tuple(cluster.subpositions[: max(0, self.max_subpositions)])
        if len(subs) < len(cluster.subpositions):
            reasons.append("subpositions_truncated")

        total_sub_weight = float(sum(float(x.weight) for x in subs))
        if total_sub_weight > capped_target and total_sub_weight > 0.0:
            scale = capped_target / total_sub_weight
            subs = tuple(
                PlannedSubPosition(
                    index=sp.index,
                    weight=float(sp.weight) * scale,
                    entry_ts=sp.entry_ts,
                    entry_price=sp.entry_price,
                    status=sp.status,
                    meta=dict(sp.meta),
                )
                for sp in subs
            )
            reasons.append("subpositions_rescaled")

        if capped_target <= 0.0 and not self.allow_zero_weight_clusters:
            reasons.append("zero_weight_rejected")
            return ClusterRiskDecision(
                approved=False,
                adjusted_target_weight=0.0,
                adjusted_subpositions=tuple(),
                reasons=tuple(reasons),
                meta={"cluster_id": cluster.cluster_id},
            )

        return ClusterRiskDecision(
            approved=True,
            adjusted_target_weight=float(capped_target),
            adjusted_subpositions=subs,
            reasons=tuple(reasons),
            meta={"cluster_id": cluster.cluster_id},
        )
