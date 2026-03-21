from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .policy_parts.contracts import PolicyDecision, PolicyState
from .policy_parts.factory import PolicyFactory


@dataclass
class PolicyModel:
    profile: str = "default"
    config: dict[str, Any] = field(default_factory=dict)

    def _decide_one(self, meta) -> PolicyDecision:
        state = PolicyState(
            ts=int(getattr(meta, "ts", 0) or 0),
            symbol=str(getattr(meta, "symbol", "") or ""),
            strategy_id=str(getattr(meta, "strategy_id", "") or ""),
            side=str(getattr(meta, "side", "flat") or "flat").lower(),
            p_win=float(getattr(meta, "p_win", 0.0) or 0.0),
            expected_return=float(getattr(meta, "expected_return", 0.0) or 0.0),
            score=float(getattr(meta, "score", 0.0) or 0.0),
            model_meta=dict(getattr(meta, "model_meta", {}) or {}),
        )
        pipeline = PolicyFactory(profile=str(self.profile), config=dict(self.config or {})).build()
        return pipeline.run(state)

    def decide_many(self, metas) -> list[PolicyDecision]:
        return [self._decide_one(m) for m in list(metas or [])]
