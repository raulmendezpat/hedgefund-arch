from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class PolicyDecision:
    ts: int
    symbol: str
    strategy_id: str
    side: str
    accept: bool
    size_mult: float
    policy_score: float
    band: str
    reason: str
    policy_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyState:
    ts: int
    symbol: str
    strategy_id: str
    side: str
    p_win: float
    expected_return: float
    score: float
    model_meta: dict[str, Any] = field(default_factory=dict)

    accept: bool = True
    size_mult: float = 0.0
    band: str = "reject"
    reason: str = "init"
    tags: dict[str, Any] = field(default_factory=dict)


class PolicyRule(Protocol):
    def apply(self, state: PolicyState) -> PolicyState: ...
