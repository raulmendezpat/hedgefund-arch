from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AllocationCandidate:
    symbol: str
    side: str
    score: float
    p_win: float = 0.5
    expected_return: float = 0.0
    base_weight: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class AllocationResult:
    weights: dict[str, float]
    meta: dict[str, Any] = field(default_factory=dict)
