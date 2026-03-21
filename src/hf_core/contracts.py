from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OpportunityCandidate:
    ts: int
    symbol: str
    strategy_id: str
    side: str
    signal_strength: float
    base_weight: float
    signal_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureRow:
    ts: int
    symbol: str
    strategy_id: str
    side: str
    values: dict[str, float | int | str]


@dataclass
class MetaScore:
    ts: int
    symbol: str
    strategy_id: str
    side: str
    p_win: float
    expected_return: float
    score: float
    model_meta: dict[str, Any] = field(default_factory=dict)
