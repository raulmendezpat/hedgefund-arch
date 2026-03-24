from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hf_core.contracts import OpportunityCandidate
from hf_core.policy import PolicyDecision


@dataclass
class SelectionRow:
    idx: int
    ts: int
    symbol: str
    strategy_id: str
    side: str
    signal_strength: float
    base_weight: float
    p_win: float
    post_ml_score: float
    competitive_score: float
    policy_score: float
    policy_band: str
    policy_reason: str
    policy_size_mult: float
    accept_in: bool
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionContext:
    rows: list[SelectionRow]
    selected_idx: list[int]
    trace_rows: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
