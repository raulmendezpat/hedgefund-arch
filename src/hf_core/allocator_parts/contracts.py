from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hf_core.contracts import OpportunityCandidate


@dataclass
class AllocatorContext:
    candidates: list[OpportunityCandidate]
    deployable_rows: list[dict[str, Any]] = field(default_factory=list)
    symbol_raw_weights: dict[str, float] = field(default_factory=dict)
    symbol_weights: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
