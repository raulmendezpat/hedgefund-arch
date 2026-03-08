from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from hf.core.types import Signal


@dataclass
class Opportunity:
    strategy_id: str
    symbol: str
    side: str
    strength: float
    timestamp: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_signal(
        cls,
        signal: Signal,
        strategy_id: Optional[str] = None,
        timestamp: Optional[int] = None,
    ) -> "Opportunity":
        meta = dict(getattr(signal, "meta", {}) or {})
        resolved_strategy_id = str(
            strategy_id
            or meta.get("strategy_id")
            or "unknown_strategy"
        )

        return cls(
            strategy_id=resolved_strategy_id,
            symbol=str(signal.symbol),
            side=str(signal.side),
            strength=float(getattr(signal, "strength", 0.0) or 0.0),
            timestamp=timestamp,
            meta=meta,
        )

    def is_active(self) -> bool:
        if self.side == "flat":
            return False
        if "skip" in (self.meta or {}):
            return False
        return True
