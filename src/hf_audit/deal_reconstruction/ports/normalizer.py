"""Port for translating raw exchange data into domain fills."""

from __future__ import annotations

from typing import Protocol

from hf_audit.deal_reconstruction.domain.models import NormalizedFill
from hf_audit.deal_reconstruction.ports.fetcher import RawTradeRecord


class FillNormalizer(Protocol):
    """Convert one raw exchange trade to a normalized fill."""

    def normalize(self, raw_trade: RawTradeRecord) -> NormalizedFill:
        """Return an exchange-independent fill."""
