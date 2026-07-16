"""Analytics port independent from exchange and reporting technology."""

from __future__ import annotations

from typing import Iterable, Mapping, Protocol

from hf_audit.deal_reconstruction.domain.models import Deal


class DealAnalytics(Protocol):
    """Calculate aggregate statistics from reconstructed deals."""

    def calculate(self, deals: Iterable[Deal]) -> Mapping[str, object]:
        """Return serializable deal statistics."""
