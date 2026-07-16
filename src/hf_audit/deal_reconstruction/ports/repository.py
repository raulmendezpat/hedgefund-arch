"""Output port for reconstructed deals."""

from __future__ import annotations

from typing import Iterable, Protocol

from hf_audit.deal_reconstruction.domain.models import Deal


class DealRepository(Protocol):
    """Persist or export reconstructed deals."""

    def save_all(self, deals: Iterable[Deal]) -> None:
        """Persist the supplied deals."""
