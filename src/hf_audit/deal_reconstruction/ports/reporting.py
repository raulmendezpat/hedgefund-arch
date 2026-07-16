"""Output port for reconstructed-deal audit reports."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable

from hf_audit.deal_reconstruction.domain.reconstruction import (
    DealReconstructionResult,
)


@runtime_checkable
class ReconstructionReportWriter(Protocol):
    """Persist a reconstruction result without owning domain logic."""

    def write(
        self,
        *,
        result: DealReconstructionResult,
        output_dir: Path,
        context: Mapping[str, object] | None = None,
    ) -> Mapping[str, Path]:
        """Write reports and return their generated paths."""
