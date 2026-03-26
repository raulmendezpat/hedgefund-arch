from __future__ import annotations

from hf_core.allocation_factory import build_allocation_engine
from hf_core.legacy_allocation_config import LegacyAllocationConfig


def build_legacy_allocation_engine(
    *,
    config: LegacyAllocationConfig,
):
    return build_allocation_engine(config=config)
