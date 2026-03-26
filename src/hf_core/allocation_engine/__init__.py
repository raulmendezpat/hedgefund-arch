from hf_core.allocation_engine.contracts import AllocationCandidate, AllocationResult
from hf_core.allocation_engine.engine import SnapshotAllocatorEngine
from hf_core.allocation_engine.factory import build_snapshot_allocator

__all__ = [
    "AllocationCandidate",
    "AllocationResult",
    "SnapshotAllocatorEngine",
    "build_snapshot_allocator",
]
from hf_core.legacy_allocation_engine import LegacyAllocationEngine


class AllocationEngine(LegacyAllocationEngine):
    """Canonical allocation engine name during migration."""
    pass
