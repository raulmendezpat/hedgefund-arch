"""Domain-specific exceptions."""


class DealReconstructionError(Exception):
    """Base exception for deal reconstruction."""


class DomainValidationError(DealReconstructionError, ValueError):
    """Raised when a domain object violates a required invariant."""


class PositionStateError(DealReconstructionError):
    """Raised when a fill is incompatible with the current position state."""
