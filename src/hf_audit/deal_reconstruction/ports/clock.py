"""Clock abstraction for deterministic orchestration tests."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol


class Clock(Protocol):
    """Provide current time without hard-coding system time."""

    def now_utc(self) -> datetime:
        """Return the current UTC time."""
