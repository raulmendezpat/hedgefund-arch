"""UTC datetime conversion utilities for exchange adapters."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any


UTC = timezone.utc


def _from_numeric_timestamp(value: Any) -> datetime:
    numeric = Decimal(str(value).strip())

    # Exchange timestamps are usually milliseconds. This also supports seconds,
    # microseconds and nanoseconds defensively.
    absolute_value = abs(numeric)

    if absolute_value >= Decimal("100000000000000000"):
        seconds = numeric / Decimal("1000000000")
    elif absolute_value >= Decimal("100000000000000"):
        seconds = numeric / Decimal("1000000")
    elif absolute_value >= Decimal("100000000000"):
        seconds = numeric / Decimal("1000")
    else:
        seconds = numeric

    return datetime.fromtimestamp(float(seconds), tz=UTC)


def to_utc_datetime(
    value: Any,
    *,
    field_name: str = "timestamp",
) -> datetime:
    """Convert ISO strings, datetimes or epoch values to timezone-aware UTC."""

    if value is None or value == "":
        raise ValueError(f"{field_name} is required")

    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{field_name} must be timezone-aware")
        return value.astimezone(UTC)

    if isinstance(value, (int, float, Decimal)):
        return _from_numeric_timestamp(value)

    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} is required")

    try:
        if text.lstrip("+-").replace(".", "", 1).isdigit():
            return _from_numeric_timestamp(text)

        normalized = text
        if normalized.endswith("Z"):
            normalized = f"{normalized[:-1]}+00:00"

        parsed = datetime.fromisoformat(normalized)

        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(f"{field_name} must contain timezone information")

        return parsed.astimezone(UTC)

    except (ValueError, OverflowError, OSError) as exc:
        raise ValueError(
            f"{field_name} is not a supported UTC timestamp: {value!r}"
        ) from exc
