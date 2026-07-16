"""Decimal conversion utilities for exchange adapters.

This module is infrastructure-only. Domain entities remain unaware of raw
exchange formats and floating-point representations.
"""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping


ZERO = Decimal("0")


def to_decimal(
    value: Any,
    *,
    default: Decimal | None = None,
    field_name: str = "value",
) -> Decimal:
    """Convert a raw value to ``Decimal`` without using binary float arithmetic."""

    if value is None or value == "":
        if default is not None:
            return default
        raise ValueError(f"{field_name} is required")

    if isinstance(value, Decimal):
        return value

    if isinstance(value, bool):
        raise ValueError(f"{field_name} cannot be boolean")

    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(
            f"{field_name} must be a valid decimal-compatible value: {value!r}"
        ) from exc


def non_negative_decimal(
    value: Any,
    *,
    default: Decimal = ZERO,
    field_name: str = "value",
) -> Decimal:
    """Return a non-negative Decimal."""

    result = to_decimal(
        value,
        default=default,
        field_name=field_name,
    )

    if result < ZERO:
        raise ValueError(f"{field_name} cannot be negative")

    return result


def absolute_decimal(
    value: Any,
    *,
    default: Decimal = ZERO,
    field_name: str = "value",
) -> Decimal:
    """Convert a potentially signed exchange cost into a positive cost."""

    return abs(
        to_decimal(
            value,
            default=default,
            field_name=field_name,
        )
    )


def parse_json_like(value: Any) -> Any:
    """Decode JSON text while preserving already-structured values."""

    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return None

    if not (
        (text.startswith("{") and text.endswith("}"))
        or (text.startswith("[") and text.endswith("]"))
    ):
        return value

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _extract_fee_values(value: Any) -> Iterable[Any]:
    """Yield fee-cost candidates from Bitget or CCXT fee structures."""

    value = parse_json_like(value)

    if value is None or value == "":
        return

    if isinstance(value, Mapping):
        preferred_keys = (
            "fee",
            "feeCost",
            "fee_cost",
            "cost",
            "amount",
            "totalFee",
            "total_fee",
        )

        yielded = False
        for key in preferred_keys:
            if key in value and value[key] not in (None, ""):
                yielded = True
                yield value[key]

        if not yielded:
            for nested_value in value.values():
                if isinstance(nested_value, (Mapping, list, tuple)):
                    yield from _extract_fee_values(nested_value)
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _extract_fee_values(item)
        return

    yield value


def extract_fee_cost(*fee_candidates: Any) -> Decimal:
    """Aggregate exchange fee candidates as a non-negative cost.

    Bitget and CCXT may represent commissions as negative numbers, positive
    costs, dictionaries, JSON text, or lists. The domain contract always stores
    ``fee`` as a non-negative cost.
    """

    total = ZERO

    for candidate in fee_candidates:
        for raw_fee in _extract_fee_values(candidate):
            try:
                total += absolute_decimal(
                    raw_fee,
                    default=ZERO,
                    field_name="fee",
                )
            except ValueError:
                continue

    return total
