"""Pure Bitget-to-domain semantic resolvers."""

from __future__ import annotations

from typing import Any

from hf_audit.deal_reconstruction.domain.enums import (
    FillAction,
    FillOrigin,
    PositionSide,
)


def _normalized_text(value: Any) -> str:
    return str(value or "").strip().lower()


def resolve_fill_action(trade_side: Any) -> FillAction:
    """Resolve Bitget ``tradeSide`` to a domain fill action."""

    normalized = _normalized_text(trade_side)

    aliases = {
        "open": FillAction.OPEN,
        "open_long": FillAction.OPEN,
        "open_short": FillAction.OPEN,
        "close": FillAction.CLOSE,
        "close_long": FillAction.CLOSE,
        "close_short": FillAction.CLOSE,
        "reduce": FillAction.CLOSE,
        "reduce_only": FillAction.CLOSE,
    }

    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Bitget tradeSide value: {trade_side!r}"
        ) from exc


def resolve_position_side(
    *,
    trade_side: Any,
    order_side: Any,
) -> PositionSide:
    """Resolve hedge-mode position direction.

    Bitget semantics:

    - ``open + buy``  -> open long
    - ``open + sell`` -> open short
    - ``close + sell`` -> close long
    - ``close + buy``  -> close short
    """

    action = resolve_fill_action(trade_side)
    side = _normalized_text(order_side)

    if side not in {"buy", "sell"}:
        raise ValueError(f"Unsupported Bitget side value: {order_side!r}")

    if action is FillAction.OPEN:
        return PositionSide.LONG if side == "buy" else PositionSide.SHORT

    return PositionSide.LONG if side == "sell" else PositionSide.SHORT


def resolve_fill_origin(value: Any) -> FillOrigin:
    """Resolve Bitget ``enterPointSource`` without leaking raw values."""

    return FillOrigin.from_exchange_value(value)
