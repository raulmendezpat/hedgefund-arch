"""Bitget implementation of the FillNormalizer infrastructure port."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from decimal import Decimal
from typing import Any

from hf_audit.deal_reconstruction.domain.enums import FillAction
from hf_audit.deal_reconstruction.domain.models import NormalizedFill
from hf_audit.deal_reconstruction.infrastructure.bitget_value_resolver import (
    resolve_fill_action,
    resolve_fill_origin,
    resolve_position_side,
)
from hf_audit.deal_reconstruction.infrastructure.datetime_utils import (
    to_utc_datetime,
)
from hf_audit.deal_reconstruction.infrastructure.decimal_utils import (
    ZERO,
    extract_fee_cost,
    to_decimal,
)


class BitgetFillNormalizer:
    """Translate one CCXT/Bitget trade record into ``NormalizedFill``.

    The adapter accepts mapping-like records and lightweight objects so it
    remains decoupled from a concrete CCXT class.
    """

    exchange_name = "bitget"

    def normalize(self, raw_trade: Any) -> NormalizedFill:
        """Return one exchange-independent domain fill."""

        record = self._record_to_mapping(raw_trade)
        info = self._mapping_or_empty(record.get("info"))

        trade_side = self._first_value(
            info,
            record,
            keys=("tradeSide", "trade_side", "action"),
        )
        order_side = self._first_value(
            info,
            record,
            keys=("side", "orderSide", "order_side"),
        )

        action = resolve_fill_action(trade_side)
        position_side = resolve_position_side(
            trade_side=trade_side,
            order_side=order_side,
        )

        symbol = self._required_text(
            self._first_value(
                record,
                info,
                keys=("symbol",),
            ),
            "symbol",
        )

        trade_id = self._required_text(
            self._first_value(
                info,
                record,
                keys=("tradeId", "trade_id", "id"),
            ),
            "trade_id",
        )

        order_id = self._required_text(
            self._first_value(
                info,
                record,
                keys=("orderId", "order_id", "order"),
            ),
            "order_id",
        )

        timestamp_value = self._first_value(
            record,
            info,
            keys=(
                "timestamp",
                "datetime",
                "cTime",
                "createdTime",
                "created_at",
            ),
        )

        quantity_value = self._first_value(
            record,
            info,
            keys=("amount", "quantity", "qty", "baseVolume", "size"),
        )
        price_value = self._first_value(
            record,
            info,
            keys=("price", "fillPrice", "fill_price"),
        )

        quantity = to_decimal(
            quantity_value,
            field_name="quantity",
        )
        price = to_decimal(
            price_value,
            field_name="price",
        )

        realised_pnl = ZERO
        if action is FillAction.CLOSE:
            realised_pnl = to_decimal(
                self._first_value(
                    info,
                    record,
                    keys=(
                        "profit",
                        "realisedPnl",
                        "realizedPnl",
                        "realised_pnl",
                        "realized_pnl",
                    ),
                    default=ZERO,
                ),
                default=ZERO,
                field_name="realised_pnl",
            )

        fee = extract_fee_cost(
            info.get("feeDetail"),
            info.get("fee"),
            record.get("fee"),
            record.get("fees"),
        )

        origin_value = self._first_value(
            info,
            record,
            keys=(
                "enterPointSource",
                "enter_point_source",
                "origin",
            ),
            default="unknown",
        )

        metadata = self._build_metadata(
            info=info,
            record=record,
            action=action,
            order_side=str(order_side or "").strip().lower(),
        )

        raw_reference = self._serialize_reference(
            {
                "trade_id": trade_id,
                "order_id": order_id,
                "symbol": symbol,
                "trade_side": str(trade_side),
                "order_side": str(order_side),
                "origin": str(origin_value),
            }
        )

        return NormalizedFill(
            exchange=self.exchange_name,
            symbol=symbol,
            trade_id=trade_id,
            order_id=order_id,
            timestamp=to_utc_datetime(
                timestamp_value,
                field_name="timestamp",
            ),
            position_side=position_side,
            action=action,
            quantity=quantity,
            price=price,
            fee=fee,
            realised_pnl=realised_pnl,
            origin=resolve_fill_origin(origin_value),
            raw_reference=raw_reference,
            metadata=metadata,
        )

    @staticmethod
    def _record_to_mapping(raw_trade: Any) -> dict[str, Any]:
        if isinstance(raw_trade, Mapping):
            return dict(raw_trade)

        if is_dataclass(raw_trade):
            return dict(asdict(raw_trade))

        if hasattr(raw_trade, "_asdict"):
            return dict(raw_trade._asdict())

        if hasattr(raw_trade, "__dict__"):
            return dict(vars(raw_trade))

        # Some wrapper records expose a raw payload explicitly.
        for attribute_name in ("payload", "data", "raw", "record"):
            value = getattr(raw_trade, attribute_name, None)
            if isinstance(value, Mapping):
                return dict(value)

        raise TypeError(
            "raw_trade must be mapping-like, a dataclass, or expose attributes"
        )

    @staticmethod
    def _mapping_or_empty(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    @staticmethod
    def _first_value(
        *sources: Mapping[str, Any],
        keys: tuple[str, ...],
        default: Any = None,
    ) -> Any:
        for source in sources:
            for key in keys:
                if key in source and source[key] not in (None, ""):
                    return source[key]
        return default

    @staticmethod
    def _required_text(value: Any, field_name: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError(f"{field_name} is required")
        return normalized

    @staticmethod
    def _serialize_reference(value: Mapping[str, Any]) -> str:
        return json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    @staticmethod
    def _build_metadata(
        *,
        info: Mapping[str, Any],
        record: Mapping[str, Any],
        action: FillAction,
        order_side: str,
    ) -> dict[str, str]:
        fields = {
            "trade_side": action.value,
            "order_side": order_side,
            "trade_scope": info.get("tradeScope", ""),
            "position_mode": info.get("posMode", ""),
            "ccxt_type": record.get("type", ""),
            "ccxt_taker_or_maker": record.get("takerOrMaker", ""),
            "ccxt_fee_currency": (
                record.get("fee", {}).get("currency", "")
                if isinstance(record.get("fee"), Mapping)
                else ""
            ),
        }

        return {
            str(key): str(value)
            for key, value in fields.items()
            if value not in (None, "")
        }
