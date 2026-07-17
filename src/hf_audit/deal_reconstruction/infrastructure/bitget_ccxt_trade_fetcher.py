"""Read-only Bitget trade fetcher backed by a CCXT exchange instance.

This adapter performs exchange read operations only. It does not create,
cancel, amend or otherwise modify orders, positions, leverage, margin mode,
take-profit plans or stop-loss plans.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from time import sleep
from typing import Any

from hf_audit.deal_reconstruction.ports.fetcher import RawTradeRecord


UTC = timezone.utc


class BitgetCcxtTradeFetcher:
    """Fetch Bitget fills through CCXT's read-only ``fetch_my_trades`` API."""

    def __init__(
        self,
        exchange: Any,
        *,
        page_limit: int = 100,
        max_pages_per_symbol: int = 100,
        rate_limit_seconds: float | None = None,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        if page_limit <= 0:
            raise ValueError("page_limit must be greater than zero")

        if max_pages_per_symbol <= 0:
            raise ValueError("max_pages_per_symbol must be greater than zero")

        if rate_limit_seconds is not None and rate_limit_seconds < 0:
            raise ValueError("rate_limit_seconds cannot be negative")

        if not hasattr(exchange, "fetch_my_trades"):
            raise TypeError(
                "exchange must expose a fetch_my_trades method"
            )

        self._exchange = exchange
        self._page_limit = int(page_limit)
        self._max_pages_per_symbol = int(max_pages_per_symbol)
        self._rate_limit_seconds = rate_limit_seconds
        self._sleeper = sleeper

    def fetch_trades(
        self,
        *,
        symbols: Iterable[str],
        start: datetime,
        end: datetime,
    ) -> Iterable[RawTradeRecord]:
        """Return unique raw fills ordered chronologically.

        ``start`` is inclusive and ``end`` is exclusive.

        Pagination advances using the greatest exchange timestamp returned by
        the preceding page. Duplicate rows returned across adjacent pages are
        removed before the records are returned.
        """

        start_utc = self._as_utc(start, field_name="start")
        end_utc = self._as_utc(end, field_name="end")

        if end_utc <= start_utc:
            raise ValueError("end must be later than start")

        start_ms = self._datetime_to_ms(start_utc)
        end_ms = self._datetime_to_ms(end_utc)

        normalized_symbols = self._normalize_symbols(symbols)

        collected: list[dict[str, object]] = []
        seen_keys: set[tuple[str, str, str, int]] = set()

        for symbol in normalized_symbols:
            cursor_ms = start_ms

            for _page_number in range(self._max_pages_per_symbol):
                rows = self._exchange.fetch_my_trades(
                    symbol=symbol,
                    since=cursor_ms,
                    limit=self._page_limit,
                )

                page = list(rows or [])

                if not page:
                    break

                greatest_timestamp_ms: int | None = None
                page_reached_end = False

                for raw_trade in page:
                    record = self._record_to_mapping(raw_trade)
                    timestamp_ms = self._extract_timestamp_ms(record)

                    if timestamp_ms is None:
                        continue

                    if (
                        greatest_timestamp_ms is None
                        or timestamp_ms > greatest_timestamp_ms
                    ):
                        greatest_timestamp_ms = timestamp_ms

                    if timestamp_ms >= end_ms:
                        page_reached_end = True
                        continue

                    if timestamp_ms < start_ms:
                        continue

                    enriched_record = dict(record)
                    enriched_record.setdefault(
                        "requested_symbol",
                        symbol,
                    )
                    enriched_record.setdefault(
                        "_source",
                        "fetch_my_trades",
                    )

                    dedupe_key = self._dedupe_key(
                        enriched_record,
                        requested_symbol=symbol,
                        timestamp_ms=timestamp_ms,
                    )

                    if dedupe_key in seen_keys:
                        continue

                    seen_keys.add(dedupe_key)
                    collected.append(enriched_record)

                if page_reached_end:
                    break

                if greatest_timestamp_ms is None:
                    break

                next_cursor_ms = greatest_timestamp_ms + 1

                if next_cursor_ms <= cursor_ms:
                    break

                cursor_ms = next_cursor_ms

                if len(page) < self._page_limit:
                    break

                delay_seconds = self._resolve_rate_limit_seconds()

                if delay_seconds > 0:
                    self._sleeper(delay_seconds)

        collected.sort(
            key=lambda record: (
                self._extract_timestamp_ms(record) or 0,
                self._clean_text(record.get("id")),
                self._clean_text(record.get("order")),
                self._clean_text(record.get("requested_symbol")),
            )
        )

        return collected

    def _resolve_rate_limit_seconds(self) -> float:
        if self._rate_limit_seconds is not None:
            return float(self._rate_limit_seconds)

        exchange_rate_limit_ms = getattr(
            self._exchange,
            "rateLimit",
            0,
        )

        try:
            return max(0.0, float(exchange_rate_limit_ms) / 1000.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _normalize_symbols(symbols: Iterable[str]) -> tuple[str, ...]:
        ordered_symbols: list[str] = []
        seen_symbols: set[str] = set()

        for raw_symbol in symbols:
            symbol = str(raw_symbol).strip()

            if not symbol:
                raise ValueError("symbols cannot contain blank values")

            if symbol in seen_symbols:
                continue

            seen_symbols.add(symbol)
            ordered_symbols.append(symbol)

        if not ordered_symbols:
            raise ValueError("at least one symbol is required")

        return tuple(ordered_symbols)

    @staticmethod
    def _as_utc(value: datetime, *, field_name: str) -> datetime:
        if not isinstance(value, datetime):
            raise TypeError(f"{field_name} must be a datetime")

        if value.tzinfo is None:
            raise ValueError(
                f"{field_name} must be timezone-aware"
            )

        return value.astimezone(UTC)

    @staticmethod
    def _datetime_to_ms(value: datetime) -> int:
        return int(value.timestamp() * 1000)

    @staticmethod
    def _record_to_mapping(
        raw_trade: object,
    ) -> Mapping[str, object]:
        if isinstance(raw_trade, Mapping):
            return raw_trade

        raise TypeError(
            "fetch_my_trades returned a non-mapping trade record"
        )

    @classmethod
    def _extract_timestamp_ms(
        cls,
        record: Mapping[str, object],
    ) -> int | None:
        direct_timestamp = cls._to_timestamp_ms(
            record.get("timestamp")
        )

        if direct_timestamp is not None:
            return direct_timestamp

        info = record.get("info")

        if isinstance(info, Mapping):
            for key in ("cTime", "uTime", "timestamp"):
                timestamp_ms = cls._to_timestamp_ms(
                    info.get(key)
                )

                if timestamp_ms is not None:
                    return timestamp_ms

        datetime_value = record.get("datetime")

        if isinstance(datetime_value, str):
            text = datetime_value.strip()

            if text:
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"

                try:
                    parsed = datetime.fromisoformat(text)
                except ValueError:
                    return None

                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)

                return cls._datetime_to_ms(
                    parsed.astimezone(UTC)
                )

        return None

    @staticmethod
    def _to_timestamp_ms(value: object) -> int | None:
        if value is None:
            return None

        try:
            numeric_value = int(float(str(value).strip()))
        except (TypeError, ValueError):
            return None

        if numeric_value <= 0:
            return None

        # Accept Unix seconds defensively, although CCXT normally returns ms.
        if numeric_value < 10_000_000_000:
            numeric_value *= 1000

        return numeric_value

    @classmethod
    def _dedupe_key(
        cls,
        record: Mapping[str, object],
        *,
        requested_symbol: str,
        timestamp_ms: int,
    ) -> tuple[str, str, str, int]:
        info = record.get("info")
        info_mapping = info if isinstance(info, Mapping) else {}

        trade_id = cls._first_text(
            record.get("id"),
            info_mapping.get("tradeId"),
            info_mapping.get("id"),
        )

        order_id = cls._first_text(
            record.get("order"),
            info_mapping.get("orderId"),
        )

        record_symbol = cls._first_text(
            record.get("symbol"),
            info_mapping.get("symbol"),
            requested_symbol,
        )

        return (
            record_symbol,
            trade_id,
            order_id,
            timestamp_ms,
        )

    @classmethod
    def _first_text(cls, *values: object) -> str:
        for value in values:
            text = cls._clean_text(value)

            if text:
                return text

        return ""

    @staticmethod
    def _clean_text(value: object) -> str:
        if value is None:
            return ""

        return str(value).strip()
