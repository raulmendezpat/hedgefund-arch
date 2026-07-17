#!/usr/bin/env python3
"""Export read-only Bitget trade history as canonical normalized-fill CSV.

The script performs exchange read operations only through CCXT
``fetch_my_trades``. It does not create, cancel or amend orders, positions,
leverage, margin mode, take-profit plans or stop-loss plans.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from hf_audit.deal_reconstruction.infrastructure.bitget_ccxt_trade_fetcher import (
    BitgetCcxtTradeFetcher,
)
from hf_audit.deal_reconstruction.infrastructure.bitget_fill_normalizer import (
    BitgetFillNormalizer,
)


UTC = timezone.utc

NORMALIZED_COLUMNS = [
    "exchange",
    "symbol",
    "trade_id",
    "order_id",
    "timestamp",
    "position_side",
    "action",
    "quantity",
    "price",
    "fee",
    "realised_pnl",
    "origin",
    "raw_reference",
    "metadata",
]

DEFAULT_SYMBOLS = [
    "AAVE/USDT:USDT",
    "ADA/USDT:USDT",
    "AVAX/USDT:USDT",
    "BCH/USDT:USDT",
    "BNB/USDT:USDT",
    "BTC/USDT:USDT",
    "DOGE/USDT:USDT",
    "DOT/USDT:USDT",
    "ENA/USDT:USDT",
    "ETH/USDT:USDT",
    "INJ/USDT:USDT",
    "LINK/USDT:USDT",
    "NEAR/USDT:USDT",
    "ONDO/USDT:USDT",
    "RENDER/USDT:USDT",
    "SOL/USDT:USDT",
    "SUI/USDT:USDT",
    "TRX/USDT:USDT",
    "XLM/USDT:USDT",
    "XRP/USDT:USDT",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Bitget trade history through CCXT read-only APIs and "
            "write canonical normalized fills."
        )
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination normalized-fill CSV.",
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated CCXT Bitget symbols.",
    )
    parser.add_argument(
        "--start",
        help=(
            "Inclusive UTC start timestamp. Accepts ISO-8601, including a "
            "trailing Z."
        ),
    )
    parser.add_argument(
        "--end",
        help=(
            "Exclusive UTC end timestamp. Defaults to the current UTC time."
        ),
    )
    parser.add_argument(
        "--lookback-hours",
        type=float,
        default=168.0,
        help=(
            "Lookback used when --start is omitted. Default: 168 hours "
            "(7 days)."
        ),
    )
    parser.add_argument(
        "--page-limit",
        type=int,
        default=100,
        help="Maximum rows requested per CCXT page.",
    )
    parser.add_argument(
        "--max-pages-per-symbol",
        type=int,
        default=100,
        help="Safety limit for pagination per symbol.",
    )
    parser.add_argument(
        "--summary-json",
        help="Optional JSON execution summary path.",
    )
    return parser.parse_args()


def first_environment_value(*names: str) -> str | None:
    """Return the first non-empty environment variable from ``names``."""

    for name in names:
        value = os.environ.get(name)
        if value:
            return value

    return None


def parse_utc_datetime(value: str) -> datetime:
    """Parse an ISO-8601 timestamp and return an aware UTC datetime."""

    cleaned = str(value).strip()

    if not cleaned:
        raise ValueError("timestamp cannot be empty")

    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"

    parsed = datetime.fromisoformat(cleaned)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)

    return parsed.astimezone(UTC)


def resolve_window(
    *,
    start_text: str | None,
    end_text: str | None,
    lookback_hours: float,
    now: datetime | None = None,
) -> tuple[datetime, datetime]:
    """Resolve and validate the requested UTC fetch window."""

    if lookback_hours <= 0:
        raise ValueError("lookback_hours must be greater than zero")

    resolved_end = (
        parse_utc_datetime(end_text)
        if end_text
        else (now or datetime.now(UTC)).astimezone(UTC)
    )

    resolved_start = (
        parse_utc_datetime(start_text)
        if start_text
        else resolved_end - timedelta(hours=lookback_hours)
    )

    if resolved_start >= resolved_end:
        raise ValueError("start must be earlier than end")

    return resolved_start, resolved_end


def parse_symbols(value: str) -> list[str]:
    """Parse, deduplicate and validate comma-separated symbols."""

    symbols: list[str] = []
    seen: set[str] = set()

    for raw_symbol in str(value).split(","):
        symbol = raw_symbol.strip()

        if not symbol or symbol in seen:
            continue

        seen.add(symbol)
        symbols.append(symbol)

    if not symbols:
        raise ValueError("at least one symbol is required")

    return symbols


def enum_or_value(value: Any) -> Any:
    """Return an enum value or the original scalar value."""

    if isinstance(value, Enum):
        return value.value

    return value


def datetime_to_utc_text(value: Any) -> str:
    """Serialize a datetime as canonical UTC ISO-8601 text."""

    if not isinstance(value, datetime):
        raise TypeError(
            "normalized fill timestamp must be a datetime instance"
        )

    aware = value

    if aware.tzinfo is None:
        aware = aware.replace(tzinfo=UTC)

    return aware.astimezone(UTC).isoformat().replace("+00:00", "Z")


def serialize_json_value(value: Any) -> str:
    """Serialize structured audit values deterministically."""

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def normalized_fill_to_row(fill: Any) -> dict[str, str]:
    """Convert one canonical NormalizedFill into the CSV contract."""

    row = {
        "exchange": str(enum_or_value(fill.exchange)),
        "symbol": str(fill.symbol),
        "trade_id": str(fill.trade_id),
        "order_id": str(fill.order_id or ""),
        "timestamp": datetime_to_utc_text(fill.timestamp),
        "position_side": str(enum_or_value(fill.position_side)),
        "action": str(enum_or_value(fill.action)),
        "quantity": str(fill.quantity),
        "price": str(fill.price),
        "fee": str(fill.fee),
        "realised_pnl": str(fill.realised_pnl),
        "origin": str(enum_or_value(fill.origin)),
        "raw_reference": serialize_json_value(fill.raw_reference),
        "metadata": serialize_json_value(fill.metadata),
    }

    missing = [
        column
        for column in NORMALIZED_COLUMNS
        if column not in row
    ]

    if missing:
        raise ValueError(
            f"normalized row is missing columns: {missing}"
        )

    return row


def load_bitget_credentials(
    *,
    secret_path: Path | None = None,
) -> tuple[dict[str, str], str]:
    """Resolve Bitget credentials without exposing their values.

    Credential precedence:

    1. Existing environment variables.
    2. The ``envelope`` object in the repository ``secret.json``.

    The returned source label contains no secret material.
    """

    api_key = first_environment_value(
        "BITGET_API_KEY",
        "BITGET_KEY",
        "CCXT_BITGET_API_KEY",
        "EXCHANGE_API_KEY",
        "API_KEY",
    )
    secret = first_environment_value(
        "BITGET_API_SECRET",
        "BITGET_SECRET",
        "CCXT_BITGET_SECRET",
        "EXCHANGE_API_SECRET",
        "API_SECRET",
        "SECRET",
    )
    password = first_environment_value(
        "BITGET_API_PASSWORD",
        "BITGET_PASSWORD",
        "CCXT_BITGET_PASSWORD",
        "EXCHANGE_API_PASSWORD",
        "API_PASSWORD",
        "PASSPHRASE",
    )

    environment_values = {
        "apiKey": api_key,
        "secret": secret,
        "password": password,
    }

    if all(environment_values.values()):
        return (
            {
                key: str(value)
                for key, value in environment_values.items()
            },
            "environment",
        )

    if secret_path is None:
        configured_path = os.environ.get(
            "BITGET_SECRET_JSON",
            "",
        ).strip()

        if configured_path:
            secret_path = Path(configured_path).expanduser()
        else:
            secret_path = (
                Path(__file__).resolve().parents[1]
                / "secret.json"
            )

    secret_path = Path(secret_path)

    envelope: Mapping[str, Any] = {}

    if secret_path.is_file():
        try:
            document = json.loads(
                secret_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "unable to load Bitget credential configuration "
                f"from {secret_path}"
            ) from exc

        if not isinstance(document, Mapping):
            raise RuntimeError(
                "Bitget credential configuration must contain "
                "a JSON object"
            )

        candidate = document.get("envelope", {})

        if isinstance(candidate, Mapping):
            envelope = candidate

    resolved = {
        "apiKey": api_key or envelope.get("apiKey"),
        "secret": secret or envelope.get("secret"),
        "password": password or envelope.get("password"),
    }

    missing = [
        name
        for name, value in (
            ("api_key", resolved["apiKey"]),
            ("secret", resolved["secret"]),
            ("password", resolved["password"]),
        )
        if not isinstance(value, str) or not value.strip()
    ]

    if missing:
        raise RuntimeError(
            "missing Bitget credential components: "
            + ", ".join(missing)
        )

    return (
        {
            key: str(value).strip()
            for key, value in resolved.items()
        },
        "secret_json",
    )


def build_exchange_from_environment() -> Any:
    """Create an authenticated read-only CCXT Bitget client."""

    credentials, credential_source = load_bitget_credentials()

    import ccxt

    exchange = ccxt.bitget(
        {
            **credentials,
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",
                "defaultSubType": "linear",
                "defaultSettle": "USDT",
                "adjustForTimeDifference": True,
            },
        }
    )

    exchange.load_markets()

    print(
        "BITGET_CREDENTIAL_SOURCE="
        f"{credential_source}"
    )

    return exchange



def normalize_records(
    raw_records: Iterable[Mapping[str, object]],
    *,
    normalizer: BitgetFillNormalizer | None = None,
) -> list[Any]:
    """Normalize every raw record without silently dropping errors."""

    active_normalizer = normalizer or BitgetFillNormalizer()
    normalized = []

    for index, raw_record in enumerate(raw_records, start=1):
        try:
            normalized.append(
                active_normalizer.normalize(raw_record)
            )
        except Exception as exc:
            raise RuntimeError(
                f"normalization failed at raw record {index}: {exc}"
            ) from exc

    normalized.sort(
        key=lambda fill: (
            fill.timestamp,
            str(fill.symbol),
            str(fill.trade_id),
        )
    )

    return normalized


def write_normalized_csv_atomic(
    path: Path,
    fills: Sequence[Any],
) -> None:
    """Write the canonical CSV atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            writer = csv.DictWriter(
                temporary_file,
                fieldnames=NORMALIZED_COLUMNS,
                extrasaction="raise",
            )
            writer.writeheader()

            for fill in fills:
                writer.writerow(normalized_fill_to_row(fill))

        temporary_path.replace(path)

    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def write_summary_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Write a JSON summary atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")

    try:
        temporary_path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()

    symbols = parse_symbols(args.symbols)
    start, end = resolve_window(
        start_text=args.start,
        end_text=args.end,
        lookback_hours=args.lookback_hours,
    )

    output_path = Path(args.output)
    exchange = build_exchange_from_environment()

    fetcher = BitgetCcxtTradeFetcher(
        exchange,
        page_limit=args.page_limit,
        max_pages_per_symbol=args.max_pages_per_symbol,
    )

    raw_records = list(
        fetcher.fetch_trades(
            symbols=symbols,
            start=start,
            end=end,
        )
    )

    normalized_fills = normalize_records(raw_records)

    write_normalized_csv_atomic(
        output_path,
        normalized_fills,
    )

    summary = {
        "status": "ok",
        "mode": "read_only",
        "exchange": "bitget",
        "start_utc": start.isoformat().replace("+00:00", "Z"),
        "end_utc": end.isoformat().replace("+00:00", "Z"),
        "symbol_count": len(symbols),
        "symbols": symbols,
        "raw_record_count": len(raw_records),
        "normalized_fill_count": len(normalized_fills),
        "output_csv": str(output_path),
        "output_columns": NORMALIZED_COLUMNS,
    }

    if args.summary_json:
        write_summary_atomic(
            Path(args.summary_json),
            summary,
        )

    print("BITGET_NORMALIZED_FILL_EXPORT")
    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
