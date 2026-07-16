#!/usr/bin/env python3
"""Read-only Bitget-native deal reconstruction audit runner.

This runner is intentionally separated from:

- exchange order execution;
- live reconciliation;
- strategy lifecycle logic;
- trading configuration;
- TP and SL management.

The runner consumes normalized exchange fills and delegates deal construction
to the exchange-independent application service.
"""

from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from hf_audit.deal_reconstruction.application import DealReconstructor
from hf_audit.deal_reconstruction.infrastructure.reconstruction_report_writer import CsvJsonReconstructionReportWriter
from hf_audit.deal_reconstruction.ports.reporting import ReconstructionReportWriter
from hf_audit.deal_reconstruction.domain.enums import (
    FillAction,
    FillOrigin,
    PositionSide,
)
from hf_audit.deal_reconstruction.domain.models import NormalizedFill


ZERO = Decimal("0")

MANUAL_ORIGINS = {
    FillOrigin.IOS,
    FillOrigin.WEB,
    FillOrigin.ANDROID,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments without coupling to exchange APIs."""

    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct exchange-native deals from a normalized fill CSV."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Normalized fill CSV produced by the Bitget fill normalizer.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where audit outputs will be written.",
    )

    parser.add_argument(
        "--expected-rows",
        type=int,
        default=None,
        help="Optional exact expected input row count.",
    )

    return parser.parse_args()


def clean_text(value: Any) -> str:
    """Return a stable stripped string for CSV and JSON values."""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def parse_decimal(
    value: Any,
    *,
    default: str = "0",
) -> Decimal:
    """Convert a scalar value to Decimal without float arithmetic."""

    raw = clean_text(value)

    return Decimal(
        raw if raw else default
    )


def parse_utc_datetime(value: Any) -> datetime:
    """Convert ISO-8601 text to an aware UTC datetime."""

    raw = clean_text(value)

    if not raw:
        raise ValueError(
            "timestamp cannot be empty"
        )

    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"

    parsed = datetime.fromisoformat(raw)

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed.astimezone(
        timezone.utc
    )


def parse_metadata(
    row: Mapping[str, Any],
) -> dict[str, str]:
    """Read metadata from either a serialized field or flattened columns."""

    metadata: dict[str, str] = {}

    raw_metadata = clean_text(
        row.get("metadata")
    )

    if raw_metadata:
        parsed: Any

        try:
            parsed = json.loads(
                raw_metadata
            )
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(
                    raw_metadata
                )
            except (SyntaxError, ValueError):
                parsed = {}

        if isinstance(parsed, Mapping):
            metadata.update(
                {
                    str(key): clean_text(value)
                    for key, value in parsed.items()
                }
            )

    for key, value in row.items():
        key_text = str(key)

        if not key_text.startswith(
            "metadata_"
        ):
            continue

        resolved_value = clean_text(
            value
        )

        if not resolved_value:
            continue

        metadata[
            key_text.removeprefix(
                "metadata_"
            )
        ] = resolved_value

    return metadata


def load_normalized_fills(
    path: Path,
) -> tuple[
    list[NormalizedFill],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load normalized fills and return fills, source frame and errors."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Input file does not exist: {path}"
        )

    source_frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
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
    }

    missing_columns = sorted(
        required_columns.difference(
            source_frame.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Normalized input is missing required columns: "
            + ", ".join(missing_columns)
        )

    fills: list[NormalizedFill] = []
    error_rows: list[dict[str, Any]] = []

    source_records = source_frame.to_dict(
        orient="records"
    )

    for row_number, row in enumerate(
        source_records,
        start=2,
    ):
        try:
            origin_value = (
                clean_text(
                    row.get("origin")
                ).lower()
                or FillOrigin.UNKNOWN.value
            )

            fill = NormalizedFill(
                exchange=clean_text(
                    row.get("exchange")
                ),
                symbol=clean_text(
                    row.get("symbol")
                ),
                trade_id=clean_text(
                    row.get("trade_id")
                ),
                order_id=clean_text(
                    row.get("order_id")
                ),
                timestamp=parse_utc_datetime(
                    row.get("timestamp")
                ),
                position_side=PositionSide(
                    clean_text(
                        row.get("position_side")
                    ).lower()
                ),
                action=FillAction(
                    clean_text(
                        row.get("action")
                    ).lower()
                ),
                quantity=parse_decimal(
                    row.get("quantity")
                ),
                price=parse_decimal(
                    row.get("price")
                ),
                fee=parse_decimal(
                    row.get("fee")
                ),
                realised_pnl=parse_decimal(
                    row.get("realised_pnl")
                ),
                origin=FillOrigin(
                    origin_value
                ),
                raw_reference=clean_text(
                    row.get("raw_reference")
                ),
                metadata=parse_metadata(
                    row
                ),
            )

            fills.append(fill)

        except Exception as exc:
            error_rows.append(
                {
                    "source_row_number": row_number,
                    "symbol": clean_text(
                        row.get("symbol")
                    ),
                    "trade_id": clean_text(
                        row.get("trade_id")
                    ),
                    "error_type": (
                        type(exc).__name__
                    ),
                    "error": str(exc),
                }
            )

    error_frame = pd.DataFrame(
        error_rows
    )

    return (
        fills,
        source_frame,
        error_frame,
    )


def run_reconstruction(
    fills: list[NormalizedFill],
):
    """Run the pure application service over normalized fills.

    This function performs no exchange access and writes no files.
    """

    reconstructor = DealReconstructor()

    return reconstructor.reconstruct(
        fills
    )


def build_in_memory_summary(
    *,
    fills: list[NormalizedFill],
    result,
) -> dict[str, Any]:
    """Build a compact validation summary without persistence."""

    manual_origins = {
        FillOrigin.IOS,
        FillOrigin.WEB,
        FillOrigin.ANDROID,
    }

    manual_input_fills = [
        fill
        for fill in fills
        if fill.origin in manual_origins
    ]

    manual_close_input_fills = [
        fill
        for fill in manual_input_fills
        if fill.action is FillAction.CLOSE
    ]

    manual_close_deals = [
        deal
        for deal in result.deals
        if deal.includes_manual_close
    ]

    completed_deal_quantity_balance_pass = all(
        deal.entry_leg.quantity
        == deal.exit_leg.quantity
        for deal in result.deals
    )

    open_position_balance_pass = all(
        (
            sum(
                (
                    fill.quantity
                    for fill in position.entry_fills
                ),
                ZERO,
            )
            - sum(
                (
                    fill.quantity
                    for fill in position.exit_fills
                ),
                ZERO,
            )
        )
        == position.open_quantity
        for position in result.open_positions
    )

    anomaly_types = sorted(
        {
            anomaly.anomaly_type
            for anomaly in result.anomalies
        }
    )

    summary = {
        "processed_fill_count": (
            result.processed_fill_count
        ),
        "completed_deal_count": (
            result.completed_deal_count
        ),
        "manual_input_fill_count": len(
            manual_input_fills
        ),
        "manual_close_input_fill_count": len(
            manual_close_input_fills
        ),
        "manual_close_deal_count": len(
            manual_close_deals
        ),
        "open_position_count": (
            result.open_position_count
        ),
        "anomaly_count": (
            result.anomaly_count
        ),
        "anomaly_types": anomaly_types,
        "completed_deal_quantity_balance_pass": (
            completed_deal_quantity_balance_pass
        ),
        "open_position_balance_pass": (
            open_position_balance_pass
        ),
        "all_completed_deal_quantities_positive": all(
            deal.quantity > ZERO
            for deal in result.deals
        ),
        "all_open_position_quantities_positive": all(
            position.open_quantity > ZERO
            for position in result.open_positions
        ),
    }

    summary["validation_pass"] = all(
        [
            summary["processed_fill_count"]
            == len(fills),
            summary[
                "completed_deal_quantity_balance_pass"
            ],
            summary[
                "open_position_balance_pass"
            ],
            summary[
                "all_completed_deal_quantities_positive"
            ],
            summary[
                "all_open_position_quantities_positive"
            ],
        ]
    )

    return summary


def print_reconstruction_preview(
    *,
    result,
    summary: Mapping[str, Any],
) -> None:
    """Print a concise preview of reconstructed domain results."""

    print()
    print(
        "IN-MEMORY RECONSTRUCTION SUMMARY"
    )
    print("-" * 70)

    for key, value in summary.items():
        print(
            f"{key}={value}"
        )

    print()
    print(
        "COMPLETED DEAL PREVIEW"
    )
    print("-" * 70)

    if not result.deals:
        print("<none>")
    else:
        for deal in result.deals[:10]:
            print(
                " | ".join(
                    [
                        deal.symbol,
                        deal.position_side.value,
                        deal.opened_at.isoformat(),
                        deal.closed_at.isoformat(),
                        f"qty={deal.quantity}",
                        (
                            "realised_pnl="
                            f"{deal.exchange_realised_pnl}"
                        ),
                        (
                            "manual_close="
                            f"{deal.includes_manual_close}"
                        ),
                    ]
                )
            )

    print()
    print(
        "OPEN POSITION PREVIEW"
    )
    print("-" * 70)

    if not result.open_positions:
        print("<none>")
    else:
        for position in result.open_positions[:10]:
            print(
                " | ".join(
                    [
                        position.symbol,
                        position.position_side.value,
                        (
                            "opened_at="
                            f"{position.opened_at.isoformat()}"
                        ),
                        (
                            "open_qty="
                            f"{position.open_quantity}"
                        ),
                        (
                            "entry_fills="
                            f"{len(position.entry_fills)}"
                        ),
                        (
                            "exit_fills="
                            f"{len(position.exit_fills)}"
                        ),
                    ]
                )
            )

    print()
    print(
        "ANOMALY PREVIEW"
    )
    print("-" * 70)

    if not result.anomalies:
        print("<none>")
    else:
        for anomaly in result.anomalies[:15]:
            print(
                " | ".join(
                    [
                        anomaly.anomaly_type,
                        anomaly.symbol,
                        anomaly.position_side.value,
                        (
                            "trade_id="
                            f"{anomaly.trade_id}"
                        ),
                        (
                            "qty="
                            f"{anomaly.quantity}"
                        ),
                    ]
                )
            )



def write_reports(
    *,
    output_dir: Path,
    result,
    writer: ReconstructionReportWriter,
) -> None:
    """Persist reconstruction reports through the injected reporting port."""

    writer.write(
        result=result,
        output_dir=output_dir,
        context={
            "runner": "audit_bitget_native_deals",
            "input_mode": "normalized_csv",
        },
    )

    print()
    print("REPORT OUTPUT")
    print("-"*70)

    for name in sorted(output_dir.glob("*")):
        print(name)

def main(
    *,
    report_writer: ReconstructionReportWriter,
) -> int:
    """Load normalized fills and run reconstruction with injected reporting."""

    args = parse_args()

    fills, source_frame, error_frame = (
        load_normalized_fills(
            args.input
        )
    )

    if args.expected_rows is not None:
        if len(source_frame) != args.expected_rows:
            raise SystemExit(
                "Expected "
                f"{args.expected_rows} rows, "
                f"found {len(source_frame)}"
            )

    print(
        f"INPUT={args.input}"
    )
    print(
        f"OUTPUT_DIR={args.output_dir}"
    )
    print(
        f"SOURCE_ROWS={len(source_frame)}"
    )
    print(
        f"NORMALIZED_FILL_COUNT={len(fills)}"
    )
    print(
        f"INPUT_ERROR_COUNT={len(error_frame)}"
    )

    if not error_frame.empty:
        print(
            error_frame.to_string(
                index=False
            )
        )
        return 1

    print(
        "BLOCK1_INPUT_LOADING=PASS"
    )

    result = run_reconstruction(
        fills
    )

    summary = build_in_memory_summary(
        fills=fills,
        result=result,
    )

    print_reconstruction_preview(
        result=result,
        summary=summary,
    )

    write_reports(
        output_dir=args.output_dir,
        result=result,
        writer=report_writer,
    )

    if not summary["validation_pass"]:
        print(
            "BLOCK2_IN_MEMORY_RECONSTRUCTION=FAIL"
        )
        return 1

    print()
    print(
        "BLOCK2_IN_MEMORY_RECONSTRUCTION=PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(
            report_writer=CsvJsonReconstructionReportWriter(),
        )
    )
