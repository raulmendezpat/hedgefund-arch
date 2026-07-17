"""Tests for the standalone normalized-fill producer."""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import SimpleNamespace


UTC = timezone.utc
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PRODUCER_PATH = (
    REPOSITORY_ROOT
    / "scripts"
    / "export_bitget_normalized_fills.py"
)

SPEC = importlib.util.spec_from_file_location(
    "export_bitget_normalized_fills",
    PRODUCER_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise RuntimeError(
        f"unable to load producer module from {PRODUCER_PATH}"
    )

PRODUCER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PRODUCER)


class FakeValue(Enum):
    VALUE = "value"


def build_fill(
    *,
    trade_id: str = "trade-1",
    timestamp: datetime | None = None,
):
    return SimpleNamespace(
        exchange="bitget",
        symbol="BTC/USDT:USDT",
        trade_id=trade_id,
        order_id="order-1",
        timestamp=timestamp or datetime(2026, 7, 17, 0, 0, tzinfo=UTC),
        position_side=FakeValue.VALUE,
        action=FakeValue.VALUE,
        quantity=Decimal("0.001"),
        price=Decimal("100000"),
        fee=Decimal("0.05"),
        realised_pnl=Decimal("1.25"),
        origin=FakeValue.VALUE,
        raw_reference={"trade_id": trade_id},
        metadata={"source": "test"},
    )


class ExportBitgetNormalizedFillsTest(unittest.TestCase):
    def test_parse_symbols_deduplicates_and_preserves_order(self):
        result = PRODUCER.parse_symbols(
            "BTC/USDT:USDT, ETH/USDT:USDT,"
            "BTC/USDT:USDT"
        )

        self.assertEqual(
            result,
            [
                "BTC/USDT:USDT",
                "ETH/USDT:USDT",
            ],
        )

    def test_resolve_window_uses_requested_lookback(self):
        end = datetime(2026, 7, 17, 1, 0, tzinfo=UTC)

        start, resolved_end = PRODUCER.resolve_window(
            start_text=None,
            end_text=None,
            lookback_hours=24,
            now=end,
        )

        self.assertEqual(
            start,
            datetime(2026, 7, 16, 1, 0, tzinfo=UTC),
        )
        self.assertEqual(resolved_end, end)

    def test_normalized_fill_to_row_matches_exact_contract(self):
        row = PRODUCER.normalized_fill_to_row(build_fill())

        self.assertEqual(
            list(row),
            PRODUCER.NORMALIZED_COLUMNS,
        )
        self.assertEqual(row["exchange"], "bitget")
        self.assertEqual(row["trade_id"], "trade-1")
        self.assertEqual(row["position_side"], "value")
        self.assertEqual(row["timestamp"], "2026-07-17T00:00:00Z")
        self.assertEqual(row["quantity"], "0.001")
        self.assertEqual(row["realised_pnl"], "1.25")
        self.assertEqual(
            row["metadata"],
            '{"source":"test"}',
        )

    def test_atomic_csv_writer_outputs_header_and_rows(self):
        fills = [
            build_fill(
                trade_id="trade-1",
                timestamp=datetime(
                    2026,
                    7,
                    17,
                    0,
                    0,
                    tzinfo=UTC,
                ),
            ),
            build_fill(
                trade_id="trade-2",
                timestamp=datetime(
                    2026,
                    7,
                    17,
                    0,
                    1,
                    tzinfo=UTC,
                ),
            ),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = (
                Path(temporary_directory)
                / "normalized_fills.csv"
            )

            PRODUCER.write_normalized_csv_atomic(
                output,
                fills,
            )

            with output.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as input_file:
                rows = list(csv.DictReader(input_file))

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            list(rows[0]),
            PRODUCER.NORMALIZED_COLUMNS,
        )
        self.assertEqual(
            [row["trade_id"] for row in rows],
            ["trade-1", "trade-2"],
        )

    def test_invalid_window_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "start must be earlier than end",
        ):
            PRODUCER.resolve_window(
                start_text="2026-07-17T01:00:00Z",
                end_text="2026-07-17T00:00:00Z",
                lookback_hours=24,
            )

    def test_credentials_fall_back_to_secret_json_envelope(self):
        import json
        import os
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp_dir:
            secret_path = Path(temp_dir) / "secret.json"
            secret_path.write_text(
                json.dumps(
                    {
                        "envelope": {
                            "apiKey": "file-api-key",
                            "secret": "file-secret",
                            "password": "file-password",
                        }
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                credentials, source = (
                    PRODUCER.load_bitget_credentials(
                        secret_path=secret_path
                    )
                )

        self.assertEqual(source, "secret_json")
        self.assertEqual(
            credentials,
            {
                "apiKey": "file-api-key",
                "secret": "file-secret",
                "password": "file-password",
            },
        )

    def test_environment_credentials_take_precedence(self):
        import json
        import os
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp_dir:
            secret_path = Path(temp_dir) / "secret.json"
            secret_path.write_text(
                json.dumps(
                    {
                        "envelope": {
                            "apiKey": "file-api-key",
                            "secret": "file-secret",
                            "password": "file-password",
                        }
                    }
                ),
                encoding="utf-8",
            )

            environment = {
                "BITGET_API_KEY": "environment-api-key",
                "BITGET_API_SECRET": "environment-secret",
                "BITGET_API_PASSWORD": "environment-password",
            }

            with patch.dict(
                os.environ,
                environment,
                clear=True,
            ):
                credentials, source = (
                    PRODUCER.load_bitget_credentials(
                        secret_path=secret_path
                    )
                )

        self.assertEqual(source, "environment")
        self.assertEqual(
            credentials,
            {
                "apiKey": "environment-api-key",
                "secret": "environment-secret",
                "password": "environment-password",
            },
        )

    def test_partial_environment_uses_secret_json_for_missing_values(
        self,
    ):
        import json
        import os
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp_dir:
            secret_path = Path(temp_dir) / "secret.json"
            secret_path.write_text(
                json.dumps(
                    {
                        "envelope": {
                            "apiKey": "file-api-key",
                            "secret": "file-secret",
                            "password": "file-password",
                        }
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {"BITGET_API_KEY": "environment-api-key"},
                clear=True,
            ):
                credentials, source = (
                    PRODUCER.load_bitget_credentials(
                        secret_path=secret_path
                    )
                )

        self.assertEqual(source, "secret_json")
        self.assertEqual(
            credentials,
            {
                "apiKey": "environment-api-key",
                "secret": "file-secret",
                "password": "file-password",
            },
        )

    def test_missing_credentials_are_rejected_without_exposing_values(
        self,
    ):
        import json
        import os
        from unittest.mock import patch

        private_value = "never-print-this-private-value"

        with tempfile.TemporaryDirectory() as temp_dir:
            secret_path = Path(temp_dir) / "secret.json"
            secret_path.write_text(
                json.dumps(
                    {
                        "envelope": {
                            "apiKey": private_value,
                        }
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(RuntimeError) as context:
                    PRODUCER.load_bitget_credentials(
                        secret_path=secret_path
                    )

        message = str(context.exception)

        self.assertIn("secret", message)
        self.assertIn("password", message)
        self.assertNotIn(private_value, message)



if __name__ == "__main__":
    unittest.main()
