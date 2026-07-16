from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd

from hf_audit.deal_reconstruction.application import (
    DealReconstructor,
)
from hf_audit.deal_reconstruction.domain.enums import (
    FillAction,
    FillOrigin,
    PositionSide,
)
from hf_audit.deal_reconstruction.domain.models import (
    NormalizedFill,
)
from hf_audit.deal_reconstruction.infrastructure.reconstruction_report_writer import (
    CsvJsonReconstructionReportWriter,
    build_asset_summary_frame,
    build_deal_frame,
    build_manual_close_deal_frame,
    build_summary,
)


BASE_TS = datetime(
    2026,
    7,
    16,
    0,
    0,
    tzinfo=timezone.utc,
)


def make_fill(
    *,
    trade_id: str,
    minute: int,
    side: PositionSide,
    action: FillAction,
    quantity: str,
    price: str,
    realised_pnl: str = "0",
    fee: str = "0",
    origin: FillOrigin = FillOrigin.API,
) -> NormalizedFill:
    return NormalizedFill(
        exchange="bitget",
        symbol="BTC/USDT:USDT",
        trade_id=trade_id,
        order_id=f"order-{trade_id}",
        timestamp=(
            BASE_TS
            + timedelta(minutes=minute)
        ),
        position_side=side,
        action=action,
        quantity=Decimal(quantity),
        price=Decimal(price),
        realised_pnl=Decimal(
            realised_pnl
        ),
        fee=Decimal(fee),
        origin=origin,
    )


def build_result():
    fills = [
        make_fill(
            trade_id="open-long",
            minute=0,
            side=PositionSide.LONG,
            action=FillAction.OPEN,
            quantity="2",
            price="100",
            fee="0.04",
        ),
        make_fill(
            trade_id="close-long",
            minute=5,
            side=PositionSide.LONG,
            action=FillAction.CLOSE,
            quantity="2",
            price="110",
            realised_pnl="20",
            fee="0.05",
            origin=FillOrigin.IOS,
        ),
        make_fill(
            trade_id="open-short",
            minute=10,
            side=PositionSide.SHORT,
            action=FillAction.OPEN,
            quantity="1",
            price="120",
            fee="0.02",
        ),
        make_fill(
            trade_id="orphan-close",
            minute=15,
            side=PositionSide.LONG,
            action=FillAction.CLOSE,
            quantity="1",
            price="111",
            realised_pnl="1",
            fee="0.01",
        ),
    ]

    return DealReconstructor().reconstruct(
        fills
    )


class ReconstructionReportWriterTest(
    unittest.TestCase
):
    def test_deal_frame_contains_manual_close(self) -> None:
        result = build_result()

        frame = build_deal_frame(
            result.deals
        )

        self.assertEqual(
            len(frame),
            1,
        )
        self.assertTrue(
            bool(
                frame.iloc[0][
                    "includes_manual_close"
                ]
            )
        )
        self.assertEqual(
            frame.iloc[0][
                "exchange_realised_pnl"
            ],
            "20",
        )
        self.assertEqual(
            frame.iloc[0][
                "total_fees"
            ],
            "0.09",
        )
        self.assertEqual(
            frame.iloc[0][
                "net_realised_pnl"
            ],
            "19.91",
        )

    def test_manual_close_frame_filters_deals(
        self,
    ) -> None:
        result = build_result()

        frame = (
            build_manual_close_deal_frame(
                result.deals
            )
        )

        self.assertEqual(
            len(frame),
            1,
        )
        self.assertEqual(
            frame.iloc[0]["exit_origins"],
            "ios",
        )

    def test_asset_summary_combines_outputs(
        self,
    ) -> None:
        result = build_result()

        frame = build_asset_summary_frame(
            result
        )

        self.assertEqual(
            len(frame),
            1,
        )

        row = frame.iloc[0]

        self.assertEqual(
            int(
                row[
                    "completed_deal_count"
                ]
            ),
            1,
        )
        self.assertEqual(
            int(
                row[
                    "manual_close_deal_count"
                ]
            ),
            1,
        )
        self.assertEqual(
            int(
                row[
                    "open_position_count"
                ]
            ),
            1,
        )
        self.assertEqual(
            int(
                row[
                    "anomaly_count"
                ]
            ),
            1,
        )

    def test_summary_validates_result(self) -> None:
        result = build_result()

        summary = build_summary(
            result=result,
            context={
                "source": "unit-test",
            },
        )

        self.assertEqual(
            summary["status"],
            "PASS",
        )
        self.assertEqual(
            summary[
                "processed_fill_count"
            ],
            4,
        )
        self.assertEqual(
            summary[
                "completed_deal_count"
            ],
            1,
        )
        self.assertEqual(
            summary[
                "manual_close_deal_count"
            ],
            1,
        )
        self.assertEqual(
            summary[
                "open_position_count"
            ],
            1,
        )
        self.assertEqual(
            summary[
                "anomaly_count"
            ],
            1,
        )
        self.assertEqual(
            summary["context"]["source"],
            "unit-test",
        )

    def test_writer_generates_all_files(self) -> None:
        result = build_result()

        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)

            paths = (
                CsvJsonReconstructionReportWriter()
                .write(
                    result=result,
                    output_dir=output_dir,
                    context={
                        "source": "unit-test",
                    },
                )
            )

            self.assertEqual(
                set(paths),
                {
                    "deals",
                    "manual_close_deals",
                    "open_positions",
                    "anomalies",
                    "summary_by_asset",
                    "summary",
                },
            )

            for path in paths.values():
                self.assertTrue(
                    path.is_file()
                )

            deals = pd.read_csv(
                paths["deals"]
            )
            manual = pd.read_csv(
                paths[
                    "manual_close_deals"
                ]
            )
            positions = pd.read_csv(
                paths["open_positions"]
            )
            anomalies = pd.read_csv(
                paths["anomalies"]
            )

            summary = json.loads(
                paths["summary"].read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                len(deals),
                1,
            )
            self.assertEqual(
                len(manual),
                1,
            )
            self.assertEqual(
                len(positions),
                1,
            )
            self.assertEqual(
                len(anomalies),
                1,
            )
            self.assertEqual(
                summary["status"],
                "PASS",
            )


if __name__ == "__main__":
    unittest.main()
