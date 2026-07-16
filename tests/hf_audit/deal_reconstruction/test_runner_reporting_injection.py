from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from hf_audit.deal_reconstruction.application import (
    DealReconstructor,
)
from hf_audit.deal_reconstruction.domain.reconstruction import (
    DealReconstructionResult,
)
from hf_audit.deal_reconstruction.ports.reporting import (
    ReconstructionReportWriter,
)


RUNNER_PATH = Path(
    "scripts/audit_bitget_native_deals.py"
)


def load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "audit_bitget_native_deals",
        RUNNER_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Unable to load audit runner"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


class FakeReportWriter:
    def __init__(self) -> None:
        self.call_count = 0
        self.result = None
        self.output_dir = None
        self.context = None

    def write(
        self,
        *,
        result: DealReconstructionResult,
        output_dir: Path,
        context: Mapping[str, Any],
    ):
        self.call_count += 1
        self.result = result
        self.output_dir = output_dir
        self.context = dict(context)

        return {}


class RunnerReportingInjectionTest(unittest.TestCase):
    def test_fake_writer_satisfies_port(self) -> None:
        self.assertIsInstance(
            FakeReportWriter(),
            ReconstructionReportWriter,
        )

    def test_write_reports_uses_injected_writer(self) -> None:
        runner = load_runner_module()
        writer = FakeReportWriter()

        result = DealReconstructor().reconstruct(
            []
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            runner.write_reports(
                output_dir=output_dir,
                result=result,
                writer=writer,
            )

        self.assertEqual(
            writer.call_count,
            1,
        )
        self.assertIs(
            writer.result,
            result,
        )
        self.assertEqual(
            writer.output_dir,
            output_dir,
        )
        self.assertEqual(
            writer.context,
            {
                "runner": "audit_bitget_native_deals",
                "input_mode": "normalized_csv",
            },
        )


if __name__ == "__main__":
    unittest.main()
