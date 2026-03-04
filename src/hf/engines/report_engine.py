from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json
import pandas as pd


@dataclass
class ReportEngine:
    """
    Consolida artefactos del pipeline en un único JSON.

    Espera (por name):
      results/pipeline_allocations_{name}.csv
      results/pipeline_equity_{name}.csv
      results/pipeline_metrics_{name}.json
    Opcional (si se corrió net):
      results/pipeline_equity_net_{name}.csv
      results/pipeline_metrics_net_{name}.json
    """

    results_dir: str = "results"

    def _p(self, rel: str) -> Path:
        return Path(self.results_dir) / rel

    def build(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        write: bool = True,
    ) -> Dict[str, Any]:
        config = dict(config or {})

        alloc_csv = self._p(f"pipeline_allocations_{name}.csv")
        eq_csv = self._p(f"pipeline_equity_{name}.csv")
        gross_json = self._p(f"pipeline_metrics_{name}.json")

        net_eq_csv = self._p(f"pipeline_equity_net_{name}.csv")
        net_json = self._p(f"pipeline_metrics_net_{name}.json")

        if not alloc_csv.exists():
            raise FileNotFoundError(f"Missing allocations CSV: {alloc_csv}")
        if not eq_csv.exists():
            raise FileNotFoundError(f"Missing equity CSV: {eq_csv}")
        if not gross_json.exists():
            raise FileNotFoundError(f"Missing gross metrics JSON: {gross_json}")

        alloc = pd.read_csv(alloc_csv)

        case_counts = {}
        if "case" in alloc.columns:
            vc = alloc["case"].value_counts(dropna=False)
            case_counts = {str(k): int(v) for k, v in vc.items()}

        exposure_pct = float("nan")
        if ("w_btc" in alloc.columns) and ("w_sol" in alloc.columns):
            wsum = pd.to_numeric(alloc["w_btc"], errors="coerce").fillna(0.0) + pd.to_numeric(
                alloc["w_sol"], errors="coerce"
            ).fillna(0.0)
            exposure_pct = float((wsum > 1e-12).mean() * 100.0)

        gross_metrics = json.loads(gross_json.read_text(encoding="utf-8"))

        net_metrics = None
        if net_json.exists():
            net_metrics = json.loads(net_json.read_text(encoding="utf-8"))

        report: Dict[str, Any] = {
            "name": name,
            "config": config,
            "case_counts": case_counts,
            "exposure_pct_from_allocations": exposure_pct,
            "gross_metrics": gross_metrics,
            "net_metrics": net_metrics,
            "artifacts": {
                "allocations_csv": str(alloc_csv),
                "equity_csv": str(eq_csv),
                "gross_metrics_json": str(gross_json),
                "net_equity_csv": str(net_eq_csv) if net_eq_csv.exists() else None,
                "net_metrics_json": str(net_json) if net_json.exists() else None,
            },
        }

        if write:
            out = self._p(f"pipeline_report_{name}.json")
            out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

        return report
