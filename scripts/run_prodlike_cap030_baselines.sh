#!/usr/bin/env bash
set -euo pipefail

cd /Users/raulmendez/Projects/hedgefund-arch
source .venv/bin/activate

run_case () {
  local NAME="$1"
  local START_TS="$2"
  local END_TS="$3"

  echo
  echo "================================================================================"
  echo "RUNNING: ${NAME}"
  echo "================================================================================"

  PYTHONPATH=src python scripts/research_runtime.py \
    --name "${NAME}" \
    --strategy-registry artifacts/rt_registry_add_trx_specialized.json \
    --selection-policy-config artifacts/selection_policy_config.calibrated.json \
    --selection-policy-profile research \
    --policy-config artifacts/policy_config.json \
    --policy-profile symmetric_v1 \
    --selection-semantics-mode research \
    --start "${START_TS}" \
    --end "${END_TS}" \
    --exchange binanceusdm \
    --cache-dir data/cache \
    --allocator-mode production_like_snapshot \
    --allocator-profile blended \
    --projection-profile research \
    --target-exposure 0.40 \
    --symbol-cap 0.28 \
    --execution-symbol-cap 0.30 \
    --runtime-prod-ml-position-sizing \
    --prodlike-allocator-apply-ml-sizing \
    --runtime-ml-size-mode calibrated \
    --runtime-ml-size-scale 6.0 \
    --runtime-ml-size-min 0.70 \
    --runtime-ml-size-max 1.30 \
    --runtime-ml-size-base 0.85 \
    --runtime-ml-size-pwin-threshold 0.46 \
    --runtime-ml-size-overrides 'bnb_trend|short|1.08' \
    --pwin-calibration-artifact artifacts/pwin_calibration_strategy_side_baseline_prodsem_3m_v3.json
}

END_TS="2026-04-03 00:00:00"

run_case "prodlike_cap030_1m_baseline"  "2026-03-04 00:00:00" "${END_TS}"
run_case "prodlike_cap030_3m_baseline"  "2026-01-03 00:00:00" "${END_TS}"
run_case "prodlike_cap030_6m_baseline"  "2025-10-03 00:00:00" "${END_TS}"
run_case "prodlike_cap030_12m_baseline" "2025-04-03 00:00:00" "${END_TS}"

python - <<'PYSUM'
import json
from pathlib import Path
import pandas as pd

names = [
    "prodlike_cap030_1m_baseline",
    "prodlike_cap030_3m_baseline",
    "prodlike_cap030_6m_baseline",
    "prodlike_cap030_12m_baseline",
]

rows = []
for name in names:
    p = Path(f"results/research_runtime_metrics_{name}.json")
    obj = json.loads(p.read_text())
    rows.append({
        "name": name,
        "total_return_pct": obj.get("total_return_pct"),
        "sharpe_annual": obj.get("sharpe_annual"),
        "max_drawdown_pct": obj.get("max_drawdown_pct"),
        "trade_count": obj.get("trade_count"),
        "win_rate_pct": obj.get("win_rate_pct"),
        "avg_active_symbols": obj.get("avg_active_symbols"),
        "avg_gross_weight": obj.get("avg_gross_weight"),
    })

df = pd.DataFrame(rows)
out = Path("results/prodlike_cap030_baseline_summary.csv")
df.to_csv(out, index=False)

print("\n=== PRODLIKE CAP 0.30 BASELINE SUMMARY ===")
print(df.to_string(index=False))
print(f"\nSAVED: {out}")
PYSUM
