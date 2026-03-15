#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/ubuntu/hedgefund-arch"
LOG_DIR="$APP_DIR/runtime/logs"
STATE_DIR="$APP_DIR/runtime/state"
LOCK_FILE="/tmp/hedgefund_arch_prod_1h.lock"
RUN_TS="$(date -u +"%Y%m%dT%H%M%SZ")"
STATUS_FILE="$STATE_DIR/last_run_status.json"
LOG_FILE="$LOG_DIR/run_${RUN_TS}.log"

mkdir -p "$LOG_DIR" "$STATE_DIR"

if [ -f "$LOCK_FILE" ]; then
  echo "{\"ts\":\"$RUN_TS\",\"status\":\"skipped\",\"reason\":\"lock_exists\"}" > "$STATUS_FILE"
  exit 0
fi

touch "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

cd "$APP_DIR"
source .venv/bin/activate

START_TS="$(date -u -d '30 days ago' +"%Y-%m-%d %H:%M:%S")"

# IMPORTANT: trading still disabled
export LIVE_TRADING=1

if python scripts/hf_pipeline_alloc.py \
  --name prod_candidate_live \
  --start "$START_TS" \
  --signal-engine registry_portfolio \
  --strategy-registry artifacts/strategy_registry_regime_v1_vol_expansion.json \
  --opportunity-selection-mode competitive \
  --allocation-engine-mode multi_strategy \
  --allocator-blend-alpha 0.40 \
  --allocator-smoothing-alpha 0.50 \
  --allocator-smoothing-snap-eps 0.02 \
  --allocator-rebalance-deadband 0.03 \
  --allocator-symbol-cap 0.25 \
  --allocator-target-exposure 0.05 \
  --ml-filter \
  --ml-model-registry artifacts/ml_registry.json \
  --ml-thresholds-path artifacts/ml_thresholds_registry_v1.json \
  >> "$LOG_FILE" 2>&1
then

  python - <<'PYLOG' >> "$LOG_FILE" 2>&1
import json
from pathlib import Path
import pandas as pd

app = Path("/home/ubuntu/hedgefund-arch")
alloc_path = app / "results" / "pipeline_allocations_prod_candidate_live.csv"
status_path = app / "runtime" / "state" / "last_run_status.json"

if alloc_path.exists():
    df = pd.read_csv(alloc_path, low_memory=False)
    if len(df) > 0:
        row = df.iloc[-1]

        wcols = [c for c in df.columns if c.startswith("w_")]
        weights = {}
        active_weights = {}
        exposure = 0.0

        for c in wcols:
            v = float(row.get(c, 0.0) or 0.0)
            weights[c] = v
            exposure += abs(v)
            if abs(v) > 1e-12:
                active_weights[c] = v

        payload = {
            "portfolio_regime": str(row.get("portfolio_regime", "unknown")),
            "portfolio_conviction": float(row.get("portfolio_conviction", 0.0) or 0.0),
            "portfolio_breadth": int(row.get("portfolio_breadth", 0) or 0),
            "portfolio_avg_strength": float(row.get("portfolio_avg_strength", 0.0) or 0.0),
            "portfolio_avg_pwin": float(row.get("portfolio_avg_pwin", 0.0) or 0.0),
            "portfolio_avg_atrp": float(row.get("portfolio_avg_atrp", 0.0) or 0.0),
            "portfolio_regime_symbol_cap_mult": float(row.get("portfolio_regime_symbol_cap_mult", 1.0) or 1.0),
            "portfolio_regime_scale_applied": float(row.get("portfolio_regime_scale_applied", 1.0) or 1.0),
            "exposure": float(exposure),
            "active_symbol_count": int(sum(1 for v in active_weights.values() if abs(v) > 1e-12)),
            "active_weights": active_weights,
        }

        print("=== PORTFOLIO SNAPSHOT ===")
        print(json.dumps(payload, indent=2, sort_keys=True))
PYLOG

  python "$APP_DIR/deploy/reconcile_live.py" >> "$LOG_FILE" 2>&1

  echo "{\"ts\":\"$RUN_TS\",\"status\":\"ok\",\"log\":\"$LOG_FILE\",\"start\":\"$START_TS\",\"live_trading\":$( [ \"${LIVE_TRADING:-0}\" = \"1\" ] && echo true || echo false )}" > "$STATUS_FILE"

else

  echo "{\"ts\":\"$RUN_TS\",\"status\":\"fail\",\"log\":\"$LOG_FILE\",\"start\":\"$START_TS\",\"live_trading\":$( [ \"${LIVE_TRADING:-0}\" = \"1\" ] && echo true || echo false )}" > "$STATUS_FILE"
  exit 1

fi
