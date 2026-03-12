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

  python "$APP_DIR/deploy/reconcile_live.py" >> "$LOG_FILE" 2>&1

  echo "{\"ts\":\"$RUN_TS\",\"status\":\"ok\",\"log\":\"$LOG_FILE\",\"start\":\"$START_TS\",\"live_trading\":$( [ \"${LIVE_TRADING:-0}\" = \"1\" ] && echo true || echo false )}" > "$STATUS_FILE"

else

  echo "{\"ts\":\"$RUN_TS\",\"status\":\"fail\",\"log\":\"$LOG_FILE\",\"start\":\"$START_TS\",\"live_trading\":$( [ \"${LIVE_TRADING:-0}\" = \"1\" ] && echo true || echo false )}" > "$STATUS_FILE"
  exit 1

fi
