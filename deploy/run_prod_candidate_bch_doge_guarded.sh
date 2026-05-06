#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$APP_DIR/runtime/logs"
STATE_DIR="$APP_DIR/runtime/state"
LOCK_FILE="/tmp/hedgefund_arch_v2_bch_doge_guarded_1h.lock"
RUN_TS="$(date -u +"%Y%m%dT%H%M%SZ")"
STATUS_FILE="$STATE_DIR/last_run_status_bch_doge_guarded.json"
LOG_FILE="$LOG_DIR/run_bch_doge_guarded_${RUN_TS}.log"

mkdir -p "$LOG_DIR" "$STATE_DIR" "$APP_DIR/results"

if [ -f "$LOCK_FILE" ]; then
  echo "{\"ts\":\"$RUN_TS\",\"status\":\"skipped\",\"reason\":\"lock_exists\",\"profile\":\"bch_doge_guarded\"}" > "$STATUS_FILE"
  exit 0
fi

touch "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

cd "$APP_DIR"
source .venv/bin/activate

START_TS="$(date -u -d '30 days ago' +"%Y-%m-%d %H:%M:%S")"
END_TS="$(date -u +"%Y-%m-%d %H:%M:%S")"

# Safety default:
# - 0 = dry-run reconcile with Bitget connection, no real order placement
# - 1 = live trading
export LIVE_TRADING="${LIVE_TRADING:-0}"

RUN_NAME="prod_v2_live_candidate_bch_doge_guarded"
REGISTRY="artifacts/asset_universe_ab/rt_registry_base_plus_bch_doge_guarded.json"
SELECTION_POLICY="artifacts/asset_universe_ab/selection_policy_config.base_plus_bch_doge_guarded.json"
PWIN_ARTIFACT="artifacts/pwin_calibration_strategy_side_baseline_prodsem_3m_v3.json"

EXTRA_ARGS=()
if [ -f "$PWIN_ARTIFACT" ]; then
  EXTRA_ARGS+=(--pwin-calibration-artifact "$PWIN_ARTIFACT")
else
  echo "WARN: pwin calibration artifact not found: $PWIN_ARTIFACT ; continuing without it" >> "$LOG_FILE"
fi

echo "======================================================================" >> "$LOG_FILE"
echo "BCH/DOGE GUARDED PROD-LIKE LAUNCHER" >> "$LOG_FILE"
echo "======================================================================" >> "$LOG_FILE"
echo "RUN_TS=$RUN_TS" >> "$LOG_FILE"
echo "START_TS=$START_TS" >> "$LOG_FILE"
echo "END_TS=$END_TS" >> "$LOG_FILE"
echo "LIVE_TRADING=${LIVE_TRADING:-0}" >> "$LOG_FILE"
echo "RUN_NAME=$RUN_NAME" >> "$LOG_FILE"
echo "REGISTRY=$REGISTRY" >> "$LOG_FILE"
echo "SELECTION_POLICY=$SELECTION_POLICY" >> "$LOG_FILE"
echo "PWIN_ARTIFACT_EXISTS=$([ -f "$PWIN_ARTIFACT" ] && echo true || echo false)" >> "$LOG_FILE"

if PYTHONPATH=src python scripts/research_runtime.py \
  --name "$RUN_NAME" \
  --strategy-registry "$REGISTRY" \
  --selection-policy-config "$SELECTION_POLICY" \
  --selection-policy-profile research \
  --policy-config artifacts/policy_config.json \
  --policy-profile symmetric_v1 \
  --selection-semantics-mode research \
  --exit-registry-json artifacts/exit_policy_registry.runtime_overrides.json \
  --start "$START_TS" \
  --end "$END_TS" \
  --exchange binanceusdm \
  --cache-dir data/cache \
  --allocator-mode production_like_snapshot \
  --allocator-profile blended \
  --projection-profile research \
  --target-exposure 0.40 \
  --symbol-cap 0.28 \
  --execution-symbol-cap 0.60 \
  --runtime-prod-ml-position-sizing \
  --prodlike-allocator-apply-ml-sizing \
  --runtime-ml-size-mode calibrated \
  --runtime-ml-size-scale 8.5 \
  --runtime-ml-size-min 0.50 \
  --runtime-ml-size-max 1.50 \
  --runtime-ml-size-base 0.70 \
  --runtime-ml-size-pwin-threshold 0.46 \
  "${EXTRA_ARGS[@]}" \
  >> "$LOG_FILE" 2>&1
then

  python - <<'PYLOG' >> "$LOG_FILE" 2>&1
import json
from pathlib import Path
import pandas as pd

app = Path.cwd()
run_name = "prod_v2_live_candidate_bch_doge_guarded"
alloc_path = app / "results" / f"research_runtime_{run_name}.csv"

if alloc_path.exists():
    df = pd.read_csv(alloc_path, low_memory=False)
    if len(df) > 0:
        row = df.iloc[-1]

        symbol_cols = sorted({
            c.replace("_execution_target_weight", "")
            for c in df.columns if c.endswith("_execution_target_weight")
        } | {
            c.replace("_cluster_target_weight", "")
            for c in df.columns if c.endswith("_cluster_target_weight")
        } | {
            c.replace("_w_after_signal_gating", "")
            for c in df.columns if c.endswith("_w_after_signal_gating")
        } | {
            c.replace("_w_after_smoothing", "")
            for c in df.columns if c.endswith("_w_after_smoothing")
        } | {
            c.replace("_w_after_ml_position_sizing", "")
            for c in df.columns if c.endswith("_w_after_ml_position_sizing")
        } | {
            c.replace("_w_raw_allocator", "")
            for c in df.columns if c.endswith("_w_raw_allocator")
        })

        active_weights = {}
        exposure = 0.0

        exec_cfg_path = app / "deploy" / "live_execution_config.json"
        exec_cfg = json.loads(exec_cfg_path.read_text()) if exec_cfg_path.exists() else {}
        symbol_overrides = exec_cfg.get("symbol_overrides", {}) or {}
        default_max_target_weight = float((exec_cfg.get("default_symbol_config", {}) or {}).get("max_target_weight", 0.25))

        for base in symbol_cols:
            exec_v = float(row.get(f"{base}_execution_target_weight", 0.0) or 0.0)
            cluster_v = float(row.get(f"{base}_cluster_target_weight", 0.0) or 0.0)
            signal_v = float(row.get(f"{base}_w_after_signal_gating", 0.0) or 0.0)
            smooth_v = float(row.get(f"{base}_w_after_smoothing", 0.0) or 0.0)
            ml_v = float(row.get(f"{base}_w_after_ml_position_sizing", 0.0) or 0.0)
            raw_alloc_v = float(row.get(f"{base}_w_raw_allocator", 0.0) or 0.0)

            source = f"{base}_execution_target_weight"
            raw_v = exec_v
            if abs(raw_v) <= 1e-12:
                source = f"{base}_cluster_target_weight"
                raw_v = cluster_v
            if abs(raw_v) <= 1e-12:
                source = f"{base}_w_after_signal_gating"
                raw_v = signal_v
            if abs(raw_v) <= 1e-12:
                source = f"{base}_w_after_smoothing"
                raw_v = smooth_v
            if abs(raw_v) <= 1e-12:
                source = f"{base}_w_after_ml_position_sizing"
                raw_v = ml_v
            if abs(raw_v) <= 1e-12:
                source = f"{base}_w_raw_allocator"
                raw_v = raw_alloc_v

            sym = base.replace("_usdt_usdt", "").upper() + "/USDT:USDT"
            max_target_weight = float(
                (symbol_overrides.get(sym, {}) or {}).get("max_target_weight", default_max_target_weight)
            )
            capped_v = max(-max_target_weight, min(max_target_weight, raw_v))

            exposure += abs(capped_v)
            if abs(capped_v) > 1e-12:
                active_weights[sym] = {
                    "source": source,
                    "raw_target_weight": raw_v,
                    "capped_target_weight": capped_v,
                    "max_target_weight": max_target_weight,
                }

        payload = {
            "run_name": run_name,
            "exposure": float(exposure),
            "active_symbol_count": int(len(active_weights)),
            "active_weights": active_weights,
        }

        print("=== PORTFOLIO SNAPSHOT BCH/DOGE GUARDED ===")
        print(json.dumps(payload, indent=2, sort_keys=True))
else:
    print(f"WARN: allocation output not found: {alloc_path}")
PYLOG

  PYTHONPATH=src python "$APP_DIR/deploy/reconcile_live.py" >> "$LOG_FILE" 2>&1

  echo "{\"ts\":\"$RUN_TS\",\"status\":\"ok\",\"profile\":\"bch_doge_guarded\",\"log\":\"$LOG_FILE\",\"start\":\"$START_TS\",\"end\":\"$END_TS\",\"live_trading\":$( [ "${LIVE_TRADING:-0}" = "1" ] && echo true || echo false )}" > "$STATUS_FILE"
else
  echo "{\"ts\":\"$RUN_TS\",\"status\":\"fail\",\"profile\":\"bch_doge_guarded\",\"log\":\"$LOG_FILE\",\"start\":\"$START_TS\",\"end\":\"$END_TS\",\"live_trading\":$( [ "${LIVE_TRADING:-0}" = "1" ] && echo true || echo false )}" > "$STATUS_FILE"
  exit 1
fi
