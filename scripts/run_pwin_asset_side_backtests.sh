#!/usr/bin/env bash
set -eo pipefail

cd /Users/raulmendez/Projects/hedgefund-arch
source .venv/bin/activate
export PYTHONPATH=src

BASE_REGISTRY=""
NEW_REGISTRY="artifacts/pwin_asset_side_models_v2_clean/pwin_asset_side_model_registry.json"

COMMON_ARGS=(
  --strategy-registry artifacts/rt_registry_add_trx_specialized.json
  --selection-policy-config artifacts/selection_policy_config.calibrated.json
  --selection-policy-profile research
  --policy-config artifacts/policy_config.json
  --policy-profile symmetric_v1
  --selection-semantics-mode research
  --exchange binanceusdm
  --cache-dir data/cache
  --allocator-mode production_like_snapshot
  --allocator-profile blended
  --projection-profile research
  --target-exposure 0.40
  --symbol-cap 0.28
  --execution-symbol-cap 0.30
  --runtime-prod-ml-position-sizing
  --prodlike-allocator-apply-ml-sizing
  --runtime-ml-size-mode calibrated
  --runtime-ml-size-scale 6.0
  --runtime-ml-size-min 0.70
  --runtime-ml-size-max 1.30
  --runtime-ml-size-base 0.85
  --runtime-ml-size-pwin-threshold 0.46
  --runtime-ml-size-overrides 'bnb_trend|short|1.08'
  --pwin-calibration-artifact artifacts/pwin_calibration_strategy_side_baseline_prodsem_3m_v3.json
)

run_case () {
  local name="$1"
  local start="$2"
  local end="$3"
  local mode="$4"

  if [[ "$mode" == "baseline" ]]; then
    unset PWIN_ASSET_SIDE_REGISTRY || true
    python scripts/research_runtime.py       --name "${name}_baseline"       --start "$start"       --end "$end"       "${COMMON_ARGS[@]}"
  else
    export PWIN_ASSET_SIDE_REGISTRY="$NEW_REGISTRY"
    python scripts/research_runtime.py       --name "${name}_asset_side_pwin"       --start "$start"       --end "$end"       "${COMMON_ARGS[@]}"
  fi
}

# 1M
run_case "pwin_cmp_1m" "2026-03-05 22:02:01" "2026-04-05 22:02:01" "baseline"
run_case "pwin_cmp_1m" "2026-03-05 22:02:01" "2026-04-05 22:02:01" "asset_side"

# 3M
run_case "pwin_cmp_3m" "2026-01-05 22:02:01" "2026-04-05 22:02:01" "baseline"
run_case "pwin_cmp_3m" "2026-01-05 22:02:01" "2026-04-05 22:02:01" "asset_side"

# 6M
run_case "pwin_cmp_6m" "2025-10-05 22:02:01" "2026-04-05 22:02:01" "baseline"
run_case "pwin_cmp_6m" "2025-10-05 22:02:01" "2026-04-05 22:02:01" "asset_side"
