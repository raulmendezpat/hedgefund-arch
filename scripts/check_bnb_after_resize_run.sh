#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/hedgefund-arch-v2"
cd "$ROOT"

source .venv/bin/activate
export PYTHONPATH=src

LATEST_LOG="$(ls -t runtime/logs/run_*.log 2>/dev/null | head -n 1 || true)"

echo "===== LATEST LOG ====="
echo "$LATEST_LOG"
echo

if [[ -n "${LATEST_LOG:-}" && -f "${LATEST_LOG:-}" ]]; then
  echo "===== BNB BLOCK IN LOG ====="
  sed -n '/=== BNB\/USDT:USDT ===/,/MARGIN_MODE_/p' "$LATEST_LOG" | sed '$d' || true
  echo

  echo "===== BNB RELEVANT LINES ====="
  grep -nE "BNB/USDT:USDT|BNBUSDTUSDT|tp2-BNB|sl-BNB|PLAN_SNAPSHOT|PROTECTIVE_PLAN|CANCEL_PLAN|PLACE_TP2_PLAN|PLACE_SL_PLAN|skip rebalance|target_qty|current_qty|target_weight|side:" "$LATEST_LOG" || true
  echo
fi

python - <<'PY'
from pathlib import Path
import pandas as pd

cand = Path("results/research_runtime_candidates_prod_v2_live_candidate.csv")
events = Path("results/research_runtime_lifecycle_events_prod_v2_live_candidate.csv")
opens = Path("results/research_runtime_lifecycle_open_positions_prod_v2_live_candidate.csv")

for p in [cand, events, opens]:
    print("=" * 120)
    print(p)
    if not p.exists():
        print("missing")
        continue
    df = pd.read_csv(p, low_memory=False)
    sub = df[df.get("symbol", "").astype(str) == "BNB/USDT:USDT"].copy() if "symbol" in df.columns else pd.DataFrame()
    print(f"rows={len(sub)}")
    if len(sub) == 0:
        continue
    keep = [c for c in [
        "ts", "symbol", "side", "strategy_id", "reason", "selected_final",
        "p_win", "post_ml_score", "post_ml_competitive_score",
        "target_weight", "qty", "bars_held", "pnl", "exit_reason"
    ] if c in sub.columns]
    print(sub[keep].tail(30).to_string(index=False))
PY
