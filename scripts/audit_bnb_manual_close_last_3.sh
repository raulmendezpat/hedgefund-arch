#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/hedgefund-arch-v2"
cd "$ROOT"

source .venv/bin/activate
export PYTHONPATH=src

echo "===== BNB MANUAL CLOSE AUDIT ====="
date -u
date
echo

mapfile -t LOGS < <(ls -t runtime/logs/run_*.log 2>/dev/null | head -n 3 || true)
if [[ ${#LOGS[@]} -eq 0 ]]; then
  echo "No encontré logs."
  exit 0
fi

echo "===== LOGS UNDER REVIEW ====="
printf '%s\n' "${LOGS[@]}"
echo

for LOG in "${LOGS[@]}"; do
  echo "===================================================================================================="
  echo "LOG: $LOG"
  echo "===================================================================================================="
  echo

  echo "----- BNB BLOCK -----"
  sed -n '/=== BNB\/USDT:USDT ===/,/MARGIN_MODE_/p' "$LOG" | sed '$d' || true
  echo

  echo "----- BNB SIGNAL / EXECUTION / PLAN LINES -----"
  grep -nE "BNB/USDT:USDT|BNBUSDTUSDT|tp2-BNB|sl-BNB|PLAN_SNAPSHOT|PROTECTIVE_PLAN|CANCEL_PLAN|PLACE_TP2_PLAN|PLACE_SL_PLAN|skip rebalance|delta_notional|target_qty|current_qty|target_weight|side:" "$LOG" || true
  echo

  echo "----- POSSIBLE MANUAL INTERVENTION / RECONCILE SYMPTOMS -----"
  grep -nEi "BNB/USDT:USDT|manual|reconcile|stale|reduce-only|reject|position.*mismatch|qty.*mismatch|plan.*replace|cancelled stale|no position|active_state" "$LOG" || true
  echo
done

echo "===== SEARCHING RECENT RESULT FILES WITH BNB ====="
python - <<'PY'
from pathlib import Path
import pandas as pd

targets = []
for p in sorted(Path("results").glob("research_runtime_lifecycle_trades_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True):
    targets.append(p)
for p in sorted(Path("results").glob("research_runtime_lifecycle_events_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True):
    targets.append(p)
for p in sorted(Path("results").glob("research_runtime_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True):
    targets.append(p)

seen = 0
for p in targets:
    if seen >= 12:
        break
    try:
        df = pd.read_csv(p)
    except Exception:
        continue

    cols = set(df.columns)
    symbol_col = None
    for c in ["symbol", "asset", "instrument"]:
        if c in cols:
            symbol_col = c
            break
    if symbol_col is None:
        continue

    mask = df[symbol_col].astype(str).eq("BNB/USDT:USDT")
    if mask.any():
        seen += 1
        sub = df.loc[mask].copy()
        print("=" * 120)
        print(p)
        keep = [c for c in [
            "ts","timestamp","symbol","side","strategy_id","event_type","action","reason",
            "pnl","qty","price","entry_price","exit_price","bars_held",
            "target_weight","target_qty","current_qty","delta_qty","delta_notional",
            "selected_final","p_win","post_ml_score","post_ml_competitive_score","exit_reason"
        ] if c in sub.columns]
        if keep:
            print(sub[keep].tail(20).to_string(index=False))
        else:
            print(sub.tail(20).to_string(index=False))
PY
echo

echo "===== LAST RUN STATUS ====="
cat runtime/state/last_run_status.json 2>/dev/null || true
echo

echo "===== RECENT FILE TIMELINE ====="
ls -lt runtime/logs | head -n 10 || true
echo
ls -lt results | head -n 30 || true
