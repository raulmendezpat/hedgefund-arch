#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/hedgefund-arch-v2"
cd "$ROOT"

source .venv/bin/activate
export PYTHONPATH=src

echo "===== HOST / TIME ====="
hostname
date -u
date
echo

echo "===== INSTANCE RESOURCES ====="
free -h
echo
nproc
echo
df -h
echo

echo "===== GIT ====="
git branch --show-current || true
git log --oneline -n 5 || true
git describe --tags --always || true
echo

echo "===== PRE-RUN STATUS ====="
cat runtime/state/last_run_status.json 2>/dev/null || true
echo

echo "===== RECENT LOGS BEFORE MANUAL RUN ====="
ls -lt runtime/logs/run_*.log 2>/dev/null | head -n 8 || true
echo

echo "===== REMOVE STALE LOCK ====="
rm -f /tmp/hedgefund_arch_v2_prod_1h.lock
ls -l /tmp/hedgefund_arch_v2_prod_1h.lock 2>/dev/null || true
echo

echo "===== MANUAL PROD RUN ====="
bash deploy/run_prod_candidate_v2.sh || true
echo

echo "===== POST-RUN STATUS ====="
cat runtime/state/last_run_status.json 2>/dev/null || true
echo

echo "===== NEWEST LOGS AFTER MANUAL RUN ====="
ls -lt runtime/logs/run_*.log 2>/dev/null | head -n 8 || true
echo

LATEST_LOG="$(ls -t runtime/logs/run_*.log 2>/dev/null | head -n 1 || true)"
echo "LATEST_LOG=$LATEST_LOG"
echo

if [[ -n "${LATEST_LOG:-}" && -f "${LATEST_LOG:-}" ]]; then
  echo "===== LAST 160 LINES OF LATEST LOG ====="
  tail -n 160 "$LATEST_LOG" || true
  echo

  echo "===== KEY EXECUTION / RECONCILE LINES ====="
  grep -nE "PORTFOLIO SNAPSHOT|EXECUTION SNAPSHOT|LIVE RECONCILIATION|BNB/USDT:USDT|CANCEL_PLAN|PLACE_|tp_action|loss_plan|profit_plan|skip rebalance|Out of memory|Killed|Traceback|ERROR|Exception" "$LATEST_LOG" || true
  echo
fi

echo "===== RECENT RESULT FILES ====="
ls -lt results | head -n 20 || true
echo

echo "===== BNB STATE FROM LATEST RESULT FILES ====="
python - <<'PY'
from pathlib import Path
import pandas as pd

targets = [
    Path("results/research_runtime_prod_v2_live_candidate.csv"),
    Path("results/research_runtime_candidates_prod_v2_live_candidate.csv"),
    Path("results/research_runtime_lifecycle_events_prod_v2_live_candidate.csv"),
    Path("results/research_runtime_lifecycle_open_positions_prod_v2_live_candidate.csv"),
    Path("results/research_runtime_lifecycle_trades_prod_v2_live_candidate.csv"),
]

for p in targets:
    print("=" * 120)
    print(p)
    if not p.exists():
        print("missing")
        continue
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception as e:
        print(f"read_error: {e}")
        continue

    if "symbol" not in df.columns:
        print("no symbol column")
        print(df.tail(5).to_string(index=False))
        continue

    sub = df[df["symbol"].astype(str) == "BNB/USDT:USDT"].copy()
    print(f"rows={len(sub)}")
    if len(sub) == 0:
        continue

    keep = [c for c in [
        "ts", "symbol", "side", "strategy_id", "selected_final", "reason",
        "p_win", "post_ml_score", "post_ml_competitive_score",
        "target_weight", "qty", "bars_held", "pnl", "exit_reason"
    ] if c in sub.columns]

    print(sub[keep].tail(20).to_string(index=False))
PY
echo

echo "===== MEMORY / OOM CHECK AFTER RUN ====="
journalctl --since "30 minutes ago" --no-pager | grep -Ei "oom|killed process|out of memory|python invoked oom-killer|cron.service: Failed with result 'oom-kill'" || true
echo

echo "===== DONE ====="
