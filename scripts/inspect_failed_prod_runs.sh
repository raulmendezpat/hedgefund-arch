#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/hedgefund-arch-v2"
cd "$ROOT"

source .venv/bin/activate
export PYTHONPATH=src

echo "===== TIME ====="
date -u
date
echo

echo "===== LAST RUN STATUS ====="
cat runtime/state/last_run_status.json 2>/dev/null || true
echo

echo "===== ZERO / SMALL LOGS ====="
ls -l runtime/logs/run_*.log 2>/dev/null | tail -n 20 || true
echo
find runtime/logs -maxdepth 1 -name 'run_*.log' -size 0 -print | sort || true
echo

echo "===== CRON ====="
crontab -l 2>/dev/null || true
echo

echo "===== RECENT SYSTEM / CRON / KILL / OOM ====="
journalctl --since "2026-04-06 03:30:00" --no-pager | \
grep -Ei 'CRON|cron|oom|killed process|out of memory|hedgefund|run_prod_candidate_v2|python|bash' || true
echo

echo "===== DISK ====="
df -h
echo
du -sh runtime runtime/logs results 2>/dev/null || true
echo

echo "===== INODES ====="
df -i
echo

echo "===== RECENT CLEANUP LOG ====="
if [[ -f runtime/logs/cleanup_runtime_space.log ]]; then
  tail -n 200 runtime/logs/cleanup_runtime_space.log
else
  echo "missing: runtime/logs/cleanup_runtime_space.log"
fi
echo

echo "===== PROD RUN SCRIPT ====="
sed -n '1,260p' deploy/run_prod_candidate_v2.sh
echo

echo "===== TRY DRY MANUAL RUN WITH TRACE ====="
bash -x deploy/run_prod_candidate_v2.sh > /tmp/prod_run_manual_trace.log 2>&1 || true
tail -n 200 /tmp/prod_run_manual_trace.log || true
echo

echo "===== NEWEST LOGS AFTER MANUAL TRACE ====="
ls -lt runtime/logs | head -n 10 || true
echo
