#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/hedgefund-arch-v2"
cd "$ROOT"

echo "===== CLEANUP START $(date -u '+%Y-%m-%d %H:%M:%S UTC') ====="

before_kb="$(du -sk . | awk '{print $1}')"

mkdir -p runtime/logs results

echo "--- disk before ---"
df -h /
du -sh results runtime runtime/logs 2>/dev/null || true
echo

echo "--- remove zero-byte run logs ---"
find runtime/logs -maxdepth 1 -name 'run_*.log' -size 0 -print -delete || true
echo

echo "--- gzip older non-empty run logs (keep last 12 plain) ---"
mapfile -t RUN_LOGS < <(ls -1t runtime/logs/run_*.log 2>/dev/null || true)
if [[ ${#RUN_LOGS[@]} -gt 12 ]]; then
  for f in "${RUN_LOGS[@]:12}"; do
    if [[ -f "$f" && ! -f "$f.gz" ]]; then
      gzip -f "$f"
      echo "gzipped: $f"
    fi
  done
fi
echo

echo "--- delete heavy regenerable traces ---"
find results -maxdepth 1 -type f \( \
  -name 'selection_trace_*.jsonl' -o \
  -name 'allocation_inputs_trace_*.jsonl' \
\) -print -delete || true
echo

echo "--- delete stale research artifacts older than 5 days ---"
python - <<'PY'
from pathlib import Path
from datetime import datetime, timedelta, UTC

root = Path("results")
now = datetime.now(UTC)
cutoff = now - timedelta(days=5)

keep_prefixes = [
    "research_runtime_prod_v2_live_candidate",
    "research_runtime_candidates_prod_v2_live_candidate",
    "research_runtime_lifecycle_",
    "research_runtime_metrics_prod_v2_live_candidate",
    "selection_stage_summary_prod_v2_live_candidate",
    "selection_reason_summary_prod_v2_live_candidate",
    "prod_reconcile_",
    "shadow_vs_live_reconcile_",
    "live_protective_",
    "blocked_delta_",
    "aave_blocked_",
]

delete_suffixes = {".csv", ".json", ".jsonl"}

deleted = []
for p in sorted(root.glob("*")):
    if not p.is_file():
        continue
    if p.suffix not in delete_suffixes:
        continue

    name = p.name

    if any(name.startswith(k) for k in keep_prefixes):
        continue

    mtime = datetime.fromtimestamp(p.stat().st_mtime, UTC)
    if mtime >= cutoff:
        continue

    if (
        name.startswith("research_runtime_")
        or name.startswith("selection_stage_summary_")
        or name.startswith("selection_reason_summary_")
        or name.startswith("asset_side_")
        or name.startswith("pwin_")
        or name.startswith("rt_research_")
    ):
        deleted.append(str(p))
        p.unlink(missing_ok=True)

print("deleted_count:", len(deleted))
for x in deleted[:200]:
    print("deleted:", x)
PY
echo

echo "--- prune old compressed logs (keep last 48) ---"
mapfile -t GZ_LOGS < <(ls -1t runtime/logs/run_*.log.gz 2>/dev/null || true)
if [[ ${#GZ_LOGS[@]} -gt 48 ]]; then
  for f in "${GZ_LOGS[@]:48}"; do
    rm -f "$f"
    echo "deleted old gz log: $f"
  done
fi
echo

after_kb="$(du -sk . | awk '{print $1}')"
freed_mb="$(( (before_kb - after_kb) / 1024 ))"

echo "--- disk after ---"
df -h /
du -sh results runtime runtime/logs 2>/dev/null || true
echo

echo "freed_mb: ${freed_mb}"
echo "===== CLEANUP END ====="
