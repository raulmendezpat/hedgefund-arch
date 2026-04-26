#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/hedgefund-arch-v2"
cd "$ROOT"

echo "===== CLEANUP SAFE NOW ====="
date -u
date
echo

echo "===== BEFORE ====="
df -h /
du -sh results runtime runtime/logs 2>/dev/null || true
echo

mkdir -p runtime/logs

echo "===== REMOVE ZERO-BYTE RUN LOGS ====="
find runtime/logs -maxdepth 1 -name 'run_*.log' -size 0 -print -delete || true
echo

echo "===== COMPRESS OLD NON-EMPTY RUN LOGS (KEEP LAST 12 PLAIN) ====="
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

echo "===== DELETE LARGE REGENERABLE TRACE FILES ====="
find results -maxdepth 1 -type f \( \
  -name 'selection_trace_*.jsonl' -o \
  -name 'allocation_inputs_trace_*.jsonl' \
\) -print -delete || true
echo

echo "===== DELETE OLD RESEARCH CSV/JSON ARTIFACTS (KEEP PROD + RECENT) ====="
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

    # borrar artefactos de research / comparativas / baselines viejos
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

echo "===== DELETE OLD COMPRESSED LOGS (KEEP LAST 48) ====="
mapfile -t GZ_LOGS < <(ls -1t runtime/logs/run_*.log.gz 2>/dev/null || true)
if [[ ${#GZ_LOGS[@]} -gt 48 ]]; then
  for f in "${GZ_LOGS[@]:48}"; do
    rm -f "$f"
    echo "deleted old gz log: $f"
  done
fi
echo

echo "===== AFTER ====="
df -h /
du -sh results runtime runtime/logs 2>/dev/null || true
echo
