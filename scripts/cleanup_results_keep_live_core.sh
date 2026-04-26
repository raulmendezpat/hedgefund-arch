#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/hedgefund-arch-v2"
cd "$ROOT"

echo "===== BEFORE ====="
df -h
echo
du -sh results runtime 2>/dev/null || true
echo
ls -lh results | sort -k5 -h | tail -n 40 || true
echo

mkdir -p runtime/archive_cleanup

# borrar traces pesados de producción actual que no son esenciales
rm -f results/allocation_inputs_trace_prod_v2_live_candidate.jsonl
rm -f results/selection_trace_prod_v2_live_candidate.jsonl

# borrar artefactos research viejos que no necesitas para producción inmediata
rm -f results/research_runtime_rt_research_best_v3_90d.csv
rm -f results/research_runtime_rt_research_best_v5_90d.csv
rm -f results/research_runtime_lifecycle_trades_rt_research_best_v3_90d.csv
rm -f results/research_runtime_lifecycle_trades_rt_research_best_v5_90d.csv
rm -f results/research_runtime_lifecycle_open_positions_rt_research_best_v3_90d.csv
rm -f results/research_runtime_lifecycle_open_positions_rt_research_best_v5_90d.csv
rm -f results/research_runtime_lifecycle_events_rt_research_best_v3_90d.csv
rm -f results/research_runtime_lifecycle_events_rt_research_best_v5_90d.csv
rm -f results/research_runtime_lifecycle_equity_rt_research_best_v3_90d.csv
rm -f results/research_runtime_lifecycle_equity_rt_research_best_v5_90d.csv
rm -f results/research_runtime_metrics_rt_research_best_v3_90d.json
rm -f results/research_runtime_metrics_rt_research_best_v5_90d.json
rm -f results/research_runtime_lifecycle_metrics_rt_research_best_v3_90d.json
rm -f results/research_runtime_lifecycle_metrics_rt_research_best_v5_90d.json

# opcional: deja solo auditorías recientes útiles
find results -maxdepth 1 -type f -name '*.jsonl' -size +10M -print -delete || true

echo
echo "===== AFTER ====="
df -h
echo
du -sh results runtime 2>/dev/null || true
echo
ls -lh results | sort -k5 -h | tail -n 40 || true
