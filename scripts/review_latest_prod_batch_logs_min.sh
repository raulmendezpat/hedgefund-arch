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

echo "===== LAST RUN STATUS ====="
cat runtime/state/last_run_status.json 2>/dev/null || true
echo

echo "===== NEWEST LOG FILES ====="
ls -lt runtime/logs/run_*.log 2>/dev/null | head -n 12 || true
echo

LATEST_MAIN="$(ls -t runtime/logs/run_*.log 2>/dev/null | grep -vE 'run_.*T[0-9]{6}Z\.log$' >/dev/null 2>&1 || true)"

python - <<'PY'
from pathlib import Path
import re, json

logs = sorted(Path("runtime/logs").glob("run_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
if not logs:
    print("No encontré logs.")
    raise SystemExit(0)

latest = logs[0]
partner = logs[1] if len(logs) > 1 else None

# si el más nuevo es el short summary chico, usar el siguiente como main
if latest.stat().st_size < 1000 and partner is not None:
    summary_log = latest
    main_log = partner
else:
    main_log = latest
    summary_log = partner if partner and partner.stat().st_size < 1000 else None

print("===== SELECTED LOGS =====")
print("main_log:", main_log)
print("summary_log:", summary_log)
print()

def extract_metric(txt, key):
    m = re.search(rf"{re.escape(key)}:\s*([\-0-9.eE]+)", txt)
    return float(m.group(1)) if m else None

if summary_log and summary_log.exists():
    print("===== SHORT SUMMARY LOG =====")
    print(summary_log.read_text(errors="replace"))
    print()

txt = main_log.read_text(errors="replace")

print("===== MAIN METRICS =====")
payload = {
    "equity_final": extract_metric(txt, "equity_final"),
    "total_return_pct": extract_metric(txt, "total_return_pct"),
    "sharpe_annual": extract_metric(txt, "sharpe_annual"),
    "max_drawdown_pct": extract_metric(txt, "max_drawdown_pct"),
    "trade_count": extract_metric(txt, "trade_count"),
    "avg_active_symbols": extract_metric(txt, "avg_active_symbols"),
    "avg_gross_weight": extract_metric(txt, "avg_gross_weight"),
}
print(json.dumps(payload, indent=2))
print()

m = re.search(r"=== PORTFOLIO SNAPSHOT ===\n(.*?)\n=== LIVE RECONCILIATION ===", txt, re.S)
print("===== PORTFOLIO SNAPSHOT =====")
print(m.group(1).strip() if m else "No encontrado.")
print()

print("===== BNB BLOCK =====")
m = re.search(r"=== BNB/USDT:USDT ===\n(.*?)(?:\nMARGIN_MODE_OK -> symbol=BTC/USDT:USDT|\n=== EXECUTION SNAPSHOT ===)", txt, re.S)
print(m.group(0).strip() if m else "No encontrado.")
print()

print("===== KEY LINES =====")
for i, line in enumerate(txt.splitlines(), start=1):
    if any(x in line for x in [
        "ORDER ->",
        "POST_ORDER_REFRESH",
        "PLACE_SL_PLAN",
        "PLACE_TP1_PLAN",
        "PLACE_TP2_PLAN",
        "PROTECTIVE_PLAN_BEFORE",
        "PROTECTIVE_PLAN_AFTER",
        "sl_action:",
        "tp_action:",
        "blocked_reason",
        "reject",
        "error",
        "Traceback",
    ]):
        print(f"{i}:{line}")
print()

print("===== LAST 80 LINES =====")
lines = txt.splitlines()
print("\n".join(lines[-80:]))
PY

echo
echo "===== RECENT OOM / CRON ====="
journalctl --since "4 hours ago" --no-pager | grep -Ei 'oom|killed process|out of memory|run_prod_candidate_v2|cron.service: Failed with result' || true
echo
