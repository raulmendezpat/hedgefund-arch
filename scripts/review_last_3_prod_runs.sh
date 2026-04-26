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

echo "===== GIT ====="
git branch --show-current || true
git log --oneline -n 5 || true
git describe --tags --always || true
echo

echo "===== LAST RUN STATUS ====="
if [[ -f runtime/state/last_run_status.json ]]; then
  cat runtime/state/last_run_status.json
else
  echo "missing: runtime/state/last_run_status.json"
fi
echo

echo "===== LAST 3 LOG FILES ====="
mapfile -t LOGS < <(ls -t runtime/logs/run_*.log 2>/dev/null | head -n 3 || true)
if [[ ${#LOGS[@]} -eq 0 ]]; then
  echo "No encontré logs en runtime/logs/"
  exit 0
fi

printf '%s\n' "${LOGS[@]}"
echo

python - <<'PY'
from pathlib import Path
import json
import re

logs = sorted(Path("runtime/logs").glob("run_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:3]

def extract_float(text, key):
    m = re.search(rf"{re.escape(key)}:\s*([-+]?[0-9]*\.?[0-9]+)", text)
    return float(m.group(1)) if m else None

def extract_int(text, key):
    m = re.search(rf"{re.escape(key)}:\s*([0-9]+)", text)
    return int(m.group(1)) if m else None

def extract_json_block(text, header):
    idx = text.find(header)
    if idx < 0:
        return None
    start = text.find("{", idx)
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

rows = []
for log in logs:
    txt = log.read_text(errors="ignore")
    portfolio_raw = extract_json_block(txt, "=== PORTFOLIO SNAPSHOT ===")
    portfolio = {}
    if portfolio_raw:
        try:
            portfolio = json.loads(portfolio_raw)
        except Exception:
            portfolio = {}

    active_weights = portfolio.get("active_weights", {}) if isinstance(portfolio, dict) else {}
    active_symbols = ",".join(sorted(active_weights.keys())) if isinstance(active_weights, dict) and active_weights else ""

    rows.append({
        "log": str(log),
        "mtime": log.stat().st_mtime,
        "equity_final": extract_float(txt, "equity_final"),
        "total_return_pct": extract_float(txt, "total_return_pct"),
        "sharpe_annual": extract_float(txt, "sharpe_annual"),
        "max_drawdown_pct": extract_float(txt, "max_drawdown_pct"),
        "trade_count": extract_int(txt, "trade_count"),
        "avg_active_symbols": extract_float(txt, "avg_active_symbols"),
        "avg_gross_weight": extract_float(txt, "avg_gross_weight"),
        "portfolio_exposure": portfolio.get("exposure") if isinstance(portfolio, dict) else None,
        "portfolio_active_symbol_count": portfolio.get("active_symbol_count") if isinstance(portfolio, dict) else None,
        "portfolio_active_symbols": active_symbols,
    })

print("===== LAST 3 RUN SUMMARY =====")
for r in rows:
    print(r)
PY
echo

for LOG in "${LOGS[@]}"; do
  echo "===================================================================================================="
  echo "LOG: $LOG"
  echo "===================================================================================================="
  echo

  echo "----- METRICS -----"
  grep -E "^(equity_start|equity_final|total_return_pct|sharpe_annual|max_drawdown_pct|vol_annual|trade_count|win_rate_pct|rows|avg_n_opps|avg_n_accepts|avg_active_symbols|avg_gross_weight):" "$LOG" || true
  echo

  echo "----- PORTFOLIO SNAPSHOT -----"
  sed -n '/=== PORTFOLIO SNAPSHOT ===/,/=== LIVE RECONCILIATION ===/p' "$LOG" || true
  echo

  echo "----- EXECUTION SNAPSHOT -----"
  sed -n '/=== EXECUTION SNAPSHOT ===/,$p' "$LOG" | sed -n '1,160p' || true
  echo

  echo "----- BNB / MANUAL-INTERVENTION RELEVANT LINES -----"
  grep -nEi "BNB/USDT:USDT|manual|reconcile|reconciliation|skip rebalance|cancel|PLACE_|CANCEL_|PROTECTIVE_PLAN|tp_action|loss_plan|profit_plan|reduce-only|reject|error|exception|traceback" "$LOG" || true
  echo

  echo "----- LAST 120 LINES -----"
  tail -n 120 "$LOG" || true
  echo
done

echo "===== LAST RESULT FILES ====="
ls -lt results | head -n 30 || true
echo

echo "===== LAST LIFECYCLE / METRICS FILES ====="
ls -lt results/research_runtime_* results/*prod* 2>/dev/null | head -n 40 || true
