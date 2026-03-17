#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/ubuntu/hedgefund-arch"
cd "$APP_DIR"

CONFIG_JSON="$APP_DIR/deploy/live_execution_config.json"
mapfile -t HEALTHCHECK_SYMBOLS < <(
  python - <<'PY2'
import json
from pathlib import Path

cfg = json.loads(Path("deploy/live_execution_config.json").read_text())
symbols = list((cfg.get("symbol_overrides") or {}).keys())
for s in symbols:
    print(s)
PY2
)

if [ "${#HEALTHCHECK_SYMBOLS[@]}" -eq 0 ]; then
  HEALTHCHECK_SYMBOLS=("BTC/USDT:USDT" "SOL/USDT:USDT")
fi

source .venv/bin/activate

echo "======================================"
echo "BOT HEALTHCHECK"
echo "======================================"
echo "server_time_utc: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo

echo "=== LAST STATUS ==="
if [ -f runtime/state/last_run_status.json ]; then
  cat runtime/state/last_run_status.json
else
  echo "missing: runtime/state/last_run_status.json"
fi
echo

echo "=== LAST LOG FILE ==="
LATEST_LOG=$(ls -1t runtime/logs/run_*.log 2>/dev/null | head -n 1 || true)
if [ -n "${LATEST_LOG:-}" ]; then
  echo "$LATEST_LOG"
  echo
  echo "--- tail last 80 lines ---"
  tail -n 80 "$LATEST_LOG"
else
  echo "no log files found"
fi
echo

export HEALTHCHECK_SYMBOLS_CSV="${HEALTHCHECK_SYMBOLS[*]}"
python - <<'PY'
import json
from pathlib import Path
import pandas as pd

from hf.legacy.ltb.utilities.bitget_futures import BitgetFutures

APP = Path("/home/ubuntu/hedgefund-arch")
secret = json.loads((APP / "secret.json").read_text())["envelope"]
bitget = BitgetFutures(secret)

print("=== ACCOUNT ===")
bal = bitget.fetch_balance()
usdt = (bal.get("USDT") or {})
print("usdt_free:", usdt.get("free"))
print("usdt_used:", usdt.get("used"))
print("usdt_total:", usdt.get("total"))
print()

print("=== POSITIONS ===")
import os
symbols = os.environ["HEALTHCHECK_SYMBOLS_CSV"].split()
for sym in symbols:
    pos = bitget.fetch_open_positions(sym)
    print(sym, "->", pos)
print()

print("=== OPEN TRIGGER ORDERS (SL/TP) ===")
import os
symbols = os.environ["HEALTHCHECK_SYMBOLS_CSV"].split()
for sym in symbols:
    try:
        orders = bitget.fetch_open_trigger_orders(sym) or []
    except Exception as e:
        orders = f"ERROR: {e}"
    print(sym, "->", orders)
print()

print("=== LATEST MODEL DECISION ===")
p = APP / "results/pipeline_allocations_prod_candidate_live.csv"
if p.exists():
    df = pd.read_csv(p)
    if len(df) > 0:
        row = df.iloc[-1]
        cols = [
            "ts",
            "w_btc", "w_sol",
            "btc_side", "sol_side",
            "btc_strategy_id", "sol_strategy_id",
            "btc_p_win", "sol_p_win",
            "btc_post_ml_score", "sol_post_ml_score",
            "btc_execution_target_weight", "sol_execution_target_weight",
        ]
        for c in cols:
            if c in df.columns:
                print(f"{c}: {row[c]}")
    else:
        print("pipeline allocations file exists but is empty")
else:
    print("missing:", p)
PY
