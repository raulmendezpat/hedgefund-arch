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
cat runtime/state/last_run_status.json 2>/dev/null || true
echo

echo "===== NEWEST LOG FILES ====="
ls -lt runtime/logs/run_*.log 2>/dev/null | head -n 20 || true
echo

python - <<'PY'
from pathlib import Path
import re
import json

logs = sorted(Path("runtime/logs").glob("run_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
logs = logs[:12]

print("===== LATEST BATCH DETECTION =====")
if not logs:
    print("No encontré logs.")
    raise SystemExit(0)

sizes = [(p.name, p.stat().st_size) for p in logs]
for name, size in sizes:
    print(f"{name} size={size}")

batch = []
started = False
for p in logs:
    size = p.stat().st_size
    if not started:
        batch.append(p)
        started = True
        continue
    if size == 0 or size < 700:
        batch.append(p)
    else:
        break

batch = sorted(batch, key=lambda p: p.stat().st_mtime)

print("\n===== BATCH UNDER REVIEW =====")
for p in batch:
    print(f"{p} size={p.stat().st_size}")

metric_patterns = {
    "equity_final": re.compile(r"equity_final:\s*([\-0-9.eE]+)"),
    "total_return_pct": re.compile(r"total_return_pct:\s*([\-0-9.eE]+)"),
    "sharpe_annual": re.compile(r"sharpe_annual:\s*([\-0-9.eE]+)"),
    "max_drawdown_pct": re.compile(r"max_drawdown_pct:\s*([\-0-9.eE]+)"),
    "trade_count": re.compile(r"trade_count:\s*([\-0-9.eE]+)"),
    "avg_active_symbols": re.compile(r"avg_active_symbols:\s*([\-0-9.eE]+)"),
    "avg_gross_weight": re.compile(r"avg_gross_weight:\s*([\-0-9.eE]+)"),
}

sym_re = re.compile(r'"([A-Z]+/USDT:USDT)"\s*:\s*\{')
action_re = re.compile(r'"action":\s*"([^"]+)"')
side_re = re.compile(r'"side":\s*"([^"]+)"')
target_re = re.compile(r'"target_weight":\s*([\-0-9.eE]+)')
qty_re = re.compile(r'"current_qty":\s*([\-0-9.eE]+)')
target_qty_re = re.compile(r'"target_qty":\s*([\-0-9.eE]+)')

for p in batch:
    print("\n" + "=" * 120)
    print(f"LOG: {p}")
    print("=" * 120)
    txt = p.read_text(errors="replace")

    print("\n----- SUMMARY -----")
    summary = {"log": str(p), "size": p.stat().st_size}
    for k, rgx in metric_patterns.items():
        m = rgx.search(txt)
        summary[k] = float(m.group(1)) if m else None
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n----- PORTFOLIO SNAPSHOT -----")
    m = re.search(r"=== PORTFOLIO SNAPSHOT ===\n(.*?)\n=== LIVE RECONCILIATION ===", txt, re.S)
    if m:
        print(m.group(1).strip())
    else:
        print("No encontrado.")

    print("\n----- EXECUTION SNAPSHOT -----")
    m = re.search(r"=== EXECUTION SNAPSHOT ===\n(\{.*?\})\s*$", txt, re.S)
    if m:
        snap = m.group(1)
        print(snap[:6000])
        if len(snap) > 6000:
            print("\n...[truncated]...")
    else:
        print("No encontrado.")

    print("\n----- SYMBOL BLOCKS WITH ORDER / RECONCILE ACTIVITY -----")
    interesting = []
    current_symbol = None
    current_block = []
    for line in txt.splitlines():
        if line.startswith("=== ") and "/USDT:USDT" in line and line.endswith(" ==="):
            if current_symbol and current_block:
                block_text = "\n".join(current_block)
                if any(x in block_text for x in [
                    "ORDER ->",
                    "POST_ORDER_REFRESH",
                    "PLACE_SL_PLAN",
                    "PLACE_TP1_PLAN",
                    "PLACE_TP2_PLAN",
                    "CANCEL_PLAN",
                    "tp_action:",
                    "reduceOnly",
                    "reject",
                    "blocked_reason",
                ]):
                    interesting.append((current_symbol, block_text))
            current_symbol = line.replace("=== ", "").replace(" ===", "").strip()
            current_block = [line]
        elif current_symbol:
            current_block.append(line)

    if current_symbol and current_block:
        block_text = "\n".join(current_block)
        if any(x in block_text for x in [
            "ORDER ->",
            "POST_ORDER_REFRESH",
            "PLACE_SL_PLAN",
            "PLACE_TP1_PLAN",
            "PLACE_TP2_PLAN",
            "CANCEL_PLAN",
            "tp_action:",
            "reduceOnly",
            "reject",
            "blocked_reason",
        ]):
            interesting.append((current_symbol, block_text))

    if interesting:
        for sym, block in interesting:
            print(f"\n### {sym}")
            print(block[:4000])
            if len(block) > 4000:
                print("...[truncated]...")
    else:
        print("No encontré bloques con actividad de ejecución.")

    print("\n----- KEY LINES -----")
    for i, line in enumerate(txt.splitlines(), start=1):
        if any(x in line for x in [
            "ORDER ->",
            "POST_ORDER_REFRESH",
            "PLACE_SL_PLAN",
            "PLACE_TP1_PLAN",
            "PLACE_TP2_PLAN",
            "CANCEL_PLAN",
            "PROTECTIVE_PLAN_BEFORE",
            "PROTECTIVE_PLAN_AFTER",
            "tp_action:",
            "action: ",
            "blocked_reason",
            "reject",
            "error",
            "exception",
            "Traceback",
            "OOM",
            "Killed",
        ]):
            print(f"{i}:{line}")

    print("\n----- LAST 120 LINES -----")
    lines = txt.splitlines()
    tail = lines[-120:] if len(lines) > 120 else lines
    print("\n".join(tail))
PY

echo
echo "===== RECENT RESULT FILES ====="
ls -lt results 2>/dev/null | head -n 30 || true
echo

echo "===== RECENT OOM / KILL / CRON ====="
journalctl --since "8 hours ago" --no-pager | grep -Ei 'oom|killed process|out of memory|python invoked oom-killer|run_prod_candidate_v2|cron.service: Failed with result' || true
echo
