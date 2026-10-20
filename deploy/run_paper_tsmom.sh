#!/usr/bin/env bash
# Hourly paper-trading cycle for time-series momentum on BTC/ETH (docs/plans/tsmom_paper_plan.md).
# Uses only public market data (Bitget + Binance); it never sends orders and needs no API keys.
#
# One-time setup:
#   PYTHONPATH=src .venv/bin/python scripts/paper_tsmom.py init --capital 10000
# launchd/cron: every hour at minute 10.
# Report (includes the 50/50 combination with the cross-sectional momentum paper):
#   PYTHONPATH=src .venv/bin/python scripts/paper_tsmom.py report
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$APP_DIR/runtime/logs"
STATE_DIR="$APP_DIR/runtime/paper/tsmom_btc_eth_v1"
mkdir -p "$LOG_DIR"

cd "$APP_DIR"
source .venv/bin/activate
export PYTHONPATH="$APP_DIR/src"

# One log file per UTC day; the Python side holds its own lock against overlapping runs.
LOG_FILE="$LOG_DIR/paper_tsmom_$(date -u +%Y%m%d).log"
python scripts/paper_tsmom.py run --state-dir "$STATE_DIR" >> "$LOG_FILE" 2>&1
