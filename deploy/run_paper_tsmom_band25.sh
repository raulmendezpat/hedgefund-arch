#!/usr/bin/env bash
# Hourly paper-trading cycle for the TSMOM BTC/ETH variant with a 0.25 band (docs/plans/tsmom_band25_hypothesis.md).
# Uses only public market data (Bitget + Binance); it never sends orders and needs no API keys.
#
# One-time setup:
#   PYTHONPATH=src .venv/bin/python scripts/paper_tsmom.py init --name tsmom_band25_v1 --capital 10000
# launchd/cron: every hour at minute 30.
# Report (includes the comparison with the primary TSMOM paper on the same days):
#   PYTHONPATH=src .venv/bin/python scripts/paper_tsmom.py report --name tsmom_band25_v1
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$APP_DIR/runtime/logs"
STATE_DIR="$APP_DIR/runtime/paper/tsmom_band25_v1"
mkdir -p "$LOG_DIR"

cd "$APP_DIR"
source .venv/bin/activate
export PYTHONPATH="$APP_DIR/src"

# One log file per UTC day; the Python side holds its own lock against overlapping runs.
LOG_FILE="$LOG_DIR/paper_tsmom_band25_$(date -u +%Y%m%d).log"
python scripts/paper_tsmom.py run --state-dir "$STATE_DIR" >> "$LOG_FILE" 2>&1
