#!/usr/bin/env bash
# Hourly paper-trading cycle for the cross-sectional momentum hypothesis.
# Uses only public market data (Bitget + Binance); it never sends orders and needs no API keys.
#
# One-time setup:
#   PYTHONPATH=src .venv/bin/python scripts/paper_xs_momentum.py init --capital 10000
# Cron (every hour at minute 5, UTC host clock assumed by the schedule logic itself):
#   5 * * * * /path/to/hedgefund-arch/deploy/run_paper_xs_momentum.sh
# Report:
#   PYTHONPATH=src .venv/bin/python scripts/paper_xs_momentum.py report
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$APP_DIR/runtime/logs"
STATE_DIR="$APP_DIR/runtime/paper/xs_momentum_v1"
mkdir -p "$LOG_DIR"

cd "$APP_DIR"
source .venv/bin/activate
export PYTHONPATH="$APP_DIR/src"

# One log file per UTC day; the Python side holds its own lock against overlapping runs.
LOG_FILE="$LOG_DIR/paper_xs_momentum_$(date -u +%Y%m%d).log"
python scripts/paper_xs_momentum.py run --state-dir "$STATE_DIR" >> "$LOG_FILE" 2>&1
