#!/usr/bin/env bash
# Hourly paper-trading cycle for the cross-sectional 10%-per-side momentum variant (docs/plans/xs_frac10_hypothesis.md).
# Uses only public market data (Bitget + Binance); it never sends orders and needs no API keys.
#
# One-time setup:
#   PYTHONPATH=src .venv/bin/python scripts/paper_xs_frac10.py init --capital 10000
# launchd/cron: every hour at minute 25.
# Report (includes the comparison with the primary momentum paper):
#   PYTHONPATH=src .venv/bin/python scripts/paper_xs_frac10.py report
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$APP_DIR/runtime/logs"
STATE_DIR="$APP_DIR/runtime/paper/xs_frac10_v1"
mkdir -p "$LOG_DIR"

cd "$APP_DIR"
source .venv/bin/activate
export PYTHONPATH="$APP_DIR/src"

# One log file per UTC day; the Python side holds its own lock against overlapping runs.
LOG_FILE="$LOG_DIR/paper_xs_frac10_$(date -u +%Y%m%d).log"
python scripts/paper_xs_frac10.py run --state-dir "$STATE_DIR" >> "$LOG_FILE" 2>&1
