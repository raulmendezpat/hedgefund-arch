#!/usr/bin/env bash
# Hourly live cycle for time-series momentum on BTC/ETH (see scripts/live_tsmom.py).
#
# MODE=dry-run (default) reads the account and logs the orders it would send; it changes nothing.
# MODE=live sends orders and also needs LIVE_TRADING=1.
# Credentials: the Python runner reads BITGET_API_KEY / BITGET_API_SECRET / BITGET_API_PASSPHRASE
# from $BITGET_TSMOM_ENV_FILE (default ~/.config/hedgefund/bitget_tsmom.env, chmod 600, never inside
# the repo) as plain KEY=VALUE data; this script never sources it. It must be a different
# subaccount from the momentum runner's.
#
# One-time setup:  PYTHONPATH=src .venv/bin/python scripts/live_tsmom.py init --mode dry-run
# Cron (minute 17, after the paper runners):
#   17 * * * * MODE=dry-run /path/to/hedgefund-arch/deploy/run_live_tsmom.sh
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${MODE:-dry-run}"
LOG_DIR="$APP_DIR/runtime/logs/live_tsmom"
mkdir -p "$LOG_DIR"

cd "$APP_DIR"
source .venv/bin/activate
export PYTHONPATH="$APP_DIR/src"

LOG_FILE="$LOG_DIR/live_tsmom_${MODE}_$(date -u +%Y%m%d).log"
python scripts/live_tsmom.py run --mode "$MODE" >> "$LOG_FILE" 2>&1
