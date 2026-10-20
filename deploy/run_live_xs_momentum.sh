#!/usr/bin/env bash
# Hourly live cycle for the cross-sectional momentum hypothesis (see scripts/live_xs_momentum.py).
#
# MODE=dry-run (default) reads the account and logs the orders it would send; it changes nothing.
# MODE=live sends orders and also needs LIVE_TRADING=1.
# Credentials: the Python runner reads BITGET_API_KEY / BITGET_API_SECRET / BITGET_API_PASSPHRASE
# from $BITGET_ENV_FILE (default ~/.config/hedgefund/bitget.env, chmod 600, never inside the repo)
# as plain KEY=VALUE data; this script never sources it.
#
# One-time setup:  PYTHONPATH=src .venv/bin/python scripts/live_xs_momentum.py init --mode dry-run
# Cron (minute 7, after the paper runner):
#   7 * * * * MODE=dry-run /path/to/hedgefund-arch/deploy/run_live_xs_momentum.sh
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${MODE:-dry-run}"
LOG_DIR="$APP_DIR/runtime/logs"
mkdir -p "$LOG_DIR"

cd "$APP_DIR"
source .venv/bin/activate
export PYTHONPATH="$APP_DIR/src"

LOG_FILE="$LOG_DIR/live_xs_momentum_${MODE}_$(date -u +%Y%m%d).log"
python scripts/live_xs_momentum.py run --mode "$MODE" >> "$LOG_FILE" 2>&1
