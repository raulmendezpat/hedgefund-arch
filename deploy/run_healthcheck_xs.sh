#!/usr/bin/env bash
# Hourly health check of the momentum runners (paper + live dry-run), with macOS notifications.
# It only reads their state and logs. Runner locations default to the sibling worktrees:
#   PAPER_DIR (default ../hedgefund-arch-paper), LIVE_DIR (default this checkout), LIVE_MODE (dry-run)
# launchd/cron: every hour at minute 20.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="${PAPER_DIR:-$(cd "$APP_DIR/.." && pwd)/hedgefund-arch-paper}"
LIVE_DIR="${LIVE_DIR:-$APP_DIR}"
LIVE_MODE="${LIVE_MODE:-dry-run}"
LOG_DIR="$APP_DIR/runtime/logs"
mkdir -p "$LOG_DIR"

cd "$APP_DIR"
source .venv/bin/activate
export PYTHONPATH="$APP_DIR/src"

python scripts/healthcheck_xs.py \
  --state-file "$APP_DIR/runtime/healthcheck/state.json" \
  --runner "paper=$PAPER_DIR/runtime/paper/xs_momentum_v1:$PAPER_DIR/runtime/logs" \
  --runner "$LIVE_MODE=$LIVE_DIR/runtime/live/xs_momentum_v1/$LIVE_MODE:$LIVE_DIR/runtime/logs" \
  >> "$LOG_DIR/healthcheck_xs_$(date -u +%Y%m%d).log" 2>&1
