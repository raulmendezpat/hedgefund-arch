#!/usr/bin/env bash
# Hourly health check of the runners (xs, TSMOM and taker-flow papers, live dry-run), with macOS notifications.
# It only reads their state and logs. Runner locations default to the sibling worktrees:
#   PAPER_DIR (default ../hedgefund-arch-paper), TSMOM_DIR (default ../hedgefund-arch-paper-tsmom),
#   TAKER_DIR (default ../hedgefund-arch-paper-taker),
#   LIVE_DIR (default this checkout), LIVE_MODE (dry-run)
# launchd/cron: every hour at minute 20.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="${PAPER_DIR:-$(cd "$APP_DIR/.." && pwd)/hedgefund-arch-paper}"
TSMOM_DIR="${TSMOM_DIR:-$(cd "$APP_DIR/.." && pwd)/hedgefund-arch-paper-tsmom}"
TAKER_DIR="${TAKER_DIR:-$(cd "$APP_DIR/.." && pwd)/hedgefund-arch-paper-taker}"
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
  --runner "tsmom=$TSMOM_DIR/runtime/paper/tsmom_btc_eth_v1:$TSMOM_DIR/runtime/logs" \
  --runner "taker=$TAKER_DIR/runtime/paper/taker_flow_v1:$TAKER_DIR/runtime/logs" \
  --runner "$LIVE_MODE=$LIVE_DIR/runtime/live/xs_momentum_v1/$LIVE_MODE:$LIVE_DIR/runtime/logs" \
  >> "$LOG_DIR/healthcheck_xs_$(date -u +%Y%m%d).log" 2>&1
