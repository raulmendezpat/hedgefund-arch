#!/usr/bin/env bash
# Weekly event and mapping review of the momentum book (docs/plans/xs_event_review.md), alerts only.
# Runs hourly; acts once per decision day after the Sunday 23:00 UTC close (Monday 00:00 UTC),
# before the 01:00 UTC fill. Needs ~/.config/hedgefund/anthropic.env.
# launchd/cron: every hour at minute 12.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Own subfolder: the health check scans runtime/logs/*.log as the dry-run's logs; this job notifies its own failures.
LOG_DIR="$APP_DIR/runtime/logs/event_review"
mkdir -p "$LOG_DIR"

cd "$APP_DIR"
source .venv/bin/activate
export PYTHONPATH="$APP_DIR/src"

python scripts/event_review_xs.py run --state-dir "$APP_DIR/runtime/event_review" \
  >> "$LOG_DIR/event_review_$(date -u +%Y%m%d).log" 2>&1
