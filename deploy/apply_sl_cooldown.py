#!/usr/bin/env python3
"""
Apply a symbol-level cooldown after a stop-loss event.

Purpose:
- Detect recent SL/stop-loss exits from the lifecycle trades CSV.
- Persist a cooldown state in runtime/state/sl_cooldown_state.json.
- For any symbol in active cooldown, zero the latest-row execution weights
  in the runtime CSV before live reconcile runs.
- Cooldown automatically expires after the configured number of hours.

This script is intentionally execution-layer only:
- It does not alter strategy selection logic.
- It does not place or cancel exchange orders.
- It only prevents new/rebalanced exposure for symbols in cooldown by zeroing
  latest runtime target weight columns before reconcile_live.py consumes them.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SL_REASON_RE = re.compile(
    r"(^|[^a-z0-9])(sl|stop_loss|stop-loss|stoploss|loss_plan)([^a-z0-9]|$)",
    re.IGNORECASE,
)

TIME_STOP_RE = re.compile(r"time[_ -]?stop", re.IGNORECASE)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none", "null"}:
        return None

    try:
        ts = pd.to_datetime(text, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


def symbol_slug(symbol: str) -> str:
    slug = symbol.lower()
    slug = slug.replace("/", "_")
    slug = slug.replace(":", "_")
    slug = slug.replace("-", "_")
    slug = re.sub(r"[^a-z0-9_]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug


def is_sl_reason(reason: Any) -> bool:
    text = str(reason or "").strip().lower()
    if not text:
        return False
    if TIME_STOP_RE.search(text):
        return False
    return bool(SL_REASON_RE.search(text))


def find_col(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def trade_key(row: pd.Series, symbol_col: str, side_col: str | None, time_col: str, reason_col: str) -> str:
    symbol = str(row.get(symbol_col, ""))
    side = str(row.get(side_col, "")) if side_col else ""
    ts = str(row.get(time_col, ""))
    reason = str(row.get(reason_col, ""))
    pnl = str(row.get("pnl", row.get("realized_pnl", row.get("net_pnl", ""))))
    return "|".join([symbol, side, ts, reason, pnl])


def zero_latest_runtime_weights(runtime_path: Path, cooldown_symbols: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "runtime_path": str(runtime_path),
        "cooldown_symbols": cooldown_symbols,
        "changed": False,
        "zeroed_columns_by_symbol": {},
    }

    if not cooldown_symbols:
        return result

    if not runtime_path.exists():
        raise FileNotFoundError(f"runtime CSV not found: {runtime_path}")

    runtime = pd.read_csv(runtime_path, low_memory=False)
    if runtime.empty:
        result["warning"] = "runtime CSV is empty"
        return result

    latest_idx = runtime.index[-1]

    target_patterns = [
        "_w_raw_allocator",
        "_w_after_ml_position_sizing",
        "_w_after_smoothing",
        "_w_after_signal_gating",
        "_cluster_target_weight",
        "_execution_target_weight",
        "_live_capped_execution_target_weight",
    ]

    exclude_patterns = [
        "_live_execution_max_target_weight",
        "_max_target_weight",
        "_cap",
        "_cap_delta",
    ]

    for symbol in cooldown_symbols:
        slug = symbol_slug(symbol)
        zero_cols: list[str] = []

        for col in runtime.columns:
            col_l = col.lower()
            if slug not in col_l:
                continue
            if any(x in col_l for x in exclude_patterns):
                continue
            if any(x in col_l for x in target_patterns):
                zero_cols.append(col)

        if zero_cols:
            runtime.loc[latest_idx, zero_cols] = 0.0
            result["zeroed_columns_by_symbol"][symbol] = zero_cols
            result["changed"] = True
        else:
            result["zeroed_columns_by_symbol"][symbol] = []

    if result["changed"]:
        backup = runtime_path.with_suffix(runtime_path.suffix + ".bak_before_sl_cooldown")
        if not backup.exists():
            backup.write_bytes(runtime_path.read_bytes())
        runtime.to_csv(runtime_path, index=False)
        result["backup"] = str(backup)

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-csv", required=True)
    parser.add_argument("--trades-csv", required=True)
    parser.add_argument("--state-json", default="runtime/state/sl_cooldown_state.json")
    parser.add_argument("--cooldown-hours", type=float, default=6.0)
    parser.add_argument("--lookback-hours", type=float, default=6.5)
    parser.add_argument("--report-json", default="")
    args = parser.parse_args()

    now = utc_now()
    runtime_path = Path(args.runtime_csv)
    trades_path = Path(args.trades_csv)
    state_path = Path(args.state_json)

    state = load_json(state_path, default={})
    if not isinstance(state, dict):
        state = {}

    state.setdefault("cooldown_hours", args.cooldown_hours)
    state.setdefault("cooldowns", {})
    state.setdefault("seen_sl_trade_keys", [])

    cooldowns: dict[str, Any] = state.get("cooldowns", {})
    seen = set(state.get("seen_sl_trade_keys", []))

    # Expire old cooldowns.
    active_cooldowns: dict[str, Any] = {}
    expired: list[str] = []

    for symbol, info in cooldowns.items():
        until = parse_dt(info.get("cooldown_until_utc"))
        if until and until > now:
            active_cooldowns[symbol] = info
        else:
            expired.append(symbol)

    new_cooldowns: list[dict[str, Any]] = []
    warnings: list[str] = []

    if trades_path.exists() and trades_path.stat().st_size > 0:
        trades = pd.read_csv(trades_path, low_memory=False)

        symbol_col = find_col(list(trades.columns), ["symbol"])
        side_col = find_col(list(trades.columns), ["side", "position_side", "hold_side"])
        reason_col = find_col(list(trades.columns), ["exit_reason", "close_reason", "reason", "exit_type"])
        time_col = find_col(
            list(trades.columns),
            ["exit_ts", "close_ts", "closed_at", "exit_time", "close_time", "timestamp", "ts"],
        )

        if not symbol_col:
            warnings.append("trades CSV has no symbol column; cannot detect SL cooldowns")
        if not reason_col:
            warnings.append("trades CSV has no exit reason column; cannot detect SL cooldowns")
        if not time_col:
            warnings.append("trades CSV has no exit timestamp column; cannot detect recent SL cooldowns")

        if symbol_col and reason_col and time_col:
            lookback_start = now - timedelta(hours=float(args.lookback_hours))

            for _, row in trades.iterrows():
                reason = row.get(reason_col)
                if not is_sl_reason(reason):
                    continue

                trade_ts = parse_dt(row.get(time_col))
                if not trade_ts:
                    continue

                if trade_ts < lookback_start:
                    continue

                key = trade_key(row, symbol_col, side_col, time_col, reason_col)
                if key in seen:
                    continue

                symbol = str(row.get(symbol_col, "")).strip()
                if not symbol:
                    continue

                cooldown_until = trade_ts + timedelta(hours=float(args.cooldown_hours))
                if cooldown_until <= now:
                    # Historical SL already outside cooldown window.
                    seen.add(key)
                    continue

                info = {
                    "symbol": symbol,
                    "cooldown_started_utc": trade_ts.isoformat(),
                    "cooldown_until_utc": cooldown_until.isoformat(),
                    "reason": str(reason),
                    "source": str(trades_path),
                    "trade_key": key,
                    "detected_at_utc": now.isoformat(),
                }

                active_cooldowns[symbol] = info
                new_cooldowns.append(info)
                seen.add(key)
    else:
        warnings.append(f"trades CSV missing or empty: {trades_path}")

    active_symbols = sorted(active_cooldowns.keys())
    zero_result = zero_latest_runtime_weights(runtime_path, active_symbols)

    state["cooldown_hours"] = args.cooldown_hours
    state["updated_at_utc"] = now.isoformat()
    state["cooldowns"] = active_cooldowns
    state["seen_sl_trade_keys"] = sorted(seen)[-5000:]
    save_json(state_path, state)

    report = {
        "status": "ok",
        "now_utc": now.isoformat(),
        "cooldown_hours": args.cooldown_hours,
        "lookback_hours": args.lookback_hours,
        "state_json": str(state_path),
        "runtime_csv": str(runtime_path),
        "trades_csv": str(trades_path),
        "expired_symbols": expired,
        "new_cooldowns": new_cooldowns,
        "active_cooldown_symbols": active_symbols,
        "zero_result": zero_result,
        "warnings": warnings,
    }

    print(json.dumps(report, indent=2, sort_keys=True))

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
