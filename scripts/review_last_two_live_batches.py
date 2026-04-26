#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path("/home/ubuntu/hedgefund-arch-v2")
LOG_DIR = ROOT / "runtime" / "logs"
RESULTS_DIR = ROOT / "results"
STATE_PATH = ROOT / "runtime" / "state" / "last_run_status.json"

MAIN_LOG_RE = re.compile(r"run_(\d{8}T\d{6}Z)\.log$")
SUMMARY_RE = re.compile(r"^RUNTIME_RUN\s*$", re.M)

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

def find_latest_logs() -> list[Path]:
    logs = sorted(LOG_DIR.glob("run_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs

def extract_runtime_run_block(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if "RUNTIME_RUN" not in text:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line == "RUNTIME_RUN":
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out

def extract_json_block(text: str, marker: str) -> Any | None:
    idx = text.find(marker)
    if idx < 0:
        return None
    sub = text[idx + len(marker):].lstrip()
    if not sub.startswith("{"):
        return None
    depth = 0
    buf = []
    for ch in sub:
        buf.append(ch)
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
    try:
        return json.loads("".join(buf))
    except Exception:
        return None

def extract_symbol_block(text: str, symbol: str) -> str:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == f"=== {symbol} ===":
            start = i
            break
    if start is None:
        return "No encontrado."
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("===") and lines[j].strip() != f"=== {symbol} ===":
            end = j
            break
    return "\n".join(lines[start:end]).strip()

def extract_key_lines(text: str) -> list[str]:
    pats = [
        r"PROTECTIVE_PLAN_BEFORE",
        r"PROTECTIVE_PLAN_AFTER",
        r"sl_action:",
        r"tp_action:",
        r"blocked_reason",
        r"action:",
        r"reduce",
        r"reject",
        r"reconcile",
        r"cancelled stale plan orders",
        r"skip rebalance",
    ]
    out = []
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        l = line.lower()
        if any(re.search(p.lower(), l) for p in pats):
            out.append(f"{i}:{line}")
    return out[:120]

def parse_main_summary(text: str) -> dict[str, Any]:
    metrics = {}
    for key in [
        "equity_final",
        "total_return_pct",
        "sharpe_annual",
        "max_drawdown_pct",
        "trade_count",
        "avg_active_symbols",
        "avg_gross_weight",
    ]:
        m = re.search(rf"^{re.escape(key)}:\s+(.+)$", text, re.M)
        if m:
            raw = m.group(1).strip()
            try:
                metrics[key] = float(raw)
            except Exception:
                metrics[key] = raw
        else:
            metrics[key] = None
    return metrics

def choose_batches() -> list[tuple[Path, Path | None]]:
    logs = find_latest_logs()
    pairs = []
    used = set()

    for log in logs:
        if log in used:
            continue
        txt = read_text(log)
        runtime_run = extract_runtime_run_block(txt)
        if runtime_run:
            main_name = runtime_run.get("log", "")
            # summary logs don't point back to main log path, so use nearest older larger log
            ts = log.stem.replace("run_", "")
            # try find nearest prior non-summary log
            candidate_main = None
            for other in logs:
                if other == log:
                    continue
                if other in used:
                    continue
                if other.stat().st_size > 2000 and other.stat().st_mtime <= log.stat().st_mtime:
                    candidate_main = other
                    break
            pairs.append((candidate_main or log, log))
            used.add(log)
            if candidate_main:
                used.add(candidate_main)
        else:
            # non-summary main log
            # pair with immediate newer tiny summary if exists
            summary = None
            for other in logs:
                if other == log or other in used:
                    continue
                txt2 = read_text(other)
                if extract_runtime_run_block(txt2):
                    if abs(other.stat().st_mtime - log.stat().st_mtime) < 600:
                        summary = other
                        break
            pairs.append((log, summary))
            used.add(log)
            if summary:
                used.add(summary)

        if len(pairs) >= 2:
            break

    # normalize unique by main log path
    normalized = []
    seen = set()
    for main, summary in pairs:
        key = str(main)
        if key in seen:
            continue
        seen.add(key)
        normalized.append((main, summary))
    return normalized[:2]

def tail(text: str, n: int = 80) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:])

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "last_two_live_batches_review.txt"

    lines: list[str] = []
    lines.append("===== LAST TWO LIVE BATCHES REVIEW =====")
    lines.append("")
    lines.append(f"root: {ROOT}")
    lines.append("")

    if STATE_PATH.exists():
        lines.append("===== LAST RUN STATUS =====")
        lines.append(read_text(STATE_PATH).strip())
        lines.append("")

    batches = choose_batches()
    if not batches:
        lines.append("No encontré logs.")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"saved: {out_path}")
        print(out_path.read_text(encoding="utf-8"))
        return

    batch_summaries = []

    for idx, (main_log, summary_log) in enumerate(batches, start=1):
        main_txt = read_text(main_log)
        summary_txt = read_text(summary_log) if summary_log and summary_log.exists() else ""

        main_metrics = parse_main_summary(main_txt)
        portfolio = extract_json_block(main_txt, "=== PORTFOLIO SNAPSHOT ===")
        execution = extract_json_block(main_txt, "=== EXECUTION SNAPSHOT ===")
        bnb_block = extract_symbol_block(main_txt, "BNB/USDT:USDT")
        btc_block = extract_symbol_block(main_txt, "BTC/USDT:USDT")
        key_lines = extract_key_lines(main_txt)
        runtime_run = extract_runtime_run_block(summary_txt) if summary_txt else {}

        batch_summaries.append({
            "batch_no": idx,
            "main_log": str(main_log),
            "summary_log": str(summary_log) if summary_log else "",
            **main_metrics,
        })

        lines.append("=" * 120)
        lines.append(f"BATCH #{idx}")
        lines.append("=" * 120)
        lines.append(f"main_log: {main_log}")
        lines.append(f"summary_log: {summary_log if summary_log else 'N/A'}")
        lines.append("")

        if runtime_run:
            lines.append("----- SHORT SUMMARY LOG -----")
            lines.append(json.dumps(runtime_run, indent=2, ensure_ascii=False, sort_keys=True))
            lines.append("")

        lines.append("----- MAIN METRICS -----")
        lines.append(json.dumps(main_metrics, indent=2, ensure_ascii=False, sort_keys=True))
        lines.append("")

        lines.append("----- PORTFOLIO SNAPSHOT -----")
        lines.append(json.dumps(portfolio, indent=2, ensure_ascii=False, sort_keys=True) if portfolio is not None else "No encontrado.")
        lines.append("")

        lines.append("----- EXECUTION SNAPSHOT -----")
        lines.append(json.dumps(execution, indent=2, ensure_ascii=False, sort_keys=True) if execution is not None else "No encontrado.")
        lines.append("")

        lines.append("----- BNB BLOCK -----")
        lines.append(bnb_block)
        lines.append("")

        lines.append("----- BTC BLOCK -----")
        lines.append(btc_block)
        lines.append("")

        lines.append("----- KEY LINES -----")
        lines.extend(key_lines if key_lines else ["No encontradas."])
        lines.append("")

        lines.append("----- LAST 80 LINES -----")
        lines.append(tail(main_txt, 80))
        lines.append("")

    if len(batch_summaries) == 2:
        a = batch_summaries[0]
        b = batch_summaries[1]
        lines.append("=" * 120)
        lines.append("DELTA BATCH #1 vs BATCH #2")
        lines.append("=" * 120)

        for k in [
            "equity_final",
            "total_return_pct",
            "sharpe_annual",
            "max_drawdown_pct",
            "trade_count",
            "avg_active_symbols",
            "avg_gross_weight",
        ]:
            va = a.get(k)
            vb = b.get(k)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                delta = va - vb
                lines.append(f"{k}: latest={va} prev={vb} delta={delta}")
            else:
                lines.append(f"{k}: latest={va} prev={vb} delta=N/A")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {out_path}")
    print()
    print(out_path.read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
