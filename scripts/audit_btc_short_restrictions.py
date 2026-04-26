#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path("/home/ubuntu/hedgefund-arch-v2")
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, bool):
            return float(v)
        s = str(v).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def safe_text(v: Any) -> str:
    return "" if v is None else str(v)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def iter_json_files() -> list[Path]:
    out: list[Path] = []
    for base in [ARTIFACTS, RESULTS]:
        if not base.exists():
            continue
        out.extend(sorted(base.rglob("*.json")))
    return out


def grep_files(patterns: list[str], roots: list[Path]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    regexes = [re.compile(p, re.IGNORECASE) for p in patterns]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".py", ".json", ".txt", ".yaml", ".yml", ".ini", ".cfg", ".sh"}:
                continue
            txt = read_text(path)
            if not txt:
                continue
            lines = txt.splitlines()
            for i, line in enumerate(lines, start=1):
                for rx in regexes:
                    if rx.search(line):
                        hits.append(
                            {
                                "file": str(path.relative_to(ROOT)),
                                "line_no": i,
                                "pattern": rx.pattern,
                                "line": line.strip(),
                            }
                        )
                        break
    return hits


def audit_registry() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(ARTIFACTS.glob("*.json")):
        obj = load_json_if_exists(p)
        if obj is None:
            continue

        if isinstance(obj, list):
            for idx, r in enumerate(obj):
                if not isinstance(r, dict):
                    continue
                symbol = safe_text(r.get("symbol"))
                strategy_id = safe_text(r.get("strategy_id") or r.get("id") or r.get("name"))
                side = safe_text(r.get("side"))
                base_weight = safe_float(r.get("base_weight"), default=float("nan"))
                enabled = r.get("enabled")
                if symbol == "BTC/USDT:USDT" or "btc" in strategy_id.lower():
                    rows.append(
                        {
                            "source": str(p.relative_to(ROOT)),
                            "entry_type": "registry_row",
                            "index": idx,
                            "symbol": symbol,
                            "strategy_id": strategy_id,
                            "side": side,
                            "base_weight": base_weight,
                            "enabled": enabled,
                            "note": "",
                        }
                    )

        elif isinstance(obj, dict):
            txt = json.dumps(obj, ensure_ascii=False)
            if "BTC/USDT:USDT" in txt or re.search(r'"btc[^"]*"', txt, re.IGNORECASE):
                rows.append(
                    {
                        "source": str(p.relative_to(ROOT)),
                        "entry_type": "json_object_match",
                        "index": "",
                        "symbol": "",
                        "strategy_id": "",
                        "side": "",
                        "base_weight": "",
                        "enabled": "",
                        "note": "JSON contiene referencias a BTC o BTC/USDT:USDT; revisar manualmente",
                    }
                )
    return rows


def audit_candidate_outputs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_files = sorted(RESULTS.glob("research_runtime_candidates_*.csv"))
    for p in candidate_files[-10:]:
        try:
            with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.DictReader(f)
                count_total = 0
                count_btc_short = 0
                selected_btc_short = 0
                for r in reader:
                    count_total += 1
                    symbol = safe_text(r.get("symbol"))
                    side = safe_text(r.get("side"))
                    if symbol == "BTC/USDT:USDT" and side == "short":
                        count_btc_short += 1
                        sel = safe_text(r.get("selected_final") or r.get("selected") or r.get("is_selected"))
                        if sel.lower() in {"1", "true", "yes"}:
                            selected_btc_short += 1
                rows.append(
                    {
                        "source": str(p.relative_to(ROOT)),
                        "entry_type": "candidate_csv",
                        "index": "",
                        "symbol": "BTC/USDT:USDT",
                        "strategy_id": "",
                        "side": "short",
                        "base_weight": "",
                        "enabled": "",
                        "note": f"btc_short_rows={count_btc_short}; btc_short_selected={selected_btc_short}; total_rows={count_total}",
                    }
                )
        except Exception as e:
            rows.append(
                {
                    "source": str(p.relative_to(ROOT)),
                    "entry_type": "candidate_csv_error",
                    "index": "",
                    "symbol": "BTC/USDT:USDT",
                    "strategy_id": "",
                    "side": "short",
                    "base_weight": "",
                    "enabled": "",
                    "note": f"error={type(e).__name__}: {e}",
                }
            )
    return rows


def summarize_restrictions(registry_rows: list[dict[str, Any]], grep_hits: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []

    btc_rows = [
        r for r in registry_rows
        if r.get("entry_type") == "registry_row"
        and (r.get("symbol") == "BTC/USDT:USDT" or "btc" in safe_text(r.get("strategy_id")).lower())
    ]

    short_rows = [r for r in btc_rows if safe_text(r.get("side")).lower() == "short"]

    if not short_rows:
        notes.append("No encontré filas explícitas BTC short en artifacts/*.json tipo registry list.")
    else:
        zero_weight = [r for r in short_rows if str(r.get("base_weight")) != "" and safe_float(r.get("base_weight"), 999.0) == 0.0]
        disabled = [r for r in short_rows if safe_text(r.get("enabled")).lower() in {"false", "0", "no"}]
        if zero_weight:
            notes.append(f"Encontré {len(zero_weight)} filas BTC short con base_weight=0.")
        if disabled:
            notes.append(f"Encontré {len(disabled)} filas BTC short con enabled=false.")
        if not zero_weight and not disabled:
            notes.append("No vi BTC short explícitamente deshabilitado por base_weight=0 ni enabled=false en los registry JSON tipo lista.")

    suspicious = []
    for h in grep_hits:
        line = safe_text(h.get("line")).lower()
        if "btc" in line and (
            "disable" in line
            or "block" in line
            or "restrict" in line
            or "guard" in line
            or "weight" in line
            or "cap" in line
            or "short" in line
        ):
            suspicious.append(h)

    notes.append(f"Hallazgos de texto potencialmente relevantes: {len(suspicious)}")
    return notes


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"source": "", "entry_type": "", "index": "", "symbol": "", "strategy_id": "", "side": "", "base_weight": "", "enabled": "", "note": ""}]
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    registry_rows = audit_registry()
    candidate_rows = audit_candidate_outputs()
    grep_hits = grep_files(
        patterns=[
            r"btc.*short",
            r"short.*btc",
            r"btc_short_guard",
            r"disable",
            r"restrict",
            r"block",
            r"symbol-cap",
            r"execution-symbol-cap",
            r"target_weight",
            r"base_weight",
        ],
        roots=[ARTIFACTS, SRC, SCRIPTS],
    )

    registry_csv = RESULTS / "btc_short_audit_registry_rows.csv"
    grep_csv = RESULTS / "btc_short_audit_grep_hits.csv"
    candidate_csv = RESULTS / "btc_short_audit_candidate_rows.csv"
    report_txt = RESULTS / "btc_short_audit_report.txt"

    write_csv(registry_csv, registry_rows)
    write_csv(candidate_csv, candidate_rows)
    write_csv(grep_csv, grep_hits if grep_hits else [{"file": "", "line_no": "", "pattern": "", "line": ""}])

    notes = summarize_restrictions(registry_rows, grep_hits)

    lines: list[str] = []
    lines.append("===== BTC SHORT AUDIT REPORT =====")
    lines.append("")
    lines.append(f"root: {ROOT}")
    lines.append("")
    lines.append("===== SUMMARY =====")
    for n in notes:
        lines.append(f"- {n}")
    lines.append("")
    lines.append("===== OUTPUTS =====")
    lines.append(str(registry_csv))
    lines.append(str(candidate_csv))
    lines.append(str(grep_csv))
    lines.append("")
    lines.append("===== REGISTRY ROWS (TOP 40) =====")
    for r in registry_rows[:40]:
        lines.append(json.dumps(r, ensure_ascii=False))
    lines.append("")
    lines.append("===== CANDIDATE FILE SUMMARY =====")
    for r in candidate_rows[:40]:
        lines.append(json.dumps(r, ensure_ascii=False))
    lines.append("")
    lines.append("===== GREP HITS (TOP 80) =====")
    for h in grep_hits[:80]:
        lines.append(json.dumps(h, ensure_ascii=False))

    report_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"saved: {registry_csv}")
    print(f"saved: {candidate_csv}")
    print(f"saved: {grep_csv}")
    print(f"saved: {report_txt}")
    print()
    print(report_txt.read_text(encoding='utf-8', errors='ignore'))


if __name__ == "__main__":
    main()
