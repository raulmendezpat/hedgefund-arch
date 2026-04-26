#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path("/home/ubuntu/hedgefund-arch-v2")
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"

TARGET_FILES = [
    ARTIFACTS / "selection_policy_config.calibrated.json",
    ARTIFACTS / "policy_config.json",
    ARTIFACTS / "rt_registry_add_trx_specialized.json",
]

TARGET_KEYS = [
    "BTC/USDT:USDT::short",
    "BTC/USDT:USDT::short::btc_trend",
    "BTC/USDT:USDT::short::btc_trend_loose",
    "btc_trend",
    "btc_trend_loose",
    "BTC/USDT:USDT",
]

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def walk(obj: Any, path: str = ""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            out.append((p, v))
            out.extend(walk(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{path}[{i}]"
            out.append((p, v))
            out.extend(walk(v, p))
    return out

def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "btc_short_config_blocks.txt"

    lines: list[str] = []
    lines.append("===== BTC SHORT CONFIG BLOCKS =====")
    lines.append("")

    for path in TARGET_FILES:
        lines.append("=" * 120)
        lines.append(f"FILE: {path}")
        if not path.exists():
            lines.append("NO EXISTE")
            lines.append("")
            continue

        try:
            obj = load_json(path)
        except Exception as e:
            lines.append(f"ERROR CARGANDO JSON: {type(e).__name__}: {e}")
            lines.append("")
            continue

        found_any = False

        if isinstance(obj, dict):
            for key in TARGET_KEYS:
                if key in obj:
                    found_any = True
                    lines.append("-" * 120)
                    lines.append(f"TOP LEVEL KEY: {key}")
                    lines.append(json.dumps(obj[key], indent=2, ensure_ascii=False, sort_keys=True))
                    lines.append("")

        walked = walk(obj)
        for p, v in walked:
            p_l = p.lower()
            if (
                "btc/usdt:usdt::short" in p_l
                or "btc_trend_loose" in p_l
                or "btc_trend" in p_l
            ):
                found_any = True
                lines.append("-" * 120)
                lines.append(f"PATH: {p}")
                try:
                    lines.append(json.dumps(v, indent=2, ensure_ascii=False, sort_keys=True))
                except Exception:
                    lines.append(repr(v))
                lines.append("")

        if not found_any:
            lines.append("NO SE ENCONTRARON BLOQUES BTC SHORT RELEVANTES")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {out_path}")
    print()
    print(out_path.read_text(encoding='utf-8'))

if __name__ == "__main__":
    main()
