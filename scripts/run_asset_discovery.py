
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
RES = ROOT / "results"

START = "2024-01-01 00:00:00"

# Candidatos para discovery.
# Mantengo símbolos líquidos y con alta probabilidad de existir en perp USDT.
CANDIDATES = [
    "AVAX/USDT:USDT",
    "ADA/USDT:USDT",
    "DOGE/USDT:USDT",
    "LTC/USDT:USDT",
    "DOT/USDT:USDT",
    "BCH/USDT:USDT",
    "TRX/USDT:USDT",
    "AAVE/USDT:USDT",
]

def slug_symbol(symbol: str) -> str:
    return (
        symbol.replace("/USDT:USDT", "")
        .replace("/USDT", "")
        .replace(":USDT", "")
        .replace("/", "_")
        .lower()
    )

def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    return p.returncode, out

def write_registry(symbol: str) -> Path:
    slug = slug_symbol(symbol)
    out = ART / f"registry_discovery_{slug}.json"
    row = [{
        "strategy_id": f"{slug}_trend",
        "symbol": symbol,
        "engine": "btc_trend_signal",
        "enabled": True,
        "base_weight": 1.0,
        "cluster_id": "discovery",
        "cluster_cap": 0.25,
        "params": {}
    }]
    out.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return out

def read_metrics(name: str) -> dict:
    p = RES / f"pipeline_metrics_{name}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return d

def read_alloc_stats(name: str) -> tuple[float, float]:
    p = RES / f"pipeline_allocations_{name}.csv"
    df = pd.read_csv(p, low_memory=False)
    wcols = [c for c in df.columns if c.startswith("w_")]
    if not wcols:
        return 0.0, 0.0
    exp = sum(df[c].abs() for c in wcols)
    return float(exp.mean()), float((exp > 0.01).mean())

def main():
    rows = []

    for symbol in CANDIDATES:
        slug = slug_symbol(symbol)
        reg = write_registry(symbol)
        name = f"discovery_{slug}"

        code, out = run([
            "python", "scripts/hf_pipeline_alloc.py",
            "--name", name,
            "--start", START,
            "--signal-engine", "registry_portfolio",
            "--strategy-registry", str(reg),
            "--opportunity-selection-mode", "competitive",
            "--allocation-engine-mode", "multi_strategy",
        ])

        if code != 0:
            rows.append({
                "symbol": symbol,
                "status": "fail",
                "return_pct": None,
                "sharpe": None,
                "maxdd": None,
                "fills": None,
                "avg_exposure": None,
                "time_in_market": None,
                "error_tail": out[-500:],
            })
            continue

        m = read_metrics(name)
        avg_exp, tim = read_alloc_stats(name)

        rows.append({
            "symbol": symbol,
            "status": "ok",
            "return_pct": m.get("total_return_pct"),
            "sharpe": m.get("sharpe_annual"),
            "maxdd": m.get("max_drawdown_pct"),
            "fills": m.get("execution_fill_count_total"),
            "avg_exposure": avg_exp,
            "time_in_market": tim,
            "error_tail": "",
        })

    df = pd.DataFrame(rows)

    def score_row(r):
        if r["status"] != "ok":
            return -1e9
        sharpe = float(r["sharpe"] or 0.0)
        ret = float(r["return_pct"] or 0.0)
        dd = abs(float(r["maxdd"] or 0.0))
        return sharpe * 10.0 + ret * 0.05 - dd * 0.10

    df["discovery_score"] = df.apply(score_row, axis=1)
    df = df.sort_values(["status", "discovery_score"], ascending=[True, False])

    out_csv = ART / "asset_discovery_results.csv"
    df.to_csv(out_csv, index=False)

    print("\\n=== ASSET DISCOVERY RESULTS ===\\n")
    print(df[[
        "symbol", "status", "return_pct", "sharpe", "maxdd",
        "fills", "avg_exposure", "time_in_market", "discovery_score"
    ]].to_string(index=False))

    print("\\nSaved ->", out_csv)

if __name__ == "__main__":
    main()
