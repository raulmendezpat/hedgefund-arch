
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

REGISTRY = "artifacts/strategy_registry.json"
MODEL = "artifacts/ml_strategy_ensemble_registry_v4.joblib"

FEATURE_DATASET = "results/ml_features_ml_dataset_btc_link_v4.csv"

THRESHOLDS = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]

def run(cmd):
    print("\nRUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)

def compute_metrics(name):

    m = ROOT / f"results/pipeline_metrics_{name}.json"
    a = ROOT / f"results/pipeline_allocations_{name}.csv"

    if not m.exists():
        return None

    d = json.loads(m.read_text())

    df = pd.read_csv(a, low_memory=False)

    exp = (
        df.get("w_btc_usdt_usdt",0).abs() +
        df.get("w_link_usdt_usdt",0).abs() +
        df.get("w_bnb_usdt_usdt",0).abs() +
        df.get("w_sol_usdt_usdt",0).abs()
    )

    return {
        "return_pct": d.get("total_return_pct"),
        "cagr_pct": d.get("cagr_pct"),
        "sharpe": d.get("sharpe_annual"),
        "maxdd": d.get("max_drawdown_pct"),
        "fills": d.get("execution_fill_count_total"),
        "avg_exposure": float(exp.mean()),
        "time_in_market": float((exp > 0.01).mean()),
    }

def main():

    results = []

    for q in THRESHOLDS:

        thr_file = f"artifacts/tmp_threshold_{int(q*100)}.json"

        run([
            "python", "scripts/build_ml_thresholds.py",
            "--input", FEATURE_DATASET,
            "--output", thr_file,
            "--quantile", str(q)
        ])

        name = f"portfolio_threshold_{int(q*100)}"

        run([
            "python", "scripts/hf_pipeline_alloc.py",
            "--name", name,
            "--start", "2024-01-01 00:00:00",
            "--signal-engine", "registry_portfolio",
            "--strategy-registry", REGISTRY,
            "--opportunity-selection-mode", "competitive",
            "--allocation-engine-mode", "multi_strategy",
            "--allocator-symbol-cap", "0.45",
            "--allocator-target-exposure", "0.05",
            "--ml-filter",
            "--ml-model-registry", MODEL,
            "--ml-thresholds-path", thr_file
        ])

        metrics = compute_metrics(name)

        if metrics:
            metrics["threshold"] = q
            results.append(metrics)

    df = pd.DataFrame(results).sort_values("threshold")

    print("\n=== THRESHOLD SWEEP RESULTS ===\n")
    print(df)

    out = ROOT / "artifacts/threshold_sweep_results.csv"
    df.to_csv(out, index=False)

    print("\nSaved ->", out)

if __name__ == "__main__":
    main()
