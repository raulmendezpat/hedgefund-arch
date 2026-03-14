
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

REGISTRY = "artifacts/strategy_registry.json"
FULL_FEATURES = "results/ml_features_ml_features_btc_link_bnb_xrp_eth_v7.csv"

WINDOWS = [
    {
        "tag": "wf_eth_2024q1_test_2024q2",
        "train_start": "2024-01-01 00:00:00",
        "train_end":   "2024-03-31 23:00:00",
        "test_start":  "2024-04-01 00:00:00",
        "test_end":    "2024-06-30 23:00:00",
    },
    {
        "tag": "wf_eth_2024q2_test_2024q3",
        "train_start": "2024-01-01 00:00:00",
        "train_end":   "2024-06-30 23:00:00",
        "test_start":  "2024-07-01 00:00:00",
        "test_end":    "2024-09-30 23:00:00",
    },
    {
        "tag": "wf_eth_2024q3_test_2024q4",
        "train_start": "2024-01-01 00:00:00",
        "train_end":   "2024-09-30 23:00:00",
        "test_start":  "2024-10-01 00:00:00",
        "test_end":    "2024-12-31 23:00:00",
    },
    {
        "tag": "wf_eth_2024q4_test_2025q1",
        "train_start": "2024-01-01 00:00:00",
        "train_end":   "2024-12-31 23:00:00",
        "test_start":  "2025-01-01 00:00:00",
        "test_end":    "2025-03-31 23:00:00",
    },
]

def run(cmd: list[str], extra_env: dict[str, str] | None = None) -> None:
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    print("\\nRUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)

def compute_window_metrics(portfolio_name: str, test_start: str, test_end: str) -> dict:
    eq_p = ROOT / f"results/pipeline_equity_{portfolio_name}.csv"
    fills_p = ROOT / f"results/execution_fills_{portfolio_name}.csv"

    eq = pd.read_csv(eq_p)
    eq["ts"] = pd.to_datetime(eq["ts"], utc=True)
    start_ts = pd.Timestamp(test_start, tz="UTC")
    end_ts = pd.Timestamp(test_end, tz="UTC")
    eq = eq[(eq["ts"] >= start_ts) & (eq["ts"] <= end_ts)].copy()

    if eq.empty or len(eq) < 2:
        return {
            "total_return_pct": None,
            "cagr_pct": None,
            "sharpe_annual": None,
            "max_drawdown_pct": None,
            "vol_annual": None,
            "execution_fill_count_total": 0,
        }

    start_equity = float(eq["equity"].iloc[0])
    end_equity = float(eq["equity"].iloc[-1])
    total_return = (end_equity / start_equity) - 1.0 if start_equity > 0 else float("nan")

    hours = max(len(eq), 1)
    years = hours / 8760.0
    cagr = ((end_equity / start_equity) ** (1.0 / years) - 1.0) if start_equity > 0 and years > 0 else float("nan")

    port_ret = eq["port_ret"].astype(float).fillna(0.0)
    mean_ret = float(port_ret.mean())
    vol = float(port_ret.std(ddof=0))
    vol_annual = vol * math.sqrt(8760.0)
    sharpe = (mean_ret / vol) * math.sqrt(8760.0) if vol > 0 else float("nan")

    running_max = eq["equity"].cummax()
    dd_pct = (eq["equity"] / running_max - 1.0) * 100.0
    max_dd_pct = float(dd_pct.min())

    fill_count = 0
    if fills_p.exists():
        fills = pd.read_csv(fills_p, low_memory=False)
        if "ts_utc" in fills.columns:
            fills["ts_utc"] = pd.to_datetime(fills["ts_utc"], utc=True, errors="coerce")
            fills = fills[(fills["ts_utc"] >= start_ts) & (fills["ts_utc"] <= end_ts)]
            fill_count = int(len(fills))

    return {
        "total_return_pct": float(total_return * 100.0),
        "cagr_pct": float(cagr * 100.0),
        "sharpe_annual": float(sharpe),
        "max_drawdown_pct": float(max_dd_pct),
        "vol_annual": float(vol_annual),
        "execution_fill_count_total": int(fill_count),
    }

def main() -> None:
    full_csv = ROOT / FULL_FEATURES
    if not full_csv.exists():
        raise SystemExit(f"No existe FULL_FEATURES: {full_csv}")

    full_df = pd.read_csv(full_csv, low_memory=False)
    full_df["ts"] = pd.to_datetime(full_df["ts"], utc=True)

    summaries = []

    for w in WINDOWS:
        tag = w["tag"]
        train_start = pd.Timestamp(w["train_start"], tz="UTC")
        train_end = pd.Timestamp(w["train_end"], tz="UTC")

        train_df = full_df[(full_df["ts"] >= train_start) & (full_df["ts"] <= train_end)].copy()
        train_csv = ROOT / f"results/{tag}_train_features.csv"
        train_df.to_csv(train_csv, index=False)

        model_path = f"artifacts/{tag}_ensemble.joblib"
        model_metrics_path = f"artifacts/{tag}_ensemble_metrics.json"
        thresholds_path = f"artifacts/{tag}_thresholds_q40.json"
        portfolio_name = f"portfolio_{tag}"

        run([
            "python", "scripts/train_ml_strategy_ensemble.py",
            "--input", str(train_csv),
            "--output-model-registry", model_path,
            "--output-metrics", model_metrics_path,
        ])

        run([
            "python", "scripts/build_ml_thresholds.py",
            "--input", str(train_csv),
            "--output", thresholds_path,
            "--quantile", "0.40",
        ])

        run([
            "python", "scripts/hf_pipeline_alloc.py",
            "--name", portfolio_name,
            "--start", w["test_start"],
            "--signal-engine", "registry_portfolio",
            "--strategy-registry", REGISTRY,
            "--opportunity-selection-mode", "competitive",
            "--allocation-engine-mode", "multi_strategy",
            "--allocator-symbol-cap", "0.45",
            "--allocator-target-exposure", "0.05",
            "--ml-filter",
            "--ml-model-registry", model_path,
            "--ml-thresholds-path", thresholds_path,
        ], {
            "HF_ALLOCATOR_TOP_N_SYMBOLS": "0",
            "HF_ALLOCATOR_APPLY_CLUSTER_CAPS": "1",
        })

        metrics = compute_window_metrics(portfolio_name, w["test_start"], w["test_end"])
        summaries.append({"tag": tag, **metrics})

    print("\\n=== FINAL ROLLING SUMMARY ETH ===")
    for s in summaries:
        print(
            s["tag"],
            "return_pct=", s["total_return_pct"],
            "cagr_pct=", s["cagr_pct"],
            "sharpe=", s["sharpe_annual"],
            "maxdd=", s["max_drawdown_pct"],
            "fills=", s["execution_fill_count_total"],
        )

    out = ROOT / "artifacts/rolling_ensemble_btc_link_bnb_xrp_eth_summary.json"
    out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("\\nSaved summary ->", out)

if __name__ == "__main__":
    main()
