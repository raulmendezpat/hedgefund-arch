
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PORTFOLIO_NAME = "portfolio_ml_btc_link_bnb_xrp_eth_v7_q40"
EQUITY_CSV = ROOT / f"results/pipeline_equity_{PORTFOLIO_NAME}.csv"
OUT_CSV = ROOT / f"artifacts/montecarlo_{PORTFOLIO_NAME}.csv"
OUT_JSON = ROOT / f"artifacts/montecarlo_{PORTFOLIO_NAME}_summary.json"

N_SIMS = 1000
BLOCK_SIZE = 24  # 24 horas si barras horarias

def compute_stats(port_ret: np.ndarray):
    eq = np.cumprod(1.0 + port_ret)
    total_return = float(eq[-1] - 1.0)

    running_max = np.maximum.accumulate(eq)
    dd = eq / running_max - 1.0
    max_dd = float(dd.min())

    mean_ret = float(np.mean(port_ret))
    vol = float(np.std(port_ret, ddof=0))
    sharpe = (mean_ret / vol) * math.sqrt(8760.0) if vol > 0 else float("nan")

    return total_return, sharpe, max_dd

def bootstrap_blocks(x: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    out = []
    while len(out) < n:
        start = int(rng.integers(0, max(1, n - block_size + 1)))
        out.extend(x[start:start + block_size].tolist())
    return np.array(out[:n], dtype=float)

def main():
    df = pd.read_csv(EQUITY_CSV)
    if "port_ret" not in df.columns:
        raise SystemExit("No encontré columna port_ret en pipeline_equity")

    port_ret = df["port_ret"].astype(float).fillna(0.0).to_numpy()
    if len(port_ret) < 100:
        raise SystemExit("Muy pocos retornos para Monte Carlo")

    rng = np.random.default_rng(42)
    rows = []

    base_total, base_sharpe, base_maxdd = compute_stats(port_ret)

    for i in range(N_SIMS):
        sim_ret = bootstrap_blocks(port_ret, BLOCK_SIZE, rng)
        total_return, sharpe, max_dd = compute_stats(sim_ret)
        rows.append({
            "sim_id": i,
            "total_return_pct": total_return * 100.0,
            "sharpe_annual": sharpe,
            "max_drawdown_pct": max_dd * 100.0,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    summary = {
        "portfolio_name": PORTFOLIO_NAME,
        "n_sims": N_SIMS,
        "block_size": BLOCK_SIZE,
        "base": {
            "total_return_pct": base_total * 100.0,
            "sharpe_annual": base_sharpe,
            "max_drawdown_pct": base_maxdd * 100.0,
        },
        "percentiles": {
            "return_pct_p05": float(out["total_return_pct"].quantile(0.05)),
            "return_pct_p25": float(out["total_return_pct"].quantile(0.25)),
            "return_pct_p50": float(out["total_return_pct"].quantile(0.50)),
            "return_pct_p75": float(out["total_return_pct"].quantile(0.75)),
            "return_pct_p95": float(out["total_return_pct"].quantile(0.95)),

            "sharpe_p05": float(out["sharpe_annual"].quantile(0.05)),
            "sharpe_p25": float(out["sharpe_annual"].quantile(0.25)),
            "sharpe_p50": float(out["sharpe_annual"].quantile(0.50)),
            "sharpe_p75": float(out["sharpe_annual"].quantile(0.75)),
            "sharpe_p95": float(out["sharpe_annual"].quantile(0.95)),

            "maxdd_p05": float(out["max_drawdown_pct"].quantile(0.05)),
            "maxdd_p25": float(out["max_drawdown_pct"].quantile(0.25)),
            "maxdd_p50": float(out["max_drawdown_pct"].quantile(0.50)),
            "maxdd_p75": float(out["max_drawdown_pct"].quantile(0.75)),
            "maxdd_p95": float(out["max_drawdown_pct"].quantile(0.95)),
        }
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\\n=== MONTE CARLO SUMMARY ===\\n")
    print(json.dumps(summary, indent=2))
    print("\\nSaved ->", OUT_CSV)
    print("Saved ->", OUT_JSON)

if __name__ == "__main__":
    main()
