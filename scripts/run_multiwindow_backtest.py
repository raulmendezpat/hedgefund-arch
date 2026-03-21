from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

from hf.data.ohlcv import fetch_ohlcv_ccxt, dt_to_ms_utc


WINDOWS = [1, 3, 6, 12, 24]


def load_registry(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Strategy registry must be a list")
    return data


def first_symbol_from_registry(path: str) -> str:
    rows = load_registry(path)
    for r in rows:
        s = str(r.get("symbol", "") or "")
        if s:
            return s
    raise ValueError("No symbol found in registry")


def detect_last_ts(symbol: str, exchange: str, cache_dir: str) -> pd.Timestamp:
    # fetch_ohlcv_ccxt no tolera start_ms=None; usamos un inicio muy temprano
    df = fetch_ohlcv_ccxt(
        symbol=symbol,
        timeframe="1h",
        start_ms=dt_to_ms_utc("2017-01-01 00:00:00"),
        end_ms=None,
        exchange_id=exchange,
        cache_dir=cache_dir,
        use_cache=True,
        refresh_if_no_end=True,
    ).copy()

    ts = pd.to_datetime(
        pd.to_numeric(df["timestamp"], errors="coerce"),
        unit="ms",
        utc=True,
        errors="coerce",
    ).dropna()

    if ts.empty:
        raise ValueError(f"No timestamps found for symbol={symbol}")

    return ts.max()


def monthly_returns_from_runtime_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").copy()

    if "equity" not in df.columns:
        raise ValueError(f"Missing equity column in {path}")

    m = df.set_index("ts")["equity"].resample("ME").last().dropna().to_frame("equity")
    m["monthly_return"] = m["equity"].pct_change().fillna(0.0)
    m = m.reset_index()
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-name", default="contextual_policy_v1")
    ap.add_argument("--strategy-registry", default="artifacts/strategy_registry_v2.json")
    ap.add_argument("--policy-config", default="artifacts/policy_config.json")
    ap.add_argument("--policy-profile", default="default")
    ap.add_argument("--exchange", default="binanceusdm")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--target-exposure", type=float, default=0.07)
    ap.add_argument("--symbol-cap", type=float, default=0.50)
    args = ap.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    ref_symbol = first_symbol_from_registry(args.strategy_registry)
    last_ts = detect_last_ts(ref_symbol, args.exchange, args.cache_dir)

    summary_rows = []
    monthly_frames = []

    print(f"reference_symbol: {ref_symbol}")
    print(f"last_ts_utc: {last_ts}")

    for months in WINDOWS:
        start_ts = (last_ts - pd.DateOffset(months=months)).floor("h")
        run_name = f"{args.base_name}_{months}m"
        start_str = start_ts.strftime("%Y-%m-%d %H:%M:%S")

        cmd = [
            "python",
            "scripts/research_runtime.py",
            "--name", run_name,
            "--start", start_str,
            "--strategy-registry", args.strategy_registry,
            "--policy-config", args.policy_config,
            "--policy-profile", args.policy_profile,
            "--exchange", args.exchange,
            "--cache-dir", args.cache_dir,
            "--target-exposure", str(args.target_exposure),
            "--symbol-cap", str(args.symbol_cap),
        ]

        print(f"\n=== RUN {months}M ===")
        print("start:", start_str)
        print("cmd:", " ".join(cmd))

        t0 = time.perf_counter()
        subprocess.run(cmd, check=True)
        wall_seconds = time.perf_counter() - t0

        metrics_path = results_dir / f"research_runtime_metrics_{run_name}.json"
        runtime_csv = results_dir / f"research_runtime_{run_name}.csv"

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        monthly = monthly_returns_from_runtime_csv(runtime_csv)
        monthly["window_months"] = months
        monthly["run_name"] = run_name

        monthly_path = results_dir / f"monthly_returns_{run_name}.csv"
        monthly.to_csv(monthly_path, index=False)
        monthly_frames.append(monthly)

        summary_rows.append({
            "window_months": months,
            "run_name": run_name,
            "start_utc": start_str,
            "end_utc": str(last_ts),
            "wall_seconds": float(wall_seconds),
            "reported_total_seconds": float(metrics.get("total_seconds", 0.0) or 0.0),
            "load_seconds": float(metrics.get("load_seconds", 0.0) or 0.0),
            "run_seconds": float(metrics.get("run_seconds", 0.0) or 0.0),
            "rows": int(metrics.get("rows", 0) or 0),
            "total_return_pct": float(metrics.get("total_return_pct", 0.0) or 0.0),
            "sharpe_annual": float(metrics.get("sharpe_annual", 0.0) or 0.0),
            "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0) or 0.0),
            "win_rate_pct": float(metrics.get("win_rate_pct", 0.0) or 0.0),
            "vol_annual": float(metrics.get("vol_annual", 0.0) or 0.0),
            "equity_final": float(metrics.get("equity_final", 0.0) or 0.0),
            "avg_n_opps": float(metrics.get("avg_n_opps", 0.0) or 0.0),
            "avg_n_accepts": float(metrics.get("avg_n_accepts", 0.0) or 0.0),
            "avg_active_symbols": float(metrics.get("avg_active_symbols", 0.0) or 0.0),
            "avg_gross_weight": float(metrics.get("avg_gross_weight", 0.0) or 0.0),
            "monthly_returns_file": str(monthly_path),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("window_months")
    summary_path = results_dir / f"multiwindow_summary_{args.base_name}.csv"
    summary_df.to_csv(summary_path, index=False)

    monthly_all = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    monthly_all_path = results_dir / f"multiwindow_monthly_returns_{args.base_name}.csv"
    monthly_all.to_csv(monthly_all_path, index=False)

    txt_path = results_dir / f"multiwindow_report_{args.base_name}.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("=== MULTIWINDOW SUMMARY ===\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n=== MONTHLY RETURNS ===\n")
        if not monthly_all.empty:
            f.write(monthly_all.to_string(index=False))
        else:
            f.write("no monthly data")

    print(f"\nsaved: {summary_path}")
    print(f"saved: {monthly_all_path}")
    print(f"saved: {txt_path}")

    print("\n=== SUMMARY ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
