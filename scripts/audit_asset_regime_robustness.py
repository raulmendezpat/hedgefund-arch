from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hf.data.ohlcv import fetch_ohlcv_ccxt, dt_to_ms_utc


WINDOWS = [1, 3, 6, 12]
RET_COLS = ["ret_1h", "ret_4h", "ret_12h", "ret_24h", "ret_48h"]


def load_registry(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Strategy registry must be a list")
    return data


def unique_symbols_from_registry(path: str) -> list[str]:
    rows = load_registry(path)
    out = []
    seen = set()
    for r in rows:
        s = str(r.get("symbol", "") or "")
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def load_symbol_coverage(symbols: list[str], exchange: str, cache_dir: str) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        df = fetch_ohlcv_ccxt(
            symbol=sym,
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
            rows.append({
                "symbol": sym,
                "rows": 0,
                "min_ts": pd.NaT,
                "max_ts": pd.NaT,
            })
            continue

        rows.append({
            "symbol": sym,
            "rows": int(len(ts)),
            "min_ts": ts.min(),
            "max_ts": ts.max(),
        })

    out = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    return out


def pick_return_col(df: pd.DataFrame) -> str:
    for c in ["ret_48h", "ret_24h", "ret_12h", "ret_4h", "ret_1h"]:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    raise ValueError("No return columns found")


def summarize_group(df: pd.DataFrame, group_cols: list[str], ret_col: str) -> pd.DataFrame:
    g = (
        df.groupby(group_cols, dropna=False)
        .agg(
            rows=("strategy_id", "size"),
            accept_rate=("accept", "mean"),
            avg_ret=(ret_col, "mean"),
            med_ret=(ret_col, "median"),
            hit_rate=(ret_col, lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            avg_pwin=("p_win", "mean"),
            avg_score=("score", "mean"),
            avg_size=("size_mult", "mean"),
        )
        .reset_index()
        .sort_values("avg_ret", ascending=False)
    )
    return g


def summarize_windows(
    attr_dir: Path,
    base_prefix: str,
    windows: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    strategy_rows = []
    strategy_side_rows = []

    for months in windows:
        candidates = [
            attr_dir / f"performance_attribution_{base_prefix}_{months}m.csv",
            attr_dir / f"performance_attribution_diag_{months}m_{base_prefix}.csv",
            attr_dir / f"performance_attribution_{base_prefix}.csv" if months == 6 else None,
        ]
        candidates = [x for x in candidates if x is not None]

        fp = next((x for x in candidates if x.exists()), None)
        if fp is None:
            print(f"SKIP window={months} no attribution file found for base_prefix={base_prefix}")
            continue

        df = pd.read_csv(fp)
        ret_col = pick_return_col(df)

        accepted = df[df["accept"] == True].copy()
        accepted["window_months"] = months

        summary_rows.append({
            "window_months": months,
            "source_file": str(fp),
            "rows_all": int(len(df)),
            "rows_accepted": int(len(accepted)),
            "accept_rate": float(df["accept"].mean()),
            "ret_col": ret_col,
            "avg_ret_accepted": float(pd.to_numeric(accepted[ret_col], errors="coerce").mean()) if len(accepted) else 0.0,
            "hit_rate_accepted": float((pd.to_numeric(accepted[ret_col], errors="coerce") > 0).mean()) if len(accepted) else 0.0,
            "avg_pwin_accepted": float(pd.to_numeric(accepted["p_win"], errors="coerce").mean()) if len(accepted) else 0.0,
            "avg_score_accepted": float(pd.to_numeric(accepted["score"], errors="coerce").mean()) if len(accepted) else 0.0,
            "avg_size_accepted": float(pd.to_numeric(accepted["size_mult"], errors="coerce").mean()) if len(accepted) else 0.0,
        })

        if len(accepted):
            by_strategy = summarize_group(accepted, ["strategy_id"], ret_col)
            by_strategy["window_months"] = months
            strategy_rows.append(by_strategy)

            by_strategy_side = summarize_group(accepted, ["strategy_id", "side"], ret_col)
            by_strategy_side["window_months"] = months
            strategy_side_rows.append(by_strategy_side)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty and "window_months" in summary_df.columns:
        summary_df = summary_df.sort_values("window_months").reset_index(drop=True)

    strategy_df = pd.concat(strategy_rows, ignore_index=True) if strategy_rows else pd.DataFrame()
    strategy_side_df = pd.concat(strategy_side_rows, ignore_index=True) if strategy_side_rows else pd.DataFrame()

    return summary_df, strategy_df, strategy_side_df


def robust_leaderboard(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    for keys, g in df.groupby(key_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(key_cols, keys)}
        row["windows_present"] = int(g["window_months"].nunique())
        row["avg_ret_mean"] = float(g["avg_ret"].mean())
        row["avg_ret_min"] = float(g["avg_ret"].min())
        row["avg_ret_max"] = float(g["avg_ret"].max())
        row["hit_rate_mean"] = float(g["hit_rate"].mean())
        row["accept_rate_mean"] = float(g["accept_rate"].mean())
        row["rows_mean"] = float(g["rows"].mean())
        row["positive_windows"] = int((g["avg_ret"] > 0).sum())
        row["negative_windows"] = int((g["avg_ret"] <= 0).sum())
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["positive_windows", "avg_ret_mean", "avg_ret_min", "hit_rate_mean"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-prefix", required=True, help="Ej: meta_patch_1")
    ap.add_argument("--strategy-registry", default="artifacts/strategy_registry_v2.json")
    ap.add_argument("--exchange", default="binanceusdm")
    ap.add_argument("--cache-dir", default="data/cache")
    args = ap.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    symbols = unique_symbols_from_registry(args.strategy_registry)
    coverage = load_symbol_coverage(symbols, args.exchange, args.cache_dir)

    common_start = coverage["min_ts"].max() if not coverage.empty else pd.NaT
    common_end = coverage["max_ts"].min() if not coverage.empty else pd.NaT

    summary_df, by_strategy_df, by_strategy_side_df = summarize_windows(
        results_dir,
        args.base_prefix,
        WINDOWS,
    )

    robust_strategy = robust_leaderboard(by_strategy_df, ["strategy_id"])
    robust_strategy_side = robust_leaderboard(by_strategy_side_df, ["strategy_id", "side"])

    coverage_path = results_dir / f"asset_coverage_{args.base_prefix}.csv"
    summary_path = results_dir / f"robustness_summary_{args.base_prefix}.csv"
    strategy_path = results_dir / f"robustness_by_strategy_{args.base_prefix}.csv"
    strategy_side_path = results_dir / f"robustness_by_strategy_side_{args.base_prefix}.csv"
    leaderboard_strategy_path = results_dir / f"robustness_leaderboard_strategy_{args.base_prefix}.csv"
    leaderboard_strategy_side_path = results_dir / f"robustness_leaderboard_strategy_side_{args.base_prefix}.csv"
    report_path = results_dir / f"robustness_report_{args.base_prefix}.txt"

    coverage.to_csv(coverage_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    by_strategy_df.to_csv(strategy_path, index=False)
    by_strategy_side_df.to_csv(strategy_side_path, index=False)
    robust_strategy.to_csv(leaderboard_strategy_path, index=False)
    robust_strategy_side.to_csv(leaderboard_strategy_side_path, index=False)

    with report_path.open("w", encoding="utf-8") as f:
        f.write("=== COVERAGE ===\n")
        if coverage.empty:
            f.write("no coverage\n")
        else:
            f.write(coverage.to_string(index=False))
            f.write("\n")
            f.write(f"\ncommon_start: {common_start}\n")
            f.write(f"common_end  : {common_end}\n")

        f.write("\n\n=== WINDOW SUMMARY ===\n")
        if summary_df.empty:
            f.write("no summary\n")
        else:
            f.write(summary_df.to_string(index=False))

        f.write("\n\n=== ROBUST STRATEGY LEADERBOARD ===\n")
        if robust_strategy.empty:
            f.write("no strategy leaderboard\n")
        else:
            f.write(robust_strategy.head(50).to_string(index=False))

        f.write("\n\n=== ROBUST STRATEGY+SIDE LEADERBOARD ===\n")
        if robust_strategy_side.empty:
            f.write("no strategy+side leaderboard\n")
        else:
            f.write(robust_strategy_side.head(80).to_string(index=False))

    print(f"saved: {coverage_path}")
    print(f"saved: {summary_path}")
    print(f"saved: {strategy_path}")
    print(f"saved: {strategy_side_path}")
    print(f"saved: {leaderboard_strategy_path}")
    print(f"saved: {leaderboard_strategy_side_path}")
    print(f"saved: {report_path}")

    print("\n=== COVERAGE ===")
    print(coverage.to_string(index=False))
    print("\ncommon_start:", common_start)
    print("common_end  :", common_end)

    if not summary_df.empty:
        print("\n=== WINDOW SUMMARY ===")
        print(summary_df.to_string(index=False))

    if not robust_strategy.empty:
        print("\n=== ROBUST STRATEGY LEADERBOARD ===")
        print(robust_strategy.head(20).to_string(index=False))

    if not robust_strategy_side.empty:
        print("\n=== ROBUST STRATEGY+SIDE LEADERADERBOARD ===")
        print(robust_strategy_side.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
