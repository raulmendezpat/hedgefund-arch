from __future__ import annotations

from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")
OUT_DATASET = RESULTS_DIR / "pwin_asset_side_dataset_v2_clean.csv"
OUT_SUMMARY = RESULTS_DIR / "pwin_asset_side_dataset_build_summary_v2_clean.csv"

TRACE_COL = "trace_candidate_id"
TARGET_COL = "label_win"

CANDIDATE_PREFIX = "research_runtime_candidates_"
TRADES_PREFIX = "research_runtime_lifecycle_trades_"

# Ex-ante only: nada que venga de selección final, scores post-ML, ni resultados del trade.
NUMERIC_FEATURES = [
    "signal_strength",
    "base_weight",
    "adx",
    "atrp",
    "rsi",
    "ret_1h_lag",
    "ret_4h_lag",
    "ret_12h_lag",
    "ret_24h_lag",
    "ema_gap_fast_slow",
    "dist_close_ema_fast",
    "dist_close_ema_slow",
    "range_pct",
    "rolling_vol_24h",
    "rolling_vol_72h",
    "atrp_zscore",
    "breakout_distance_up",
    "breakout_distance_down",
    "pullback_depth",
    "btc_ret_24h_lag",
    "btc_rolling_vol_24h",
    "btc_atrp",
    "btc_adx",
]

CATEGORICAL_FEATURES = [
    "symbol",
    "side",
    "strategy_id",
]

PASS_COLS = [
    "ts",
    TRACE_COL,
]

TRADE_KEEP_COLS = [
    TRACE_COL,
    "symbol",
    "strategy_id",
    "side",
    "entry_ts",
    "exit_ts",
    "pnl",
    "exit_reason",
]

EXCLUDE_RUN_PATTERNS = [
    "btcguard_audit",
    "btcguard_fixfinal",
    "btcguard_obsfix",
    "btcguard_v3",
    "btcguard_v4",
    "btcguard_v5",
    "btcguard_v6",
    "btcguard_diag",
]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _norm_trace(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if TRACE_COL not in out.columns:
        out[TRACE_COL] = ""
    out[TRACE_COL] = out[TRACE_COL].fillna("").astype(str).str.strip()
    return out


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _candidate_run_name(path: Path) -> str:
    name = path.name
    if not name.startswith(CANDIDATE_PREFIX) or not name.endswith(".csv"):
        return ""
    return name[len(CANDIDATE_PREFIX):-4]


def _trade_path_for_run(run_name: str) -> Path:
    return RESULTS_DIR / f"{TRADES_PREFIX}{run_name}.csv"


def _should_skip_run(run_name: str) -> bool:
    s = str(run_name).lower()
    return any(pat in s for pat in EXCLUDE_RUN_PATTERNS)


def _candidate_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in PASS_COLS + CATEGORICAL_FEATURES + NUMERIC_FEATURES:
        if c in df.columns and c not in cols:
            cols.append(c)
    return cols


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    candidate_files = sorted(RESULTS_DIR.glob(f"{CANDIDATE_PREFIX}*.csv"))
    if not candidate_files:
        raise SystemExit("No candidate files found under results/")

    built_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for cand_path in candidate_files:
        run_name = _candidate_run_name(cand_path)
        if not run_name:
            continue

        if _should_skip_run(run_name):
            continue

        trade_path = _trade_path_for_run(run_name)
        if not trade_path.exists():
            continue

        cand = _safe_read_csv(cand_path)
        tr = _safe_read_csv(trade_path)

        cand = _norm_trace(cand)
        tr = _norm_trace(tr)

        cand = cand[cand[TRACE_COL] != ""].copy()
        tr = tr[tr[TRACE_COL] != ""].copy()

        cand_cols = _candidate_cols(cand)
        cand = cand[cand_cols].copy()

        trade_cols = [c for c in TRADE_KEEP_COLS if c in tr.columns]
        tr = tr[trade_cols].copy()

        if "pnl" not in tr.columns:
            continue

        tr["pnl"] = pd.to_numeric(tr["pnl"], errors="coerce")
        tr = tr.dropna(subset=["pnl"]).copy()
        tr[TARGET_COL] = (tr["pnl"] > 0).astype(int)

        # Evitar explosión por duplicados de trace id
        cand = cand.drop_duplicates(subset=[TRACE_COL], keep="first").copy()
        tr = tr.drop_duplicates(subset=[TRACE_COL], keep="first").copy()

        merged = cand.merge(
            tr,
            on=TRACE_COL,
            how="inner",
            suffixes=("_cand", "_trade"),
        )

        if merged.empty:
            summary_rows.append(
                {
                    "run_name": run_name,
                    "candidate_rows": int(len(cand)),
                    "trade_rows": int(len(tr)),
                    "merged_rows": 0,
                    "win_rate": float("nan"),
                }
            )
            continue

        # Resolver columnas base si quedaron con sufijos
        for base_col in ["symbol", "strategy_id", "side"]:
            cand_col = f"{base_col}_cand"
            trade_col = f"{base_col}_trade"
            if base_col not in merged.columns:
                if cand_col in merged.columns:
                    merged[base_col] = merged[cand_col]
                elif trade_col in merged.columns:
                    merged[base_col] = merged[trade_col]

        merged["run_name"] = run_name

        keep_cols = []
        for c in PASS_COLS + CATEGORICAL_FEATURES + NUMERIC_FEATURES + [
            "run_name",
            TARGET_COL,
            "pnl",
            "exit_reason",
            "entry_ts",
            "exit_ts",
        ]:
            if c in merged.columns and c not in keep_cols:
                keep_cols.append(c)

        merged = merged[keep_cols].copy()
        merged = _to_numeric(merged, NUMERIC_FEATURES + ["pnl"])
        merged = merged.dropna(subset=["symbol", "side", "strategy_id", TARGET_COL]).copy()

        built_frames.append(merged)

        summary_rows.append(
            {
                "run_name": run_name,
                "candidate_rows": int(len(cand)),
                "trade_rows": int(len(tr)),
                "merged_rows": int(len(merged)),
                "win_rate": float(pd.to_numeric(merged[TARGET_COL], errors="coerce").mean()),
            }
        )

    if not built_frames:
        raise SystemExit("No merged rows were produced")

    full = pd.concat(built_frames, ignore_index=True)

    # Deduplicar por trace id global
    full = full.drop_duplicates(subset=[TRACE_COL], keep="first").copy()

    # Mantener solo grupos con muestra útil
    grp = (
        full.groupby(["symbol", "side"], dropna=False)
        .agg(
            rows=(TARGET_COL, "size"),
            positives=(TARGET_COL, "sum"),
        )
        .reset_index()
    )
    grp["negatives"] = grp["rows"] - grp["positives"]

    valid_groups = grp[
        (grp["rows"] >= 80) &
        (grp["positives"] >= 12) &
        (grp["negatives"] >= 12)
    ][["symbol", "side"]].copy()

    full = full.merge(valid_groups, on=["symbol", "side"], how="inner")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["merged_rows", "candidate_rows", "trade_rows"],
        ascending=[False, False, False],
    )
    summary_df.to_csv(OUT_SUMMARY, index=False)

    full.to_csv(OUT_DATASET, index=False)

    print(f"saved: {OUT_SUMMARY}")
    print()
    print("===== BUILD SUMMARY =====")
    print(summary_df.to_string(index=False))

    print()
    print("===== DATASET SUMMARY =====")
    print("rows:", len(full))
    print("symbols:", sorted(full["symbol"].astype(str).unique().tolist()))
    print("sides:", sorted(full["side"].astype(str).unique().tolist()))

    by_group = (
        full.groupby(["symbol", "side"], dropna=False)
        .agg(
            rows=(TARGET_COL, "size"),
            win_rate=(TARGET_COL, "mean"),
            mean_pnl=("pnl", "mean"),
        )
        .reset_index()
        .sort_values(["rows", "symbol", "side"], ascending=[False, True, True])
    )
    print()
    print("by symbol/side:")
    print(by_group.to_string(index=False))

    print()
    print(f"saved: {OUT_DATASET}")


if __name__ == "__main__":
    main()
