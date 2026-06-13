from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "adx",
    "range_pct",
    "breakout_distance_up",
    "breakout_distance_down",
    "dist_close_ema_fast",
    "dist_close_ema_slow",
    "pullback_depth",
    "ret_1h_lag",
    "ret_4h_lag",
    "ret_12h_lag",
    "ret_24h_lag",
    "ema_gap_fast_slow",
    "atrp",
    "atrp_zscore",
    "rolling_vol_24h",
    "rolling_vol_72h",
    "btc_adx",
    "btc_atrp",
    "btc_ret_24h_lag",
    "btc_rolling_vol_24h",
]

CATEGORICAL_FEATURES = [
    "symbol",
    "side",
    "strategy_id",
]

LABEL_COL = "target_win"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing input CSV: {path}")
    return pd.read_csv(path, low_memory=False)


def _normalise_symbol(value: object) -> str:
    return str(value or "").replace("/USDT:USDT", "").replace("USDT", "").strip().upper()


def build_dataset(candidates_path: Path, trades_path: Path, out_csv: Path, out_manifest: Path) -> None:
    candidates = _read_csv(candidates_path)
    trades = _read_csv(trades_path)

    required_candidate_cols = ["trace_candidate_id"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing_candidate = [c for c in required_candidate_cols if c not in candidates.columns]
    if missing_candidate:
        raise SystemExit(f"Candidates CSV missing required columns: {missing_candidate}")

    required_trade_cols = ["trace_candidate_id", "pnl"]
    missing_trade = [c for c in required_trade_cols if c not in trades.columns]
    if missing_trade:
        raise SystemExit(f"Trades CSV missing required columns: {missing_trade}")

    if candidates["trace_candidate_id"].duplicated().any():
        dup = int(candidates["trace_candidate_id"].duplicated().sum())
        raise SystemExit(f"Candidates CSV has duplicate trace_candidate_id rows: {dup}")

    if trades["trace_candidate_id"].duplicated().any():
        dup = int(trades["trace_candidate_id"].duplicated().sum())
        raise SystemExit(f"Trades CSV has duplicate trace_candidate_id rows: {dup}")

    trade_cols = [
        c for c in [
            "trace_candidate_id",
            "pnl",
            "exit_reason",
            "entry_ts",
            "exit_ts",
            "bars_held",
            "entry_px",
            "exit_px",
        ]
        if c in trades.columns
    ]

    joined = candidates.merge(
        trades[trade_cols],
        on="trace_candidate_id",
        how="inner",
        suffixes=("", "_trade"),
    ).copy()

    if joined.empty:
        raise SystemExit("Join produced zero rows.")

    joined["pnl"] = pd.to_numeric(joined["pnl"], errors="coerce")
    joined = joined[joined["pnl"].notna()].copy()
    joined[LABEL_COL] = (joined["pnl"] > 0).astype(int)

    # Make numeric features model-safe.
    for col in NUMERIC_FEATURES:
        joined[col] = pd.to_numeric(joined[col], errors="coerce")
        joined[col] = joined[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Make categorical features model-safe and stable.
    for col in CATEGORICAL_FEATURES:
        joined[col] = joined[col].astype(str).fillna("")

    # Short symbol code is useful for audits only. The model uses full symbol.
    joined["symbol_code"] = joined["symbol"].map(_normalise_symbol)

    output_cols = [
        "trace_candidate_id",
        "ts",
        "symbol",
        "symbol_code",
        "side",
        "strategy_id",
        *NUMERIC_FEATURES,
        "pnl",
        LABEL_COL,
        *[c for c in ["exit_reason", "entry_ts", "exit_ts", "bars_held"] if c in joined.columns],
    ]

    output_cols = [c for c in output_cols if c in joined.columns]
    out = joined[output_cols].copy()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(out_csv, index=False)

    group_summary = (
        out.groupby(["symbol", "side", "strategy_id"], dropna=False)
        .agg(
            rows=(LABEL_COL, "size"),
            win_rate=(LABEL_COL, "mean"),
            pnl_sum=("pnl", "sum"),
            pnl_avg=("pnl", "mean"),
        )
        .reset_index()
        .sort_values(["pnl_sum"], ascending=False)
    )

    manifest = {
        "version": "v0_56_candidate_quality_dataset_builder",
        "status": "DATASET_BUILT",
        "candidates_path": str(candidates_path),
        "trades_path": str(trades_path),
        "output_csv": str(out_csv),
        "label_col": LABEL_COL,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "rows": int(len(out)),
        "candidate_rows": int(len(candidates)),
        "trade_rows": int(len(trades)),
        "joined_rows": int(len(joined)),
        "trade_coverage_pct": float(round(100.0 * len(out) / max(len(trades), 1), 6)),
        "target_distribution": {
            str(k): int(v)
            for k, v in out[LABEL_COL].value_counts(dropna=False).sort_index().items()
        },
        "groups": json.loads(group_summary.to_json(orient="records")),
        "notes": [
            "Non-production dataset builder for first crypto discovery universe.",
            "Labels are created from lifecycle trade pnl: target_win = pnl > 0.",
            "Uses trace_candidate_id to join candidate features to realized trades.",
            "Training/promotion requires larger 1M/3M/6M datasets and A/B validation.",
        ],
    }

    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates-csv", required=True)
    ap.add_argument("--trades-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-manifest", required=True)
    args = ap.parse_args()

    build_dataset(
        candidates_path=Path(args.candidates_csv),
        trades_path=Path(args.trades_csv),
        out_csv=Path(args.out_csv),
        out_manifest=Path(args.out_manifest),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
