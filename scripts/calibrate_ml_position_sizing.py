from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _build_bins(df: pd.DataFrame, pwin_col: str, bins: int) -> pd.DataFrame:
    work = df[[pwin_col]].copy()
    work["_bin"] = pd.qcut(
        work[pwin_col],
        q=min(bins, max(2, work[pwin_col].nunique())),
        duplicates="drop",
    )
    return work


def _interval_bounds(interval) -> tuple[float, float]:
    if interval is None or pd.isna(interval):
        return (0.0, 1.0)
    return (float(interval.left), float(interval.right))


def calibrate_position_sizing(
    df: pd.DataFrame,
    *,
    pwin_col: str,
    ret_col: str,
    label_col: str,
    bins: int,
    max_mult: float,
    min_mult: float,
) -> pd.DataFrame:
    work = df.copy()

    work[pwin_col] = pd.to_numeric(work[pwin_col], errors="coerce")
    work[ret_col] = pd.to_numeric(work[ret_col], errors="coerce")
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce")

    work = work.dropna(subset=[pwin_col, ret_col, label_col]).copy()
    work = work[np.isfinite(work[pwin_col]) & np.isfinite(work[ret_col]) & np.isfinite(work[label_col])].copy()

    if "selected_by_opportunity_selector" in work.columns:
        work = work[work["selected_by_opportunity_selector"].fillna(0).astype(float) == 1].copy()

    if work.empty:
        raise ValueError("No hay filas válidas luego del filtrado para calibración.")

    binned = _build_bins(work, pwin_col=pwin_col, bins=bins)
    work["_bin"] = binned["_bin"]

    grouped = (
        work.groupby("_bin", observed=True)
        .agg(
            samples=(pwin_col, "size"),
            pwin_min=(pwin_col, "min"),
            pwin_max=(pwin_col, "max"),
            pwin_mean=(pwin_col, "mean"),
            avg_ret=(ret_col, "mean"),
            median_ret=(ret_col, "median"),
            ret_std=(ret_col, "std"),
            win_rate=(label_col, "mean"),
        )
        .reset_index()
    )

    grouped["ret_std"] = grouped["ret_std"].fillna(0.0)
    grouped["sharpe_like"] = grouped.apply(
        lambda r: 0.0 if float(r["ret_std"]) <= 1e-12 else float(r["avg_ret"]) / float(r["ret_std"]),
        axis=1,
    )

    def _score_row(r) -> float:
        avg_ret = float(r["avg_ret"])
        win_rate = float(r["win_rate"])

        if avg_ret <= 0.0:
            return 0.0

        win_edge = max(0.0, win_rate - 0.5)
        return avg_ret * (1.0 + 2.0 * win_edge)

    grouped["raw_score"] = grouped.apply(_score_row, axis=1)

    max_score = float(grouped["raw_score"].max()) if not grouped.empty else 0.0

    grouped = grouped.sort_values("pwin_mean").reset_index(drop=True)

    positive_mask = grouped["raw_score"] > 0.0
    positive_scores = grouped.loc[positive_mask, "raw_score"]

    grouped["size_mult"] = 0.0

    if not positive_scores.empty:
        min_positive = float(positive_scores.min())
        max_positive = float(positive_scores.max())

        if max_positive <= min_positive:
            grouped.loc[positive_mask, "size_mult"] = max_mult
        else:
            grouped.loc[positive_mask, "size_mult"] = grouped.loc[positive_mask, "raw_score"].apply(
                lambda x: min_mult + (max_mult - min_mult) * ((float(x) - min_positive) / (max_positive - min_positive))
            )

    grouped["size_mult"] = grouped["size_mult"].clip(lower=0.0, upper=max_mult)

    grouped.loc[grouped["avg_ret"] <= 0.0, "size_mult"] = 0.0

    mins: List[float] = []
    maxs: List[float] = []
    for interval in grouped["_bin"]:
        lo, hi = _interval_bounds(interval)
        mins.append(lo)
        maxs.append(hi)

    grouped["bin_min"] = mins
    grouped["bin_max"] = maxs

    cols = [
        "bin_min",
        "bin_max",
        "samples",
        "pwin_min",
        "pwin_max",
        "pwin_mean",
        "avg_ret",
        "median_ret",
        "ret_std",
        "sharpe_like",
        "win_rate",
        "raw_score",
        "size_mult",
    ]
    return grouped[cols].copy()


def build_artifact(
    calib: pd.DataFrame,
    *,
    dataset_path: str,
    pwin_col: str,
    ret_col: str,
    label_col: str,
    bins: int,
    min_mult: float,
    max_mult: float,
) -> Dict:
    rows = []
    for _, r in calib.iterrows():
        rows.append(
            {
                "min": _safe_float(r["bin_min"]),
                "max": _safe_float(r["bin_max"], 1.0),
                "samples": int(r["samples"]),
                "pwin_mean": _safe_float(r["pwin_mean"]),
                "avg_ret": _safe_float(r["avg_ret"]),
                "win_rate": _safe_float(r["win_rate"]),
                "size": _safe_float(r["size_mult"]),
            }
        )

    if rows:
        rows[0]["min"] = 0.0
        rows[-1]["max"] = 1.0

    return {
        "type": "ml_position_size_map",
        "version": 1,
        "source_dataset": dataset_path,
        "pwin_col": pwin_col,
        "ret_col": ret_col,
        "label_col": label_col,
        "bins_requested": int(bins),
        "min_mult": float(min_mult),
        "max_mult": float(max_mult),
        "bins": rows,
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Calibrar ML position sizing desde dataset histórico.")
    ap.add_argument(
        "--dataset",
        type=str,
        default="results/ml_features_registry_ml_dataset_v1.csv",
        help="CSV histórico con p_win y retornos futuros.",
    )
    ap.add_argument(
        "--output-json",
        type=str,
        default="artifacts/ml_position_size_map_v1.json",
        help="Artifact JSON de salida.",
    )
    ap.add_argument(
        "--output-csv",
        type=str,
        default="artifacts/ml_position_size_calibration_v1.csv",
        help="Resumen tabular de calibración.",
    )
    ap.add_argument("--pwin-col", type=str, default="p_win")
    ap.add_argument("--ret-col", type=str, default="future_ret_6")
    ap.add_argument("--label-col", type=str, default="y_win_selected_6")
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--min-mult", type=float, default=0.0)
    ap.add_argument("--max-mult", type=float, default=1.2)
    return ap.parse_args()


def main():
    args = parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe dataset: {dataset_path}")

    df = pd.read_csv(dataset_path)

    calib = calibrate_position_sizing(
        df,
        pwin_col=args.pwin_col,
        ret_col=args.ret_col,
        label_col=args.label_col,
        bins=args.bins,
        min_mult=float(args.min_mult),
        max_mult=float(args.max_mult),
    )

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    artifact = build_artifact(
        calib,
        dataset_path=str(dataset_path),
        pwin_col=args.pwin_col,
        ret_col=args.ret_col,
        label_col=args.label_col,
        bins=args.bins,
        min_mult=float(args.min_mult),
        max_mult=float(args.max_mult),
    )

    calib.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(f"Saved CSV  -> {out_csv}")
    print(f"Saved JSON -> {out_json}")
    print()
    print(calib.to_string(index=False))


if __name__ == "__main__":
    main()
