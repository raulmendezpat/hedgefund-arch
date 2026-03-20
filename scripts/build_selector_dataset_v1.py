from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_col(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _find_symbol_prefixes(alloc_df: pd.DataFrame) -> list[str]:
    prefixes: list[str] = []
    for c in alloc_df.columns:
        if c.startswith("w_") and c.endswith("_usdt_usdt"):
            prefixes.append(c[len("w_"):])
    return sorted(set(prefixes))


def _symbol_from_prefix(prefix: str) -> str:
    return prefix.replace("_usdt_usdt", "").upper()


def _ccxt_symbol_from_prefix(prefix: str) -> str:
    return f"{_symbol_from_prefix(prefix)}/USDT:USDT"


def _first_existing(cols: list[str], all_cols: set[str]) -> Optional[str]:
    for c in cols:
        if c in all_cols:
            return c
    return None


def _build_symbol_column_map(alloc_df: pd.DataFrame, prefix: str) -> dict[str, Optional[str]]:
    cols = set(alloc_df.columns)

    mapping = {
        "weight": _first_existing([f"w_{prefix}"], cols),
        "side": _first_existing(
            [f"{prefix}_side", f"{prefix}_cluster_side"],
            cols,
        ),
        "strategy_id": _first_existing(
            [f"{prefix}_strategy_id", f"{prefix}_cluster_strategy_id"],
            cols,
        ),
        "strength": _first_existing([f"{prefix}_strength"], cols),
        "base_weight": _first_existing([f"{prefix}_base_weight"], cols),
        "p_win": _first_existing([f"{prefix}_p_win", f"{prefix}_cluster_p_win"], cols),
        "competitive_score": _first_existing(
            [f"{prefix}_competitive_score", f"{prefix}_cluster_competitive_score"],
            cols,
        ),
        "post_ml_score": _first_existing(
            [f"{prefix}_post_ml_score", f"{prefix}_cluster_post_ml_score"],
            cols,
        ),
        "engine": _first_existing([f"{prefix}_engine", f"{prefix}_cluster_engine"], cols),
        "registry_symbol": _first_existing(
            [f"{prefix}_registry_symbol", f"{prefix}_cluster_registry_symbol"],
            cols,
        ),
        "execution_target_weight": _first_existing([f"{prefix}_execution_target_weight"], cols),
        "cluster_target_weight": _first_existing([f"{prefix}_cluster_target_weight"], cols),
        "adx": _first_existing([f"{prefix}_adx"], cols),
        "atrp": _first_existing([f"{prefix}_atrp"], cols),
        "bb_width": _first_existing([f"{prefix}_bb_width"], cols),
        "ml_position_size_mult": _first_existing([f"{prefix}_ml_position_size_mult"], cols),
    }
    return mapping


def _signed_forward_return(ret_series: pd.Series, weight_series: pd.Series, horizon: int) -> pd.Series:
    gross = pd.Series(1.0, index=ret_series.index, dtype="float64")
    for h in range(1, horizon + 1):
        gross = gross * (1.0 + ret_series.shift(-h).fillna(0.0))
    fwd = gross - 1.0

    side_sign = np.sign(weight_series.fillna(0.0))
    signed = fwd * side_sign
    signed = signed.where(side_sign != 0.0, 0.0)
    return signed


def _load_metrics_cost_proxy(run_name: str, results_dir: Path, default_cost_bps: float) -> float:
    metrics_path = results_dir / f"pipeline_metrics_{run_name}.json"
    if not metrics_path.exists():
        return default_cost_bps / 10000.0

    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        avg_cost_pct = float(data.get("avg_execution_cost_pct", 0.0) or 0.0)
        if avg_cost_pct > 0:
            return avg_cost_pct
    except Exception:
        pass

    return default_cost_bps / 10000.0


def build_dataset_for_run(
    run_name: str,
    results_dir: Path,
    edge_min_3: float,
    edge_min_6: float,
    default_cost_bps: float,
) -> pd.DataFrame:
    alloc_path = results_dir / f"pipeline_allocations_{run_name}.csv"
    perf_path = results_dir / f"pipeline_equity_{run_name}.csv"

    if not alloc_path.exists():
        raise FileNotFoundError(f"No existe {alloc_path}")
    if not perf_path.exists():
        raise FileNotFoundError(f"No existe {perf_path}")

    alloc = pd.read_csv(alloc_path, low_memory=False).reset_index(drop=True)
    perf = pd.read_csv(perf_path, low_memory=False).reset_index(drop=True)

    if len(alloc) != len(perf):
        raise ValueError(
            f"len mismatch en {run_name}: allocations={len(alloc)} vs equity={len(perf)}"
        )

    cost_proxy = _load_metrics_cost_proxy(run_name, results_dir, default_cost_bps)

    prefixes = _find_symbol_prefixes(alloc)
    rows: list[pd.DataFrame] = []

    base_common = pd.DataFrame({
        "run": run_name,
        "row_idx": alloc.index.astype(int),
        "ts": _safe_col(alloc, "ts"),
        "ts_utc": _safe_col(alloc, "ts_utc"),
        "portfolio_regime": _safe_col(alloc, "portfolio_regime", "unknown").astype(str),
        "portfolio_breadth": _to_num(_safe_col(alloc, "portfolio_breadth", 0.0)).fillna(0.0),
        "portfolio_avg_pwin": _to_num(_safe_col(alloc, "portfolio_avg_pwin", 0.0)).fillna(0.0),
        "portfolio_avg_atrp": _to_num(_safe_col(alloc, "portfolio_avg_atrp", 0.0)).fillna(0.0),
        "portfolio_avg_strength": _to_num(_safe_col(alloc, "portfolio_avg_strength", 0.0)).fillna(0.0),
        "portfolio_conviction": _to_num(_safe_col(alloc, "portfolio_conviction", 0.0)).fillna(0.0),
        "portfolio_regime_scale_applied": _to_num(
            _safe_col(alloc, "portfolio_regime_scale_applied", 1.0)
        ).fillna(1.0),
    })

    for prefix in prefixes:
        sym = _symbol_from_prefix(prefix)
        ccxt_sym = _ccxt_symbol_from_prefix(prefix)
        colmap = _build_symbol_column_map(alloc, prefix)

        weight_col = colmap["weight"]
        side_col = colmap["side"]
        strat_col = colmap["strategy_id"]

        if weight_col is None or side_col is None or strat_col is None:
            continue

        weight = _to_num(_safe_col(alloc, weight_col, 0.0)).fillna(0.0)
        side = _safe_col(alloc, side_col, "flat").astype(str).str.lower().fillna("flat")
        strategy_id = _safe_col(alloc, strat_col, "").astype(str).fillna("")

        ret_col = f"ret_{ccxt_sym}"
        if ret_col not in perf.columns:
            continue

        asset_ret = _to_num(perf[ret_col]).fillna(0.0)

        candidate = base_common.copy()
        candidate["symbol"] = sym
        candidate["symbol_ccxt"] = ccxt_sym
        candidate["strategy_id"] = strategy_id
        candidate["side"] = side
        candidate["weight"] = weight.abs()
        candidate["signed_weight"] = weight
        candidate["abs_weight"] = weight.abs()

        for feat_name in [
            "strength",
            "base_weight",
            "p_win",
            "competitive_score",
            "post_ml_score",
            "engine",
            "registry_symbol",
            "execution_target_weight",
            "cluster_target_weight",
            "adx",
            "atrp",
            "bb_width",
            "ml_position_size_mult",
        ]:
            src = colmap.get(feat_name)
            if src is None:
                candidate[feat_name] = np.nan
            else:
                candidate[feat_name] = alloc[src]

        num_cols = [
            "strength",
            "base_weight",
            "p_win",
            "competitive_score",
            "post_ml_score",
            "execution_target_weight",
            "cluster_target_weight",
            "adx",
            "atrp",
            "bb_width",
            "ml_position_size_mult",
        ]
        for c in num_cols:
            candidate[c] = _to_num(candidate[c]).fillna(0.0)

        candidate["ret_1"] = _signed_forward_return(asset_ret, weight, 1)
        candidate["ret_3"] = _signed_forward_return(asset_ret, weight, 3)
        candidate["ret_6"] = _signed_forward_return(asset_ret, weight, 6)

        candidate["ret_net_3"] = candidate["ret_3"] - float(cost_proxy)
        candidate["ret_net_6"] = candidate["ret_6"] - float(cost_proxy)

        candidate["y_win_3"] = (candidate["ret_3"] > 0.0).astype(int)
        candidate["y_win_6"] = (candidate["ret_6"] > 0.0).astype(int)

        candidate["y_edge_net_pos_3"] = (candidate["ret_net_3"] > float(edge_min_3)).astype(int)
        candidate["y_edge_net_pos_6"] = (candidate["ret_net_6"] > float(edge_min_6)).astype(int)

        mask = (
            candidate["strategy_id"].astype(str).str.len() > 0
        ) & (
            candidate["side"].isin(["long", "short"])
        ) & (
            candidate["abs_weight"] > 1e-12
        )

        candidate = candidate.loc[mask].copy()
        if len(candidate) == 0:
            continue

        rows.append(candidate)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    lead = [
        "run",
        "row_idx",
        "ts",
        "ts_utc",
        "symbol",
        "symbol_ccxt",
        "strategy_id",
        "side",
        "abs_weight",
        "signed_weight",
        "strength",
        "base_weight",
        "p_win",
        "competitive_score",
        "post_ml_score",
        "engine",
        "registry_symbol",
        "portfolio_regime",
        "portfolio_breadth",
        "portfolio_avg_pwin",
        "portfolio_avg_atrp",
        "portfolio_avg_strength",
        "portfolio_conviction",
        "portfolio_regime_scale_applied",
        "ret_1",
        "ret_3",
        "ret_6",
        "ret_net_3",
        "ret_net_6",
        "y_win_3",
        "y_win_6",
        "y_edge_net_pos_3",
        "y_edge_net_pos_6",
    ]
    tail = [c for c in out.columns if c not in lead]
    out = out[lead + tail]

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build selector training dataset v1 from backtest outputs.")
    ap.add_argument(
        "--runs",
        required=True,
        help="Comma-separated run names",
    )
    ap.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing pipeline outputs",
    )
    ap.add_argument(
        "--out",
        default="results/selector_training_dataset_v1.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--edge-min-3",
        type=float,
        default=0.0010,
        help="Minimum net edge threshold for y_edge_net_pos_3",
    )
    ap.add_argument(
        "--edge-min-6",
        type=float,
        default=0.0015,
        help="Minimum net edge threshold for y_edge_net_pos_6",
    )
    ap.add_argument(
        "--default-cost-bps",
        type=float,
        default=8.0,
        help="Fallback flat cost proxy in bps",
    )
    args = ap.parse_args()

    runs = [x.strip() for x in str(args.runs).split(",") if x.strip()]
    results_dir = Path(args.results_dir)
    out_path = Path(args.out)

    all_parts: list[pd.DataFrame] = []
    for run in runs:
        part = build_dataset_for_run(
            run_name=run,
            results_dir=results_dir,
            edge_min_3=float(args.edge_min_3),
            edge_min_6=float(args.edge_min_6),
            default_cost_bps=float(args.default_cost_bps),
        )
        print(f"[{run}] rows={len(part)}")
        if len(part):
            all_parts.append(part)

    if not all_parts:
        raise SystemExit("No se generaron filas para ningún run.")

    df = pd.concat(all_parts, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\n=== DATASET SUMMARY ===")
    print("rows:", len(df))
    print("runs:", sorted(df["run"].astype(str).unique().tolist()))
    print("symbols:", sorted(df["symbol"].astype(str).unique().tolist()))
    print("strategy_ids:", df["strategy_id"].nunique())
    print("\nside counts:")
    print(df["side"].value_counts(dropna=False).to_string())
    print("\ny_win_3 mean by side:")
    print(df.groupby("side")["y_win_3"].mean().to_string())
    print("\ny_edge_net_pos_3 mean by side:")
    print(df.groupby("side")["y_edge_net_pos_3"].mean().to_string())
    print("\nSaved ->", out_path)


if __name__ == "__main__":
    main()
