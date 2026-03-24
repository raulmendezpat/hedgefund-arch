from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_trace(path: str) -> pd.DataFrame:
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["gross_weight"] = pd.to_numeric(df.get("gross_weight"), errors="coerce").fillna(0.0)
    df["n_accepts"] = pd.to_numeric(df.get("n_accepts"), errors="coerce").fillna(0).astype(int)
    df["n_alloc_inputs"] = pd.to_numeric(df.get("n_alloc_inputs"), errors="coerce").fillna(0).astype(int)
    df["n_weighted"] = pd.to_numeric(df.get("n_weighted"), errors="coerce").fillna(0).astype(int)
    return df


def load_runtime(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if "portfolio_return" in df.columns:
        df["portfolio_return"] = pd.to_numeric(df["portfolio_return"], errors="coerce")
    else:
        df["portfolio_return"] = np.nan

    if "equity" in df.columns:
        df["equity"] = pd.to_numeric(df["equity"], errors="coerce").ffill().fillna(1000.0)
        eq_ret = df["equity"].pct_change().fillna(0.0)
        df["portfolio_return"] = df["portfolio_return"].fillna(eq_ret)
    else:
        df["portfolio_return"] = df["portfolio_return"].fillna(0.0)

    return df


def load_candidates(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    for c in [
        "p_win", "expected_return", "score", "policy_score", "base_weight",
        "ret_1h_lag", "ret_4h_lag", "ret_12h_lag", "ret_24h_lag"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "accept" in df.columns:
        df["accept"] = df["accept"].fillna(False).astype(bool)
    return df


def explode_weights(trace_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in trace_df.iterrows():
        ts = r["ts"]
        weights = dict(r.get("weights", {}) or {})
        accepted_scores = dict(r.get("accepted_scores", {}) or {})
        accepted_pwins = dict(r.get("accepted_pwins", {}) or {})
        selected_meta = dict(r.get("selected_meta", {}) or {})

        for sym, w in weights.items():
            meta = dict(selected_meta.get(sym, {}) or {})
            rows.append({
                "ts": ts,
                "symbol": str(sym),
                "weight": float(w or 0.0),
                "abs_weight": abs(float(w or 0.0)),
                "accepted_score": float(accepted_scores.get(sym, np.nan)) if sym in accepted_scores else np.nan,
                "accepted_pwin": float(accepted_pwins.get(sym, np.nan)) if sym in accepted_pwins else np.nan,
                "expected_return": float(meta.get("expected_return", np.nan)) if meta else np.nan,
                "score_final": float(meta.get("score_final", np.nan)) if meta else np.nan,
                "side": str(meta.get("side", "")) if meta else "",
            })
    return pd.DataFrame(rows)


def qcut_safe(series: pd.Series, q: int, prefix: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.nunique() < max(2, q):
        return pd.Series(np.where(s.notna(), f"{prefix}_all", np.nan), index=series.index)
    try:
        return pd.qcut(s, q=q, duplicates="drop").astype(str)
    except Exception:
        return pd.Series(np.where(s.notna(), f"{prefix}_all", np.nan), index=series.index)


def summarize_symbol_alpha(pos_df: pd.DataFrame) -> pd.DataFrame:
    if pos_df.empty:
        return pd.DataFrame()

    g = pos_df.groupby("symbol", dropna=False).agg(
        rows=("weight", "size"),
        mean_weight=("weight", "mean"),
        mean_abs_weight=("abs_weight", "mean"),
        mean_pwin=("accepted_pwin", "mean"),
        mean_expected_return=("expected_return", "mean"),
        mean_score_final=("score_final", "mean"),
        mean_weighted_return=("weighted_return", "mean"),
        sum_weighted_return=("weighted_return", "sum"),
        hit_rate=("weighted_return", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0).mean())),
    ).reset_index()

    return g.sort_values("sum_weighted_return", ascending=False)


def summarize_bucket_alpha(pos_df: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    if pos_df.empty or col not in pos_df.columns:
        return pd.DataFrame()

    x = pos_df.copy()
    x[f"{label}_bucket"] = qcut_safe(x[col], q=5, prefix=label)
    g = x.groupby(f"{label}_bucket", dropna=False).agg(
        rows=("weighted_return", "size"),
        mean_weight=("abs_weight", "mean"),
        mean_signal=("accepted_pwin", "mean"),
        mean_expected_return=("expected_return", "mean"),
        mean_score_final=("score_final", "mean"),
        mean_weighted_return=("weighted_return", "mean"),
        sum_weighted_return=("weighted_return", "sum"),
        hit_rate=("weighted_return", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0).mean())),
    ).reset_index()
    return g.sort_values("sum_weighted_return", ascending=False)


def summarize_snapshots(snap_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if snap_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    cols = [c for c in ["ts", "portfolio_return", "gross_weight", "n_accepts", "n_alloc_inputs", "n_weighted"] if c in snap_df.columns]
    base = snap_df[cols].copy()

    best = base.sort_values("portfolio_return", ascending=False).head(20)
    worst = base.sort_values("portfolio_return", ascending=True).head(20)
    return best, worst


def summarize_side_mix(pos_df: pd.DataFrame) -> pd.DataFrame:
    if pos_df.empty or "side" not in pos_df.columns:
        return pd.DataFrame()

    x = pos_df.copy()
    x["side"] = x["side"].astype(str).str.lower()

    g = x.groupby("side", dropna=False).agg(
        rows=("side", "size"),
        mean_weight=("weight", "mean"),
        mean_abs_weight=("abs_weight", "mean"),
        mean_pwin=("accepted_pwin", "mean"),
        mean_expected_return=("expected_return", "mean"),
        mean_score_final=("score_final", "mean"),
        mean_weighted_return=("weighted_return", "mean"),
        sum_weighted_return=("weighted_return", "sum"),
        hit_rate=("weighted_return", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0).mean())),
    ).reset_index()

    total = float(g["rows"].sum()) if not g.empty else 0.0
    if total > 0:
        g["row_pct"] = g["rows"] / total

    return g.sort_values("rows", ascending=False)


def build_position_runs(pos_df: pd.DataFrame) -> pd.DataFrame:
    if pos_df.empty:
        return pd.DataFrame()

    x = pos_df.copy().sort_values(["symbol", "side", "ts"]).reset_index(drop=True)
    x["is_active"] = x["abs_weight"] > 0

    run_rows = []

    for (symbol, side), g in x.groupby(["symbol", "side"], dropna=False):
        g = g.sort_values("ts").reset_index(drop=True)
        if g.empty:
            continue

        active = False
        run_start = None
        prev_ts = None
        run_returns = []
        run_weights = []

        for _, r in g.iterrows():
            ts = r["ts"]
            is_active = bool(r["is_active"])
            pr = float(r.get("portfolio_return", 0.0) or 0.0)
            w = float(r.get("weight", 0.0) or 0.0)

            if is_active and not active:
                active = True
                run_start = ts
                run_returns = [pr]
                run_weights = [w]
            elif is_active and active:
                run_returns.append(pr)
                run_weights.append(w)
            elif (not is_active) and active:
                run_end = prev_ts if prev_ts is not None else ts
                hours = 0.0
                if run_start is not None and run_end is not None:
                    hours = float((run_end - run_start).total_seconds() / 3600.0) + 1.0

                gross_ret = float(sum(run_returns))
                weighted_ret = float(sum((rw * rr) for rw, rr in zip(run_weights, run_returns)))

                run_rows.append({
                    "symbol": str(symbol),
                    "side": str(side),
                    "start_ts": run_start,
                    "end_ts": run_end,
                    "duration_hours": hours,
                    "bars": len(run_returns),
                    "gross_return_sum": gross_ret,
                    "weighted_return_sum": weighted_ret,
                    "mean_abs_weight": float(np.mean(np.abs(run_weights))) if run_weights else 0.0,
                })

                active = False
                run_start = None
                run_returns = []
                run_weights = []

            prev_ts = ts

        if active:
            run_end = prev_ts
            hours = 0.0
            if run_start is not None and run_end is not None:
                hours = float((run_end - run_start).total_seconds() / 3600.0) + 1.0

            gross_ret = float(sum(run_returns))
            weighted_ret = float(sum((rw * rr) for rw, rr in zip(run_weights, run_returns)))

            run_rows.append({
                "symbol": str(symbol),
                "side": str(side),
                "start_ts": run_start,
                "end_ts": run_end,
                "duration_hours": hours,
                "bars": len(run_returns),
                "gross_return_sum": gross_ret,
                "weighted_return_sum": weighted_ret,
                "mean_abs_weight": float(np.mean(np.abs(run_weights))) if run_weights else 0.0,
            })

    return pd.DataFrame(run_rows)


def summarize_run_durations(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()

    g = run_df.groupby("side", dropna=False).agg(
        runs=("side", "size"),
        avg_duration_hours=("duration_hours", "mean"),
        median_duration_hours=("duration_hours", "median"),
        max_duration_hours=("duration_hours", "max"),
        avg_bars=("bars", "mean"),
        avg_gross_return_sum=("gross_return_sum", "mean"),
        avg_weighted_return_sum=("weighted_return_sum", "mean"),
        hit_rate=("weighted_return_sum", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0).mean())),
    ).reset_index()

    total = float(g["runs"].sum()) if not g.empty else 0.0
    if total > 0:
        g["run_pct"] = g["runs"] / total

    return g.sort_values("runs", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Run basename, e.g. smoke_allocator_refactor_v4")
    args = ap.parse_args()

    name = str(args.name)

    runtime_csv = f"results/research_runtime_{name}.csv"
    candidates_csv = f"results/research_runtime_candidates_{name}.csv"
    trace_jsonl = f"results/allocation_trace_{name}.jsonl"

    trace_df = load_trace(trace_jsonl)
    runtime_df = load_runtime(runtime_csv)
    cand_df = load_candidates(candidates_csv)

    pos_df = explode_weights(trace_df)

    if pos_df.empty:
        print("No weighted positions found in allocation trace.")
        return

    rt = runtime_df[["ts", "portfolio_return"]].copy()
    pos_df = pos_df.merge(rt, on="ts", how="left")
    pos_df["portfolio_return"] = pd.to_numeric(pos_df["portfolio_return"], errors="coerce").fillna(0.0)
    pos_df["weighted_return"] = pos_df["weight"] * pos_df["portfolio_return"]

    snap_df = trace_df.merge(rt, on="ts", how="left")
    snap_df["portfolio_return"] = pd.to_numeric(snap_df["portfolio_return"], errors="coerce").fillna(0.0)

    symbol_alpha = summarize_symbol_alpha(pos_df)
    pwin_alpha = summarize_bucket_alpha(pos_df, "accepted_pwin", "pwin")
    er_alpha = summarize_bucket_alpha(pos_df, "expected_return", "er")
    score_alpha = summarize_bucket_alpha(pos_df, "score_final", "score")
    side_mix = summarize_side_mix(pos_df)
    run_df = build_position_runs(pos_df)
    run_duration = summarize_run_durations(run_df)

    best_snaps, worst_snaps = summarize_snapshots(snap_df)

    out_dir = Path("results")
    symbol_alpha_path = out_dir / f"analysis_symbol_alpha_{name}.csv"
    pwin_alpha_path = out_dir / f"analysis_pwin_bucket_alpha_{name}.csv"
    er_alpha_path = out_dir / f"analysis_er_bucket_alpha_{name}.csv"
    score_alpha_path = out_dir / f"analysis_score_bucket_alpha_{name}.csv"
    best_snaps_path = out_dir / f"analysis_best_snapshots_{name}.csv"
    worst_snaps_path = out_dir / f"analysis_worst_snapshots_{name}.csv"
    side_mix_path = out_dir / f"analysis_side_mix_{name}.csv"
    run_duration_path = out_dir / f"analysis_run_duration_{name}.csv"
    run_detail_path = out_dir / f"analysis_run_detail_{name}.csv"

    symbol_alpha.to_csv(symbol_alpha_path, index=False)
    pwin_alpha.to_csv(pwin_alpha_path, index=False)
    er_alpha.to_csv(er_alpha_path, index=False)
    score_alpha.to_csv(score_alpha_path, index=False)
    best_snaps.to_csv(best_snaps_path, index=False)
    worst_snaps.to_csv(worst_snaps_path, index=False)
    side_mix.to_csv(side_mix_path, index=False)
    run_duration.to_csv(run_duration_path, index=False)
    run_df.to_csv(run_detail_path, index=False)

    print("saved:", symbol_alpha_path)
    print("saved:", pwin_alpha_path)
    print("saved:", er_alpha_path)
    print("saved:", score_alpha_path)
    print("saved:", best_snaps_path)
    print("saved:", worst_snaps_path)
    print("saved:", side_mix_path)
    print("saved:", run_duration_path)
    print("saved:", run_detail_path)

    print("\n=== SYMBOL ALPHA ===")
    print(symbol_alpha.head(20).to_string(index=False) if not symbol_alpha.empty else "(empty)")

    print("\n=== PWIN BUCKET ALPHA ===")
    print(pwin_alpha.to_string(index=False) if not pwin_alpha.empty else "(empty)")

    print("\n=== EXPECTED RETURN BUCKET ALPHA ===")
    print(er_alpha.to_string(index=False) if not er_alpha.empty else "(empty)")

    print("\n=== SCORE BUCKET ALPHA ===")
    print(score_alpha.to_string(index=False) if not score_alpha.empty else "(empty)")

    print("\n=== SIDE MIX ===")
    print(side_mix.to_string(index=False) if not side_mix.empty else "(empty)")

    print("\n=== RUN DURATION ===")
    print(run_duration.to_string(index=False) if not run_duration.empty else "(empty)")

    print("\n=== BEST SNAPSHOTS ===")
    print(best_snaps.head(10).to_string(index=False) if not best_snaps.empty else "(empty)")

    print("\n=== WORST SNAPSHOTS ===")
    print(worst_snaps.head(10).to_string(index=False) if not worst_snaps.empty else "(empty)")


if __name__ == "__main__":
    main()
