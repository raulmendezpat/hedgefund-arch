from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd


def safe_literal_dict(x):
    if pd.isna(x):
        return {}
    s = str(x).strip()
    if not s or s == "{}":
        return {}
    try:
        out = ast.literal_eval(s)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def legacy_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    wcols = [c for c in df.columns if c.startswith("w_")]
    weights = df[wcols].apply(pd.to_numeric, errors="coerce").fillna(0.0) if wcols else pd.DataFrame(index=df.index)

    selector_rows = []
    if "selector_time_by_symbol" in df.columns:
        for _, r in df.iterrows():
            payload = safe_literal_dict(r["selector_time_by_symbol"])
            for sym, meta in payload.items():
                if not isinstance(meta, dict):
                    continue
                selector_rows.append({
                    "ts": int(r.get("ts", 0) or 0),
                    "symbol": sym,
                    "strategy_id": str(meta.get("strategy_id", "") or ""),
                    "side": str(meta.get("side", "flat") or "flat"),
                    "accept": bool(meta.get("accept", False)),
                    "score": float(meta.get("score", 0.0) or 0.0),
                    "pwin": float(meta.get("pwin", 0.0) or 0.0),
                })
    s = pd.DataFrame(selector_rows)

    out = pd.DataFrame({
        "ts": pd.to_numeric(df["ts"], errors="coerce"),
        "legacy_active_symbols": (weights.abs() > 1e-12).sum(axis=1) if len(weights.columns) else 0,
        "legacy_gross_weight": weights.abs().sum(axis=1) if len(weights.columns) else 0.0,
    })

    if len(s):
        agg = s.groupby("ts").agg(
            legacy_candidates=("symbol", "size"),
            legacy_accepts=("accept", lambda x: int(pd.Series(x).astype(bool).sum())),
            legacy_avg_score=("score", "mean"),
            legacy_avg_pwin=("pwin", "mean"),
        ).reset_index()
        out = out.merge(agg, on="ts", how="left")
    else:
        out["legacy_candidates"] = 0
        out["legacy_accepts"] = 0
        out["legacy_avg_score"] = 0.0
        out["legacy_avg_pwin"] = 0.0

    out = out.fillna(0.0)

    summary = {
        "rows": int(len(out)),
        "gross_weight_mean": float(pd.to_numeric(out["legacy_gross_weight"], errors="coerce").mean()),
        "active_symbols_mean": float(pd.to_numeric(out["legacy_active_symbols"], errors="coerce").mean()),
        "candidates_mean": float(pd.to_numeric(out["legacy_candidates"], errors="coerce").mean()),
        "accepts_mean": float(pd.to_numeric(out["legacy_accepts"], errors="coerce").mean()),
        "zero_weight_rows": int((pd.to_numeric(out["legacy_gross_weight"], errors="coerce") <= 1e-12).sum()),
    }
    return out, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    legacy_alloc = Path(f"results/pipeline_allocations_{args.name}.csv")
    alloc_audit = Path(f"results/allocator_audit_{args.name}.csv")
    pre_audit = Path(f"results/pre_allocation_audit_{args.name}.csv")

    if not legacy_alloc.exists():
        raise SystemExit(f"Missing {legacy_alloc}")
    if not alloc_audit.exists():
        raise SystemExit(f"Missing {alloc_audit}")
    if not pre_audit.exists():
        raise SystemExit(f"Missing {pre_audit}")

    legacy_df = pd.read_csv(legacy_alloc, low_memory=False)
    alloc_df = pd.read_csv(alloc_audit)
    pre_df = pd.read_csv(pre_audit)

    legacy_ts_df, legacy_summary_dict = legacy_summary(legacy_df)

    policy_ts = (
        pre_df.groupby("ts").agg(
            candidates=("symbol", "size"),
            accepts=("accept", lambda x: int(pd.Series(x).astype(bool).sum())),
            avg_score=("score_out", "mean"),
            avg_pwin=("p_win_out", "mean"),
            avg_weight_out=("base_weight_out", "mean"),
        ).reset_index()
    )

    merged = legacy_ts_df.merge(alloc_df, on="ts", how="outer").merge(policy_ts, on="ts", how="outer")
    merged = merged.sort_values("ts").fillna(0.0)

    out_path = Path(f"results/compare_pipelines_{args.name}.csv")
    merged.to_csv(out_path, index=False)

    print("saved:", out_path)

    print("\n=== LEGACY SUMMARY ===")
    print(json.dumps(legacy_summary_dict, indent=2))

    print("\n=== CLEAN SUMMARY ===")
    print(json.dumps({
        "rows": int(len(alloc_df)),
        "gross_weight_mean": float(pd.to_numeric(alloc_df["gross_weight"], errors="coerce").mean()),
        "active_symbols_mean": float(pd.to_numeric(alloc_df["active_symbols"], errors="coerce").mean()),
        "accepts_mean": float(pd.to_numeric(alloc_df["n_after_policy"], errors="coerce").mean()),
        "zero_weight_rows": int((pd.to_numeric(alloc_df["gross_weight"], errors="coerce") <= 1e-12).sum()),
    }, indent=2))

    print("\n=== DELTA CLEAN - LEGACY ===")
    print(json.dumps({
        "gross_weight_mean_delta": float(pd.to_numeric(alloc_df["gross_weight"], errors="coerce").mean() - pd.to_numeric(legacy_ts_df["legacy_gross_weight"], errors="coerce").mean()),
        "active_symbols_mean_delta": float(pd.to_numeric(alloc_df["active_symbols"], errors="coerce").mean() - pd.to_numeric(legacy_ts_df["legacy_active_symbols"], errors="coerce").mean()),
        "accepts_mean_delta": float(pd.to_numeric(alloc_df["n_after_policy"], errors="coerce").mean() - pd.to_numeric(legacy_ts_df["legacy_accepts"], errors="coerce").mean()),
        "zero_weight_rows_delta": int((pd.to_numeric(alloc_df["gross_weight"], errors="coerce") <= 1e-12).sum() - (pd.to_numeric(legacy_ts_df["legacy_gross_weight"], errors="coerce") <= 1e-12).sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
