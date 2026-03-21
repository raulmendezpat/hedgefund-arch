from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd

from hf_core import AllocationBridge, Allocator, FeatureBuilder, MetaModel, OpportunityCandidate, PolicyModel, SignalEngine


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


def build_candidates_from_allocations(df: pd.DataFrame) -> list[OpportunityCandidate]:
    out: list[OpportunityCandidate] = []
    if "selector_time_by_symbol" not in df.columns:
        return out

    for _, r in df.iterrows():
        ts = int(r.get("ts", 0) or 0)
        portfolio_context = {
            "portfolio_regime": str(r.get("portfolio_regime", "unknown") or "unknown"),
            "portfolio_breadth": float(pd.to_numeric(pd.Series([r.get("portfolio_breadth", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
            "portfolio_avg_pwin": float(pd.to_numeric(pd.Series([r.get("portfolio_avg_pwin", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
            "portfolio_avg_atrp": float(pd.to_numeric(pd.Series([r.get("portfolio_avg_atrp", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
            "portfolio_avg_strength": float(pd.to_numeric(pd.Series([r.get("portfolio_avg_strength", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
            "portfolio_conviction": float(pd.to_numeric(pd.Series([r.get("portfolio_conviction", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
            "portfolio_regime_scale_applied": float(pd.to_numeric(pd.Series([r.get("portfolio_regime_scale_applied", 1.0)]), errors="coerce").fillna(1.0).iloc[0]),
        }
        payload = safe_literal_dict(r["selector_time_by_symbol"])
        for sym, meta in payload.items():
            if not isinstance(meta, dict):
                continue
            out.append(
                OpportunityCandidate(
                    ts=ts,
                    symbol=str(sym),
                    strategy_id=str(meta.get("strategy_id", "") or ""),
                    side=str(meta.get("side", "flat") or "flat").lower(),
                    signal_strength=float(meta.get("score", 0.0) or 0.0),
                    base_weight=float(meta.get("mult", 0.0) or 0.0),
                    signal_meta={
                        "p_win": float(meta.get("pwin", 0.0) or 0.0),
                        "competitive_score": float(meta.get("score", 0.0) or 0.0),
                        "post_ml_score": float(meta.get("score", 0.0) or 0.0),
                        **portfolio_context,
                    },
                )
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    alloc_path = Path(f"results/pipeline_allocations_{args.name}.csv")
    if not alloc_path.exists():
        raise SystemExit(f"Missing file: {alloc_path}")

    df = pd.read_csv(alloc_path, low_memory=False)
    raw = build_candidates_from_allocations(df)
    if not raw:
        raise SystemExit("No candidates found")

    eng = SignalEngine()
    fb = FeatureBuilder()
    mm = MetaModel()
    pm = PolicyModel()
    bridge = AllocationBridge()
    allocator = Allocator(target_exposure=1.0, symbol_cap=0.35)

    rows = []
    by_ts: dict[int, list[OpportunityCandidate]] = {}
    for c in raw:
        by_ts.setdefault(int(c.ts), []).append(c)

    for ts in sorted(by_ts.keys()):
        batch = by_ts.get(ts, [])
        norm = eng.build_candidates(ts=ts, selected_opps_for_alloc=batch)

        feature_rows = []
        for c in norm:
            meta = dict(c.signal_meta or {})
            portfolio_context = {
                "portfolio_regime": meta.get("portfolio_regime", "unknown"),
                "portfolio_breadth": meta.get("portfolio_breadth", 0.0),
                "portfolio_avg_pwin": meta.get("portfolio_avg_pwin", 0.0),
                "portfolio_avg_atrp": meta.get("portfolio_avg_atrp", 0.0),
                "portfolio_avg_strength": meta.get("portfolio_avg_strength", 0.0),
                "portfolio_conviction": meta.get("portfolio_conviction", 0.0),
                "portfolio_regime_scale_applied": meta.get("portfolio_regime_scale_applied", 1.0),
            }
            feature_rows.append(fb.build_feature_row(candidate=c, portfolio_context=portfolio_context))

        scores = mm.predict_many(feature_rows)
        decisions = pm.decide_many(scores)
        alloc_inputs = bridge.apply(candidates=norm, decisions=decisions)
        alloc = allocator.allocate(candidates=alloc_inputs)

        rows.append({
            "ts": ts,
            "n_raw_batch": len(batch),
            "n_norm": len(norm),
            "n_after_policy": len(alloc_inputs),
            "n_weights": len(alloc.weights),
            "gross_weight": float(sum(abs(v) for v in alloc.weights.values())),
            "active_symbols": int(sum(1 for v in alloc.weights.values() if abs(float(v)) > 1e-12)),
            "alloc_case": str(alloc.meta.get("case", "")),
        })

    out = pd.DataFrame(rows)
    out_path = Path(f"results/allocator_audit_{args.name}.csv")
    out.to_csv(out_path, index=False)

    print("saved:", out_path)
    print("\n=== DESCRIBE ===")
    print(out[["n_raw_batch", "n_norm", "n_after_policy", "n_weights", "gross_weight", "active_symbols"]].describe().to_string())
    print("\n=== ALLOC CASES ===")
    print(out["alloc_case"].value_counts(dropna=False).to_string())
    print("\n=== SAMPLE ===")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
