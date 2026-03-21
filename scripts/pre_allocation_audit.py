from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd

from hf_core import AllocationBridge, FeatureBuilder, MetaModel, OpportunityCandidate, PolicyModel, SignalEngine


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
    raw_candidates = build_candidates_from_allocations(df)
    print("raw_candidates:", len(raw_candidates))
    if not raw_candidates:
        raise SystemExit("No candidates found")

    eng = SignalEngine()
    fb = FeatureBuilder()
    mm = MetaModel()
    pm = PolicyModel()
    bridge = AllocationBridge()

    norm = eng.build_candidates(ts=int(raw_candidates[0].ts), selected_opps_for_alloc=raw_candidates)

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

    out_rows = []
    for c, s, d in zip(norm, scores, decisions):
        out_rows.append({
            "ts": c.ts,
            "symbol": c.symbol,
            "strategy_id": c.strategy_id,
            "side": c.side,
            "signal_strength": c.signal_strength,
            "base_weight_in": c.base_weight,
            "p_win_out": s.p_win,
            "expected_return_out": s.expected_return,
            "score_out": s.score,
            "accept": d.accept,
            "size_mult": d.size_mult,
            "band": d.band,
            "reason": d.reason,
            "base_weight_out": (c.base_weight * d.size_mult) if d.accept else 0.0,
        })

    out_df = pd.DataFrame(out_rows)
    out_path = Path(f"results/pre_allocation_audit_{args.name}.csv")
    out_df.to_csv(out_path, index=False)

    print("saved:", out_path)
    print("allocation_inputs:", len(alloc_inputs))

    print("\n=== ACCEPT SUMMARY ===")
    print(out_df["accept"].value_counts(dropna=False).to_string())

    print("\n=== BAND SUMMARY ===")
    print(out_df["band"].value_counts(dropna=False).to_string())

    print("\n=== BY STRATEGY ===")
    print(
        out_df.groupby("strategy_id").agg(
            rows=("symbol", "size"),
            accept_rate=("accept", "mean"),
            avg_p_win=("p_win_out", "mean"),
            avg_expected_return=("expected_return_out", "mean"),
            avg_score=("score_out", "mean"),
            avg_size=("size_mult", "mean"),
            avg_weight_out=("base_weight_out", "mean"),
        ).sort_values("avg_score", ascending=False).to_string()
    )

    print("\n=== BY SIDE ===")
    print(
        out_df.groupby("side").agg(
            rows=("symbol", "size"),
            accept_rate=("accept", "mean"),
            avg_p_win=("p_win_out", "mean"),
            avg_expected_return=("expected_return_out", "mean"),
            avg_score=("score_out", "mean"),
            avg_size=("size_mult", "mean"),
            avg_weight_out=("base_weight_out", "mean"),
        ).to_string()
    )


if __name__ == "__main__":
    main()
