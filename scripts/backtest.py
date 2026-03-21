from __future__ import annotations

import argparse
import ast
import json
import math
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


def sym_key(sym: str) -> str:
    return str(sym).lower().replace("/", "_").replace(":", "_").replace("-", "_")


def compute_metrics(port_ret: pd.Series, equity: pd.Series, periods_per_year: float = 8760.0) -> dict:
    port_ret = pd.to_numeric(port_ret, errors="coerce").fillna(0.0)
    equity = pd.to_numeric(equity, errors="coerce").ffill().fillna(1000.0)

    total_return_pct = (float(equity.iloc[-1]) / float(equity.iloc[0]) - 1.0) * 100.0 if len(equity) else 0.0
    mean_r = float(port_ret.mean()) if len(port_ret) else 0.0
    std_r = float(port_ret.std(ddof=0)) if len(port_ret) else 0.0
    sharpe = (mean_r / std_r) * math.sqrt(periods_per_year) if std_r > 0 else 0.0

    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    max_dd_pct = float(dd.min()) * 100.0 if len(dd) else 0.0

    years = len(port_ret) / periods_per_year if periods_per_year > 0 else 0.0
    if years > 0 and float(equity.iloc[0]) > 0 and float(equity.iloc[-1]) > 0:
        cagr = ((float(equity.iloc[-1]) / float(equity.iloc[0])) ** (1.0 / years) - 1.0) * 100.0
    else:
        cagr = 0.0

    win_rate_pct = float((port_ret > 0).mean()) * 100.0 if len(port_ret) else 0.0
    vol_annual = std_r * math.sqrt(periods_per_year) if std_r > 0 else 0.0

    return {
        "total_return_pct": total_return_pct,
        "cagr_pct": cagr,
        "sharpe_annual": sharpe,
        "max_drawdown_pct": max_dd_pct,
        "vol_annual": vol_annual,
        "equity_final": float(equity.iloc[-1]) if len(equity) else 1000.0,
        "win_rate_pct": win_rate_pct,
    }


def build_candidates_from_row(r: pd.Series) -> list[OpportunityCandidate]:
    payload = safe_literal_dict(r.get("selector_time_by_symbol"))
    ts = int(pd.to_numeric(pd.Series([r.get("ts", 0)]), errors="coerce").fillna(0).iloc[0])

    portfolio_context = {
        "portfolio_regime": str(r.get("portfolio_regime", "unknown") or "unknown"),
        "portfolio_breadth": float(pd.to_numeric(pd.Series([r.get("portfolio_breadth", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
        "portfolio_avg_pwin": float(pd.to_numeric(pd.Series([r.get("portfolio_avg_pwin", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
        "portfolio_avg_atrp": float(pd.to_numeric(pd.Series([r.get("portfolio_avg_atrp", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
        "portfolio_avg_strength": float(pd.to_numeric(pd.Series([r.get("portfolio_avg_strength", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
        "portfolio_conviction": float(pd.to_numeric(pd.Series([r.get("portfolio_conviction", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
        "portfolio_regime_scale_applied": float(pd.to_numeric(pd.Series([r.get("portfolio_regime_scale_applied", 1.0)]), errors="coerce").fillna(1.0).iloc[0]),
    }

    out: list[OpportunityCandidate] = []
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
    equity_path = Path(f"results/pipeline_equity_{args.name}.csv")
    legacy_metrics_path = Path(f"results/pipeline_metrics_{args.name}.json")

    if not alloc_path.exists():
        raise SystemExit(f"Missing {alloc_path}")
    if not equity_path.exists():
        raise SystemExit(f"Missing {equity_path}")

    alloc_df = pd.read_csv(alloc_path, low_memory=False)
    eq_df = pd.read_csv(equity_path, low_memory=False)

    if len(alloc_df) != len(eq_df):
        raise SystemExit(f"Length mismatch: allocations={len(alloc_df)} equity={len(eq_df)}")

    eng = SignalEngine()
    fb = FeatureBuilder()
    mm = MetaModel()
    pm = PolicyModel()
    bridge = AllocationBridge()
    allocator = Allocator(target_exposure=1.0, symbol_cap=0.35)

    weight_rows = []

    for i in range(len(alloc_df)):
        r = alloc_df.iloc[i]
        candidates = build_candidates_from_row(r)
        ts = int(pd.to_numeric(pd.Series([r.get("ts", 0)]), errors="coerce").fillna(i).iloc[0])

        norm = eng.build_candidates(ts=ts, selected_opps_for_alloc=candidates)

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

        row = {
            "row_id": i,
            "n_raw_batch": len(candidates),
            "n_norm": len(norm),
            "n_after_policy": len(alloc_inputs),
            "gross_weight": float(sum(abs(v) for v in alloc.weights.values())),
            "active_symbols": int(sum(1 for v in alloc.weights.values() if abs(float(v)) > 1e-12)),
            "alloc_case": str(alloc.meta.get("case", "")),
        }
        for sym, w in dict(alloc.weights).items():
            row[f"w_{sym_key(sym)}"] = float(w)
        weight_rows.append(row)

    weights_df = pd.DataFrame(weight_rows)
    eq_df = eq_df.reset_index(drop=True).copy()
    eq_df["row_id"] = range(len(eq_df))

    merged = eq_df.merge(weights_df, on="row_id", how="left")

    for c in merged.columns:
        if c.startswith("w_") or c in {"n_raw_batch", "n_norm", "n_after_policy", "gross_weight", "active_symbols"}:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)

    ret_cols = [c for c in merged.columns if str(c).startswith("ret_")]
    gross_ret = pd.Series(0.0, index=merged.index, dtype="float64")
    for c in ret_cols:
        sym = c.replace("ret_", "", 1)
        wcol = f"w_{sym_key(sym)}"
        if wcol in merged.columns:
            gross_ret = gross_ret + pd.to_numeric(merged[wcol], errors="coerce").fillna(0.0) * pd.to_numeric(merged[c], errors="coerce").fillna(0.0)

    wcols = [c for c in weights_df.columns if c.startswith("w_")]
    if wcols:
        wd = weights_df[wcols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        turnover = wd.diff().abs()
        if len(turnover):
            turnover.iloc[0] = wd.iloc[0].abs()
        turnover = turnover.sum(axis=1)
    else:
        turnover = pd.Series(0.0, index=weights_df.index)

    merged["turnover"] = pd.to_numeric(turnover, errors="coerce").fillna(0.0)

    if "execution_cost_rate" in merged.columns:
        merged["execution_cost_rate"] = pd.to_numeric(merged["execution_cost_rate"], errors="coerce").fillna(0.0)
    else:
        merged["execution_cost_rate"] = 0.0

    merged["gross_port_ret"] = gross_ret
    merged["execution_cost_drag_pct"] = merged["turnover"] * merged["execution_cost_rate"]
    merged["port_ret"] = merged["gross_port_ret"] - merged["execution_cost_drag_pct"]

    equity = pd.Series(1000.0, index=merged.index, dtype="float64")
    for i in range(1, len(merged)):
        equity.iloc[i] = equity.iloc[i - 1] * (1.0 + float(merged.loc[i, "port_ret"]))
    merged["equity"] = equity

    metrics = compute_metrics(merged["port_ret"], merged["equity"])
    metrics["execution_turnover_sum"] = float(pd.to_numeric(merged["turnover"], errors="coerce").fillna(0.0).sum())
    metrics["total_execution_cost_drag_pct"] = float(pd.to_numeric(merged["execution_cost_drag_pct"], errors="coerce").fillna(0.0).sum())
    metrics["active_symbols_mean"] = float(pd.to_numeric(merged["active_symbols"], errors="coerce").fillna(0.0).mean())
    metrics["gross_weight_mean"] = float(pd.to_numeric(merged["gross_weight"], errors="coerce").fillna(0.0).mean())
    metrics["zero_weight_rows"] = int((pd.to_numeric(merged["gross_weight"], errors="coerce").fillna(0.0) <= 1e-12).sum())

    out_csv = Path(f"results/backtest_{args.name}.csv")
    merged.to_csv(out_csv, index=False)
    out_json = Path(f"results/backtest_metrics_{args.name}.json")
    out_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    print(f"saved: {out_csv}")
    print(f"saved: {out_json}")

    print("\n=== CLEAN METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if legacy_metrics_path.exists():
        legacy = json.loads(legacy_metrics_path.read_text())
        keys = [
            "total_return_pct",
            "sharpe_annual",
            "max_drawdown_pct",
            "execution_turnover_sum",
            "total_execution_cost_drag_pct",
            "equity_final",
            "win_rate_pct",
        ]
        print("\n=== LEGACY METRICS ===")
        for k in keys:
            print(f"{k}: {legacy.get(k)}")

        print("\n=== DIFF CLEAN - LEGACY ===")
        for k in keys:
            lv = legacy.get(k)
            vv = metrics.get(k)
            if isinstance(lv, (int, float)) and isinstance(vv, (int, float)):
                print(f"{k}: {vv - lv}")
            else:
                print(f"{k}: n/a")


if __name__ == "__main__":
    main()
