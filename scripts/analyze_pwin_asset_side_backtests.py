from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")

CASES = [
    ("pwin_cmp_1m_baseline", "pwin_cmp_1m_asset_side_pwin"),
    ("pwin_cmp_3m_baseline", "pwin_cmp_3m_asset_side_pwin"),
    ("pwin_cmp_6m_baseline", "pwin_cmp_6m_asset_side_pwin"),
]

METRIC_KEYS = [
    "total_return_pct",
    "sharpe_annual",
    "max_drawdown_pct",
    "trade_count",
    "win_rate_pct",
    "avg_active_symbols",
    "avg_gross_weight",
]


def load_metrics(name: str) -> dict:
    p = RESULTS_DIR / f"research_runtime_metrics_{name}.json"
    if not p.exists():
        return {"name": name, "missing": True}
    d = json.loads(p.read_text())
    d["name"] = name
    return d


def load_candidates(name: str) -> pd.DataFrame:
    p = RESULTS_DIR / f"research_runtime_candidates_{name}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    if "trace_candidate_id" in df.columns:
        df["trace_candidate_id"] = df["trace_candidate_id"].fillna("").astype(str).str.strip()
    return df


def summarize_override(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    out = {"rows": int(len(df))}
    for c in [
        "p_win_asset_side_override_enabled",
        "p_win_asset_side_override_applied",
    ]:
        if c in df.columns:
            out[c] = int(df[c].fillna(False).astype(bool).sum())

    if "p_win_ml_raw_pre_asset_side_override" in df.columns and "p_win_ml_raw_post_asset_side_override" in df.columns:
        a = pd.to_numeric(df["p_win_ml_raw_pre_asset_side_override"], errors="coerce")
        b = pd.to_numeric(df["p_win_ml_raw_post_asset_side_override"], errors="coerce")
        diff = (b - a).abs()
        out["override_changed_rows"] = int((diff > 1e-12).fillna(False).sum())
        out["mean_abs_pwin_delta"] = float(diff.mean())

    if "p_win_asset_side_override_group_key" in df.columns:
        vc = df["p_win_asset_side_override_group_key"].fillna("").astype(str)
        vc = vc[vc != ""].value_counts()
        out["top_groups"] = vc.head(10).to_dict()

    return out


def main() -> None:
    rows = []

    for base_name, new_name in CASES:
        base = load_metrics(base_name)
        new = load_metrics(new_name)

        row = {
            "baseline": base_name,
            "new": new_name,
        }

        for k in METRIC_KEYS:
            row[f"base_{k}"] = base.get(k)
            row[f"new_{k}"] = new.get(k)

        for k in ["total_return_pct", "sharpe_annual", "max_drawdown_pct", "win_rate_pct"]:
            try:
                row[f"delta_{k}"] = float(new.get(k)) - float(base.get(k))
            except Exception:
                row[f"delta_{k}"] = None

        rows.append(row)

        print()
        print("=" * 120)
        print(base_name, "vs", new_name)
        print("=" * 120)
        for k in METRIC_KEYS:
            print(f"{k}: base={base.get(k)} | new={new.get(k)}")

        cand_new = load_candidates(new_name)
        summary = summarize_override(cand_new)

        print()
        print("override summary:")
        for k, v in summary.items():
            print(f"{k}: {v}")

        if not cand_new.empty and "p_win_asset_side_override_applied" in cand_new.columns:
            applied = cand_new[cand_new["p_win_asset_side_override_applied"].fillna(False).astype(bool)].copy()
            show_cols = [c for c in [
                "ts","symbol","strategy_id","side","p_win",
                "p_win_ml_raw_pre_asset_side_override",
                "p_win_ml_raw_post_asset_side_override",
                "p_win_asset_side_override_model_name",
                "p_win_asset_side_override_group_key",
                "selected_final",
            ] if c in applied.columns]
            if len(applied):
                print()
                print("applied sample:")
                print(applied[show_cols].head(20).to_string(index=False))

    out = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "pwin_asset_side_backtest_comparison_v1.csv"
    out.to_csv(out_csv, index=False)
    print()
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
