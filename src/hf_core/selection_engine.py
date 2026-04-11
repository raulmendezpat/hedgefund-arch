from __future__ import annotations

import math
import pandas as pd


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def _pct_rank(s: pd.Series) -> pd.Series:
    s = _num(s)
    if len(s) <= 1:
        return pd.Series([1.0] * len(s), index=s.index, dtype=float)
    return s.rank(method="average", pct=True).astype(float)


def compute_enhanced_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_defaults = {
        "symbol": "",
        "strategy_id": "",
        "side": "flat",
        "strength": 0.0,
        "p_win": 0.0,
        "base_weight": 0.0,
        "competitive_score": 0.0,
        "post_ml_score": 0.0,
    }
    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default

    df["side"] = df["side"].astype(str).str.lower().fillna("flat")
    df = df[df["side"].isin(["long", "short"])].copy()

    if df.empty:
        df["enhanced_score"] = []
        df["accept_ranked"] = []
        return df

    df["p_win"] = _num(df["p_win"])
    df["strength"] = _num(df["strength"])
    df["base_weight"] = _num(df["base_weight"]).abs()
    df["competitive_score"] = _num(df["competitive_score"])
    df["post_ml_score"] = _num(df["post_ml_score"])

    out = []

    for side, g in df.groupby("side", sort=False):
        g = g.copy()

        g["pwin_rank"] = _pct_rank(g["p_win"])
        g["postml_rank"] = _pct_rank(g["post_ml_score"])
        g["comp_rank"] = _pct_rank(g["competitive_score"])
        g["strength_rank"] = _pct_rank(g["strength"])
        g["weight_rank"] = _pct_rank(g["base_weight"])

        g["quality_core"] = (
            0.42 * g["pwin_rank"] +
            0.33 * g["postml_rank"] +
            0.15 * g["comp_rank"] +
            0.07 * g["strength_rank"] +
            0.03 * g["weight_rank"]
        )

        g["consistency_bonus"] = (
            (g["pwin_rank"] * g["postml_rank"]) ** 0.5
        )

        g["enhanced_score"] = (
            0.80 * g["quality_core"] +
            0.20 * g["consistency_bonus"]
        )

        g["side_count"] = int(len(g))
        g["side_rank_desc"] = g["enhanced_score"].rank(ascending=False, method="first").astype(int)
        g["side_rank_pct"] = g["enhanced_score"].rank(ascending=False, pct=True, method="first").astype(float)

        out.append(g)

    out_df = pd.concat(out, ignore_index=False).sort_index()
    return out_df


def apply_cross_sectional_ranking(
    df: pd.DataFrame,
    top_pct: float = 0.20,
) -> pd.DataFrame:
    df = df.copy()

    if df.empty:
        df["accept_ranked"] = []
        return df

    if "enhanced_score" not in df.columns:
        df = compute_enhanced_score(df)

    accepted_idx = []

    for side, g in df.groupby("side", sort=False):
        g = g.copy()
        n = len(g)

        if n <= 0:
            continue

        top_k = max(1, int(math.ceil(n * float(top_pct))))
        score_floor = float(g["enhanced_score"].quantile(max(0.0, 1.0 - float(top_pct))))

        # Single dynamic gate:
        # accept only the top_pct by enhanced_score / side.
        # No extra fixed p_win / post_ml quantile floors here.
        keep = (
            (g["side_rank_desc"] <= top_k) &
            (g["enhanced_score"] >= score_floor)
        )

        if not keep.any():
            keep = g["side_rank_desc"] == 1

        accepted_idx.extend(g.index[keep].tolist())

    df["accept_ranked"] = df.index.isin(accepted_idx)
    return df
