from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATES_CSV = "results/research_runtime_candidates_diag_features_v1.csv"
OUT_DIR = "artifacts/symbol_similarity"


FEATURE_COLS = [
    "signal_strength",
    "adx",
    "atrp",
    "base_weight",
    "p_win",
    "expected_return",
    "policy_score",
    "size_mult",
    "ret_1h_lag",
    "ret_4h_lag",
    "ret_12h_lag",
    "ret_24h_lag",
    "ema_gap_fast_slow",
    "dist_close_ema_fast",
    "dist_close_ema_slow",
    "range_pct",
    "rolling_vol_24h",
    "rolling_vol_72h",
    "atrp_zscore",
    "breakout_distance_up",
    "breakout_distance_down",
    "pullback_depth",
    "btc_ret_24h_lag",
    "btc_rolling_vol_24h",
    "btc_atrp",
    "btc_adx",
]

RET_COL = "ret_24h_lag"
PWIN_COL = "p_win"


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    z = pd.concat([x, y], axis=1).dropna()
    if len(z) < 20:
        return np.nan
    c = z.iloc[:, 0].corr(z.iloc[:, 1])
    if pd.isna(c):
        return np.nan
    return float(c)


def clipped_corr(v: float) -> float:
    if pd.isna(v):
        return 0.0
    return float(max(-1.0, min(1.0, v)))


def build_feature_profile(df_sym: pd.DataFrame) -> pd.Series:
    out = {}
    for c in FEATURE_COLS:
        if c not in df_sym.columns:
            continue
        s = pd.to_numeric(df_sym[c], errors="coerce")
        out[f"{c}__mean"] = float(s.mean()) if s.notna().any() else 0.0
        out[f"{c}__std"] = float(s.std()) if s.notna().any() else 0.0
        out[f"{c}__p25"] = float(s.quantile(0.25)) if s.notna().any() else 0.0
        out[f"{c}__p50"] = float(s.quantile(0.50)) if s.notna().any() else 0.0
        out[f"{c}__p75"] = float(s.quantile(0.75)) if s.notna().any() else 0.0
    return pd.Series(out, dtype=float)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CANDIDATES_CSV)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["symbol"] = df["symbol"].astype(str)

    for c in FEATURE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    symbols = sorted(df["symbol"].dropna().unique().tolist())

    profiles = {}
    by_symbol = {}

    for sym in symbols:
        g = df[df["symbol"].eq(sym)].copy().sort_values("ts")
        by_symbol[sym] = g
        profiles[sym] = build_feature_profile(g)

    profile_df = pd.DataFrame(profiles).T.fillna(0.0)

    rows = []
    for a in symbols:
        for b in symbols:
            ga = by_symbol[a]
            gb = by_symbol[b]

            feat_sim = cosine_similarity(
                profile_df.loc[a].to_numpy(dtype=float),
                profile_df.loc[b].to_numpy(dtype=float),
            )

            ret_corr = safe_corr(
                ga.set_index("ts")[RET_COL] if RET_COL in ga.columns else pd.Series(dtype=float),
                gb.set_index("ts")[RET_COL] if RET_COL in gb.columns else pd.Series(dtype=float),
            )

            pwin_corr = safe_corr(
                ga.set_index("ts")[PWIN_COL] if PWIN_COL in ga.columns else pd.Series(dtype=float),
                gb.set_index("ts")[PWIN_COL] if PWIN_COL in gb.columns else pd.Series(dtype=float),
            )

            score = (
                0.50 * clipped_corr(feat_sim) +
                0.30 * clipped_corr(pwin_corr) +
                0.20 * clipped_corr(ret_corr)
            )

            rows.append({
                "symbol_a": a,
                "symbol_b": b,
                "feature_similarity": float(feat_sim),
                "pwin_corr": None if pd.isna(pwin_corr) else float(pwin_corr),
                "return_corr": None if pd.isna(ret_corr) else float(ret_corr),
                "similarity_score": float(score),
            })

    sim_df = pd.DataFrame(rows).sort_values(["symbol_a", "similarity_score"], ascending=[True, False])

    matrix = sim_df.pivot(index="symbol_a", columns="symbol_b", values="similarity_score")
    matrix_csv = out_dir / "symbol_similarity_matrix.csv"
    sim_df_csv = out_dir / "symbol_similarity_pairs.csv"

    matrix.to_csv(matrix_csv)
    sim_df.to_csv(sim_df_csv, index=False)

    best_rows = []
    fallback_map = {}

    for sym in symbols:
        g = sim_df[(sim_df["symbol_a"].eq(sym)) & (~sim_df["symbol_b"].eq(sym))].copy()
        if g.empty:
            continue
        best = g.iloc[0]
        fallback_map[sym] = {
            "fallback_symbol": str(best["symbol_b"]),
            "similarity_score": float(best["similarity_score"]),
            "feature_similarity": float(best["feature_similarity"]),
            "pwin_corr": None if pd.isna(best["pwin_corr"]) else float(best["pwin_corr"]),
            "return_corr": None if pd.isna(best["return_corr"]) else float(best["return_corr"]),
        }
        best_rows.append({
            "symbol": sym,
            **fallback_map[sym],
        })

    fallback_csv = out_dir / "symbol_fallback_map.csv"
    pd.DataFrame(best_rows).sort_values("symbol").to_csv(fallback_csv, index=False)

    fallback_json = out_dir / "symbol_fallback_map.json"
    fallback_json.write_text(json.dumps(fallback_map, indent=2), encoding="utf-8")

    print("saved:", sim_df_csv)
    print("saved:", matrix_csv)
    print("saved:", fallback_csv)
    print("saved:", fallback_json)

    print("\n=== BEST FALLBACK BY SYMBOL ===")
    if best_rows:
        print(pd.DataFrame(best_rows).sort_values("symbol").to_string(index=False))
    else:
        print("(empty)")


if __name__ == "__main__":
    main()
