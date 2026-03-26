from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd

from .contracts import DatasetBundle, SplitBundle


DEFAULT_NUMERIC_FEATURES = [
    "signal_strength",
    "base_weight",
    "score",
    "policy_score",
    "expected_return",
    "adx",
    "atrp",
    "rsi",
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
    "portfolio_breadth",
    "portfolio_avg_pwin",
    "portfolio_avg_atrp",
    "portfolio_avg_strength",
    "portfolio_conviction",
    "p_win_math_v1",
    "p_win_hybrid_v1",
]

DEFAULT_CATEGORICAL_FEATURES = [
    "strategy_id",
    "symbol",
    "side",
    "band",
    "reason",
    "policy_profile",
    "portfolio_regime",
]


def _to_utc_dt(series: pd.Series) -> pd.Series:
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="ms", utc=True, errors="coerce")

    s_num = pd.to_numeric(s, errors="coerce")
    share_numeric = float(s_num.notna().mean()) if len(s_num) else 0.0
    if share_numeric >= 0.95:
        return pd.to_datetime(s_num, unit="ms", utc=True, errors="coerce")

    return pd.to_datetime(s, utc=True, errors="coerce")


def _ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)
    return out


def _ensure_categorical(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = ""
        out[c] = out[c].fillna("").astype(str).str.strip()
    return out


def build_runtime_trade_dataset(
    candidates_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    *,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> DatasetBundle:
    numeric_features = list(numeric_features or DEFAULT_NUMERIC_FEATURES)
    categorical_features = list(categorical_features or DEFAULT_CATEGORICAL_FEATURES)

    cand = candidates_df.copy()
    trd = trades_df.copy()

    cand["ts_dt"] = _to_utc_dt(cand["ts"])
    trd["entry_dt"] = _to_utc_dt(trd["entry_ts"])

    for c in ["symbol", "strategy_id", "side"]:
        cand[c] = cand[c].fillna("").astype(str).str.strip()
        trd[c] = trd[c].fillna("").astype(str).str.strip()

    if "side" in cand.columns:
        cand["side"] = cand["side"].str.lower()
    if "side" in trd.columns:
        trd["side"] = trd["side"].str.lower()

    join_left = ["strategy_id", "symbol", "side", "ts_dt"]
    join_right = ["strategy_id", "symbol", "side", "entry_dt"]

    merged = cand.merge(
        trd,
        left_on=join_left,
        right_on=join_right,
        how="inner",
        suffixes=("_cand", "_trade"),
    ).copy()

    if merged.empty:
        raise RuntimeError("Join candidates->trades devolvió 0 filas.")

    merged["pnl"] = pd.to_numeric(merged["pnl"], errors="coerce").fillna(0.0)
    merged["is_win"] = (merged["pnl"] > 0.0).astype(int)

    merged = _ensure_numeric(merged, numeric_features)
    merged = _ensure_categorical(merged, categorical_features)

    keep_cols = (
        ["entry_dt", "pnl", "is_win", "symbol", "strategy_id", "side"]
        + numeric_features
        + categorical_features
    )
    keep_cols = list(dict.fromkeys(keep_cols))

    df = merged[keep_cols].copy()
    df = df.dropna(subset=["entry_dt"]).sort_values("entry_dt").reset_index(drop=True)

    return DatasetBundle(
        df=df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        key_cols=["symbol", "side"],
    )


def time_split(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.70,
    valid_frac: float = 0.15,
) -> SplitBundle:
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac inválido")
    if not 0.0 < valid_frac < 1.0:
        raise ValueError("valid_frac inválido")
    if train_frac + valid_frac >= 1.0:
        raise ValueError("train_frac + valid_frac debe ser < 1.0")

    x = df.sort_values("entry_dt").reset_index(drop=True).copy()
    n = len(x)
    if n < 60:
        raise RuntimeError(f"Dataset demasiado pequeño para split robusto: {n}")

    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + valid_frac))

    train_df = x.iloc[:i1].copy()
    valid_df = x.iloc[i1:i2].copy()
    test_df = x.iloc[i2:].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        raise RuntimeError("Split temporal produjo una partición vacía.")

    return SplitBundle(train_df=train_df, valid_df=valid_df, test_df=test_df)
