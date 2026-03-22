from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.astype(float) / b.astype(float).replace(0.0, np.nan)


def build_symbol_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_index()

    close = x["close"].astype(float)
    high = x["high"].astype(float)
    low = x["low"].astype(float)

    out = pd.DataFrame(index=x.index)

    out["ret_1h_lag"] = close.pct_change(1)
    out["ret_4h_lag"] = close.pct_change(4)
    out["ret_12h_lag"] = close.pct_change(12)
    out["ret_24h_lag"] = close.pct_change(24)

    ema_fast = x["ema_fast"].astype(float) if "ema_fast" in x.columns else close.ewm(span=20, adjust=False).mean()
    ema_slow = x["ema_slow"].astype(float) if "ema_slow" in x.columns else close.ewm(span=200, adjust=False).mean()

    out["ema_gap_fast_slow"] = _safe_div(ema_fast - ema_slow, ema_slow)
    out["dist_close_ema_fast"] = _safe_div(close - ema_fast, ema_fast)
    out["dist_close_ema_slow"] = _safe_div(close - ema_slow, ema_slow)

    out["range_pct"] = _safe_div(high - low, close)
    out["rolling_vol_24h"] = close.pct_change().rolling(24, min_periods=12).std()
    out["rolling_vol_72h"] = close.pct_change().rolling(72, min_periods=24).std()

    atrp = x["atrp"].astype(float) if "atrp" in x.columns else pd.Series(0.0, index=x.index)
    atrp_mean_72 = atrp.rolling(72, min_periods=24).mean()
    atrp_std_72 = atrp.rolling(72, min_periods=24).std()
    out["atrp_zscore"] = (atrp - atrp_mean_72) / atrp_std_72.replace(0.0, np.nan)

    hh_24 = high.rolling(24, min_periods=12).max()
    ll_24 = low.rolling(24, min_periods=12).min()
    out["breakout_distance_up"] = _safe_div(hh_24 - close, close)
    out["breakout_distance_down"] = _safe_div(close - ll_24, close)

    mid_24 = (hh_24 + ll_24) / 2.0
    out["pullback_depth"] = _safe_div(close - mid_24, close)

    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return out


def merge_cross_asset_features(
    target_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    prefix: str = "btc_",
) -> pd.DataFrame:
    out = target_df.copy()
    btc = btc_df.copy()

    keep = [c for c in ["ret_24h_lag", "rolling_vol_24h", "atrp", "adx"] if c in btc.columns]
    rename = {c: f"{prefix}{c}" for c in keep}
    btc = btc[keep].rename(columns=rename)

    out = out.join(btc, how="left")
    return out
