from __future__ import annotations

import numpy as np
import pandas as pd

# Raw point-in-time columns carried on each labeled candidate (see build_ev_label_dataset.FEATURE_COLS).
UNSIGNED = [
    "adx", "atrp", "atrp_zscore", "range_pct", "rolling_vol_24h", "rolling_vol_72h",
    "btc_rolling_vol_24h", "btc_atrp", "btc_adx",
]
# Directional columns: multiplied by +1 for longs and -1 for shorts, so "positive" always
# means "in favour of the trade" and one model can serve both sides.
SIGNED = [
    "ret_1h_lag", "ret_4h_lag", "ret_12h_lag", "ret_24h_lag",
    "ema_gap_fast_slow", "dist_close_ema_fast", "dist_close_ema_slow",
    "pullback_depth", "btc_ret_24h_lag",
]

MODEL_FEATURES = (
    UNSIGNED
    + [f"sgn_{c}" for c in SIGNED]
    + [
        "is_long",
        "room_to_extreme",
        "risk_frac",
        "expected_cost_R",
        "xs_n_candidates",
        "xs_long_share",
        "xs_rank_sgn_ret_24h",
        "xs_rank_sgn_ema_gap",
        "xs_rank_atrp",
        "hour_utc",
    ]
)


def build_model_frame(data: pd.DataFrame, *, round_trip_cost: float, stop_atr_mult_col: str = "stop_atr_mult") -> pd.DataFrame:
    """Model features from labeled candidates. Uses only information known at the signal bar close.

    risk_frac/expected_cost_R come from the exit policy's stop distance and the ATR at entry,
    both known before the order is sent.
    """
    x = pd.DataFrame(index=data.index)
    sign = np.where(data["side"].eq("long"), 1.0, -1.0)
    for c in UNSIGNED:
        x[c] = pd.to_numeric(data[c], errors="coerce")
    for c in SIGNED:
        x[f"sgn_{c}"] = sign * pd.to_numeric(data[c], errors="coerce")

    x["is_long"] = (sign > 0).astype(float)
    # Distance to the 24h extreme in the trade direction (room left before the recent high/low).
    x["room_to_extreme"] = np.where(sign > 0, data["breakout_distance_up"], data["breakout_distance_down"])
    stop_mult = pd.to_numeric(data[stop_atr_mult_col], errors="coerce")
    x["risk_frac"] = stop_mult * pd.to_numeric(data["atrp"], errors="coerce")
    x["expected_cost_R"] = round_trip_cost / x["risk_frac"].replace(0.0, np.nan)

    # Cross-sectional context among all candidates generated at the same bar.
    g = data.groupby("timestamp")
    x["xs_n_candidates"] = g["side"].transform("size").astype(float)
    x["xs_long_share"] = g["side"].transform(lambda s: s.eq("long").mean())
    tmp = pd.DataFrame({"ts": data["timestamp"], "r": x["sgn_ret_24h_lag"], "e": x["sgn_ema_gap_fast_slow"], "a": x["atrp"]})
    gt = tmp.groupby("ts")
    x["xs_rank_sgn_ret_24h"] = gt["r"].rank(pct=True)
    x["xs_rank_sgn_ema_gap"] = gt["e"].rank(pct=True)
    x["xs_rank_atrp"] = gt["a"].rank(pct=True)

    x["hour_utc"] = pd.to_datetime(data["timestamp"], unit="ms", utc=True).dt.hour.astype(float)
    return x[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
