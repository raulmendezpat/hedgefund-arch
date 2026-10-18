from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from hf.data.ohlcv import _cache_path, drop_unclosed_candles
from hf.pipeline.run_portfolio import _adx, _atr, _ema
from hf_core.ml.feature_expansion import build_symbol_feature_frame, merge_cross_asset_features

BTC_SYMBOL = "BTC/USDT:USDT"


def load_closed_bars(
    symbol: str,
    *,
    cache_dir: str = "data/cache",
    exchange: str = "binanceusdm",
    timeframe: str = "1h",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Read the OHLCV cache only (no network), keeping closed candles, indexed by UTC bar open."""
    path = _cache_path(cache_dir, exchange, symbol, timeframe)
    df = pd.read_csv(path)
    df["timestamp"] = df["timestamp"].astype("int64")
    df = drop_unclosed_candles(df, timeframe)
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("ts")
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    return df


def build_feature_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """Bars plus the same point-in-time features research_runtime feeds the signal engines.

    Every value at row t uses bars <= t only (rolling/ewm windows, no forward shifts).
    """
    out = bars.copy()
    close = out["close"].astype(float)
    atr = _atr(out, 14)
    out["adx"] = _adx(out, 14)
    out["atr"] = atr
    out["atrp"] = atr / close.replace(0.0, np.nan)
    out["ema_fast"] = _ema(close, 20)
    out["ema_slow"] = _ema(close, 200)
    extra = build_symbol_feature_frame(out)
    for c in extra.columns:
        if c not in out.columns:
            out[c] = extra[c]
    return out


def build_symbol_frames(bars_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    frames = {sym: build_feature_frame(df) for sym, df in bars_by_symbol.items()}
    if BTC_SYMBOL in frames:
        btc = frames[BTC_SYMBOL]
        for sym in list(frames):
            frames[sym] = merge_cross_asset_features(frames[sym], btc, prefix="btc_")
    return frames
