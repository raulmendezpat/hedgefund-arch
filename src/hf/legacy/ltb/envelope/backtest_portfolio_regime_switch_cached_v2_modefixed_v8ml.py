#!/usr/bin/env python3
"""

[V6 PATCH] Optional V6 logic enabled in simulate_trend/simulate_bbrsi (ATR% filters, BB width, RSI turn, cooldown, BE/trailing).
Portfolio backtest (V8ML-enabled): BTC Trend (EMA pullback + ADX + ATR TP/SL) + SOL BBRSI (mean reversion)
- Loads params from bot files (AST parse `params = {...}`)
- Fetches OHLCV via ccxt (Bitget public)
- Runs each strategy on its own sub-balance (weights) and then combines equity curves
- Outputs:
  results/portfolio_trades_<name>.csv (combined trade log)
  results/portfolio_equity_<name>.csv (combined equity curve)
  results/portfolio_monthly_<name>.csv (month-by-month returns)
- Supports leverage override per bot to test 1x/2x/3x without editing bot files.
"""

from __future__ import annotations
import os
import argparse, ast
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import joblib  # type: ignore
except Exception:
    joblib = None

MODEL_DIR = Path(__file__).resolve().parents[1] / 'models'

# ----------------- V8ML gate helpers -----------------
_ML_MODEL_CACHE: Dict[str, Dict] = {}

def _load_ml_bundle(model_path: str) -> Optional[Dict]:
    if not model_path:
        return None
    if model_path in _ML_MODEL_CACHE:
        return _ML_MODEL_CACHE[model_path]
    if joblib is None:
        return None
    p = Path(model_path)
    if not p.exists():
        return None
    try:
        bundle = joblib.load(p)
        if isinstance(bundle, dict):
            _ML_MODEL_CACHE[model_path] = bundle
            return bundle
        return None
    except Exception:
        return None

def _ml_pwin(bundle: Dict, X: np.ndarray, w_lr: float, w_rf: float) -> float:
    # X shape (1, n)
    p_lr = None
    p_rf = None
    try:
        lr = bundle.get("lr")
        if lr is not None:
            p_lr = float(lr.predict_proba(X)[0, 1])
    except Exception:
        p_lr = None
    try:
        rf = bundle.get("rf")
        if rf is not None:
            p_rf = float(rf.predict_proba(X)[0, 1])
    except Exception:
        p_rf = None

    # If only one model is available, use it.
    if p_lr is None and p_rf is None:
        return float("nan")
    if p_lr is None:
        return p_rf  # type: ignore
    if p_rf is None:
        return p_lr

    wsum = max(1e-9, float(w_lr) + float(w_rf))
    return (float(w_lr) * p_lr + float(w_rf) * p_rf) / wsum

def _ml_gate_pass(params: Dict, feature_vec: List[float]) -> bool:
    mf = params.get("ml_filter", None)
    if not mf or not mf.get("enabled", False):
        return True

    model_path = str(mf.get("model_path", "")).strip()
    threshold = float(mf.get("threshold", 0.5))
    w_lr = float(mf.get("w_lr", 0.5))
    w_rf = float(mf.get("w_rf", 0.5))
    fail_safe = str(mf.get("fail_safe", "skip_entries"))

    bundle = _load_ml_bundle(model_path)
    if bundle is None:
        # mimic prod behavior
        return False if fail_safe == "skip_entries" else True

    X = np.asarray(feature_vec, dtype=float).reshape(1, -1)
    pwin = _ml_pwin(bundle, X, w_lr, w_rf)
    if not np.isfinite(pwin):
        return False if fail_safe == "skip_entries" else True

    passed = (pwin >= threshold)

    # ---- debug dump (local) ----
    try:
        from pathlib import Path as _P
        _P("results").mkdir(exist_ok=True)
        with open("results/ml_gate_pwin_dump.csv", "a") as f:
            # pwin, threshold, passed(0/1), n_features
            f.write(f"{pwin:.10f},{threshold:.6f},{1 if passed else 0},{X.shape[1]}\n")
    except Exception:
        pass
    # ----------------------------

    return passed

try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None


# ----------------- OHLCV disk cache -----------------
def _sanitize_symbol(symbol: str) -> str:
    # e.g. 'BTC/USDT:USDT' -> 'BTC_USDT_USDT'
    return symbol.replace("/", "_").replace(":", "_")

def _cache_path(cache_dir: str, exchange_id: str, symbol: str, timeframe: str) -> Path:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    fname = f"{exchange_id}__{_sanitize_symbol(symbol)}__{timeframe}.csv.gz"
    return cache_root / fname

def _load_cached_ohlcv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        # defensive: ensure correct columns/types
        need = ["timestamp","open","high","low","close","volume"]
        for c in need:
            if c not in df.columns:
                return None
        df["timestamp"] = df["timestamp"].astype("int64")
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df[need]
    except Exception:
        return None

def _save_cached_ohlcv(path: Path, df: pd.DataFrame) -> None:
    try:
        out = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        out.to_csv(path, index=False, compression="gzip")
    except Exception:
        # cache is best-effort
        pass

def _merge_ohlcv(a: Optional[pd.DataFrame], b: pd.DataFrame) -> pd.DataFrame:
    if a is None or a.empty:
        return b.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out = pd.concat([a, b], ignore_index=True)
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out


# ----------------- helpers -----------------

def compute_dynamic_weights(
    btc_regime_on: bool,
    sol_regime_on: bool,
    both_btc_weight: float = 0.75,
    prev_w_btc: float = 1.0,
    prev_w_sol: float = 0.0,
    sol_min_weight_off: float = 0.0,
):
    """Return (w_btc, w_sol) per candle.

    Rules:
    - BTC on, SOL off -> 100% BTC
    - BTC off, SOL on -> 100% SOL
    - both on -> both_btc_weight / (1-both_btc_weight)
    - both off -> keep previous weights (sticky)
    """
    if btc_regime_on and (not sol_regime_on):
        return 1.0, 0.0
    if (not btc_regime_on) and sol_regime_on:
        return 0.0, 1.0
    if btc_regime_on and sol_regime_on:
        w_btc = float(both_btc_weight)
        return w_btc, 1.0 - w_btc
    # both off
    # Optional floor: allow SOL to keep a minimum weight even when sol_regime_on is False.
    # This prevents alloc_w=0 for SOL trades when signals still exist.
    sol_floor = max(0.0, min(1.0, float(sol_min_weight_off)))
    if sol_floor > 0.0:
        ws = max(float(prev_w_sol), sol_floor)
        ws = min(ws, 1.0)
        return 1.0 - ws, ws
    return float(prev_w_btc), float(prev_w_sol)

def parse_params_from_bot_file(bot_file: str) -> Dict:
    src = Path(bot_file).read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(src, filename=bot_file)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "params":
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, dict):
                        raise ValueError("params is not a dict")
                    return value
    raise ValueError(f"Could not find `params = {{...}}` in {bot_file}")

def to_ms(ts: str) -> int:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def fetch_ohlcv_ccxt(symbol: str, timeframe: str, start_ms: int, end_ms: Optional[int], exchange_id: str = "bitget",
                    cache_dir: str = ".cache/ohlcv", use_cache: bool = True, refresh_if_no_end: bool = False) -> pd.DataFrame:
    """
    Fetch OHLCV via ccxt with an optional on-disk cache.

    Cache behavior:
    - Cache is keyed by (exchange_id, symbol, timeframe) and stored as CSV.GZ under cache_dir.
    - If cached data fully covers [start_ms, end_ms] it is used directly.
    - If partially covered, only the missing range(s) are fetched and the cache is updated.
    - If end_ms is None (open-ended), cached data is used and only refreshed to latest if refresh_if_no_end=True.
    """
    if ccxt is None:
        raise RuntimeError("ccxt not installed. Install with: pip install ccxt")

    cache_path = _cache_path(cache_dir, exchange_id, symbol, timeframe) if use_cache else None
    cached = _load_cached_ohlcv(cache_path) if (use_cache and cache_path is not None) else None

    desired_start = int(start_ms)
    desired_end = int(end_ms) if end_ms is not None else None

    # Decide what to fetch (if anything)
    fetch_ranges = []  # list of (since, until) where until can be None
    if cached is None or cached.empty:
        fetch_ranges.append((desired_start, desired_end))
    else:
        have_min = int(cached["timestamp"].min())
        have_max = int(cached["timestamp"].max())

        if desired_start < have_min:
            fetch_ranges.append((desired_start, have_min - 1))

        if desired_end is not None:
            if desired_end > have_max:
                fetch_ranges.append((have_max + 1, desired_end))
        else:
            if refresh_if_no_end:
                fetch_ranges.append((have_max + 1, None))

    # Fetch missing data
    if fetch_ranges:
        ex_class = getattr(ccxt, exchange_id)
        ex = ex_class({"enableRateLimit": True})
        limit = 200
        all_new = []
        for since, until in fetch_ranges:
            rows = []
            cur = int(since)
            while True:
                batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cur, limit=limit)
                if not batch:
                    break
                rows.extend(batch)
                last = int(batch[-1][0])
                if last == cur:
                    break
                cur = last + 1
                if until is not None and cur >= int(until):
                    break
            if rows:
                df_new = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
                df_new = df_new.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                if until is not None:
                    df_new = df_new[df_new["timestamp"] <= int(until)].reset_index(drop=True)
                all_new.append(df_new)

        if all_new:
            new_df = pd.concat(all_new, ignore_index=True)
            new_df = new_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            cached = _merge_ohlcv(cached, new_df)
            if use_cache and cache_path is not None:
                _save_cached_ohlcv(cache_path, cached)

    if cached is None:
        cached = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    out = cached
    # Slice to requested range
    out = out[out["timestamp"] >= desired_start]
    if desired_end is not None:
        out = out[out["timestamp"] <= desired_end]
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def rolling_std(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).std(ddof=0)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_s)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_s)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()

# ----------------- trade record -----------------
@dataclass
class Trade:
    strategy: str
    symbol: str
    side: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    reason: str


    # --- regime3 audit fields (entry snapshot) ---
    adx: float = float("nan")
    atrp: float = float("nan")
    slope_atr: float = float("nan")
# ----------------- BBRSI simulation (single entry) -----------------
def simulate_bbrsi(df: pd.DataFrame, params: Dict, initial_balance: float, maker_fee: float, taker_fee: float, leverage_override: Optional[float]=None):
    """BBRSI mean-reversion simulator with optional V6 filters (ATR%, BB width, RSI turn) + cooldown after close.

    Notes:
    - Backtest engine reads `params` from bot file; this function implements the V6 behavior so results differ from baseline.
    - Cooldown is expressed in candles here (1 candle ~= 1 run in production for 1H).
    """
    df = df.copy()
    close = df["close"].astype(float)

    bb_period = int(params.get("bb_period", 20))
    bb_std = float(params.get("bb_std", 2.0))
    rsi_period = int(params.get("rsi_period", 14))
    rsi_long_max = float(params.get("rsi_long_max", 35))
    rsi_short_min = float(params.get("rsi_short_min", 65))

    adx_period = int(params.get("adx_period", 14))
    # v4 used adx_max; v6 uses adx_soft/adx_hard sizing gate (if present)
    adx_max = float(params.get("adx_max", 18))
    adx_soft = params.get("adx_soft", None)
    adx_hard = params.get("adx_hard", None)

    atr_period = int(params.get("atr_period", 14))
    stop_atr_mult = float(params.get("stop_atr_mult", 1.2))

    # V6 filters (optional)
    atrp_min = params.get("atrp_min", None)
    atrp_max = params.get("atrp_max", None)
    bb_width_min = params.get("bb_width_min", None)
    bb_width_max = params.get("bb_width_max", None)
    require_rsi_turn = bool(params.get("require_rsi_turn", False))
    cooldown_after_close = int(params.get("cooldown_after_close_runs", 0))

    leverage = float(params.get("leverage", 1))
    if leverage_override is not None:
        leverage = float(leverage_override)

    pos_pct = float(params.get("position_size_percentage", 20)) / 100.0
    use_longs = bool(params.get("use_longs", True))
    use_shorts = bool(params.get("use_shorts", True))
    symbol = params.get("symbol", "")

    basis = sma(close, bb_period)
    sd = rolling_std(close, bb_period)
    bb_up = basis + bb_std * sd
    bb_low = basis - bb_std * sd
    rs = rsi(close, rsi_period)
    ax = adx(df, adx_period)
    at = atr(df, atr_period)

    df["basis"]=basis; df["bb_up"]=bb_up; df["bb_low"]=bb_low; df["rsi"]=rs; df["adx"]=ax; df["atr"]=at

    balance = initial_balance
    pos_side: Optional[str] = None
    entry_adx=float("nan"); entry_atrp=float("nan")
    entry_px = 0.0
    entry_ts_open = 0
    qty = 0.0
    entry_atr = 0.0

    cooldown_left = 0

    trades: List[Trade] = []
    equity_rows = []

    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        row = df.iloc[i]
        ts = int(row["timestamp"])
        h = float(row["high"]); l = float(row["low"]); c = float(row["close"])

        mtm=0.0
        if pos_side=="long":
            mtm=(c-entry_px)*qty
        elif pos_side=="short":
            mtm=(entry_px-c)*qty
        equity_rows.append({"timestamp": ts, "equity": balance+mtm, "balance": balance, "pos_side": pos_side or "", "qty": qty})

        if any(prev[k]!=prev[k] for k in ["basis","bb_up","bb_low","rsi","adx","atr"]):
            continue

        # manage open position
        if pos_side is not None:
            tp = float(prev["basis"])
            if pos_side=="long":
                sl = entry_px - stop_atr_mult * entry_atr
                if l <= sl <= h:
                    exit_reason="sl"; exit_price=sl
                elif l <= tp <= h:
                    exit_reason="tp"; exit_price=tp
                else:
                    continue
                fee = exit_price*qty*taker_fee
                pnl = (exit_price-entry_px)*qty - fee
                balance += (exit_price-entry_px)*qty
                balance -= fee
                trades.append(Trade("bbrsi", symbol, "long", entry_ts_open, ts, entry_px, exit_price, qty, pnl, exit_reason, adx=entry_adx, atrp=entry_atrp, slope_atr=float("nan")))
            else:
                sl = entry_px + stop_atr_mult * entry_atr
                if l <= sl <= h:
                    exit_reason="sl"; exit_price=sl
                elif l <= tp <= h:
                    exit_reason="tp"; exit_price=tp
                else:
                    continue
                fee = exit_price*qty*taker_fee
                pnl = (entry_px-exit_price)*qty - fee
                balance += (entry_px-exit_price)*qty
                balance -= fee
                trades.append(Trade("bbrsi", symbol, "short", entry_ts_open, ts, entry_px, exit_price, qty, pnl, exit_reason, adx=entry_adx, atrp=entry_atrp, slope_atr=float("nan")))
            pos_side=None; entry_px=0.0; qty=0.0; entry_atr=0.0; entry_ts_open=0; entry_adx=float("nan"); entry_atrp=float("nan"); entry_ts_open=0

            # V6: start cooldown after closing a position
            if cooldown_after_close > 0:
                cooldown_left = cooldown_after_close
            continue

        # no position: cooldown gate
        if cooldown_left > 0:
            cooldown_left -= 1
            continue

        # optional ATR% filter
        atr_v = float(prev["atr"])
        close_v = float(prev["close"])
        if atr_v!=atr_v or close_v<=0:
            continue
        atrp = atr_v / close_v
        if atrp_min is not None and atrp < float(atrp_min):
            continue
        if atrp_max is not None and atrp > float(atrp_max):
            continue

        # optional BB width filter
        basis_v = float(prev["basis"])
        bb_up_v = float(prev["bb_up"])
        bb_low_v = float(prev["bb_low"])
        if basis_v!=basis_v or basis_v<=0:
            continue
        bb_width = (bb_up_v - bb_low_v) / basis_v
        if bb_width_min is not None and bb_width < float(bb_width_min):
            continue
        if bb_width_max is not None and bb_width > float(bb_width_max):
            continue

        # ADX gate / sizing
        adx_v = float(prev["adx"])
        # Regime hard cap: respect adx_max even if adx_soft/adx_hard sizing is enabled
        if adx_v > adx_max:
            continue
        if adx_soft is not None and adx_hard is not None:
            adx_soft_f = float(adx_soft)
            adx_hard_f = float(adx_hard)
            if adx_v >= adx_hard_f:
                continue
            if adx_v <= adx_soft_f:
                adx_scale = 1.0
            else:
                adx_scale = max(0.0, min(1.0, 1.0 - (adx_v - adx_soft_f) / max(1e-9, (adx_hard_f - adx_soft_f))))
        else:
            if adx_v >= adx_max:
                continue
            adx_scale = 1.0

        notional = balance * pos_pct * leverage * adx_scale

        # RSI turn confirmation (optional) on signal candle (prev) vs i-2
        rsi_prevprev = None
        if require_rsi_turn and i >= 2:
            try:
                rsi_prevprev = float(df.iloc[i-2]["rsi"])
            except Exception:
                rsi_prevprev = None

        # Entries modeled as limit fill at band price if touched in current candle
        if use_longs and float(prev["close"]) <= bb_low_v and float(prev["rsi"]) <= rsi_long_max:
            if require_rsi_turn and (rsi_prevprev is not None) and (rsi_prevprev == rsi_prevprev):
                if float(prev["rsi"]) < rsi_prevprev:
                    continue
            # V8ML gate (optional): skip low-quality signals
            feature_vec = [
                float(adx_v),
                float(atrp),
                float(bb_width),
                float(prev["rsi"]),
                float(prev["rsi"] - (rsi_prevprev if (rsi_prevprev is not None and rsi_prevprev==rsi_prevprev) else float(prev["rsi"]))),
                float((float(prev["close"]) - bb_low_v) / max(1e-9, atr_v)),
                float((bb_up_v - float(prev["close"])) / max(1e-9, atr_v)),
                1.0,
            ]
            if not _ml_gate_pass(params, feature_vec):
                continue
            entry = bb_low_v
            if l <= entry <= h:
                q = notional / entry
                balance -= notional * maker_fee
                pos_side="long"; entry_px=entry; qty=q; entry_atr=atr_v; entry_ts_open=ts; entry_ts_open=ts
                entry_adx=adx_v
                entry_atrp=atrp
        elif use_shorts and float(prev["close"]) >= bb_up_v and float(prev["rsi"]) >= rsi_short_min:
            if require_rsi_turn and (rsi_prevprev is not None) and (rsi_prevprev == rsi_prevprev):
                if float(prev["rsi"]) > rsi_prevprev:
                    continue
            # V8ML gate (optional): skip low-quality signals
            feature_vec = [
                float(adx_v),
                float(atrp),
                float(bb_width),
                float(prev["rsi"]),
                float(prev["rsi"] - (rsi_prevprev if (rsi_prevprev is not None and rsi_prevprev==rsi_prevprev) else float(prev["rsi"]))),
                float((float(prev["close"]) - bb_low_v) / max(1e-9, atr_v)),
                float((bb_up_v - float(prev["close"])) / max(1e-9, atr_v)),
                -1.0,
            ]
            if not _ml_gate_pass(params, feature_vec):
                continue
            entry = bb_up_v
            if l <= entry <= h:
                q = notional / entry
                balance -= notional * maker_fee
                pos_side="short"; entry_px=entry; qty=q; entry_atr=atr_v; entry_ts_open=ts; entry_ts_open=ts
                entry_adx=adx_v
                entry_atrp=atrp

    return pd.DataFrame([t.__dict__ for t in trades]), pd.DataFrame(equity_rows)

def simulate_trend(df: pd.DataFrame, params: Dict, initial_balance: float, maker_fee: float, taker_fee: float, leverage_override: Optional[float]=None):
    """EMA pullback trend simulator aligned to PROD v6 behavior.

    Key points (kept incremental / no refactor):
    - Uses ADX hysteresis if params provide (adx_enter_min/adx_exit_min), otherwise falls back to adx_min.
    - Uses entry_buffer_atr_mult (if present) for pullback threshold around EMA fast.
    - Exits recompute TP/SL each candle using *current* ATR (like PROD refresh exits), then apply BE + trailing.
    - Cooldown expressed in candles (1H candle ~= 1 run).
    """
    df = df.copy()
    close = df["close"].astype(float)

    ema_fast_n = int(params.get("ema_fast", 20))
    ema_slow_n = int(params.get("ema_slow", 200))
    adx_period = int(params.get("adx_period", 14))

    # Baseline used adx_min. PROD uses enter/exit hysteresis.
    adx_min_fallback = float(params.get("adx_min", 20.0))
    adx_enter_min = float(params.get("adx_enter_min", adx_min_fallback))
    adx_exit_min = float(params.get("adx_exit_min", adx_enter_min))

    atr_period = int(params.get("atr_period", 14))
    stop_atr_mult = float(params.get("stop_atr_mult", 1.5))
    tp_atr_mult = float(params.get("tp_atr_mult", 2.0))

    # optional V6 params
    atrp_min = params.get("atrp_min", None)
    atrp_max = params.get("atrp_max", None)
    min_ema_sep_atr = params.get("min_ema_sep_atr", None)
    ema_slope_lookback = int(params.get("ema_slope_lookback", 0))
    min_ema_slope_atr = params.get("min_ema_slope_atr", None)

    entry_buffer_atr_mult = float(params.get("entry_buffer_atr_mult", 0.0))

    breakeven_activate_atr = params.get("breakeven_activate_atr", None)
    breakeven_offset_atr = params.get("breakeven_offset_atr", None)
    trail_activate_atr = params.get("trail_activate_atr", None)
    trail_stop_atr_mult = params.get("trail_stop_atr_mult", None)

    cooldown_after_close = int(params.get("cooldown_after_close_runs", 0))

    leverage = float(params.get("leverage", 1))
    if leverage_override is not None:
        leverage = float(leverage_override)

    pos_pct = float(params.get("position_size_percentage", 20)) / 100.0
    use_longs = bool(params.get("use_longs", True))
    use_shorts = bool(params.get("use_shorts", True))
    symbol = params.get("symbol", "")

    df["ema_fast"] = ema(close, ema_fast_n)
    df["ema_slow"] = ema(close, ema_slow_n)
    df["adx"] = adx(df, adx_period)
    df["atr"] = atr(df, atr_period)

    balance = initial_balance
    pos_side: Optional[str] = None
    entry_ts_open = 0  # timestamp of entry candle open (for Trade log)
    entry_px = 0.0
    qty = 0.0

    cooldown_left = 0
    trend_regime = False  # ADX hysteresis

    trades: List[Trade] = []
    equity_rows = []

    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        row = df.iloc[i]
        ts = int(row["timestamp"])
        h = float(row["high"]); l = float(row["low"]); c = float(row["close"])

        mtm = 0.0
        if pos_side == "long":
            mtm = (c - entry_px) * qty
        elif pos_side == "short":
            mtm = (entry_px - c) * qty
        equity_rows.append({"timestamp": ts, "equity": balance + mtm, "balance": balance, "pos_side": pos_side or "", "qty": qty})

        if any(prev[k] != prev[k] for k in ["ema_fast", "ema_slow", "adx", "atr", "close"]):
            continue

        # ----------------------------
        # Manage open position (exits)
        # ----------------------------
        if pos_side is not None:
            atr_now = float(prev["atr"])
            close_now = float(prev["close"])
            if atr_now != atr_now or atr_now <= 0:
                continue

            # Base TP/SL refreshed using current ATR (mirrors PROD "refresh exits")
            if pos_side == "long":
                tp = entry_px + tp_atr_mult * atr_now
                sl = entry_px - stop_atr_mult * atr_now
                move_atr = (close_now - entry_px) / atr_now if atr_now > 0 else 0.0

                # v6: break-even
                if breakeven_activate_atr is not None and breakeven_offset_atr is not None and move_atr >= float(breakeven_activate_atr):
                    sl = max(sl, entry_px + float(breakeven_offset_atr) * atr_now)

                # v6: trailing
                if trail_activate_atr is not None and trail_stop_atr_mult is not None and move_atr >= float(trail_activate_atr):
                    sl = max(sl, close_now - float(trail_stop_atr_mult) * atr_now)

            else:
                tp = entry_px - tp_atr_mult * atr_now
                sl = entry_px + stop_atr_mult * atr_now
                move_atr = (entry_px - close_now) / atr_now if atr_now > 0 else 0.0

                # v6: break-even
                if breakeven_activate_atr is not None and breakeven_offset_atr is not None and move_atr >= float(breakeven_activate_atr):
                    sl = min(sl, entry_px - float(breakeven_offset_atr) * atr_now)

                # v6: trailing
                if trail_activate_atr is not None and trail_stop_atr_mult is not None and move_atr >= float(trail_activate_atr):
                    sl = min(sl, close_now + float(trail_stop_atr_mult) * atr_now)

            # Candle-touch fill model
            if l <= sl <= h:
                exit_reason = "sl"; exit_price = sl
            elif l <= tp <= h:
                exit_reason = "tp"; exit_price = tp
            else:
                continue

            fee = exit_price * qty * taker_fee
            if pos_side == "long":
                pnl = (exit_price - entry_px) * qty - fee
                balance += (exit_price - entry_px) * qty
                trades.append(Trade("trend", symbol, "long", entry_ts_open, ts, entry_px, exit_price, qty, pnl, exit_reason, adx=entry_adx, atrp=entry_atrp, slope_atr=entry_slope_atr))
            else:
                pnl = (entry_px - exit_price) * qty - fee
                balance += (entry_px - exit_price) * qty
                trades.append(Trade("trend", symbol, "short", entry_ts_open, ts, entry_px, exit_price, qty, pnl, exit_reason, adx=entry_adx, atrp=entry_atrp, slope_atr=entry_slope_atr))

            balance -= fee
            pos_side = None; entry_px = 0.0; qty = 0.0; entry_adx=float("nan"); entry_atrp=float("nan"); entry_slope_atr=float("nan")

            if cooldown_after_close > 0:
                cooldown_left = cooldown_after_close
            continue

        # ----------------------------
        # No position: cooldown gate
        # ----------------------------
        if cooldown_left > 0:
            cooldown_left -= 1
            continue

        ema_fast_v = float(prev["ema_fast"])
        ema_slow_v = float(prev["ema_slow"])
        close_v = float(prev["close"])
        atr_v = float(prev["atr"])
        adx_v = float(prev["adx"])

        if any(x != x for x in [ema_fast_v, ema_slow_v, close_v, atr_v, adx_v]) or atr_v <= 0 or close_v <= 0:
            continue

        # v6: ATR% filter
        atrp = atr_v / close_v
        if atrp_min is not None and atrp < float(atrp_min):
            continue
        if atrp_max is not None and atrp > float(atrp_max):
            continue

        # ADX hysteresis regime (PROD-like)
        if (not trend_regime) and adx_v < adx_enter_min:
            continue
        if trend_regime and adx_v < adx_exit_min:
            trend_regime = False
            continue
        if (not trend_regime) and adx_v >= adx_enter_min:
            trend_regime = True

        # v6: EMA separation filter
        if min_ema_sep_atr is not None:
            sep = abs(ema_fast_v - ema_slow_v)
            if sep < float(min_ema_sep_atr) * atr_v:
                continue

        # v6: EMA slope filter
        slope = None
        if ema_slope_lookback and (i - 1 - ema_slope_lookback) >= 0:
            ema_prev = float(df.iloc[i-1-ema_slope_lookback]["ema_fast"])
            if ema_prev == ema_prev:
                slope = ema_fast_v - ema_prev
        min_slope = float(min_ema_slope_atr) * atr_v if min_ema_slope_atr is not None else None

        notional = balance * pos_pct * leverage
        buf = entry_buffer_atr_mult * atr_v

        # Entry modeled as limit fill at EMA fast if touched in current candle
        if use_longs and ema_fast_v > ema_slow_v and close_v <= (ema_fast_v - buf):
            if min_slope is not None and slope is not None and slope < min_slope:
                continue
            # V8ML gate (optional): skip low-quality signals
            feature_vec = [
                float(adx_v),
                float(atrp),
                float(abs(ema_fast_v - ema_slow_v) / max(1e-9, atr_v)),
                float((ema_fast_v - close_v) / max(1e-9, atr_v)),
                1.0,
            ]
            if not _ml_gate_pass(params, feature_vec):
                continue
            entry = ema_fast_v
            if l <= entry <= h:
                qty = notional / entry
                balance -= notional * maker_fee
                pos_side = "long"; entry_px = entry
                entry_ts_open = ts
                entry_adx = adx_v
                entry_atrp = atrp if "atrp" in locals() else float("nan")
                entry_slope_atr = (slope/atr_v if atr_v else float("nan"))
        elif use_shorts and ema_fast_v < ema_slow_v and close_v >= (ema_fast_v + buf):
            if min_slope is not None and slope is not None and (-slope) < min_slope:
                continue
            # V8ML gate (optional): skip low-quality signals
            feature_vec = [
                float(adx_v),
                float(atrp),
                float(abs(ema_fast_v - ema_slow_v) / max(1e-9, atr_v)),
                float((ema_fast_v - close_v) / max(1e-9, atr_v)),
                -1.0,
            ]
            if not _ml_gate_pass(params, feature_vec):
                continue
            entry = ema_fast_v
            if l <= entry <= h:
                qty = notional / entry
                balance -= notional * maker_fee
                pos_side = "short"; entry_px = entry
                entry_adx = adx_v
                entry_atrp = atrp if "atrp" in locals() else float("nan")
                entry_slope_atr = (slope/atr_v if atr_v else float("nan"))
                entry_ts_open = ts

    return pd.DataFrame([t.__dict__ for t in trades]), pd.DataFrame(equity_rows)

def summarize_equity(equity: pd.Series) -> Dict:
    if equity.empty:
        return {"error":"no equity"}
    start=float(equity.iloc[0])
    end=float(equity.iloc[-1])
    dd = equity.cummax() - equity
    max_dd=float(dd.max()) if len(dd) else 0.0
    return {
        "start_equity": start,
        "end_equity": end,
        "return_pct": (end/start-1)*100 if start else 0.0,
        "max_drawdown": max_dd,
        "max_drawdown_pct": (max_dd/float(equity.cummax().max()))*100 if len(dd) else 0.0,
    }

def _ts_to_month(ts_ms: int) -> str:
    # month in YYYY-MM from ms timestamp
    return pd.to_datetime(int(ts_ms), unit="ms", utc=True).strftime("%Y-%m")

def trade_stats(trades_df: pd.DataFrame) -> Dict[str, float]:
    """Compute simple trade statistics from a trades DataFrame."""
    if trades_df is None or trades_df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "median_pnl": 0.0,
            "profit_factor": 0.0,
            "avg_hold_hours": 0.0,
        }
    pnl = trades_df["pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    hold_h = (trades_df["exit_ts"].astype(float) - trades_df["entry_ts"].astype(float)) / (1000.0 * 3600.0)

    gp = wins.sum()
    gl = -losses.sum()
    pf = float(gp / gl) if gl > 0 else float("inf") if gp > 0 else 0.0

    return {
        "trades": int(len(trades_df)),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "win_rate_pct": float((pnl > 0).mean() * 100.0),
        "total_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
        "median_pnl": float(pnl.median()),
        "profit_factor": pf,
        "avg_hold_hours": float(hold_h.mean()) if len(hold_h) else 0.0,
    }

def trades_by_month(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly aggregation: counts + PnL by (month, strategy, symbol)."""
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=[
            "month","strategy","symbol","trades","wins","losses","win_rate_pct","pnl_sum","pnl_avg"
        ])
    df = trades_df.copy()
    df["month"] = df["exit_ts"].apply(_ts_to_month)
    df["win"] = (df["pnl"].astype(float) > 0).astype(int)
    agg = df.groupby(["month","strategy","symbol"], as_index=False).agg(
        trades=("pnl","size"),
        wins=("win","sum"),
        pnl_sum=("pnl","sum"),
        pnl_avg=("pnl","mean"),
    )
    agg["losses"] = agg["trades"] - agg["wins"]
    agg["win_rate_pct"] = (agg["wins"] / agg["trades"] * 100.0).round(2)
    # nice ordering
    agg = agg.sort_values(["month","strategy","symbol"]).reset_index(drop=True)
    return agg[["month","strategy","symbol","trades","wins","losses","win_rate_pct","pnl_sum","pnl_avg"]]

def max_drawdown_details(equity: pd.Series) -> Dict[str, float]:
    """Return max drawdown with peak/trough timestamps (indices)."""
    eq = equity.astype(float).values
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    trough_i = int(np.argmax(dd))
    peak_i = int(np.argmax(eq[:trough_i+1])) if trough_i >= 0 else 0
    max_dd = float(dd[trough_i]) if len(dd) else 0.0
    max_dd_pct = float((max_dd / peak[peak_i]) * 100.0) if peak[peak_i] > 0 else 0.0
    return {"max_drawdown": max_dd, "max_drawdown_pct": max_dd_pct, "dd_peak_i": peak_i, "dd_trough_i": trough_i}

def monthly_returns_from_equity(equity_df: pd.DataFrame) -> pd.DataFrame:
    tmp = equity_df.copy()
    tmp["dt"] = pd.to_datetime(tmp["timestamp"], unit="ms", utc=True)
    tmp["month"] = tmp["dt"].dt.strftime("%Y-%m")
    month_end = tmp.groupby("month", as_index=False).tail(1)[["month","equity"]].rename(columns={"equity":"equity_end"})
    month_end = month_end.sort_values("month").reset_index(drop=True)
    month_end["equity_start"] = month_end["equity_end"].shift(1)
    month_end.loc[0, "equity_start"] = float(tmp["equity"].iloc[0])
    month_end["return_pct"] = (month_end["equity_end"]/month_end["equity_start"] - 1)*100
    return month_end[["month","return_pct","equity_end"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--btc-bot", required=True, help="BTC trend bot file (e.g., run_btc_trend_1h_v3.py)")
    ap.add_argument("--sol-bot", required=True, help="SOL BBRSI bot file (e.g., run_sol_bbrsi_1h_v3.py)")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--exchange", default="bitget")
    ap.add_argument("--cache-dir", default=".cache/ohlcv", help="OHLCV cache directory (CSV.GZ per symbol/timeframe)")
    ap.add_argument("--no-cache", action="store_true", help="Disable OHLCV cache (always fetch from exchange)")
    ap.add_argument("--refresh-cache", action="store_true", help="If --end is not set, refresh cached OHLCV to latest")
    ap.add_argument("--initial", type=float, default=1000.0)

    ap.add_argument("--btc-weight", type=float, default=0.5, help="fraction of initial equity allocated to BTC bot")
    ap.add_argument("--sol-weight", type=float, default=0.5, help="fraction of initial equity allocated to SOL bot")
    ap.add_argument("--both-btc-weight", type=float, default=0.75, help="When both regimes are ON: allocate this fraction to BTC (rest to SOL)")
    ap.add_argument("--sol-min-weight-off", type=float, default=0.0, help="If >0: allow SOL a minimum weight even when sol_regime_on is False (prevents alloc_w=0 when SOL trades)")

    ap.add_argument("--btc-leverage", type=float, default=None, help="override leverage for BTC bot (e.g., 1,2,3)")
    ap.add_argument("--sol-leverage", type=float, default=None, help="override leverage for SOL bot (e.g., 1,2,3)")

    ap.add_argument("--btc-threshold", type=float, default=None, help="override ML gate threshold for BTC (e.g., 0.80)")
    ap.add_argument("--sol-threshold", type=float, default=None, help="override ML gate threshold for SOL (e.g., 0.62)")

    # --- Regime filters (Option 3) ---
    ap.add_argument("--regime3", action="store_true", help="Enable Option-3 hybrid regime switch (BBRSI in high vol + low/mid ADX; TREND in high ADX + slope)")
    ap.add_argument("--sol_atrp_min_regime", type=float, default=None, help="When --regime3: only allow SOL BBRSI entries if ATR%% >= this")
    ap.add_argument("--sol_adx_max_regime", type=float, default=None, help="When --regime3: only allow SOL BBRSI entries if ADX <= this (low/mid trend)")
    ap.add_argument("--btc_adx_min_regime", type=float, default=None, help="When --regime3: only allow BTC TREND entries if ADX >= this (trend regime)")
    ap.add_argument("--btc_min_slope_atr_regime", type=float, default=None, help="When --regime3: override BTC min_ema_slope_atr for stronger trend confirmation")

    ap.add_argument("--maker-fee", type=float, default=0.0002)
    ap.add_argument("--taker-fee", type=float, default=0.0006)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--name", default="btc_trend__sol_bbrsi", help="name suffix for output files")
    args = ap.parse_args()

    if abs((args.btc_weight + args.sol_weight) - 1.0) > 1e-6:
        raise ValueError("btc-weight + sol-weight must equal 1.0")

    btc_params = parse_params_from_bot_file(args.btc_bot)
    sol_params = parse_params_from_bot_file(args.sol_bot)


    # Optional CLI overrides for ML gate thresholds (per symbol)
    if args.btc_threshold is not None:
        btc_params.setdefault('ml_filter', {})
        btc_params['ml_filter']['threshold'] = float(args.btc_threshold)

    if args.sol_threshold is not None:
        sol_params.setdefault('ml_filter', {})
        sol_params['ml_filter']['threshold'] = float(args.sol_threshold)
    # Apply Option-3 regime overrides (only if enabled)
    # NOTE: Regime3 is used ONLY for portfolio allocation flags.
    # We DO NOT mutate bot params here (avoid filtering entries / changing signals).
    if args.regime3:
        pass

    start_ms = to_ms(args.start)
    end_ms = to_ms(args.end) if args.end else None

    # Fetch data per symbol
    btc_df = fetch_ohlcv_ccxt(btc_params["symbol"], btc_params.get("timeframe","1h"), start_ms, end_ms, args.exchange, cache_dir=args.cache_dir, use_cache=(not args.no_cache), refresh_if_no_end=args.refresh_cache)
    sol_df = fetch_ohlcv_ccxt(sol_params["symbol"], sol_params.get("timeframe","1h"), start_ms, end_ms, args.exchange, cache_dir=args.cache_dir, use_cache=(not args.no_cache), refresh_if_no_end=args.refresh_cache)

    btc_init = args.initial * args.btc_weight
    sol_init = args.initial * args.sol_weight

    btc_trades, btc_eq = simulate_trend(btc_df, btc_params, btc_init, args.maker_fee, args.taker_fee, args.btc_leverage)
    sol_trades, sol_eq = simulate_bbrsi(sol_df, sol_params, sol_init, args.maker_fee, args.taker_fee, args.sol_leverage)

    # Combine equity curves by timestamp (outer join, forward fill each, sum)
    btc_eq2 = btc_eq[["timestamp","equity"]].rename(columns={"equity":"equity_btc"})
    sol_eq2 = sol_eq[["timestamp","equity"]].rename(columns={"equity":"equity_sol"})
    comb = pd.merge(btc_eq2, sol_eq2, on="timestamp", how="outer").sort_values("timestamp").reset_index(drop=True)
    comb["equity_btc"] = comb["equity_btc"].ffill()
    comb["equity_sol"] = comb["equity_sol"].ffill()
    # Fill initial gaps with initial allocations
    comb["equity_btc"] = comb["equity_btc"].fillna(btc_init)
    comb["equity_sol"] = comb["equity_sol"].fillna(sol_init)
    comb["equity"] = comb["equity_btc"] + comb["equity_sol"]

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

        # --- ensure empty trade dfs still have expected columns (avoid KeyError on sort) ---
    MIN_TRADE_COLS = [
        "strategy","symbol","side",
        "entry_ts","exit_ts","entry_price","exit_price","qty","pnl","reason",
        "adx","atrp","slope_atr",
    ]
    for _df in [btc_trades, sol_trades]:
        if _df is not None and _df.empty:
            for c in MIN_TRADE_COLS:
                if c not in _df.columns:
                    _df[c] = []
    # -------------------------------------------------------------------------------

    trades_all = pd.concat([btc_trades, sol_trades], ignore_index=True)
    trades_all = trades_all.sort_values(["exit_ts","strategy"]).reset_index(drop=True)

    trades_path = out / f"portfolio_trades_{args.name}.csv"
    equity_path = out / f"portfolio_equity_{args.name}.csv"
    monthly_path = out / f"portfolio_monthly_{args.name}.csv"

    trades_all.to_csv(trades_path, index=False)

    # --- Regime coverage flags (observability; no impact on equity/trades) ---
    # SOL regime: ATR% >= sol_atrp_min_regime AND ADX <= sol_adx_max_regime
    # BTC regime: ADX >= btc_adx_min_regime AND slope_atr_proxy >= btc_min_slope_atr_regime

    sol_tmp = sol_df.copy()
    sol_adx_p = int(sol_params.get('adx_period', 14))
    sol_atr_p = int(sol_params.get('atr_period', 14))
    sol_tmp['adx_calc'] = adx(sol_tmp, sol_adx_p)
    sol_tmp['atr_calc'] = atr(sol_tmp, sol_atr_p)
    sol_tmp['atrp_calc'] = sol_tmp['atr_calc'] / sol_tmp['close'].astype(float).replace(0.0, float('nan'))

    sol_atrp_min = float(args.sol_atrp_min_regime) if args.sol_atrp_min_regime is not None else float('nan')
    sol_adx_max  = float(args.sol_adx_max_regime)  if args.sol_adx_max_regime  is not None else float('nan')

    btc_tmp = btc_df.copy()
    btc_adx_p = int(btc_params.get('adx_period', 14))
    btc_atr_p = int(btc_params.get('atr_period', 14))
    ema_fast_n = int(btc_params.get('ema_fast', 20))
    ema_slow_n = int(btc_params.get('ema_slow', 200))

    btc_tmp['adx_calc'] = adx(btc_tmp, btc_adx_p)
    btc_tmp['atr_calc'] = atr(btc_tmp, btc_atr_p)
    btc_tmp['ema_fast_calc'] = ema(btc_tmp['close'].astype(float), ema_fast_n)
    btc_tmp['ema_slow_calc'] = ema(btc_tmp['close'].astype(float), ema_slow_n)
    btc_tmp['slope_atr_proxy'] = (btc_tmp['ema_fast_calc'] - btc_tmp['ema_slow_calc']).abs() / btc_tmp['atr_calc'].replace(0.0, float('nan'))

    btc_adx_min = float(args.btc_adx_min_regime) if args.btc_adx_min_regime is not None else float('nan')
    btc_slope_min = float(args.btc_min_slope_atr_regime) if args.btc_min_slope_atr_regime is not None else float('nan')

    flags = comb[['timestamp']].copy()
    flags = flags.merge(sol_tmp[['timestamp','adx_calc','atrp_calc']].rename(columns={'adx_calc':'sol_adx','atrp_calc':'sol_atrp'}), on='timestamp', how='left')
    flags = flags.merge(btc_tmp[['timestamp','adx_calc','slope_atr_proxy']].rename(columns={'adx_calc':'btc_adx','slope_atr_proxy':'btc_slope_atr_proxy'}), on='timestamp', how='left')

    flags['sol_regime_on'] = flags['sol_adx'].notna() & flags['sol_atrp'].notna() & (flags['sol_atrp'] >= sol_atrp_min) & (flags['sol_adx'] <= sol_adx_max)
    flags['btc_regime_on'] = flags['btc_adx'].notna() & flags['btc_slope_atr_proxy'].notna() & (flags['btc_adx'] >= btc_adx_min) & (flags['btc_slope_atr_proxy'] >= btc_slope_min)

    flags_path = os.path.join(args.outdir, f'portfolio_regime_flags_{args.name}.csv')
    flags.to_csv(flags_path, index=False)
    print(f'Saved -> {flags_path}')

    # --- Dynamic allocation & realized portfolio equity (Regime3, trade-aware) ---
    if args.regime3:
        import numpy as np

        # merge regime flags onto combined equity timeline
        comb = comb.merge(flags[['timestamp','sol_regime_on','btc_regime_on']], on='timestamp', how='left')
        comb[['sol_regime_on','btc_regime_on']] = comb[['sol_regime_on','btc_regime_on']].fillna(False)

        # Candle weights (for visibility/debug; portfolio equity will be realized trade-aware)
        w_btc_list = []
        w_sol_list = []
        prev_w_btc, prev_w_sol = 1.0, 0.0
        both_w = float(args.both_btc_weight)

        for _, r in comb.iterrows():
            wb, ws = compute_dynamic_weights(
                bool(r['btc_regime_on']),
                bool(r['sol_regime_on']),
                both_btc_weight=both_w,
                prev_w_btc=prev_w_btc,
                prev_w_sol=prev_w_sol,
                sol_min_weight_off=float(getattr(args, 'sol_min_weight_off', 0.0)),
            )
            w_btc_list.append(wb)
            w_sol_list.append(ws)
            prev_w_btc, prev_w_sol = wb, ws

        comb['w_btc'] = pd.Series(w_btc_list, index=comb.index).astype(float)
        comb['w_sol'] = pd.Series(w_sol_list, index=comb.index).astype(float)

        # ---- Trade-aware realized equity (MODELO B) ----
        # Scale trade PnL from bot-initial allocations -> portfolio allocation at entry.
        eq0 = float(args.initial)

        # Prepare timestamp arrays for alignment (backward)
        comb_ts = comb['timestamp'].dropna().astype(int).values
        flags_idx = comb[['timestamp','btc_regime_on','sol_regime_on']].drop_duplicates('timestamp').set_index('timestamp')

        def weight_at_entry(entry_ts_ms: int):
            # align entry to candle (backward)
            i = int(np.searchsorted(comb_ts, int(entry_ts_ms), side='right') - 1)
            if i < 0:
                i = 0
            elif i >= len(comb_ts):
                i = len(comb_ts) - 1
            ts = int(comb_ts[i])
            r = flags_idx.loc[ts]
            # Use the precomputed sticky weights for this candle (avoids losing prev_w state inside weight_at_entry)
            try:
                wb = float(comb.loc[comb['timestamp'] == ts, 'w_btc'].iloc[0])
                ws = float(comb.loc[comb['timestamp'] == ts, 'w_sol'].iloc[0])
            except Exception:
                wb, ws = compute_dynamic_weights(
                    bool(r['btc_regime_on']),
                    bool(r['sol_regime_on']),
                    both_btc_weight=both_w,
                    sol_min_weight_off=float(getattr(args, 'sol_min_weight_off', 0.0)),
                )

            # Enforce minimum SOL weight when SOL is allowed to trade off-regime.
            # NOTE: Do NOT reference `ta` here (it's defined later). We apply the floor unconditionally when ws==0.
            sol_floor = float(getattr(args, 'sol_min_weight_off', 0.0))
            if sol_floor > 0.0 and ((not np.isfinite(float(ws))) or (float(ws) <= 0.0)):
                ws = sol_floor
                wb = 1.0 - ws
            return ts, float(wb), float(ws)

        # Realized PnL by (aligned) exit timestamp
        pnl_by_exit = {}
        pnl_by_exit_raw = {}

        ta = trades_all.copy()
        if (not ta.empty) and ('entry_ts' in ta.columns) and ('exit_ts' in ta.columns) and ('pnl' in ta.columns):
            ta = ta.dropna(subset=['entry_ts','exit_ts','pnl']).copy()
            ta['entry_ts'] = ta['entry_ts'].astype(int)
            ta['exit_ts']  = ta['exit_ts'].astype(int)
            ta['pnl']      = ta['pnl'].astype(float)

            # Determine instrument for each trade
            def _is_sol(row):
                sym = str(row.get('symbol','')).upper()
                strat = str(row.get('strategy','')).lower()
                return ('SOL' in sym) or ('sol' in strat)

            # Add allocation diagnostics columns
            alloc_w = []
            alloc_cap = []
            pnl_scaled = []
            entry_ts_aligned = []
            exit_ts_aligned = []

            for _, tr in ta.iterrows():
                ent_raw = int(tr['entry_ts'])
                ex_raw  = int(tr['exit_ts'])

                ent_candle, wb, ws = weight_at_entry(ent_raw)

                # pick weight and bot_init based on instrument
                if _is_sol(tr):
                    w = ws
                    bot_init = float(sol_init) if float(sol_init) > 0 else 1.0
                else:
                    w = wb
                    bot_init = float(btc_init) if float(btc_init) > 0 else 1.0

                cap = eq0 * w
                scale = cap / bot_init
                pnl_s = float(tr['pnl']) * scale

                # align exit to candle (backward)
                j = int(np.searchsorted(comb_ts, ex_raw, side='right') - 1)
                if j < 0:
                    j = 0
                elif j >= len(comb_ts):
                    j = len(comb_ts) - 1
                ex_candle = int(comb_ts[j])

                alloc_w.append(w)
                alloc_cap.append(cap)
                pnl_scaled.append(pnl_s)
                entry_ts_aligned.append(ent_candle)
                exit_ts_aligned.append(ex_candle)

                pnl_by_exit[ex_candle] = pnl_by_exit.get(ex_candle, 0.0) + pnl_s
                pnl_by_exit_raw[ex_candle] = pnl_by_exit_raw.get(ex_candle, 0.0) + float(tr['pnl'])

            ta['alloc_w'] = alloc_w
            ta['alloc_cap'] = alloc_cap
            ta['pnl_scaled'] = pnl_scaled
            ta['entry_ts_aligned'] = entry_ts_aligned
            ta['exit_ts_aligned'] = exit_ts_aligned

            # overwrite trades_all for export/debug
            trades_all = ta

            # Re-save trades CSV after allocation diagnostics are attached
            try:
                trades_all.to_csv(trades_path, index=False)
            except Exception as e:
                print(f"[WARN] failed to re-save trades CSV: {e}")

        comb['alloc_pnl'] = comb['timestamp'].astype(int).map(pnl_by_exit).fillna(0.0).astype(float)
        comb['pnl_raw'] = comb['timestamp'].astype(int).map(pnl_by_exit_raw).fillna(0.0).astype(float)

        # IMPORTANT:
        # - `equity` reflects the *real* (unallocated) portfolio equity if each trade were funded with `--initial`.
        # - `equity_alloc` reflects the equity after applying dynamic capital allocation weights (alloc_w).
        comb['equity_alloc'] = eq0 + comb['alloc_pnl'].cumsum()
        comb['equity'] = eq0 + comb['pnl_raw'].cumsum()
        comb['r_port'] = comb['equity'].pct_change().fillna(0.0)
        comb['r_port_alloc'] = comb['equity_alloc'].pct_change().fillna(0.0)

        # effective weights for viewing (previous candle)
        comb['w_btc_eff'] = comb['w_btc'].shift(1).fillna(comb['w_btc'])
        comb['w_sol_eff'] = comb['w_sol'].shift(1).fillna(comb['w_sol'])

        # placeholders (not used in MODELO B, but keep columns stable)
        comb['r_btc'] = 0.0
        comb['r_sol'] = 0.0
# write equity (after dynamic allocation if enabled)
    comb.to_csv(equity_path, index=False)
    # ---------------------------------------------
    monthly_df = monthly_returns_from_equity(comb[["timestamp","equity"]])
    monthly_df_alloc = None
    if "equity_alloc" in comb.columns:
        _tmp = comb[["timestamp","equity_alloc"]].rename(columns={"equity_alloc":"equity"})
        monthly_df_alloc = monthly_returns_from_equity(_tmp)
    monthly_df.to_csv(monthly_path, index=False)

    print(f"\n=== Portfolio: {args.name} ===")
    met = summarize_equity(comb["equity"])
    ddx = max_drawdown_details(comb["equity"])
    for k,v in met.items():
        print(f"{k}: {v}")

    if "equity_alloc" in comb.columns:
        print(f"\n=== Portfolio (ALLOC): {args.name} ===")
        met_a = summarize_equity(comb["equity_alloc"])
        ddx_a = max_drawdown_details(comb["equity_alloc"])
        for k,v in met_a.items():
            print(f"{k}: {v}")

    # Drawdown timing (UTC) - RAW equity
    try:
        peak_ts = int(comb.iloc[int(ddx["dd_peak_i"])]["timestamp"])
        trough_ts = int(comb.iloc[int(ddx["dd_trough_i"])]["timestamp"])
        peak_dt = pd.to_datetime(peak_ts, unit="ms", utc=True)
        trough_dt = pd.to_datetime(trough_ts, unit="ms", utc=True)
        print(f"max_dd_window_utc: peak={peak_dt} -> trough={trough_dt}")
    except Exception:
        pass

    # Drawdown timing (UTC) - ALLOC equity
    if "equity_alloc" in comb.columns:
        try:
            peak_ts = int(comb.iloc[int(ddx_a["dd_peak_i"])]["timestamp"])
            trough_ts = int(comb.iloc[int(ddx_a["dd_trough_i"])]["timestamp"])
            peak_dt = pd.to_datetime(peak_ts, unit="ms", utc=True)
            trough_dt = pd.to_datetime(trough_ts, unit="ms", utc=True)
            print(f"max_dd_window_alloc_utc: peak={peak_dt} -> trough={trough_dt}")
        except Exception:
            pass

    # Trade stats
    print("\n=== Trade statistics (overall) ===")
    st_all = trade_stats(trades_all)
    for k in ["trades","wins","losses","win_rate_pct","total_pnl","avg_pnl","median_pnl","profit_factor","avg_hold_hours"]:
        print(f"{k}: {st_all[k]}")

    for strat in sorted(trades_all["strategy"].unique()) if not trades_all.empty else []:
        st = trade_stats(trades_all[trades_all["strategy"] == strat])
        print(f"\n--- {strat.upper()} ---")
        for k in ["trades","wins","losses","win_rate_pct","total_pnl","avg_pnl","median_pnl","profit_factor","avg_hold_hours"]:
            print(f"{k}: {st[k]}")

    # Monthly returns + trades per month
    print("\n=== Monthly returns + trades ===")
    t_by_m = trades_all.copy()
    if not t_by_m.empty:
        t_by_m["month"] = t_by_m["exit_ts"].apply(_ts_to_month)
        m_tot = t_by_m.groupby("month", as_index=False).agg(trades=("pnl","size"), pnl_sum=("pnl","sum"))
        m_btc = t_by_m[t_by_m["strategy"]=="trend"].groupby("month", as_index=False).agg(btc_trades=("pnl","size"))
        m_sol = t_by_m[t_by_m["strategy"]=="bbrsi"].groupby("month", as_index=False).agg(sol_trades=("pnl","size"))
        m = monthly_df.merge(m_tot, on="month", how="left").merge(m_btc, on="month", how="left").merge(m_sol, on="month", how="left")
        m[["trades","pnl_sum","btc_trades","sol_trades"]] = m[["trades","pnl_sum","btc_trades","sol_trades"]].fillna(0)
    else:
        m = monthly_df.copy()
        m["trades"] = 0
        m["pnl_sum"] = 0.0
        m["btc_trades"] = 0
        m["sol_trades"] = 0

    for _, r in m.iterrows():
        alloc_end = None
        if monthly_df_alloc is not None:
            try:
                alloc_end = float(monthly_df_alloc[monthly_df_alloc["month"] == r["month"]]["equity_end"].iloc[0])
            except Exception:
                alloc_end = None
        extra = f" | equity_alloc_end={alloc_end:.2f}" if alloc_end is not None else ""
        print(f"{r['month']}: {r['return_pct']:.2f}% | trades={int(r['trades'])} (btc={int(r['btc_trades'])}, sol={int(r['sol_trades'])}) | pnl_sum={float(r['pnl_sum']):.2f} | equity_end={r['equity_end']:.2f}{extra}")

    # Detailed monthly trade aggregation
    t_monthly = trades_by_month(trades_all)
    trades_monthly_path = out / f"portfolio_trades_monthly_{args.name}.csv"
    t_monthly.to_csv(trades_monthly_path, index=False)

    # Save a compact stats CSV for quick comparisons
    stats_path = out / f"portfolio_stats_{args.name}.csv"
    stats_rows = []
    stats_rows.append({"scope":"overall", **st_all, **met})
    for strat in sorted(trades_all["strategy"].unique()) if not trades_all.empty else []:
        st = trade_stats(trades_all[trades_all["strategy"] == strat])
        stats_rows.append({"scope":strat, **st})
    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)

    print(f"\nSaved -> {trades_path}")
    print(f"Saved -> {equity_path}")
    print(f"Saved -> {monthly_path}")
    print(f"Saved -> {trades_monthly_path}")
    print(f"Saved -> {stats_path}")

if __name__ == "__main__":
    main()
