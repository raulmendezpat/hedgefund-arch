from __future__ import annotations
import json

import argparse
from dataclasses import asdict
from typing import Dict, Optional

import pandas as pd
import numpy as np

from hf.core.types import Candle, Allocation, Signal
from hf.engines.alloc_regime import RegimeAllocator
from hf.engines.portfolio_engine import SimplePortfolioEngine
from hf.engines.portfolio_metrics import PortfolioMetricsEngine
from hf.engines.report_engine import ReportEngine
from hf.engines.execution_simulator import ExecutionCostModel, ExecutionSimulator
from hf.engines.regime_regime3 import Regime3Engine

from hf.engines.signals import PortfolioSignalEngine, RegistryPortfolioSignalEngine, FlatSignalEngine, BtcTrendSignalEngine, SolBbrsiSignalEngine
from hf.engines.signals.sol_vol_breakout_signal import SolVolBreakoutSignalEngine
from hf.engines.signals.sol_trend_pullback_signal import SolTrendPullbackSignalEngine
from hf.engines.opportunity_book import select_opportunities, compute_competitive_score, compute_post_ml_competitive_score
from hf.engines.ml_filter import FEATURE_COLUMNS, apply_ml_filter_to_signals, build_feature_row, load_model, load_model_registry, predict_proba
from hf.engines.ml_position_sizer import MlPositionSizingEngine
from hf.engines.legacy_wrappers import (
    LEGACY_SYMBOLS,
    PlaceholderSignalEngine,
    StaticRegimeEngine,
    DynamicAllocator,
)
from hf.data.ohlcv import fetch_ohlcv_ccxt, dt_to_ms_utc



def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=int(span), adjust=False).mean()

def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Wilder smoothing
    return tr.ewm(alpha=1.0/float(period), adjust=False).mean()

def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = _atr(df, period)
    atr_safe = atr.replace(0.0, np.nan)

    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0/float(period), adjust=False).mean() / atr_safe
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0/float(period), adjust=False).mean() / atr_safe

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1.0/float(period), adjust=False).mean()
    return adx


def _rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder RSI (sin deps externas)."""
    close = series.astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / float(period), adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / float(period), adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def _bbands(series: pd.Series, period: int, std_mult: float):
    close = series.astype(float)
    mid = close.rolling(window=int(period), min_periods=int(period)).mean()
    std = close.rolling(window=int(period), min_periods=int(period)).std(ddof=0)
    up = mid + float(std_mult) * std
    low = mid - float(std_mult) * std
    return mid, up, low


def _donchian(df: pd.DataFrame, lookback: int):
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    dc_high = high.shift(1).rolling(window=int(lookback), min_periods=int(lookback)).max()
    dc_low = low.shift(1).rolling(window=int(lookback), min_periods=int(lookback)).min()
    return dc_high, dc_low


def _row_to_candle(ts: int, row: pd.Series, features: dict[str, float] | None = None) -> Candle:
    # ts in ms
    return Candle(
        ts=pd.to_datetime(int(ts), unit="ms", utc=True),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row.get("volume", 0.0)),
        features=features,
    )


def run(
    name: str,
    start: str,
    end: Optional[str],
    exchange: str,
    cache_dir: str,
    refresh_cache: bool,
    trades_csv: str,
    sol_atrp_min: float,
    sol_adx_max: float,
    btc_adx_min: float,
    btc_slope_min: float,
    signal_engine: str = "flat",
    sticky_flat: bool = False,
    # allocator policy knobs
    both_btc_weight: float = 0.75,
    sticky_when_off: bool = False,
    fallback_btc_weight: float = 1.0,
    fallback_sol_weight: float = 0.0,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    # SOL SignalEngine (BBRSI) knobs (separados de Regime3)
    sol_rsi_long_max: float = 36.0,
    sol_rsi_short_min: float = 64.0,
    sol_adx_hard_signal: float = 24.0,
    sol_atrp_min_signal: float = 0.003279,
    sol_atrp_max_signal: float = 0.0350,
    sol_bb_width_min: float = 0.0041,
    sol_bb_width_max: float = 0.120,
    sol_bb_period: int = 20,
    sol_bb_std: float = 2.0,
    sol_vol_breakout_lookback: int = 20,
    sol_vol_adx_min: float = 18.0,
    sol_vol_atrp_min: float = 0.008,
    sol_vol_atrp_max: float = 0.080,
    sol_vol_range_expansion_min: float = 1.10,
    sol_vol_confirm_close_buffer: float = 0.0,
    sol_trend_pullback_rsi_long_min: float = 40.0,
    sol_trend_pullback_rsi_long_max: float = 55.0,
    sol_trend_pullback_rsi_short_min: float = 45.0,
    sol_trend_pullback_rsi_short_max: float = 60.0,
    sol_trend_pullback_ema_pullback_max: float = 0.015,
    sol_trend_pullback_atrp_min: float = 0.004,
    sol_trend_pullback_atrp_max: float = 0.050,
    sol_trend_pullback_require_adx: bool = False,
    sol_trend_pullback_adx_min: float = 18.0,
    ml_filter: bool = False,
    ml_model_path: Optional[str] = None,
    ml_model_registry: Optional[str] = None,
    ml_threshold: float = 0.55,
    ml_export_features: bool = False,
    ml_features_out: Optional[str] = None,
    ml_position_sizing: bool = False,
    ml_size_scale: float = 1.0,
    ml_size_min: float = 0.0,
    ml_size_max: float = 1.0,
    ml_size_mode: str = "linear_edge",
    ml_size_base: float = 0.25,
    ml_size_pwin_threshold: float = 0.55,
    strategy_registry_path: str = "artifacts/strategy_registry.json",
    opportunity_selection_mode: str = "best_per_symbol",
) -> pd.DataFrame:
    start_ms = dt_to_ms_utc(start)
    end_ms = dt_to_ms_utc(end) if end else None

    btc_sym = LEGACY_SYMBOLS["BTC"]
    sol_sym = LEGACY_SYMBOLS["SOL"]

    btc = fetch_ohlcv_ccxt(btc_sym, "1h", start_ms, end_ms, exchange_id=exchange, cache_dir=cache_dir, use_cache=True, refresh_if_no_end=refresh_cache)
    sol = fetch_ohlcv_ccxt(sol_sym, "1h", start_ms, end_ms, exchange_id=exchange, cache_dir=cache_dir, use_cache=True, refresh_if_no_end=refresh_cache)

    if btc.empty or sol.empty:
        raise SystemExit("OHLCV empty for BTC or SOL. Check cache_dir/exchange/symbol/timeframe.")

    btc = btc.set_index("timestamp").sort_index()
    sol = sol.set_index("timestamp").sort_index()

    # common timestamps only (safe)
    common_ts = btc.index.intersection(sol.index)
    if len(common_ts) < 10:
        raise SystemExit(f"Not enough overlapping candles: {len(common_ts)}")


    # --- feature calc (for Regime3Engine) ---
    # BTC
    btc_close = btc["close"].astype(float)
    btc_atr = _atr(btc, 14)
    btc_adx = _adx(btc, 14)
    btc_ema_fast = _ema(btc_close, 20)
    btc_ema_slow = _ema(btc_close, 200)

    # SOL
    sol_close = sol["close"].astype(float)
    sol_atr = _atr(sol, 14)
    sol_adx = _adx(sol, 14)
    sol_atrp = sol_atr / sol_close.replace(0.0, np.nan)

    # SOL SignalEngine indicators (BBRSI)
    _sol_delta = sol_close.diff()
    _sol_gain = _sol_delta.clip(lower=0.0)
    _sol_loss = (-_sol_delta.clip(upper=0.0))
    _sol_avg_gain = _sol_gain.rolling(int(sol_bb_period), min_periods=int(sol_bb_period)).mean()
    _sol_avg_loss = _sol_loss.rolling(int(sol_bb_period), min_periods=int(sol_bb_period)).mean()
    _sol_rs = _sol_avg_gain / _sol_avg_loss.replace(0.0, np.nan)
    sol_rsi = 100.0 - (100.0 / (1.0 + _sol_rs))
    sol_ema_fast = _ema(sol_close, 20)
    sol_ema_slow = _ema(sol_close, 50)

    sol_bb_mid = sol_close.rolling(int(sol_bb_period), min_periods=int(sol_bb_period)).mean()
    _sol_bb_stddev = sol_close.rolling(int(sol_bb_period), min_periods=int(sol_bb_period)).std(ddof=0)
    sol_bb_up = sol_bb_mid + float(sol_bb_std) * _sol_bb_stddev
    sol_bb_low = sol_bb_mid - float(sol_bb_std) * _sol_bb_stddev
    sol_bb_width = (sol_bb_up - sol_bb_low) / sol_bb_mid.replace(0.0, np.nan)

    sol_vol_dc_high, sol_vol_dc_low = _donchian(sol, int(sol_vol_breakout_lookback))
    sol_atr_ema20 = _ema(sol_atr, 20)
    sol_range_expansion = sol_atr / sol_atr_ema20.replace(0.0, np.nan)

    # SignalEngine layer (default: flat placeholder)
    if signal_engine == "btc_trend":
        sig_engine = BtcTrendSignalEngine(adx_min=float(btc_adx_min))
    elif signal_engine == "sol_bbrsi":
        sig_engine = SolBbrsiSignalEngine(
            rsi_long_max=float(locals().get("sol_rsi_long_max", 36.0)),
            rsi_short_min=float(locals().get("sol_rsi_short_min", 64.0)),
            adx_hard=float(locals().get("sol_adx_hard_signal", 24.0)),
            atrp_min=float(locals().get("sol_atrp_min_signal", 0.003279)),
            atrp_max=float(locals().get("sol_atrp_max_signal", 0.0350)),
            bb_width_min=float(locals().get("sol_bb_width_min", 0.0041)),
            bb_width_max=float(locals().get("sol_bb_width_max", 0.120)),
        )
    elif signal_engine == "sol_vol_breakout":
        sig_engine = SolVolBreakoutSignalEngine(
            breakout_lookback=int(locals().get("sol_vol_breakout_lookback", 20)),
            adx_min=float(locals().get("sol_vol_adx_min", 18.0)),
            atrp_min=float(locals().get("sol_vol_atrp_min", 0.008)),
            atrp_max=float(locals().get("sol_vol_atrp_max", 0.080)),
            range_expansion_min=float(locals().get("sol_vol_range_expansion_min", 1.10)),
            confirm_close_buffer=float(locals().get("sol_vol_confirm_close_buffer", 0.0)),
        )
    elif signal_engine == "sol_trend_pullback":
        sig_engine = SolTrendPullbackSignalEngine(
            rsi_long_min=float(locals().get("sol_trend_pullback_rsi_long_min", 40.0)),
            rsi_long_max=float(locals().get("sol_trend_pullback_rsi_long_max", 55.0)),
            rsi_short_min=float(locals().get("sol_trend_pullback_rsi_short_min", 45.0)),
            rsi_short_max=float(locals().get("sol_trend_pullback_rsi_short_max", 60.0)),
            ema_pullback_max=float(locals().get("sol_trend_pullback_ema_pullback_max", 0.015)),
            atrp_min=float(locals().get("sol_trend_pullback_atrp_min", 0.004)),
            atrp_max=float(locals().get("sol_trend_pullback_atrp_max", 0.050)),
            require_adx=bool(locals().get("sol_trend_pullback_require_adx", False)),
            adx_min=float(locals().get("sol_trend_pullback_adx_min", 18.0)),
        )
    elif signal_engine == "portfolio":
        sig_engine = PortfolioSignalEngine(
            registry_path=str(strategy_registry_path),
            btc_engine=BtcTrendSignalEngine(adx_min=float(btc_adx_min)),
            sol_engine=SolBbrsiSignalEngine(
                rsi_long_max=float(locals().get("sol_rsi_long_max", 36.0)),
                rsi_short_min=float(locals().get("sol_rsi_short_min", 64.0)),
                adx_hard=float(locals().get("sol_adx_hard_signal", 24.0)),
                atrp_min=float(locals().get("sol_atrp_min_signal", 0.003279)),
                atrp_max=float(locals().get("sol_atrp_max_signal", 0.0350)),
                bb_width_min=float(locals().get("sol_bb_width_min", 0.0041)),
                bb_width_max=float(locals().get("sol_bb_width_max", 0.120)),
            ),
        )
    elif signal_engine == "registry_portfolio":
        sig_engine = RegistryPortfolioSignalEngine(
            registry_path=str(strategy_registry_path),
            selection_mode=str(opportunity_selection_mode),
            engine_factories={
                "btc_trend_signal": lambda cfg: BtcTrendSignalEngine(
                    adx_min=float((cfg.get("params", {}) or {}).get("adx_min", btc_adx_min))
                ),
                "sol_bbrsi_signal": lambda cfg: SolBbrsiSignalEngine(
                    rsi_long_max=float((cfg.get("params", {}) or {}).get("rsi_long_max", sol_rsi_long_max)),
                    rsi_short_min=float((cfg.get("params", {}) or {}).get("rsi_short_min", sol_rsi_short_min)),
                    adx_hard=float((cfg.get("params", {}) or {}).get("adx_hard", sol_adx_hard_signal)),
                    atrp_min=float((cfg.get("params", {}) or {}).get("atrp_min", sol_atrp_min_signal)),
                    atrp_max=float((cfg.get("params", {}) or {}).get("atrp_max", sol_atrp_max_signal)),
                    bb_width_min=float((cfg.get("params", {}) or {}).get("bb_width_min", sol_bb_width_min)),
                    bb_width_max=float((cfg.get("params", {}) or {}).get("bb_width_max", sol_bb_width_max)),
                ),
                "sol_vol_breakout_signal": lambda cfg: SolVolBreakoutSignalEngine(
                    breakout_lookback=int((cfg.get("params", {}) or {}).get("breakout_lookback", sol_vol_breakout_lookback)),
                    adx_min=float((cfg.get("params", {}) or {}).get("adx_min", sol_vol_adx_min)),
                    atrp_min=float((cfg.get("params", {}) or {}).get("atrp_min", sol_vol_atrp_min)),
                    atrp_max=float((cfg.get("params", {}) or {}).get("atrp_max", sol_vol_atrp_max)),
                    range_expansion_min=float((cfg.get("params", {}) or {}).get("range_expansion_min", sol_vol_range_expansion_min)),
                    confirm_close_buffer=float((cfg.get("params", {}) or {}).get("confirm_close_buffer", sol_vol_confirm_close_buffer)),
                ),
                "sol_trend_pullback_signal": lambda cfg: SolTrendPullbackSignalEngine(
                    rsi_long_min=float((cfg.get("params", {}) or {}).get("rsi_long_min", sol_trend_pullback_rsi_long_min)),
                    rsi_long_max=float((cfg.get("params", {}) or {}).get("rsi_long_max", sol_trend_pullback_rsi_long_max)),
                    rsi_short_min=float((cfg.get("params", {}) or {}).get("rsi_short_min", sol_trend_pullback_rsi_short_min)),
                    rsi_short_max=float((cfg.get("params", {}) or {}).get("rsi_short_max", sol_trend_pullback_rsi_short_max)),
                    ema_pullback_max=float((cfg.get("params", {}) or {}).get("ema_pullback_max", sol_trend_pullback_ema_pullback_max)),
                    atrp_min=float((cfg.get("params", {}) or {}).get("atrp_min", sol_trend_pullback_atrp_min)),
                    atrp_max=float((cfg.get("params", {}) or {}).get("atrp_max", sol_trend_pullback_atrp_max)),
                    require_adx=bool((cfg.get("params", {}) or {}).get("require_adx", sol_trend_pullback_require_adx)),
                    adx_min=float((cfg.get("params", {}) or {}).get("adx_min", sol_trend_pullback_adx_min)),
                ),
            },
        )
    else:
        sig_engine = FlatSignalEngine()
    reg_engine = Regime3Engine(
        sol_atrp_min=float(sol_atrp_min),
        sol_adx_max=float(sol_adx_max),
        btc_adx_min=float(btc_adx_min),
        btc_slope_min=float(btc_slope_min),
    )
    allocator = RegimeAllocator(
        both_btc_weight=float(both_btc_weight),
        sticky_when_off=bool(sticky_when_off),
        fallback_btc_weight=float(fallback_btc_weight),
        fallback_sol_weight=float(fallback_sol_weight),
        btc_symbol=btc_sym,
        sol_symbol=sol_sym,
    )

    ml_enabled_for_scores = bool(ml_filter or ml_position_sizing)
    ml_model = load_model(ml_model_path) if ml_enabled_for_scores else None
    ml_model_registry_loaded = load_model_registry(ml_model_registry) if ml_enabled_for_scores else {}
    ml_rejected_counts_by_symbol = {}
    ml_position_sized_counts_by_symbol = {}
    ml_feature_rows = []
    ml_position_sizer = MlPositionSizingEngine(
        scale=float(ml_size_scale),
        min_mult=float(ml_size_min),
        max_mult=float(ml_size_max),
        mode=str(ml_size_mode),
        base_size=float(ml_size_base),
        pwin_threshold=float(ml_size_pwin_threshold),
    ) if bool(ml_position_sizing) else None

    ml_score_position_sizer = ml_position_sizer or (
        MlPositionSizingEngine(
            scale=float(ml_size_scale),
            min_mult=float(ml_size_min),
            max_mult=float(ml_size_max),
            mode=str(ml_size_mode),
            base_size=float(ml_size_base),
            pwin_threshold=float(ml_size_pwin_threshold),
        ) if ml_enabled_for_scores else None
    )

    prev_alloc: Optional[Allocation] = None
    rows = []
    opportunity_rows = []
    selected_opportunity_rows = []
    final_selected_rows = []

    # PortfolioEngine buffers (alineados 1:1 con common_ts)
    candles_by_symbol = {btc_sym: [], sol_sym: []}
    allocs = []
    # signal gating counters (for report)
    # no reset: preserve counters collected during loop (safe)
    try:
        signal_gate_applied_any = bool(signal_gate_applied_any)
    except UnboundLocalError:
        signal_gate_applied_any = False
    try:
        signal_gated_counts = dict(signal_gated_counts)
    except UnboundLocalError:
        signal_gated_counts = {}
    # --- signal diagnostics (counts) ---
    signal_side_counts = {}  # sym -> {side -> n}
    signal_skip_counts = {}  # sym -> n
    # --- end signal diagnostics ---


    for ts in common_ts:
        candles: Dict[str, Candle] = {
            btc_sym: _row_to_candle(ts, btc.loc[ts], features={
                'adx': float(btc_adx.loc[ts]) if ts in btc_adx.index else float('nan'),
                'atr': float(btc_atr.loc[ts]) if ts in btc_atr.index else float('nan'),
                'ema_fast': float(btc_ema_fast.loc[ts]) if ts in btc_ema_fast.index else float('nan'),
                'ema_slow': float(btc_ema_slow.loc[ts]) if ts in btc_ema_slow.index else float('nan'),
            }),
            sol_sym: _row_to_candle(ts, sol.loc[ts], features={
                'adx': float(sol_adx.loc[ts]) if ts in sol_adx.index else float('nan'),
                'atr': float(sol_atr.loc[ts]) if ts in sol_atr.index else float('nan'),
                'atrp': float(sol_atrp.loc[ts]) if ts in sol_atrp.index else float('nan'),
                'rsi': float(sol_rsi.loc[ts]) if ts in sol_rsi.index else float('nan'),
                'bb_mid': float(sol_bb_mid.loc[ts]) if ts in sol_bb_mid.index else float('nan'),
                'bb_up': float(sol_bb_up.loc[ts]) if ts in sol_bb_up.index else float('nan'),
                'bb_low': float(sol_bb_low.loc[ts]) if ts in sol_bb_low.index else float('nan'),
                'bb_width': float(sol_bb_width.loc[ts]) if ts in sol_bb_width.index else float('nan'),
                'ema_fast': float(sol_ema_fast.loc[ts]) if ts in sol_ema_fast.index else float('nan'),
                'ema_slow': float(sol_ema_slow.loc[ts]) if ts in sol_ema_slow.index else float('nan'),
                'donchian_high': float(sol_vol_dc_high.loc[ts]) if ts in sol_vol_dc_high.index else float('nan'),
                'donchian_low': float(sol_vol_dc_low.loc[ts]) if ts in sol_vol_dc_low.index else float('nan'),
                'range_expansion': float(sol_range_expansion.loc[ts]) if ts in sol_range_expansion.index else float('nan'),
            }),
        }

        # collect candles for PortfolioEngine
        candles_by_symbol[btc_sym].append(candles[btc_sym])
        candles_by_symbol[sol_sym].append(candles[sol_sym])

        signals = sig_engine.generate(candles)

        if signal_engine == "registry_portfolio":
            _last_opps = list(getattr(sig_engine, "last_opportunities", []) or [])

            if ml_enabled_for_scores and _last_opps:
                from hf.engines.ml_filter import select_model_for_signal

                for _opp in _last_opps:
                    _meta = dict(getattr(_opp, "meta", {}) or {})
                    _side = str(getattr(_opp, "side", "flat"))
                    _sym = str(getattr(_opp, "symbol", ""))

                    if _side == "flat":
                        _opp.meta = _meta
                        continue

                    _candle = candles.get(_sym)
                    if _candle is None:
                        _opp.meta = _meta
                        continue

                    _tmp_sig = Signal(
                        symbol=_sym,
                        side=_side,
                        strength=float(getattr(_opp, "strength", 0.0) or 0.0),
                        meta=_meta,
                    )
                    _feat_row = build_feature_row(_sym, _candle, _tmp_sig)
                    _chosen_model = select_model_for_signal(ml_model, ml_model_registry_loaded, _sym, _side)
                    _p_win = float(predict_proba(_chosen_model, _feat_row))
                    _meta["p_win"] = _p_win

                    if ml_score_position_sizer is not None:
                        _meta["ml_position_size_mult"] = float(ml_score_position_sizer.size_from_pwin(_p_win))
                        _meta["ml_position_size_scale"] = float(ml_score_position_sizer.scale)

                    _opp.meta = _meta

            _selected_opps = select_opportunities(_last_opps, mode=str(opportunity_selection_mode)) if _last_opps else []

            _rebuilt_signals = {}
            for _opp in _selected_opps:
                _rebuilt_signals[str(_opp.symbol)] = Signal(
                    symbol=str(_opp.symbol),
                    side=str(getattr(_opp, "side", "flat")),
                    strength=float(getattr(_opp, "strength", 0.0) or 0.0),
                    meta=dict(getattr(_opp, "meta", {}) or {}),
                )

            for _sym in candles.keys():
                if _sym not in _rebuilt_signals:
                    _rebuilt_signals[_sym] = Signal(
                        symbol=_sym,
                        side="flat",
                        strength=0.0,
                        meta={"engine": "opportunity_adapter", "skip": "no_opportunity"},
                    )

            signals = _rebuilt_signals

            for _opp in _last_opps:
                _meta = dict(getattr(_opp, "meta", {}) or {})
                opportunity_rows.append({
                    "ts": int(ts),
                    "ts_utc": pd.to_datetime(int(ts), unit="ms", utc=True).isoformat(),
                    "strategy_id": str(getattr(_opp, "strategy_id", "")),
                    "symbol": str(getattr(_opp, "symbol", "")),
                    "side": str(getattr(_opp, "side", "flat")),
                    "strength": float(getattr(_opp, "strength", 0.0) or 0.0),
                    "engine": _meta.get("engine"),
                    "registry_symbol": _meta.get("registry_symbol"),
                    "base_weight": float(_meta.get("base_weight", 1.0) or 1.0),
                    "p_win": float(_meta.get("p_win", 0.0) or 0.0),
                    "ml_position_size_mult": float(_meta.get("ml_position_size_mult", 0.0) or 0.0),
                    "is_active": bool(_opp.is_active()) if hasattr(_opp, "is_active") else False,
                    "competitive_score": float(compute_competitive_score(_opp)),
                    "post_ml_score": float(compute_post_ml_competitive_score(_opp)),
                })

            for _opp in _selected_opps:
                _meta = dict(getattr(_opp, "meta", {}) or {})
                _p_win = float(_meta.get("p_win", 0.0) or 0.0)
                _ml_mult = float(_meta.get("ml_position_size_mult", 0.0) or 0.0)

                selected_opportunity_rows.append({
                    "ts": int(ts),
                    "ts_utc": pd.to_datetime(int(ts), unit="ms", utc=True).isoformat(),
                    "strategy_id": str(getattr(_opp, "strategy_id", "")),
                    "symbol": str(getattr(_opp, "symbol", "")),
                    "side": str(getattr(_opp, "side", "flat")),
                    "strength": float(getattr(_opp, "strength", 0.0) or 0.0),
                    "engine": _meta.get("engine"),
                    "registry_symbol": _meta.get("registry_symbol"),
                    "base_weight": float(_meta.get("base_weight", 1.0) or 1.0),
                    "p_win": _p_win,
                    "ml_position_size_mult": _ml_mult,
                    "is_active": bool(_opp.is_active()) if hasattr(_opp, "is_active") else False,
                    "competitive_score": float(compute_competitive_score(_opp)),
                    "post_ml_score": float(compute_post_ml_competitive_score(_opp)),
                })
        raw_signals = dict(signals or {})

        if bool(ml_export_features):
            for _sym, _sig in (raw_signals or {}).items():
                if _sig is None:
                    continue
                _meta = dict(getattr(_sig, "meta", {}) or {})
                if "skip" in _meta:
                    continue
                _side = getattr(_sig, "side", "flat")
                if _side == "flat":
                    continue
                _candle = candles.get(_sym)
                if _candle is None:
                    continue
                _feat_row = build_feature_row(_sym, _candle, _sig)
                _side_preview = getattr(_sig, "side", "flat")
                _chosen_preview_model = None
                if ml_enabled_for_scores:
                    from hf.engines.ml_filter import select_model_for_signal
                    _chosen_preview_model = select_model_for_signal(ml_model, ml_model_registry_loaded, _sym, _side_preview)
                _p_win_preview = predict_proba(_chosen_preview_model, _feat_row)
                ml_feature_rows.append({
                    "ts": int(ts),
                    "symbol": _sym,
                    "side_raw": _side,
                    "side_final": _side,
                    "p_win": float(_p_win_preview),
                    "ml_rejected": 0,
                    **{col: float(_feat_row.get(col, 0.0)) for col in FEATURE_COLUMNS},
                })

        if bool(ml_filter):
            signals, _ml_rejected = apply_ml_filter_to_signals(
                candles=candles,
                signals=signals,
                model=ml_model,
                threshold=float(ml_threshold),
                model_registry=ml_model_registry_loaded,
            )
            for _sym, _n in (_ml_rejected or {}).items():
                ml_rejected_counts_by_symbol[_sym] = int(ml_rejected_counts_by_symbol.get(_sym, 0)) + int(_n)

            if bool(ml_export_features) and ml_feature_rows:
                _ts_now = int(ts)
                _idx = {}
                for i in range(len(ml_feature_rows) - 1, -1, -1):
                    _r = ml_feature_rows[i]
                    if int(_r.get("ts", -1)) != _ts_now:
                        break
                    _idx[(int(_r["ts"]), _r["symbol"])] = i

                for _sym, _sig in (signals or {}).items():
                    _k = (_ts_now, _sym)
                    if _k not in _idx:
                        continue
                    _row = ml_feature_rows[_idx[_k]]
                    _row["side_final"] = getattr(_sig, "side", "flat")
                    _meta2 = dict(getattr(_sig, "meta", {}) or {})
                    _row["p_win"] = float(_meta2.get("p_win", _row.get("p_win", 0.0)))
                    _row["ml_rejected"] = 1 if _meta2.get("reason") == "ml_filter" else 0

        if ml_position_sizer is not None:
            signals, _ml_sized = ml_position_sizer.apply_to_signals(
                candles=candles,
                signals=signals,
                model=ml_model,
                model_registry=ml_model_registry_loaded,
            )
            for _sym, _mult in (_ml_sized or {}).items():
                if float(_mult) != 1.0:
                    ml_position_sized_counts_by_symbol[_sym] = int(ml_position_sized_counts_by_symbol.get(_sym, 0)) + 1

        # --- count signal sides + skip flags ---
        for _sym, _sig in (signals or {}).items():
            _side = getattr(_sig, 'side', 'flat') if _sig is not None else 'flat'
            signal_side_counts.setdefault(_sym, {})[_side] = int(signal_side_counts.get(_sym, {}).get(_side, 0)) + 1
            _meta = getattr(_sig, 'meta', None) if _sig is not None else None
            if isinstance(_meta, dict) and ('skip' in _meta):
                signal_skip_counts[_sym] = int(signal_skip_counts.get(_sym, 0)) + 1
        # --- end counts ---
        regimes = reg_engine.evaluate(candles, signals)
        alloc = allocator.allocate(candles, signals, regimes, prev_alloc)

        if ml_position_sizer is not None:
            alloc = ml_position_sizer.apply_to_allocation(
                allocation=alloc,
                signals=signals,
            )

        # --- Signal gating (solo para engines reales, NO para 'flat') ---
        # Si el SignalEngine aplica al símbolo (meta no contiene 'skip') y la señal es flat,
        # forzamos el weight a 0.0. Esto permite que SignalEngine afecte allocations/equity
        # sin cambiar el comportamiento default (flat).
        if signal_engine != "flat":
            w0 = dict(alloc.weights)
            w2 = dict(w0)
            gated = {}
            for sym2, wgt in list(w0.items()):
                sig2 = signals.get(sym2)
                meta2 = getattr(sig2, "meta", None) if sig2 is not None else None
                if isinstance(meta2, dict) and ("skip" in meta2):
                    continue  # engine no aplica a este símbolo
                side2 = getattr(sig2, "side", "flat") if sig2 is not None else "flat"
                if side2 == "flat" and float(wgt) != 0.0:
                    w2[sym2] = 0.0
                    gated[sym2] = True

            # Detecta cambios reales (aunque gated esté vacío por mismatch de keys)
            changed = any(float(w2.get(k, 0.0)) != float(w0.get(k, 0.0)) for k in set(w0.keys()) | set(w2.keys()))
            if changed:
                alloc = Allocation(
                    weights=w2,
                    meta={
                        **dict(getattr(alloc, "meta", {}) or {}),
                        "signal_gate_applied": True,
                        "signal_gated": gated,
                    },
                )
        # --- end signal gating ---
        prev_alloc = alloc
        allocs.append(alloc)

        btc_meta = dict(getattr(signals.get(btc_sym), "meta", {}) or {})
        sol_meta = dict(getattr(signals.get(sol_sym), "meta", {}) or {})

        final_selected_rows.append({
            "ts": int(ts),
            "ts_utc": pd.to_datetime(int(ts), unit="ms", utc=True).isoformat(),
            "symbol": str(btc_sym),
            "strategy_id": btc_meta.get("strategy_id"),
            "engine": btc_meta.get("engine"),
            "registry_symbol": btc_meta.get("registry_symbol"),
            "side": str(getattr(signals.get(btc_sym), "side", "flat")),
            "strength": float(getattr(signals.get(btc_sym), "strength", 0.0) or 0.0),
            "base_weight": float(btc_meta.get("base_weight", 1.0) or 1.0),
            "p_win": float(btc_meta.get("p_win", 0.0) or 0.0),
            "ml_position_size_mult": float(btc_meta.get("ml_position_size_mult", 0.0) or 0.0),
            "competitive_score": float(btc_meta.get("competitive_score", 0.0) or 0.0),
            "post_ml_score": float((btc_meta.get("competitive_score", 0.0) or 0.0) * (btc_meta.get("p_win", 0.0) or 0.0)),
            "is_active": bool(getattr(signals.get(btc_sym), "is_active")()) if hasattr(signals.get(btc_sym), "is_active") else False,
        })

        final_selected_rows.append({
            "ts": int(ts),
            "ts_utc": pd.to_datetime(int(ts), unit="ms", utc=True).isoformat(),
            "symbol": str(sol_sym),
            "strategy_id": sol_meta.get("strategy_id"),
            "engine": sol_meta.get("engine"),
            "registry_symbol": sol_meta.get("registry_symbol"),
            "side": str(getattr(signals.get(sol_sym), "side", "flat")),
            "strength": float(getattr(signals.get(sol_sym), "strength", 0.0) or 0.0),
            "base_weight": float(sol_meta.get("base_weight", 1.0) or 1.0),
            "p_win": float(sol_meta.get("p_win", 0.0) or 0.0),
            "ml_position_size_mult": float(sol_meta.get("ml_position_size_mult", 0.0) or 0.0),
            "competitive_score": float(sol_meta.get("competitive_score", 0.0) or 0.0),
            "post_ml_score": float((sol_meta.get("competitive_score", 0.0) or 0.0) * (sol_meta.get("p_win", 0.0) or 0.0)),
            "is_active": bool(getattr(signals.get(sol_sym), "is_active")()) if hasattr(signals.get(sol_sym), "is_active") else False,
        })

        rows.append({
            "ts": int(ts),
            "ts_utc": pd.to_datetime(int(ts), unit="ms", utc=True).isoformat(),
            "w_btc": float(alloc.weights.get(btc_sym, 0.0)),
            "w_sol": float(alloc.weights.get(sol_sym, 0.0)),
            "btc_side": str(getattr(signals.get(btc_sym), "side", "flat")),
            "sol_side": str(getattr(signals.get(sol_sym), "side", "flat")),
            "btc_strength": float(getattr(signals.get(btc_sym), "strength", 0.0) or 0.0),
            "sol_strength": float(getattr(signals.get(sol_sym), "strength", 0.0) or 0.0),
            "btc_p_win": float(btc_meta.get("p_win", 0.0) or 0.0),
            "sol_p_win": float(sol_meta.get("p_win", 0.0) or 0.0),
            "btc_post_ml_score": float((btc_meta.get("competitive_score", 0.0) or 0.0) * (btc_meta.get("p_win", 0.0) or 0.0)),
            "sol_post_ml_score": float((sol_meta.get("competitive_score", 0.0) or 0.0) * (sol_meta.get("p_win", 0.0) or 0.0)),
            "btc_ml_size_mult": float(btc_meta.get("ml_position_size_mult", 0.0) or 0.0),
            "sol_ml_size_mult": float(sol_meta.get("ml_position_size_mult", 0.0) or 0.0),
            "case": (alloc.meta or {}).get("case", ""),
            "btc_strategy_id": btc_meta.get("strategy_id"),
            "btc_engine": btc_meta.get("engine"),
            "btc_reason": ((getattr(signals.get(btc_sym), "meta", {}) or {}).get("reason") or (getattr(signals.get(btc_sym), "meta", {}) or {}).get("skip")),
            "btc_registry_symbol": btc_meta.get("registry_symbol"),
            "btc_base_weight": float(btc_meta.get("base_weight", 1.0) or 1.0),
            "btc_competitive_score": float(btc_meta.get("competitive_score", 0.0) or 0.0),
            "sol_strategy_id": sol_meta.get("strategy_id"),
            "sol_engine": sol_meta.get("engine"),
            "sol_reason": ((getattr(signals.get(sol_sym), "meta", {}) or {}).get("reason") or (getattr(signals.get(sol_sym), "meta", {}) or {}).get("skip")),
            "sol_adx": float(((getattr(signals.get(sol_sym), "meta", {}) or {}).get("adx")) or 0.0),
            "sol_atrp": float(((getattr(signals.get(sol_sym), "meta", {}) or {}).get("atrp")) or 0.0),
            "sol_bb_width": float(((getattr(signals.get(sol_sym), "meta", {}) or {}).get("bb_width")) or 0.0),
            "sol_registry_symbol": sol_meta.get("registry_symbol"),
            "sol_base_weight": float(sol_meta.get("base_weight", 1.0) or 1.0),
            "sol_competitive_score": float(sol_meta.get("competitive_score", 0.0) or 0.0),
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"results/pipeline_allocations_{name}.csv", index=False)

    if opportunity_rows:
        pd.DataFrame(opportunity_rows).to_csv(f"results/opportunity_book_{name}.csv", index=False)

    if selected_opportunity_rows:
        _sel_engine_name = "portfolio_registry" if str(signal_engine) == "registry_portfolio" else str(signal_engine)
        pd.DataFrame(selected_opportunity_rows).to_csv(
            f"results/opportunity_book_{_sel_engine_name}_sel_{opportunity_selection_mode}.csv",
            index=False,
        )

    if final_selected_rows:
        _sel_engine_name = "portfolio_registry" if str(signal_engine) == "registry_portfolio" else str(signal_engine)
        pd.DataFrame(final_selected_rows).to_csv(
            f"results/opportunity_book_{_sel_engine_name}_sel_{opportunity_selection_mode}_final.csv",
            index=False,
        )

    if bool(ml_export_features):
        _ml_out = ml_features_out or f"results/ml_features_{name}.csv"
        _ml_df = pd.DataFrame(ml_feature_rows)

        if not _ml_df.empty:
            _close_map = {
                btc_sym: btc["close"].astype(float),
                sol_sym: sol["close"].astype(float),
            }

            _parts = []
            for _sym, _g in _ml_df.groupby("symbol", sort=False):
                _g = _g.sort_values("ts").copy()
                _close_series = _close_map.get(_sym)
                if _close_series is None:
                    _parts.append(_g)
                    continue

                _px_now = _g["ts"].map(_close_series.to_dict()).astype(float)

                for _h in (1, 3, 6, 12):
                    _future_close = _g["ts"].map(_close_series.shift(-_h).to_dict()).astype(float)
                    _ret = (_future_close / _px_now) - 1.0

                    _g[f"future_ret_{_h}"] = _ret

                    _is_long = _g["side_raw"] == "long"
                    _is_short = _g["side_raw"] == "short"

                    _y = pd.Series(0, index=_g.index, dtype="int64")
                    _y.loc[_is_long & (_ret > 0)] = 1
                    _y.loc[_is_short & (_ret < 0)] = 1
                    _g[f"y_win_{_h}"] = _y

                _parts.append(_g)

            _ml_df = pd.concat(_parts, ignore_index=True)

        _ml_df.to_csv(_ml_out, index=False)

    # Portfolio performance (equity/drawdown) - research-grade, no fees/slippage
    pe = SimplePortfolioEngine(initial_equity=1000.0)
    perf = pe.run(candles_by_symbol=candles_by_symbol, allocations=allocs, symbols=(btc_sym, sol_sym))
    perf.reset_index().to_csv(f"results/pipeline_equity_{name}.csv", index=False)

    # Portfolio metrics (hedge-fund style summary)
    metrics = PortfolioMetricsEngine(risk_free_rate_annual=0.0).compute(perf_df=perf.reset_index(), alloc_df=df)
    with open(f"results/pipeline_metrics_{name}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    # Net-of-costs simulation (turnover-based fees+slippage)
    _fee_bps = float(locals().get('fee_bps', 0.0) or 0.0)
    _slip_bps = float(locals().get('slippage_bps', 0.0) or 0.0)
    if _fee_bps > 0.0 or _slip_bps > 0.0:
        sim = ExecutionSimulator(
            cost_model=ExecutionCostModel(fee_bps=_fee_bps, slippage_bps=_slip_bps),
            initial_equity=float(perf['equity'].iloc[0]) if 'equity' in perf.columns else 1000.0,
        )
        net = sim.apply_costs(perf_df=perf.reset_index(), alloc_df=df)
        net.to_csv(f"results/pipeline_equity_net_{name}.csv", index=False)

        perf_net = pd.DataFrame({
            'ts': net['ts'],
            'equity': net['net_equity'],
            'port_ret': net['net_ret'],
            'drawdown_pct': net['net_drawdown_pct'],
        })
        metrics_net = PortfolioMetricsEngine(risk_free_rate_annual=0.0).compute(perf_df=perf_net, alloc_df=df)
        with open(f"results/pipeline_metrics_net_{name}.json", "w", encoding="utf-8") as f:
            json.dump(metrics_net, f, indent=2, sort_keys=True)

    
        # --- signal gating summary for report (derived from signal counts; robust) ---
    # Usamos los contadores agregados del loop:
    #   flats_not_skipped = side_counts['flat'] - skip_counts
    # Esto evita depender de alloc.meta, que puede no persistir en algunos caminos.
    signal_gate_applied_any = False
    signal_gated_counts = {}
    for _sym, _sc in (signal_side_counts or {}).items():
        _flat = int((_sc or {}).get('flat', 0) or 0)
        _skip = int((signal_skip_counts or {}).get(_sym, 0) or 0)
        _flats_not_skipped = _flat - _skip
        if _flats_not_skipped > 0:
            signal_gate_applied_any = True
            signal_gated_counts[_sym] = int(_flats_not_skipped)
    # --- end gating summary ---
# Consolidated report (gross + net + case counts + config)
    ReportEngine().build(
        name=name,
        config={
            'both_btc_weight': float(both_btc_weight),
            'sticky_when_off': bool(sticky_when_off),
            'fallback_btc_weight': float(fallback_btc_weight),
            'fallback_sol_weight': float(fallback_sol_weight),
            'fee_bps': float(fee_bps),
            'slippage_bps': float(slippage_bps),
            'sol_atrp_min': float(sol_atrp_min),
            'sol_adx_max': float(sol_adx_max),
            'sol_vol_breakout_lookback': int(sol_vol_breakout_lookback),
            'sol_vol_adx_min': float(sol_vol_adx_min),
            'sol_vol_atrp_min': float(sol_vol_atrp_min),
            'sol_vol_atrp_max': float(sol_vol_atrp_max),
            'sol_vol_range_expansion_min': float(sol_vol_range_expansion_min),
            'sol_vol_confirm_close_buffer': float(sol_vol_confirm_close_buffer),
            'sol_trend_pullback_rsi_long_min': float(sol_trend_pullback_rsi_long_min),
            'sol_trend_pullback_rsi_long_max': float(sol_trend_pullback_rsi_long_max),
            'sol_trend_pullback_rsi_short_min': float(sol_trend_pullback_rsi_short_min),
            'sol_trend_pullback_rsi_short_max': float(sol_trend_pullback_rsi_short_max),
            'sol_trend_pullback_ema_pullback_max': float(sol_trend_pullback_ema_pullback_max),
            'sol_trend_pullback_atrp_min': float(sol_trend_pullback_atrp_min),
            'sol_trend_pullback_atrp_max': float(sol_trend_pullback_atrp_max),
            'sol_trend_pullback_require_adx': bool(sol_trend_pullback_require_adx),
            'sol_trend_pullback_adx_min': float(sol_trend_pullback_adx_min),
            'btc_adx_min': float(btc_adx_min),
            'btc_slope_min': float(btc_slope_min),
            'signal_engine': str(signal_engine),
            'signal_gate_applied': bool(signal_gate_applied_any),
            'signal_gated_counts': dict(signal_gated_counts),
            'signal_side_counts': dict(signal_side_counts),
            'signal_skip_counts': dict(signal_skip_counts),
            'ml_filter_enabled': bool(ml_filter),
            'ml_position_sizing_enabled': bool(ml_position_sizing),
            'ml_size_mode': str(ml_size_mode),
            'ml_size_scale': float(ml_size_scale),
            'ml_size_min': float(ml_size_min),
            'ml_size_max': float(ml_size_max),
            'ml_size_base': float(ml_size_base),
            'ml_size_pwin_threshold': float(ml_size_pwin_threshold),
            'ml_model_path': ml_model_path,
            'ml_model_registry_path': ml_model_registry,
            'ml_threshold': float(ml_threshold),
            'ml_rejected_counts_by_symbol': dict(ml_rejected_counts_by_symbol),
            'ml_position_sized_counts_by_symbol': dict(ml_position_sized_counts_by_symbol),
        },
        write=True,
    )

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pipeline")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--exchange", default="bitget")
    ap.add_argument("--cache-dir", default=".cache/ohlcv")
    ap.add_argument("--both-btc-weight", type=float, default=0.75)
    ap.add_argument("--sticky-when-off", action="store_true", help="If set: when both regimes are OFF, keep previous weights")
    ap.add_argument("--fallback-btc-weight", type=float, default=1.0)
    ap.add_argument("--fallback-sol-weight", type=float, default=0.0)
    ap.add_argument("--fee-bps", type=float, default=0.0, help="Execution fee in bps applied to turnover (net simulation)")
    ap.add_argument("--slippage-bps", type=float, default=0.0, help="Execution slippage in bps applied to turnover (net simulation)")
    ap.add_argument("--trades-csv", default="results/portfolio_trades_v8ml_regime3_flags.csv")
    ap.add_argument("--refresh-cache", action="store_true")

    # Regime3 thresholds (core)
    ap.add_argument("--sol-atrp-min", type=float, default=0.0030)
    ap.add_argument("--sol-adx-max", type=float, default=24.0)

    # SOL SignalEngine (BBRSI) thresholds (separados de Regime3)
    ap.add_argument("--sol-rsi-long-max", type=float, default=36.0)
    ap.add_argument("--sol-rsi-short-min", type=float, default=64.0)
    ap.add_argument("--sol-adx-hard-signal", type=float, default=24.0)
    ap.add_argument("--sol-atrp-min-signal", type=float, default=0.003279)
    ap.add_argument("--sol-atrp-max-signal", type=float, default=0.0350)
    ap.add_argument("--sol-bb-width-min", type=float, default=0.0041)
    ap.add_argument("--sol-bb-width-max", type=float, default=0.120)
    ap.add_argument("--sol-bb-period", type=int, default=20)
    ap.add_argument("--sol-bb-std", type=float, default=2.0)
    ap.add_argument("--sol-vol-breakout-lookback", type=int, default=20)
    ap.add_argument("--sol-vol-adx-min", type=float, default=18.0)
    ap.add_argument("--sol-vol-atrp-min", type=float, default=0.008)
    ap.add_argument("--sol-vol-atrp-max", type=float, default=0.080)
    ap.add_argument("--sol-vol-range-expansion-min", type=float, default=1.10)
    ap.add_argument("--sol-trend-pullback-rsi-long-min", type=float, default=40.0)
    ap.add_argument("--sol-trend-pullback-rsi-long-max", type=float, default=55.0)
    ap.add_argument("--sol-trend-pullback-rsi-short-min", type=float, default=45.0)
    ap.add_argument("--sol-trend-pullback-rsi-short-max", type=float, default=60.0)
    ap.add_argument("--sol-trend-pullback-ema-pullback-max", type=float, default=0.015)
    ap.add_argument("--sol-trend-pullback-atrp-min", type=float, default=0.004)
    ap.add_argument("--sol-trend-pullback-atrp-max", type=float, default=0.050)
    ap.add_argument("--sol-trend-pullback-require-adx", action="store_true")
    ap.add_argument("--sol-trend-pullback-adx-min", type=float, default=18.0)
    ap.add_argument("--sol-vol-confirm-close-buffer", type=float, default=0.0)
    ap.add_argument("--ml-filter", action="store_true", help="Enable optional ML probability-of-win filter on raw signals.")
    ap.add_argument("--ml-model-path", default=None, help="Path to serialized ML model (pickle/joblib-compatible pickle load).")
    ap.add_argument("--ml-model-registry", default=None, help="Optional JSON registry mapping 'SYMBOL|side' to model path.")
    ap.add_argument("--ml-threshold", type=float, default=0.55, help="Reject signal if p_win < threshold.")
    ap.add_argument("--ml-export-features", action="store_true", help="Export ML feature rows for raw/final signals.")
    ap.add_argument("--ml-features-out", default=None, help="Optional CSV path for exported ML features.")
    ap.add_argument("--ml-position-sizing", action="store_true", help="Enable ML-based position sizing using p_win -> size multiplier.")
    ap.add_argument("--ml-size-mode", type=str, default="linear_edge", choices=["linear_edge", "calibrated"], help="Sizing transform mode.")
    ap.add_argument("--ml-size-scale", type=float, default=1.0, help="Scale factor for the selected sizing mode.")
    ap.add_argument("--ml-size-min", type=float, default=0.0, help="Minimum size multiplier after transform.")
    ap.add_argument("--ml-size-max", type=float, default=1.0, help="Maximum size multiplier after transform.")
    ap.add_argument("--ml-size-base", type=float, default=0.25, help="Base position size for calibrated mode.")
    ap.add_argument("--ml-size-pwin-threshold", type=float, default=0.55, help="Probability threshold used by calibrated mode.")
    ap.add_argument("--btc-adx-min", type=float, default=18.0)
    ap.add_argument("--strategy-registry", default="artifacts/strategy_registry.json", help="Strategy registry JSON used by registry_portfolio.")
    ap.add_argument(
        "--opportunity-selection-mode",
        default="best_per_symbol",
        choices=["best_per_symbol", "all", "competitive"],
        help="Opportunity selection mode for registry_portfolio.",
    )
    ap.add_argument("--btc-slope-min", type=float, default=1.5)
    ap.add_argument(
        "--signal-engine",
        choices=["flat", "btc_trend", "sol_bbrsi", "sol_vol_breakout", "portfolio", "registry_portfolio", "sol_trend_pullback"],
        default="flat",
        help="Signal engine to use: flat, btc_trend, sol_bbrsi, portfolio, registry_portfolio (default: flat placeholder).",
    )

    ap.add_argument("--sticky-flat", action="store_true", help="If set: keep previous weights when flat (no active trades)")
    args = ap.parse_args()

    df = run(
        name=args.name,
        start=args.start,
        end=args.end,
        exchange=args.exchange,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
        trades_csv=args.trades_csv,
        sol_atrp_min=float(args.sol_atrp_min),
        sol_adx_max=float(args.sol_adx_max),
        btc_adx_min=float(args.btc_adx_min),
        btc_slope_min=float(args.btc_slope_min),
        signal_engine=str(args.signal_engine),
        sticky_flat=bool(args.sticky_flat),
        both_btc_weight=float(args.both_btc_weight),
        sticky_when_off=bool(args.sticky_when_off),
        fallback_btc_weight=float(args.fallback_btc_weight),
        fallback_sol_weight=float(args.fallback_sol_weight),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        sol_rsi_long_max=float(args.sol_rsi_long_max),
        sol_rsi_short_min=float(args.sol_rsi_short_min),
        sol_adx_hard_signal=float(args.sol_adx_hard_signal),
        sol_atrp_min_signal=float(args.sol_atrp_min_signal),
        sol_atrp_max_signal=float(args.sol_atrp_max_signal),
        sol_bb_width_min=float(args.sol_bb_width_min),
        sol_bb_width_max=float(args.sol_bb_width_max),
        sol_bb_period=int(args.sol_bb_period),
        sol_bb_std=float(args.sol_bb_std),
        sol_vol_breakout_lookback=int(args.sol_vol_breakout_lookback),
        sol_vol_adx_min=float(args.sol_vol_adx_min),
        sol_vol_atrp_min=float(args.sol_vol_atrp_min),
        sol_vol_atrp_max=float(args.sol_vol_atrp_max),
        sol_vol_range_expansion_min=float(args.sol_vol_range_expansion_min),
        sol_vol_confirm_close_buffer=float(args.sol_vol_confirm_close_buffer),
        sol_trend_pullback_rsi_long_min=float(args.sol_trend_pullback_rsi_long_min),
        sol_trend_pullback_rsi_long_max=float(args.sol_trend_pullback_rsi_long_max),
        sol_trend_pullback_rsi_short_min=float(args.sol_trend_pullback_rsi_short_min),
        sol_trend_pullback_rsi_short_max=float(args.sol_trend_pullback_rsi_short_max),
        sol_trend_pullback_ema_pullback_max=float(args.sol_trend_pullback_ema_pullback_max),
        sol_trend_pullback_atrp_min=float(args.sol_trend_pullback_atrp_min),
        sol_trend_pullback_atrp_max=float(args.sol_trend_pullback_atrp_max),
        sol_trend_pullback_require_adx=bool(args.sol_trend_pullback_require_adx),
        sol_trend_pullback_adx_min=float(args.sol_trend_pullback_adx_min),
        ml_filter=bool(args.ml_filter),
        ml_model_path=args.ml_model_path,
        ml_model_registry=args.ml_model_registry,
        ml_threshold=float(args.ml_threshold),
        ml_export_features=bool(args.ml_export_features),
        ml_features_out=args.ml_features_out,
        ml_position_sizing=bool(args.ml_position_sizing),
        ml_size_scale=float(args.ml_size_scale),
        ml_size_min=float(args.ml_size_min),
        ml_size_max=float(args.ml_size_max),
        ml_size_mode=str(args.ml_size_mode),
        ml_size_base=float(args.ml_size_base),
        ml_size_pwin_threshold=float(args.ml_size_pwin_threshold),
        strategy_registry_path=str(args.strategy_registry),
        opportunity_selection_mode=str(args.opportunity_selection_mode),
    )
    print(f"Saved -> results/pipeline_allocations_{args.name}.csv (rows={len(df)})")

if __name__ == "__main__":
    main()
