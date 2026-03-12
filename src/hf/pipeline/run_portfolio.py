from __future__ import annotations
import json

import argparse
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Optional

import pandas as pd
import numpy as np

from hf.core.types import Candle, Allocation, Signal
from hf.engines.alloc_regime import RegimeAllocator
from hf.engines.alloc_multi_strategy import MultiStrategyAllocator
from hf.engines.portfolio_engine import SimplePortfolioEngine
from hf.engines.portfolio_metrics import PortfolioMetricsEngine
from hf.engines.report_engine import ReportEngine
from hf.engines.execution_simulator import ExecutionCostModel, ExecutionSimulator
from hf.engines.regime_regime3 import Regime3Engine

from hf.engines.signals import PortfolioSignalEngine, RegistryPortfolioSignalEngine, FlatSignalEngine, BtcTrendSignalEngine, SolBbrsiSignalEngine
from hf.engines.signals.sol_vol_breakout_signal import SolVolBreakoutSignalEngine
from hf.engines.signals.sol_trend_pullback_signal import SolTrendPullbackSignalEngine
from hf.engines.signals.sol_extreme_mr_signal import SolExtremeMrSignalEngine
from hf.engines.signals.sol_vol_compression_signal import SolVolCompressionSignalEngine
from hf.engines.signals.sol_vol_expansion_signal import SolVolExpansionSignalEngine
from hf.engines.opportunity_book import select_opportunities, compute_competitive_score, compute_post_ml_competitive_score
from hf.engines.ml_filter import FEATURE_COLUMNS, apply_ml_filter_to_signals, build_feature_row, load_model, load_model_registry, predict_proba
from hf.engines.ml_position_sizer import MlPositionSizingEngine
from hf.engines.subposition_planner import SubPositionPlanner
from hf.engines.position_cluster import PositionClusterBuilder
from hf.engines.cluster_risk_engine import ClusterRiskEngine
from hf.engines.execution_planner import ExecutionPlanner
from hf.execution.order_simulator import OrderSimulator
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


def _parse_subpos_weights(raw: Optional[str]) -> Optional[list[float]]:
    if raw is None:
        return None
    parts = [x.strip() for x in str(raw).split(",")]
    vals = []
    for part in parts:
        if not part:
            continue
        vals.append(float(part))
    return vals or None


def _parse_float_list(raw: Optional[str]) -> Optional[list[float]]:
    if raw is None:
        return None
    parts = [x.strip() for x in str(raw).split(",")]
    vals = []
    for part in parts:
        if not part:
            continue
        vals.append(float(part))
    return vals or None


def _parse_int_list(raw: Optional[str]) -> Optional[list[int]]:
    if raw is None:
        return None
    parts = [x.strip() for x in str(raw).split(",")]
    vals = []
    for part in parts:
        if not part:
            continue
        vals.append(int(part))
    return vals or None


def _normalize_subpos_weights(weights: list[float], total_target_weight: float) -> list[float]:
    total = float(sum(float(x) for x in weights))
    if total <= 0.0:
        return []
    scale = float(total_target_weight) / total
    return [float(x) * scale for x in weights]


def _plan_subpositions_for_symbol(
    *,
    planner: SubPositionPlanner,
    symbol: str,
    strategy_id: str,
    side: str,
    total_target_weight: float,
    raw_custom_weights: Optional[str],
    meta: Optional[dict],
) -> dict:
    plan = planner.plan(
        symbol=str(symbol),
        strategy_id=str(strategy_id or ""),
        side=str(side or "flat"),
        total_target_weight=float(total_target_weight or 0.0),
        opportunity_meta=dict(meta or {}),
    )

    custom_weights = _parse_subpos_weights(raw_custom_weights)
    if custom_weights is not None:
        custom_weights = [float(x) for x in custom_weights if float(x) > 0.0]
        if float(total_target_weight or 0.0) > 0.0 and custom_weights:
            plan.slices = _normalize_subpos_weights(custom_weights, float(total_target_weight or 0.0))

    return {
        "count": int(len(plan.slices)),
        "weights": ",".join(f"{float(x):.8f}" for x in plan.slices),
        "plan": plan,
    }


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
    ml_thresholds_path: Optional[str] = None,
    ml_export_features: bool = False,
    ml_features_out: Optional[str] = None,
    ml_position_sizing: bool = False,
    ml_size_scale: float = 1.0,
    ml_size_min: float = 0.0,
    ml_size_max: float = 1.0,
    ml_size_mode: str = "linear_edge",
    ml_size_base: float = 0.25,
    ml_size_pwin_threshold: float = 0.55,
    ml_size_artifact_path: str = "artifacts/ml_position_size_map_v1.json",
    strategy_registry_path: str = "artifacts/strategy_registry.json",
    opportunity_selection_mode: str = "best_per_symbol",
    allocation_engine_mode: str = "regime",
    strategy_score_power: float = 1.0,
    strategy_symbol_score_agg: str = "sum",
    allocator_blend_alpha: float = 0.40,
    allocator_rebalance_deadband: float = 0.0,
    allocator_symbol_cap: float = 1.0,
    allocator_max_step_per_bar: float = 1.0,
    portfolio_riskoff_filter: bool = False,
    portfolio_riskoff_btc_adx_min: float = 18.0,
    portfolio_riskoff_btc_slope_min: float = 1.5,
    portfolio_riskoff_sol_atrp_max: float = 0.035,
    portfolio_risk_scale_enable: bool = False,
    portfolio_risk_scale_atrp_low: float = 0.010,
    portfolio_risk_scale_atrp_high: float = 0.025,
    portfolio_risk_scale_floor: float = 0.35,
    strategy_regime_gating: bool = False,
    allocator_smoothing_alpha: float = 0.50,
    allocator_smoothing_snap_eps: float = 0.02,
    btc_subpos_count: int = 1,
    sol_subpos_count: int = 1,
    btc_subpos_weights: Optional[str] = None,
    sol_subpos_weights: Optional[str] = None,
    execution_mode: str = "market",
    execution_ladder_offsets: Optional[str] = None,
    execution_time_offsets: Optional[str] = None,
    execution_slippage_bps: float = 2.0,
    execution_size_slippage_factor: float = 10.0,
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
                "sol_extreme_mr_signal": lambda cfg: SolExtremeMrSignalEngine(
                    **dict(cfg.get("params", {}) or {})
                ),
                "sol_vol_compression_signal": lambda cfg: SolVolCompressionSignalEngine(
                    **dict(cfg.get("params", {}) or {})
                ),
                "sol_vol_expansion_signal": lambda cfg: SolVolExpansionSignalEngine(
                    **dict(cfg.get("params", {}) or {})
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
    multi_allocator = MultiStrategyAllocator(
        score_power=float(strategy_score_power),
        symbol_score_agg=str(strategy_symbol_score_agg),
        weight_blend_alpha=float(allocator_blend_alpha),
        rebalance_deadband=float(allocator_rebalance_deadband),
        symbol_cap=float(allocator_symbol_cap),
    )

    ml_enabled_for_scores = bool(ml_filter or ml_position_sizing or ml_model_path or ml_model_registry)
    ml_model = load_model(ml_model_path) if ml_enabled_for_scores else None
    ml_model_registry_loaded = load_model_registry(ml_model_registry) if ml_enabled_for_scores else {}

    ml_thresholds_map = {}
    if ml_thresholds_path:
        try:
            _thr_payload = json.loads(Path(ml_thresholds_path).read_text(encoding="utf-8"))
            if isinstance(_thr_payload, dict):
                if isinstance(_thr_payload.get("thresholds"), dict):
                    ml_thresholds_map = {
                        str(k): float(v)
                        for k, v in (_thr_payload.get("thresholds") or {}).items()
                    }
                else:
                    ml_thresholds_map = {
                        str(k): float(v)
                        for k, v in _thr_payload.items()
                        if k != "quantile"
                    }
        except Exception:
            ml_thresholds_map = {}

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
        artifact_path=str(ml_size_artifact_path),
    ) if bool(ml_position_sizing) else None

    ml_score_position_sizer = ml_position_sizer or (
        MlPositionSizingEngine(
            scale=float(ml_size_scale),
            min_mult=float(ml_size_min),
            max_mult=float(ml_size_max),
            mode=str(ml_size_mode),
            base_size=float(ml_size_base),
            pwin_threshold=float(ml_size_pwin_threshold),
            artifact_path=str(ml_size_artifact_path),
        ) if ml_enabled_for_scores else None
    )

    prev_alloc: Optional[Allocation] = None
    btc_subpos_planner = SubPositionPlanner(slices=max(1, int(btc_subpos_count)))
    sol_subpos_planner = SubPositionPlanner(slices=max(1, int(sol_subpos_count)))
    cluster_builder = PositionClusterBuilder()
    cluster_risk_engine = ClusterRiskEngine(
        max_cluster_weight=1.0,
        max_subpositions=max(int(btc_subpos_count), int(sol_subpos_count), 1),
        allow_zero_weight_clusters=True,
    )
    execution_planner = ExecutionPlanner()
    
    order_sim = OrderSimulator()
    order_sim.slippage.base_slippage_bps = float(execution_slippage_bps)
    order_sim.slippage.size_slippage_factor = float(execution_size_slippage_factor)

    parsed_execution_ladder_offsets = _parse_float_list(execution_ladder_offsets)
    parsed_execution_time_offsets = _parse_int_list(execution_time_offsets)
    rows = []
    opportunity_rows = []
    selected_opportunity_rows = []
    strategy_allocation_rows = []
    final_selected_rows = []
    execution_rows = []

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
        selected_opps_for_alloc = []
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
                    _strategy_id = str(getattr(_opp, "strategy_id", "") or "")

                    if _side == "flat":
                        _meta["p_win"] = 0.0
                        _meta["ml_threshold"] = 0.0
                        _meta["ml_rejected"] = 0
                        _meta["ml_position_size_mult"] = 0.0
                        _opp.meta = _meta
                        continue

                    _candle = candles.get(_sym)
                    if _candle is None:
                        _meta["p_win"] = 0.0
                        _meta["ml_threshold"] = 0.0
                        _meta["ml_rejected"] = 0
                        _meta["ml_position_size_mult"] = 0.0
                        _opp.meta = _meta
                        continue

                    _meta["strategy_id"] = _strategy_id
                    _meta["competitive_score"] = float(compute_competitive_score(_opp))

                    _tmp_sig = Signal(
                        symbol=_sym,
                        side=_side,
                        strength=float(getattr(_opp, "strength", 0.0) or 0.0),
                        meta=_meta,
                    )
                    _feat_row = build_feature_row(_sym, _candle, _tmp_sig)
                    _chosen_model = select_model_for_signal(ml_model, ml_model_registry_loaded, _sym, _side)
                    _p_win = float(predict_proba(_chosen_model, _feat_row))

                    _thr = float(ml_threshold)
                    if ml_thresholds_map and _strategy_id:
                        try:
                            _thr = float(ml_thresholds_map.get(_strategy_id, _thr))
                        except Exception:
                            _thr = float(ml_threshold)

                    _meta["p_win"] = _p_win
                    _meta["ml_threshold"] = _thr

                    if bool(ml_filter) and _p_win < _thr:
                        _meta["ml_rejected"] = 1
                        _meta["ml_position_size_mult"] = 0.0
                        _opp.side = "flat"
                        _opp.strength = 0.0
                    else:
                        _meta["ml_rejected"] = 0
                        if ml_score_position_sizer is not None:
                            _meta["ml_position_size_mult"] = float(ml_score_position_sizer.size_from_pwin(_p_win))
                            _meta["ml_position_size_scale"] = float(ml_score_position_sizer.scale)
                        else:
                            _meta["ml_position_size_mult"] = 1.0

                    _opp.meta = _meta

            _selected_opps = select_opportunities(_last_opps, mode=str(opportunity_selection_mode)) if _last_opps else []
            selected_opps_for_alloc = list(_selected_opps)

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

            if bool(ml_export_features):
                _selected_keys = {
                    (
                        str(getattr(_opp, "strategy_id", "")),
                        str(getattr(_opp, "symbol", "")),
                        str(getattr(_opp, "side", "flat")),
                        round(float(getattr(_opp, "strength", 0.0) or 0.0), 10),
                    )
                    for _opp in (_selected_opps or [])
                }

                for _opp in _last_opps:
                    _meta = dict(getattr(_opp, "meta", {}) or {})
                    _sym = str(getattr(_opp, "symbol", ""))
                    _side = str(getattr(_opp, "side", "flat"))
                    _strategy_id = str(getattr(_opp, "strategy_id", ""))
                    _strength = float(getattr(_opp, "strength", 0.0) or 0.0)

                    if _side == "flat":
                        continue

                    _candle = candles.get(_sym)
                    if _candle is None:
                        continue

                    _tmp_sig = Signal(
                        symbol=_sym,
                        side=_side,
                        strength=_strength,
                        meta=_meta,
                    )
                    _feat_row = build_feature_row(_sym, _candle, _tmp_sig)

                    _chosen_preview_model = None
                    if ml_enabled_for_scores:
                        from hf.engines.ml_filter import select_model_for_signal
                        _chosen_preview_model = select_model_for_signal(
                            ml_model,
                            ml_model_registry_loaded,
                            _sym,
                            _side,
                        )

                    _p_win_preview = float(_meta.get("p_win", 0.0) or 0.0)
                    if _p_win_preview <= 0.0:
                        _p_win_preview = float(predict_proba(_chosen_preview_model, _feat_row))

                    _row_key = (
                        _strategy_id,
                        _sym,
                        _side,
                        round(_strength, 10),
                    )
                    _selected_flag = 1 if _row_key in _selected_keys else 0

                    ml_feature_rows.append({
                        "ts": int(ts),
                        "ts_utc": pd.to_datetime(int(ts), unit="ms", utc=True).isoformat(),
                        "strategy_id": _strategy_id,
                        "engine": _meta.get("engine"),
                        "registry_symbol": _meta.get("registry_symbol"),
                        "symbol": _sym,
                        "side_raw": _side,
                        "side_final": _side if _selected_flag else "flat",
                        "selected_by_opportunity_selector": int(_selected_flag),
                        "ml_rejected": 0,
                        "p_win": float(_p_win_preview),
                        "strength": _strength,
                        "base_weight": float(_meta.get("base_weight", 1.0) or 1.0),
                        "competitive_score": float(compute_competitive_score(_opp)),
                        "post_ml_score": float(compute_post_ml_competitive_score(_opp)),
                        **{col: float(_feat_row.get(col, 0.0)) for col in FEATURE_COLUMNS},
                    })

        raw_signals = dict(signals or {})

        if bool(ml_export_features) and str(signal_engine) != "registry_portfolio":
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
                threshold_map=ml_thresholds_map,
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

        if bool(strategy_regime_gating) and selected_opps_for_alloc:
            _btc_regime_on = bool(getattr(regimes.get(btc_sym), "on", False))
            _sol_regime_on = bool(getattr(regimes.get(sol_sym), "on", False))

            _gated_opps = []
            for _opp in list(selected_opps_for_alloc):
                _sid = str(getattr(_opp, "strategy_id", "") or "")
                _allow = True

                if _sid == "sol_bbrsi":
                    _allow = _sol_regime_on
                elif _sid in {"btc_trend", "btc_trend_loose", "sol_trend_pullback"}:
                    _allow = _btc_regime_on

                if _allow:
                    _gated_opps.append(_opp)

            selected_opps_for_alloc = list(_gated_opps)

        if (
            str(signal_engine) == "registry_portfolio"
            and str(allocation_engine_mode) == "multi_strategy"
            and selected_opps_for_alloc
        ):
            alloc = multi_allocator.allocate_from_opportunities(
                candles=candles,
                opportunities=selected_opps_for_alloc,
                prev_allocation=prev_alloc,
            )
        else:
            alloc = allocator.allocate(candles, signals, regimes, prev_alloc)

        _alloc_meta = dict(getattr(alloc, "meta", {}) or {})
        _strategy_weights = dict(_alloc_meta.get("strategy_weights", {}) or {})
        _symbol_budget = dict(_alloc_meta.get("symbol_budget", {}) or {})

        alloc_raw_weights = dict(getattr(alloc, "weights", {}) or {})
        alloc_after_ml_weights = dict(alloc_raw_weights)
        alloc_after_smoothing_weights = dict(alloc_raw_weights)
        alloc_after_signal_gating_weights = dict(alloc_raw_weights)

        for _k, _w in _strategy_weights.items():
            if "::" in _k:
                _symbol, _strategy_id = _k.split("::", 1)
            else:
                _symbol, _strategy_id = _k, "unknown_strategy"

            strategy_allocation_rows.append({
                "ts": int(ts),
                "symbol": str(_symbol),
                "strategy_id": str(_strategy_id),
                "strategy_weight": float(_w or 0.0),
                "symbol_budget": float(_symbol_budget.get(_symbol, 0.0) or 0.0),
                "allocation_case": str(_alloc_meta.get("case", "")),
            })

        if ml_position_sizer is not None:
            alloc = ml_position_sizer.apply_to_allocation(
                allocation=alloc,
                signals=signals,
            )

        alloc_after_ml_weights = dict(getattr(alloc, "weights", {}) or {})

        # Suavizado conservador de transiciones en multi_strategy.
        _alloc_case_now = str((getattr(alloc, "meta", {}) or {}).get("case", ""))
        if prev_alloc is not None and _alloc_case_now == "multi_strategy" and float(allocator_smoothing_alpha) > 0.0:
            _alpha = float(allocator_smoothing_alpha)
            _snap_eps = float(allocator_smoothing_snap_eps)
            _keys = set(dict(prev_alloc.weights).keys()) | set(dict(alloc.weights).keys())
            _smoothed_weights = {}
            for _k in _keys:
                _prev_w = float(dict(prev_alloc.weights).get(_k, 0.0) or 0.0)
                _curr_w = float(dict(alloc.weights).get(_k, 0.0) or 0.0)
                _w = _prev_w + _alpha * (_curr_w - _prev_w)
                if abs(_w) < _snap_eps:
                    _w = 0.0
                _smoothed_weights[_k] = _w

            alloc = Allocation(
                weights=_smoothed_weights,
                meta=dict(getattr(alloc, "meta", {}) or {}),
            )

        alloc_after_smoothing_weights = dict(getattr(alloc, "weights", {}) or {})

        # Guardrail operativo: limitar cambio máximo por símbolo por barra.
        if prev_alloc is not None and float(allocator_max_step_per_bar) < 1.0:
            _step_cap = max(0.0, float(allocator_max_step_per_bar))
            _prev_w = dict(getattr(prev_alloc, "weights", {}) or {})
            _curr_w = dict(getattr(alloc, "weights", {}) or {})
            _keys = set(_prev_w.keys()) | set(_curr_w.keys())
            _clamped = {}
            for _k in _keys:
                _pw = float(_prev_w.get(_k, 0.0) or 0.0)
                _cw = float(_curr_w.get(_k, 0.0) or 0.0)
                _delta = _cw - _pw
                if _delta > _step_cap:
                    _cw = _pw + _step_cap
                elif _delta < -_step_cap:
                    _cw = _pw - _step_cap
                _clamped[_k] = _cw

            alloc = Allocation(
                weights=_clamped,
                meta={
                    **dict(getattr(alloc, "meta", {}) or {}),
                    "max_step_guardrail_applied": True,
                    "max_step_per_bar": float(_step_cap),
                },
            )
            alloc_after_smoothing_weights = dict(_clamped)

        # Portfolio-level risk-off filter:
        # usar fuente real de ATRP: candle dict -> atributo -> signal.meta.
        if bool(portfolio_riskoff_filter):
            _sol_candle = candles.get(sol_sym)
            _sol_sig = signals.get(sol_sym)

            _sol_atrp_now = 0.0
            try:
                if isinstance(_sol_candle, dict) and ("atrp" in _sol_candle):
                    _sol_atrp_now = float(_sol_candle.get("atrp", 0.0) or 0.0)
                elif _sol_candle is not None and hasattr(_sol_candle, "atrp"):
                    _sol_atrp_now = float(getattr(_sol_candle, "atrp", 0.0) or 0.0)
                else:
                    _sol_meta = dict(getattr(_sol_sig, "meta", {}) or {}) if _sol_sig is not None else {}
                    _sol_atrp_now = float(_sol_meta.get("atrp", 0.0) or 0.0)
            except Exception:
                _sol_atrp_now = 0.0

            _riskoff = _sol_atrp_now > float(portfolio_riskoff_sol_atrp_max)

            if _riskoff:
                alloc = Allocation(
                    weights={k: 0.0 for k in dict(getattr(alloc, "weights", {}) or {}).keys()},
                    meta={
                        **dict(getattr(alloc, "meta", {}) or {}),
                        "portfolio_riskoff_applied": True,
                        "portfolio_riskoff_reason": {
                            "sol_atrp": float(_sol_atrp_now),
                        },
                    },
                )
                alloc_after_smoothing_weights = dict(getattr(alloc, "weights", {}) or {})

        # Continuous portfolio risk scaling based on SOL ATRP.
        if bool(portfolio_risk_scale_enable):
            _sol_candle = candles.get(sol_sym)
            _sol_sig = signals.get(sol_sym)

            _sol_atrp_now = 0.0
            try:
                if isinstance(_sol_candle, dict) and ("atrp" in _sol_candle):
                    _sol_atrp_now = float(_sol_candle.get("atrp", 0.0) or 0.0)
                elif _sol_candle is not None and hasattr(_sol_candle, "atrp"):
                    _sol_atrp_now = float(getattr(_sol_candle, "atrp", 0.0) or 0.0)
                else:
                    _sol_meta = dict(getattr(_sol_sig, "meta", {}) or {}) if _sol_sig is not None else {}
                    _sol_atrp_now = float(_sol_meta.get("atrp", 0.0) or 0.0)
            except Exception:
                _sol_atrp_now = 0.0

            _low = float(portfolio_risk_scale_atrp_low)
            _high = float(portfolio_risk_scale_atrp_high)
            _floor = max(0.0, min(1.0, float(portfolio_risk_scale_floor)))

            if _high <= _low:
                _risk_mult = 1.0 if _sol_atrp_now <= _low else _floor
            elif _sol_atrp_now <= _low:
                _risk_mult = 1.0
            elif _sol_atrp_now >= _high:
                _risk_mult = _floor
            else:
                _span = _high - _low
                _x = (_sol_atrp_now - _low) / _span
                _risk_mult = 1.0 - _x * (1.0 - _floor)

            _scaled_weights = {
                _k: float(_w) * float(_risk_mult)
                for _k, _w in dict(getattr(alloc, "weights", {}) or {}).items()
            }
            alloc = Allocation(
                weights=_scaled_weights,
                meta={
                    **dict(getattr(alloc, "meta", {}) or {}),
                    "portfolio_risk_scale_applied": True,
                    "portfolio_risk_scale_mult": float(_risk_mult),
                    "portfolio_risk_scale_sol_atrp": float(_sol_atrp_now),
                },
            )
            alloc_after_smoothing_weights = dict(_scaled_weights)

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
        alloc_after_signal_gating_weights = dict(getattr(alloc, "weights", {}) or {})
        btc_meta = dict(getattr(signals.get(btc_sym), "meta", {}) or {})
        sol_meta = dict(getattr(signals.get(sol_sym), "meta", {}) or {})

        btc_subpos_info = _plan_subpositions_for_symbol(
            planner=btc_subpos_planner,
            symbol=btc_sym,
            strategy_id=str(btc_meta.get("strategy_id", "") or ""),
            side=str(getattr(signals.get(btc_sym), "side", "flat")),
            total_target_weight=float(alloc.weights.get(btc_sym, 0.0) or 0.0),
            raw_custom_weights=btc_subpos_weights,
            meta=btc_meta,
        )
        sol_subpos_info = _plan_subpositions_for_symbol(
            planner=sol_subpos_planner,
            symbol=sol_sym,
            strategy_id=str(sol_meta.get("strategy_id", "") or ""),
            side=str(getattr(signals.get(sol_sym), "side", "flat")),
            total_target_weight=float(alloc.weights.get(sol_sym, 0.0) or 0.0),
            raw_custom_weights=sol_subpos_weights,
            meta=sol_meta,
        )

        btc_cluster = cluster_builder.build_from_weights(
            cluster_id=f"{int(ts)}::{btc_sym}::{str(btc_meta.get('strategy_id', '') or '')}",
            symbol=btc_sym,
            strategy_id=str(btc_meta.get("strategy_id", "") or ""),
            side=str(getattr(signals.get(btc_sym), "side", "flat")),
            target_weight=float(alloc.weights.get(btc_sym, 0.0) or 0.0),
            weights=[float(x) for x in getattr(btc_subpos_info["plan"], "slices", [])],
            meta={
                **btc_meta,
                "ts": int(ts),
            },
        )
        sol_cluster = cluster_builder.build_from_weights(
            cluster_id=f"{int(ts)}::{sol_sym}::{str(sol_meta.get('strategy_id', '') or '')}",
            symbol=sol_sym,
            strategy_id=str(sol_meta.get("strategy_id", "") or ""),
            side=str(getattr(signals.get(sol_sym), "side", "flat")),
            target_weight=float(alloc.weights.get(sol_sym, 0.0) or 0.0),
            weights=[float(x) for x in getattr(sol_subpos_info["plan"], "slices", [])],
            meta={
                **sol_meta,
                "ts": int(ts),
            },
        )

        btc_cluster_risk = cluster_risk_engine.evaluate(btc_cluster)
        sol_cluster_risk = cluster_risk_engine.evaluate(sol_cluster)

        btc_cluster_for_execution = btc_cluster.__class__(
            cluster_id=str(btc_cluster.cluster_id),
            symbol=str(btc_cluster.symbol),
            strategy_id=str(btc_cluster.strategy_id),
            side=str(btc_cluster.side),
            target_weight=float(btc_cluster_risk.adjusted_target_weight),
            subpositions=tuple(btc_cluster_risk.adjusted_subpositions),
            entry_schedule=tuple(getattr(btc_cluster, "entry_schedule", ()) or ()),
            exit_schedule=tuple(getattr(btc_cluster, "exit_schedule", ()) or ()),
            risk_limits=dict(getattr(btc_cluster, "risk_limits", {}) or {}),
            meta={
                **dict(getattr(btc_cluster, "meta", {}) or {}),
                "risk_adjusted": True,
            },
        )
        sol_cluster_for_execution = sol_cluster.__class__(
            cluster_id=str(sol_cluster.cluster_id),
            symbol=str(sol_cluster.symbol),
            strategy_id=str(sol_cluster.strategy_id),
            side=str(sol_cluster.side),
            target_weight=float(sol_cluster_risk.adjusted_target_weight),
            subpositions=tuple(sol_cluster_risk.adjusted_subpositions),
            entry_schedule=tuple(getattr(sol_cluster, "entry_schedule", ()) or ()),
            exit_schedule=tuple(getattr(sol_cluster, "exit_schedule", ()) or ()),
            risk_limits=dict(getattr(sol_cluster, "risk_limits", {}) or {}),
            meta={
                **dict(getattr(sol_cluster, "meta", {}) or {}),
                "risk_adjusted": True,
            },
        )

        btc_execution_plan = execution_planner.build_plan(
            cluster=btc_cluster_for_execution,
            execution_mode=str(execution_mode),
            default_order_type="market",
            time_in_force="GTC",
            ladder_limit_offsets=parsed_execution_ladder_offsets,
            time_sliced_offsets=parsed_execution_time_offsets,
            meta={"ts": int(ts), "risk_approved": bool(btc_cluster_risk.approved)},
        )
        sol_execution_plan = execution_planner.build_plan(
            cluster=sol_cluster_for_execution,
            execution_mode=str(execution_mode),
            default_order_type="market",
            time_in_force="GTC",
            ladder_limit_offsets=parsed_execution_ladder_offsets,
            time_sliced_offsets=parsed_execution_time_offsets,
            meta={"ts": int(ts), "risk_approved": bool(sol_cluster_risk.approved)},
        )
        btc_fills = order_sim.simulate_plan(
            plan=btc_execution_plan,
            bar_index=int(len(allocs)),
            open_price=float(candles[btc_sym].open),
            high_price=float(candles[btc_sym].high),
            low_price=float(candles[btc_sym].low),
            close_price=float(candles[btc_sym].close),
        )
        sol_fills = order_sim.simulate_plan(
            plan=sol_execution_plan,
            bar_index=int(len(allocs)),
            open_price=float(candles[sol_sym].open),
            high_price=float(candles[sol_sym].high),
            low_price=float(candles[sol_sym].low),
            close_price=float(candles[sol_sym].close),
        )
        raw_execution_fills = list(btc_fills) + list(sol_fills)

        raw_execution_rows = []
        for _fill in raw_execution_fills:
            raw_execution_rows.append({
                "ts": int(ts),
                "ts_utc": pd.to_datetime(int(ts), unit="ms", utc=True).isoformat(),
                "order_id": str(_fill.order_id),
                "symbol": str(_fill.symbol),
                "side": str(getattr(_fill, "side", "")),
                "bar_index": int(getattr(_fill, "bar_index", len(allocs))),
                "filled_weight": float(getattr(_fill, "filled_weight", 0.0) or 0.0),
                "expected_price": float(getattr(_fill, "expected_price", 0.0) or 0.0),
                "fill_price": float(getattr(_fill, "fill_price", 0.0) or 0.0),
                "execution_cost_bps": float(getattr(_fill, "execution_cost_bps", 0.0) or 0.0),
                "execution_cost_pct": float(getattr(_fill, "execution_cost_pct", 0.0) or 0.0),
            })
        execution_rows.extend(raw_execution_rows)

        execution_fills = [
            f for f in raw_execution_fills
            if str(getattr(f, "side", "")).lower() not in {"", "flat", "none", "neutral"}
            and float(getattr(f, "filled_weight", 0.0) or 0.0) > 0.0
            and float(getattr(f, "expected_price", 0.0) or 0.0) > 0.0
        ]
        execution_fill_count = int(len(execution_fills))

        _exec_cost_bps_vals = [
            float(getattr(f, "execution_cost_bps", 0.0) or 0.0)
            for f in execution_fills
            if getattr(f, "execution_cost_bps", None) is not None
        ]
        _exec_cost_pct_vals = [
            float(getattr(f, "execution_cost_pct", 0.0) or 0.0)
            for f in execution_fills
            if getattr(f, "execution_cost_pct", None) is not None
        ]

        avg_execution_cost_bps_ts = (
            float(sum(_exec_cost_bps_vals) / len(_exec_cost_bps_vals))
            if _exec_cost_bps_vals else 0.0
        )
        avg_execution_cost_pct_ts = (
            float(sum(_exec_cost_pct_vals) / len(_exec_cost_pct_vals))
            if _exec_cost_pct_vals else 0.0
        )

        for _fill in execution_fills:
            execution_rows.append({
                "ts": int(ts),
                "ts_utc": pd.to_datetime(int(ts), unit="ms", utc=True).isoformat(),
                "order_id": str(_fill.order_id),
                "symbol": str(_fill.symbol),
                "side": str(_fill.side),
                "bar_index": int(_fill.bar_index),
                "filled_weight": float(_fill.filled_weight),
                "expected_price": float(getattr(_fill, "expected_price", 0.0) or 0.0),
                "fill_price": float(_fill.fill_price),
                "execution_cost_bps": float(getattr(_fill, "execution_cost_bps", 0.0) or 0.0),
                "execution_cost_pct": float(getattr(_fill, "execution_cost_pct", 0.0) or 0.0),
            })

        alloc = Allocation(
            weights=dict(alloc.weights),
            meta={
                **dict(getattr(alloc, "meta", {}) or {}),
                "btc_subposition_plan": {
                    "count": int(btc_subpos_info["count"]),
                    "weights": [float(x) for x in getattr(btc_subpos_info["plan"], "slices", [])],
                },
                "sol_subposition_plan": {
                    "count": int(sol_subpos_info["count"]),
                    "weights": [float(x) for x in getattr(sol_subpos_info["plan"], "slices", [])],
                },
                "btc_position_cluster": {
                    "cluster_id": str(btc_cluster.cluster_id),
                    "strategy_id": str(btc_cluster.strategy_id),
                    "side": str(btc_cluster.side),
                    "target_weight": float(btc_cluster.target_weight),
                    "planned_weight": float(btc_cluster.planned_weight),
                    "subposition_count": int(btc_cluster.subposition_count),
                },
                "sol_position_cluster": {
                    "cluster_id": str(sol_cluster.cluster_id),
                    "strategy_id": str(sol_cluster.strategy_id),
                    "side": str(sol_cluster.side),
                    "target_weight": float(sol_cluster.target_weight),
                    "planned_weight": float(sol_cluster.planned_weight),
                    "subposition_count": int(sol_cluster.subposition_count),
                },
                "btc_cluster_risk": {
                    "approved": bool(btc_cluster_risk.approved),
                    "adjusted_target_weight": float(btc_cluster_risk.adjusted_target_weight),
                    "adjusted_subposition_count": int(len(btc_cluster_risk.adjusted_subpositions)),
                    "reasons": list(btc_cluster_risk.reasons),
                },
                "sol_cluster_risk": {
                    "approved": bool(sol_cluster_risk.approved),
                    "adjusted_target_weight": float(sol_cluster_risk.adjusted_target_weight),
                    "adjusted_subposition_count": int(len(sol_cluster_risk.adjusted_subpositions)),
                    "reasons": list(sol_cluster_risk.reasons),
                },
                "btc_execution_plan": {
                    "cluster_id": str(btc_execution_plan.cluster_id),
                    "slice_count": int(btc_execution_plan.slice_count),
                    "planned_weight": float(btc_execution_plan.planned_weight),
                    "target_weight": float(btc_execution_plan.total_target_weight),
                    "execution_mode": str((btc_execution_plan.meta or {}).get("execution_mode", str(execution_mode))),
                    "order_type": str(btc_execution_plan.slices[0].order_type if btc_execution_plan.slices else ""),
                    "time_in_force": str(btc_execution_plan.slices[0].time_in_force if btc_execution_plan.slices else "GTC"),
                    "time_offsets": "|".join(str(int(x.time_offset_bars)) for x in btc_execution_plan.slices),
                    "ladder_offsets": "|".join(str(float((x.meta or {}).get("limit_offset_pct", 0.0))) for x in btc_execution_plan.slices),
                },
                "sol_execution_plan": {
                    "cluster_id": str(sol_execution_plan.cluster_id),
                    "slice_count": int(sol_execution_plan.slice_count),
                    "planned_weight": float(sol_execution_plan.planned_weight),
                    "target_weight": float(sol_execution_plan.total_target_weight),
                    "execution_mode": str((sol_execution_plan.meta or {}).get("execution_mode", str(execution_mode))),
                    "order_type": str(sol_execution_plan.slices[0].order_type if sol_execution_plan.slices else ""),
                    "time_in_force": str(sol_execution_plan.slices[0].time_in_force if sol_execution_plan.slices else "GTC"),
                    "time_offsets": "|".join(str(int(x.time_offset_bars)) for x in sol_execution_plan.slices),
                    "ladder_offsets": "|".join(str(float((x.meta or {}).get("limit_offset_pct", 0.0))) for x in sol_execution_plan.slices),
                },
                "execution_fill_count": int(execution_fill_count),
                "avg_execution_cost_bps": float(avg_execution_cost_bps_ts),
                "avg_execution_cost_pct": float(avg_execution_cost_pct_ts),
            },
        )

        _prev_alloc_for_rows = prev_alloc
        prev_alloc = alloc
        allocs.append(alloc)

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

        _prev_w_btc = float(_prev_alloc_for_rows.weights.get(btc_sym, 0.0)) if _prev_alloc_for_rows is not None else 0.0
        _prev_w_sol = float(_prev_alloc_for_rows.weights.get(sol_sym, 0.0)) if _prev_alloc_for_rows is not None else 0.0
        _curr_w_btc = float(alloc.weights.get(btc_sym, 0.0))
        _curr_w_sol = float(alloc.weights.get(sol_sym, 0.0))

        rows.append({
            "ts": int(ts),
            "ts_utc": pd.to_datetime(int(ts), unit="ms", utc=True).isoformat(),
            "w_btc": _curr_w_btc,
            "w_sol": _curr_w_sol,
            "prev_w_btc": _prev_w_btc,
            "prev_w_sol": _prev_w_sol,
            "dw_btc": abs(_curr_w_btc - _prev_w_btc),
            "dw_sol": abs(_curr_w_sol - _prev_w_sol),

            "btc_w_raw_allocator": float(alloc_raw_weights.get(btc_sym, 0.0) or 0.0),
            "sol_w_raw_allocator": float(alloc_raw_weights.get(sol_sym, 0.0) or 0.0),

            "btc_w_after_ml_position_sizing": float(alloc_after_ml_weights.get(btc_sym, 0.0) or 0.0),
            "sol_w_after_ml_position_sizing": float(alloc_after_ml_weights.get(sol_sym, 0.0) or 0.0),

            "btc_w_after_smoothing": float(alloc_after_smoothing_weights.get(btc_sym, 0.0) or 0.0),
            "sol_w_after_smoothing": float(alloc_after_smoothing_weights.get(sol_sym, 0.0) or 0.0),

            "btc_w_after_signal_gating": float(alloc_after_signal_gating_weights.get(btc_sym, 0.0) or 0.0),
            "sol_w_after_signal_gating": float(alloc_after_signal_gating_weights.get(sol_sym, 0.0) or 0.0),

            "pipeline_weight_order": "raw->ml->smoothing->signal_gating",
            "allocation_case": str((alloc.meta or {}).get("case", "")),
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
            "btc_subpos_count": int(btc_subpos_info["count"]),
            "sol_subpos_count": int(sol_subpos_info["count"]),
            "btc_subpos_weights": str(btc_subpos_info["weights"]),
            "sol_subpos_weights": str(sol_subpos_info["weights"]),
            "btc_cluster_id": str(((alloc.meta or {}).get("btc_position_cluster", {}) or {}).get("cluster_id", "")),
            "sol_cluster_id": str(((alloc.meta or {}).get("sol_position_cluster", {}) or {}).get("cluster_id", "")),
            "btc_cluster_strategy_id": str(((alloc.meta or {}).get("btc_position_cluster", {}) or {}).get("strategy_id", "")),
            "sol_cluster_strategy_id": str(((alloc.meta or {}).get("sol_position_cluster", {}) or {}).get("strategy_id", "")),
            "btc_cluster_side": str(((alloc.meta or {}).get("btc_position_cluster", {}) or {}).get("side", "")),
            "sol_cluster_side": str(((alloc.meta or {}).get("sol_position_cluster", {}) or {}).get("side", "")),
            "btc_cluster_target_weight": float((((alloc.meta or {}).get("btc_position_cluster", {}) or {}).get("target_weight", 0.0) or 0.0)),
            "sol_cluster_target_weight": float((((alloc.meta or {}).get("sol_position_cluster", {}) or {}).get("target_weight", 0.0) or 0.0)),
            "btc_cluster_planned_weight": float((((alloc.meta or {}).get("btc_position_cluster", {}) or {}).get("planned_weight", 0.0) or 0.0)),
            "sol_cluster_planned_weight": float((((alloc.meta or {}).get("sol_position_cluster", {}) or {}).get("planned_weight", 0.0) or 0.0)),
            "btc_cluster_subposition_count": int((((alloc.meta or {}).get("btc_position_cluster", {}) or {}).get("subposition_count", 0) or 0)),
            "sol_cluster_subposition_count": int((((alloc.meta or {}).get("sol_position_cluster", {}) or {}).get("subposition_count", 0) or 0)),
            "btc_cluster_risk_approved": int(bool((((alloc.meta or {}).get("btc_cluster_risk", {}) or {}).get("approved", False)))),
            "sol_cluster_risk_approved": int(bool((((alloc.meta or {}).get("sol_cluster_risk", {}) or {}).get("approved", False)))),
            "btc_cluster_risk_target_weight": float((((alloc.meta or {}).get("btc_cluster_risk", {}) or {}).get("adjusted_target_weight", 0.0) or 0.0)),
            "sol_cluster_risk_target_weight": float((((alloc.meta or {}).get("sol_cluster_risk", {}) or {}).get("adjusted_target_weight", 0.0) or 0.0)),
            "btc_cluster_risk_subposition_count": int((((alloc.meta or {}).get("btc_cluster_risk", {}) or {}).get("adjusted_subposition_count", 0) or 0)),
            "sol_cluster_risk_subposition_count": int((((alloc.meta or {}).get("sol_cluster_risk", {}) or {}).get("adjusted_subposition_count", 0) or 0)),
            "btc_cluster_risk_reasons": "|".join(((alloc.meta or {}).get("btc_cluster_risk", {}) or {}).get("reasons", []) or []),
            "sol_cluster_risk_reasons": "|".join(((alloc.meta or {}).get("sol_cluster_risk", {}) or {}).get("reasons", []) or []),
            "btc_execution_slice_count": int((((alloc.meta or {}).get("btc_execution_plan", {}) or {}).get("slice_count", 0) or 0)),
            "sol_execution_slice_count": int((((alloc.meta or {}).get("sol_execution_plan", {}) or {}).get("slice_count", 0) or 0)),
            "btc_execution_planned_weight": float((((alloc.meta or {}).get("btc_execution_plan", {}) or {}).get("planned_weight", 0.0) or 0.0)),
            "sol_execution_planned_weight": float((((alloc.meta or {}).get("sol_execution_plan", {}) or {}).get("planned_weight", 0.0) or 0.0)),
            "btc_execution_target_weight": float((((alloc.meta or {}).get("btc_execution_plan", {}) or {}).get("target_weight", 0.0) or 0.0)),
            "sol_execution_target_weight": float((((alloc.meta or {}).get("sol_execution_plan", {}) or {}).get("target_weight", 0.0) or 0.0)),
            "btc_execution_mode": str((((alloc.meta or {}).get("btc_execution_plan", {}) or {}).get("execution_mode", "") or "")),
            "sol_execution_mode": str((((alloc.meta or {}).get("sol_execution_plan", {}) or {}).get("execution_mode", "") or "")),
            "btc_execution_order_type": str((((alloc.meta or {}).get("btc_execution_plan", {}) or {}).get("order_type", "") or "")),
            "sol_execution_order_type": str((((alloc.meta or {}).get("sol_execution_plan", {}) or {}).get("order_type", "") or "")),
            "btc_execution_time_offsets": str((((alloc.meta or {}).get("btc_execution_plan", {}) or {}).get("time_offsets", "") or "")),
            "sol_execution_time_offsets": str((((alloc.meta or {}).get("sol_execution_plan", {}) or {}).get("time_offsets", "") or "")),
            "btc_execution_ladder_offsets": str((((alloc.meta or {}).get("btc_execution_plan", {}) or {}).get("ladder_offsets", "") or "")),
            "sol_execution_ladder_offsets": str((((alloc.meta or {}).get("sol_execution_plan", {}) or {}).get("ladder_offsets", "") or "")),
            "execution_fill_count": int(((alloc.meta or {}).get("execution_fill_count", 0) or 0)),
            "execution_slippage_bps": float(((alloc.meta or {}).get("execution_slippage_bps", execution_slippage_bps) or execution_slippage_bps)),
            "execution_size_slippage_factor": float(((alloc.meta or {}).get("execution_size_slippage_factor", execution_size_slippage_factor) or execution_size_slippage_factor)),
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

    if strategy_allocation_rows:
        pd.DataFrame(strategy_allocation_rows).to_csv(
            f"results/strategy_allocations_{name}.csv",
            index=False,
        )

    if execution_rows:
        pd.DataFrame(execution_rows).to_csv(
            f"results/execution_fills_{name}.csv",
            index=False,
        )

    if bool(ml_export_features):
        _ml_out = ml_features_out or f"results/ml_features_{name}.csv"
        _ml_df = pd.DataFrame(ml_feature_rows)

        if not _ml_df.empty:
            _sort_cols = [c for c in ["symbol", "strategy_id", "ts"] if c in _ml_df.columns]
            if _sort_cols:
                _ml_df = _ml_df.sort_values(_sort_cols).reset_index(drop=True)

            _close_map = {
                btc_sym: btc["close"].astype(float),
                sol_sym: sol["close"].astype(float),
            }

            _parts = []
            _group_cols = [c for c in ["symbol", "strategy_id"] if c in _ml_df.columns]
            if not _group_cols:
                _group_cols = ["symbol"]

            for _, _g in _ml_df.groupby(_group_cols, sort=False):
                _g = _g.sort_values("ts").copy()
                _sym = str(_g["symbol"].iloc[0])
                _close_series = _close_map.get(_sym)

                if _close_series is None:
                    _parts.append(_g)
                    continue

                _px_now = _g["ts"].map(_close_series.to_dict()).astype(float)

                for _h in (1, 3, 6, 12):
                    _future_close = _g["ts"].map(_close_series.shift(-_h).to_dict()).astype(float)
                    _ret = (_future_close / _px_now) - 1.0

                    _g[f"future_close_{_h}"] = _future_close
                    _g[f"future_ret_{_h}"] = _ret

                    _is_long = _g["side_raw"] == "long"
                    _is_short = _g["side_raw"] == "short"
                    _valid = _future_close.notna()

                    _y = pd.Series(np.nan, index=_g.index, dtype="float64")
                    _y.loc[_valid & _is_long & (_ret > 0)] = 1.0
                    _y.loc[_valid & _is_long & ~(_ret > 0)] = 0.0
                    _y.loc[_valid & _is_short & (_ret < 0)] = 1.0
                    _y.loc[_valid & _is_short & ~(_ret < 0)] = 0.0
                    _g[f"y_win_{_h}"] = _y

                _parts.append(_g)

            _ml_df = pd.concat(_parts, ignore_index=True)

            if "selected_by_opportunity_selector" in _ml_df.columns:
                for _h in (1, 3, 6, 12):
                    _ml_df[f"y_win_selected_{_h}"] = np.where(
                        _ml_df["selected_by_opportunity_selector"].eq(1),
                        _ml_df[f"y_win_{_h}"],
                        np.nan,
                    )

            if "post_ml_score" in _ml_df.columns and "competitive_score" in _ml_df.columns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    _ml_df["ml_edge_ratio"] = np.where(
                        _ml_df["competitive_score"].abs() > 1e-12,
                        _ml_df["post_ml_score"] / _ml_df["competitive_score"],
                        np.nan,
                    )

            _preferred_h = 6
            if f"y_win_{_preferred_h}" in _ml_df.columns:
                _ml_df["y_win"] = _ml_df[f"y_win_{_preferred_h}"]
                _ml_df["label_horizon"] = _preferred_h

            _head_cols = [
                c for c in [
                    "ts",
                    "ts_utc",
                    "strategy_id",
                    "engine",
                    "registry_symbol",
                    "symbol",
                    "side_raw",
                    "side_final",
                    "selected_by_opportunity_selector",
                    "strength",
                    "base_weight",
                    "competitive_score",
                    "post_ml_score",
                    "p_win",
                    "ml_edge_ratio",
                    "label_horizon",
                    "y_win",
                ]
                if c in _ml_df.columns
            ]
            _tail_cols = [c for c in _ml_df.columns if c not in _head_cols]
            _ml_df = _ml_df[_head_cols + _tail_cols]

        _ml_df.to_csv(_ml_out, index=False)

    # Portfolio performance (equity/drawdown)
    pe = SimplePortfolioEngine(initial_equity=1000.0)
    perf = pe.run(candles_by_symbol=candles_by_symbol, allocations=allocs, symbols=(btc_sym, sol_sym))
    perf_out = perf.reset_index().copy()
    perf_out["gross_port_ret"] = pd.to_numeric(perf_out["port_ret"], errors="coerce").fillna(0.0)
    perf_out["gross_equity"] = pd.to_numeric(perf_out["equity"], errors="coerce").fillna(0.0)
    perf_out["gross_drawdown_pct"] = pd.to_numeric(perf_out["drawdown_pct"], errors="coerce").fillna(0.0)
    perf_out["execution_turnover"] = 0.0
    perf_out["execution_cost_rate"] = 0.0
    perf_out["execution_cost_drag_pct"] = 0.0

    if execution_rows:
        _exec_df = pd.DataFrame(execution_rows)
        _cost_bps = pd.to_numeric(_exec_df["execution_cost_bps"], errors="coerce").fillna(0.0)
        _cost_pct = pd.to_numeric(_exec_df["execution_cost_pct"], errors="coerce").fillna(0.0)

        metrics_execution_fill_count_total = int(len(_exec_df))
        metrics_avg_execution_cost_bps = float(_cost_bps.mean()) if len(_exec_df) else 0.0
        metrics_avg_execution_cost_pct = float(_cost_pct.mean()) if len(_exec_df) else 0.0
        metrics_max_execution_cost_bps = float(_cost_bps.max()) if len(_exec_df) else 0.0
        metrics_min_execution_cost_bps = float(_cost_bps.min()) if len(_exec_df) else 0.0

        _cost_source = _exec_df.drop_duplicates(subset=["ts", "order_id", "symbol", "side", "bar_index"]).copy()
        _cost_source = _cost_source[
            pd.to_numeric(_cost_source["filled_weight"], errors="coerce").fillna(0.0).abs() > 0.0
        ].copy()
        _cost_source["_ts_key"] = pd.to_datetime(
            pd.to_numeric(_cost_source["ts"], errors="coerce"),
            unit="ms",
            utc=True,
            errors="coerce",
        )
        _cost_source["_unit_cost_pct"] = pd.to_numeric(
            _cost_source["execution_cost_pct"], errors="coerce"
        ).fillna(0.0).abs()
        _cost_ts = (
            _cost_source.groupby("_ts_key", as_index=False)["_unit_cost_pct"]
            .mean()
            .rename(columns={"_unit_cost_pct": "execution_cost_rate"})
        )

        _weight_cols = [c for c in df.columns if str(c).startswith("w_")]
        _weights = df[_weight_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        _turnover = _weights.diff().abs()
        if len(_turnover):
            _turnover.iloc[0] = _weights.iloc[0].abs()
        _turnover = _turnover.sum(axis=1)
        _turnover_df = pd.DataFrame(
            {
                "_ts_key": pd.to_datetime(
                    pd.to_numeric(df["ts"], errors="coerce"),
                    unit="ms",
                    utc=True,
                    errors="coerce",
                ),
                "execution_turnover": _turnover,
            }
        )

        perf_out["_ts_key"] = pd.to_datetime(perf_out["ts"], utc=True, errors="coerce")
        perf_out = perf_out.merge(_turnover_df, on="_ts_key", how="left", suffixes=("", "_new"))
        perf_out = perf_out.merge(_cost_ts, on="_ts_key", how="left", suffixes=("", "_new"))

        if "execution_turnover_new" in perf_out.columns:
            perf_out["execution_turnover"] = perf_out["execution_turnover_new"]
            perf_out = perf_out.drop(columns=["execution_turnover_new"])
        if "execution_cost_rate_new" in perf_out.columns:
            perf_out["execution_cost_rate"] = perf_out["execution_cost_rate_new"]
            perf_out = perf_out.drop(columns=["execution_cost_rate_new"])

        perf_out["execution_turnover"] = pd.to_numeric(perf_out["execution_turnover"], errors="coerce").fillna(0.0)
        perf_out["execution_cost_rate"] = pd.to_numeric(perf_out["execution_cost_rate"], errors="coerce").fillna(0.0)
        perf_out["execution_cost_drag_pct"] = perf_out["execution_turnover"] * perf_out["execution_cost_rate"]
        perf_out["port_ret"] = perf_out["gross_port_ret"] - perf_out["execution_cost_drag_pct"]

        _equity = pd.Series(1000.0, index=perf_out.index, dtype="float64")
        for i in range(1, len(perf_out)):
            _equity.iloc[i] = _equity.iloc[i - 1] * (1.0 + float(perf_out.loc[i, "port_ret"]))
        perf_out["equity"] = _equity.values

        _peak = _equity.cummax()
        _dd = (_equity / _peak) - 1.0
        perf_out["drawdown"] = _dd.values
        perf_out["drawdown_pct"] = (_dd * 100.0).values
        perf_out = perf_out.drop(columns=["_ts_key"])
    else:
        metrics_execution_fill_count_total = 0
        metrics_avg_execution_cost_bps = 0.0
        metrics_avg_execution_cost_pct = 0.0
        metrics_max_execution_cost_bps = 0.0
        metrics_min_execution_cost_bps = 0.0

    perf_out.to_csv(f"results/pipeline_equity_{name}.csv", index=False)

    # Portfolio metrics (hedge-fund style summary)
    metrics = PortfolioMetricsEngine(risk_free_rate_annual=0.0).compute(
        perf_df=perf_out[["ts", "equity", "port_ret", "drawdown_pct"]],
        alloc_df=df,
    )
    metrics["execution_fill_count_total"] = int(metrics_execution_fill_count_total)
    metrics["avg_execution_cost_bps"] = float(metrics_avg_execution_cost_bps)
    metrics["avg_execution_cost_pct"] = float(metrics_avg_execution_cost_pct)
    metrics["max_execution_cost_bps"] = float(metrics_max_execution_cost_bps)
    metrics["min_execution_cost_bps"] = float(metrics_min_execution_cost_bps)
    metrics["total_execution_cost_drag_pct"] = float(perf_out["execution_cost_drag_pct"].sum())
    metrics["execution_turnover_sum"] = float(pd.to_numeric(perf_out.get("execution_turnover", 0.0), errors="coerce").fillna(0.0).sum())
    metrics["equity_final"] = float(pd.to_numeric(perf_out["equity"], errors="coerce").iloc[-1])
    metrics["gross_equity_final"] = float(pd.to_numeric(perf_out["equity"], errors="coerce").iloc[-1])

    with open(f"results/pipeline_metrics_{name}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    # Net-of-costs simulation (turnover-based fees+slippage)
    _fee_bps = float(locals().get('fee_bps', 0.0) or 0.0)
    _slip_bps = float(locals().get('slippage_bps', 0.0) or 0.0)
    if _fee_bps > 0.0 or _slip_bps > 0.0:
        sim = ExecutionSimulator(
            cost_model=ExecutionCostModel(fee_bps=_fee_bps, slippage_bps=_slip_bps),
            initial_equity=float(perf_out['equity'].iloc[0]) if 'equity' in perf_out.columns else 1000.0,
        )
        net = sim.apply_costs(perf_df=perf_out, alloc_df=df)
        net.to_csv(f"results/pipeline_equity_net_{name}.csv", index=False)

        turnover_diag = net.copy()
        for _c in ["turnover", "effective_turnover", "cost", "gross_ret", "net_ret"]:
            if _c in turnover_diag.columns:
                turnover_diag[_c] = pd.to_numeric(turnover_diag[_c], errors="coerce").fillna(0.0)

        turnover_diag["cost_bps_on_turnover"] = 0.0
        _mask_turn = pd.to_numeric(turnover_diag["turnover"], errors="coerce").fillna(0.0) > 0.0
        turnover_diag.loc[_mask_turn, "cost_bps_on_turnover"] = (
            turnover_diag.loc[_mask_turn, "cost"] / turnover_diag.loc[_mask_turn, "turnover"]
        ) * 10000.0
        turnover_diag.to_csv(f"results/turnover_diagnostics_{name}.csv", index=False)

        perf_net = pd.DataFrame({
            'ts': net['ts'],
            'equity': net['net_equity'],
            'port_ret': net['net_ret'],
            'drawdown_pct': net['net_drawdown_pct'],
        })
        metrics_net = PortfolioMetricsEngine(risk_free_rate_annual=0.0).compute(perf_df=perf_net, alloc_df=df)
        metrics_net["execution_fill_count_total"] = metrics.get("execution_fill_count_total", 0)
        metrics_net["avg_execution_cost_bps"] = metrics.get("avg_execution_cost_bps", 0.0)
        metrics_net["avg_execution_cost_pct"] = metrics.get("avg_execution_cost_pct", 0.0)
        metrics_net["max_execution_cost_bps"] = metrics.get("max_execution_cost_bps", 0.0)
        metrics_net["min_execution_cost_bps"] = metrics.get("min_execution_cost_bps", 0.0)
        metrics_net["execution_turnover_sum"] = float(pd.to_numeric(net.get("turnover", 0.0), errors="coerce").fillna(0.0).sum())
        metrics_net["gross_equity_final"] = float(pd.to_numeric(net["gross_equity"], errors="coerce").iloc[-1]) if "gross_equity" in net.columns else float(metrics.get("gross_equity_final", metrics.get("end_equity", 0.0)))
        metrics_net["equity_final"] = float(pd.to_numeric(net["net_equity"], errors="coerce").iloc[-1]) if "net_equity" in net.columns else float(metrics_net.get("end_equity", 0.0))
        metrics_net["fee_bps"] = float(_fee_bps)
        metrics_net["slippage_bps"] = float(_slip_bps)
        metrics_net["turnover_cost_drag_pct"] = float(
            pd.to_numeric(net["cost"], errors="coerce").fillna(0.0).sum() * 100.0
        )

        with open(f"results/pipeline_metrics_net_{name}.json", "w", encoding="utf-8") as f:
            json.dump(metrics_net, f, indent=2, sort_keys=True)

        with open(f"results/pipeline_metrics_{name}.json", "w", encoding="utf-8") as f:
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
            'allocation_engine_mode': str(allocation_engine_mode),
            'strategy_score_power': float(strategy_score_power),
            'strategy_symbol_score_agg': str(strategy_symbol_score_agg),
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
    ap.add_argument("--ml-thresholds-path", default=None, help="Optional JSON with per-strategy ML thresholds.")
    ap.add_argument("--ml-export-features", action="store_true", help="Export ML feature rows for raw/final signals.")
    ap.add_argument("--ml-features-out", default=None, help="Optional CSV path for exported ML features.")
    ap.add_argument("--ml-position-sizing", action="store_true", help="Enable ML-based position sizing using p_win -> size multiplier.")
    ap.add_argument("--ml-size-mode", type=str, default="linear_edge", choices=["linear_edge", "calibrated", "artifact_map"], help="Sizing transform mode.")
    ap.add_argument("--ml-size-artifact-path", type=str, default="artifacts/ml_position_size_map_v1.json", help="Artifact JSON path for artifact_map sizing mode.")
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
    ap.add_argument(
        "--allocation-engine-mode",
        default="regime",
        choices=["regime", "multi_strategy"],
        help="Allocation mode: legacy regime allocator or multi-strategy soft allocator.",
    )
    ap.add_argument(
        "--strategy-score-power",
        type=float,
        default=1.0,
        help="Exponent applied to competitive score before intra-symbol normalization.",
    )
    ap.add_argument(
        "--strategy-symbol-score-agg",
        default="sum",
        choices=["sum", "max"],
        help="How to aggregate strategy scores into a symbol budget.",
    )
    ap.add_argument(
        "--allocator-blend-alpha",
        type=float,
        default=0.40,
        help="Blend alpha for MultiStrategyAllocator portfolio inertia.",
    )
    ap.add_argument(
        "--allocator-rebalance-deadband",
        type=float,
        default=0.0,
        help="Minimum absolute weight change required before allocator rebalance is applied.",
    )
    ap.add_argument(
        "--allocator-symbol-cap",
        type=float,
        default=1.0,
        help="Maximum total budget assigned to a single symbol in MultiStrategyAllocator.",
    )
    ap.add_argument(
        "--allocator-max-step-per-bar",
        type=float,
        default=1.0,
        help="Maximum absolute weight change allowed per symbol in a single bar after smoothing.",
    )
    ap.add_argument(
        "--portfolio-riskoff-filter",
        action="store_true",
        help="Disable all portfolio exposure when broad regime conditions are hostile.",
    )
    ap.add_argument(
        "--portfolio-riskoff-btc-adx-min",
        type=float,
        default=18.0,
        help="Minimum BTC ADX required to allow exposure.",
    )
    ap.add_argument(
        "--portfolio-riskoff-btc-slope-min",
        type=float,
        default=1.5,
        help="Minimum BTC slope required to allow exposure.",
    )
    ap.add_argument(
        "--portfolio-riskoff-sol-atrp-max",
        type=float,
        default=0.035,
        help="Maximum SOL ATRP allowed to keep exposure enabled.",
    )
    ap.add_argument(
        "--portfolio-risk-scale-enable",
        action="store_true",
        help="Enable continuous portfolio exposure scaling based on SOL ATRP.",
    )
    ap.add_argument(
        "--portfolio-risk-scale-atrp-low",
        type=float,
        default=0.010,
        help="SOL ATRP level below which exposure stays at 100 percent.",
    )
    ap.add_argument(
        "--portfolio-risk-scale-atrp-high",
        type=float,
        default=0.025,
        help="SOL ATRP level above which exposure is clamped to the floor multiplier.",
    )
    ap.add_argument(
        "--portfolio-risk-scale-floor",
        type=float,
        default=0.35,
        help="Minimum exposure multiplier when SOL ATRP is very high.",
    )
    ap.add_argument(
        "--strategy-regime-gating",
        action="store_true",
        help="Gate registry strategies by regime before MultiStrategyAllocator.",
    )
    ap.add_argument(
        "--allocator-smoothing-alpha",
        type=float,
        default=0.50,
        help="Post-allocation smoothing alpha in pipeline; set 0.0 to disable.",
    )
    ap.add_argument(
        "--allocator-smoothing-snap-eps",
        type=float,
        default=0.02,
        help="Snap-to-zero epsilon after post-allocation smoothing.",
    )
    ap.add_argument("--btc-subpos-count", type=int, default=1, help="Number of planned BTC subpositions.")
    ap.add_argument("--sol-subpos-count", type=int, default=1, help="Number of planned SOL subpositions.")
    ap.add_argument("--btc-subpos-weights", default=None, help="Optional comma-separated BTC subposition weights.")
    ap.add_argument("--sol-subpos-weights", default=None, help="Optional comma-separated SOL subposition weights.")
    ap.add_argument(
        "--execution-mode",
        default="market",
        choices=["market", "ladder_limit", "time_sliced"],
        help="Execution planning mode for cluster slices.",
    )
    ap.add_argument(
        "--execution-ladder-offsets",
        default=None,
        help="Optional comma-separated offsets for ladder_limit mode, e.g. 0.0,0.002,0.004",
    )
    ap.add_argument(
        "--execution-time-offsets",
        default=None,
        help="Optional comma-separated bar offsets for time_sliced mode, e.g. 0,2,4",
    )
    ap.add_argument(
        "--execution-slippage-bps",
        type=float,
        default=2.0,
        help="Execution slippage base in bps for OrderSimulator fills.",
    )
    ap.add_argument(
        "--execution-size-slippage-factor",
        type=float,
        default=10.0,
        help="Additional execution slippage proportional to order size.",
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
        ml_thresholds_path=args.ml_thresholds_path,
        ml_export_features=bool(args.ml_export_features),
        ml_features_out=args.ml_features_out,
        ml_position_sizing=bool(args.ml_position_sizing),
        ml_size_scale=float(args.ml_size_scale),
        ml_size_min=float(args.ml_size_min),
        ml_size_max=float(args.ml_size_max),
        ml_size_mode=str(args.ml_size_mode),
        ml_size_base=float(args.ml_size_base),
        ml_size_pwin_threshold=float(args.ml_size_pwin_threshold),
        ml_size_artifact_path=str(args.ml_size_artifact_path),
        strategy_registry_path=str(args.strategy_registry),
        opportunity_selection_mode=str(args.opportunity_selection_mode),
        allocation_engine_mode=str(args.allocation_engine_mode),
        strategy_score_power=float(args.strategy_score_power),
        strategy_symbol_score_agg=str(args.strategy_symbol_score_agg),
        allocator_blend_alpha=float(args.allocator_blend_alpha),
        allocator_rebalance_deadband=float(args.allocator_rebalance_deadband),
        allocator_symbol_cap=float(args.allocator_symbol_cap),
        allocator_max_step_per_bar=float(args.allocator_max_step_per_bar),
        portfolio_riskoff_filter=bool(args.portfolio_riskoff_filter),
        portfolio_riskoff_btc_adx_min=float(args.portfolio_riskoff_btc_adx_min),
        portfolio_riskoff_btc_slope_min=float(args.portfolio_riskoff_btc_slope_min),
        portfolio_riskoff_sol_atrp_max=float(args.portfolio_riskoff_sol_atrp_max),
        portfolio_risk_scale_enable=bool(args.portfolio_risk_scale_enable),
        portfolio_risk_scale_atrp_low=float(args.portfolio_risk_scale_atrp_low),
        portfolio_risk_scale_atrp_high=float(args.portfolio_risk_scale_atrp_high),
        portfolio_risk_scale_floor=float(args.portfolio_risk_scale_floor),
        strategy_regime_gating=bool(args.strategy_regime_gating),
        allocator_smoothing_alpha=float(args.allocator_smoothing_alpha),
        allocator_smoothing_snap_eps=float(args.allocator_smoothing_snap_eps),
        btc_subpos_count=int(args.btc_subpos_count),
        sol_subpos_count=int(args.sol_subpos_count),
        btc_subpos_weights=args.btc_subpos_weights,
        sol_subpos_weights=args.sol_subpos_weights,
        execution_mode=str(args.execution_mode),
        execution_ladder_offsets=args.execution_ladder_offsets,
        execution_time_offsets=args.execution_time_offsets,
        execution_slippage_bps=float(args.execution_slippage_bps),
        execution_size_slippage_factor=float(args.execution_size_slippage_factor),
    )
    print(f"Saved -> results/pipeline_allocations_{args.name}.csv (rows={len(df)})")

if __name__ == "__main__":
    main()
