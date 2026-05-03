from __future__ import annotations
import json
import pickle
import os

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
from hf.engines.signals.link_trend_signal import LinkTrendSignalEngine
from hf.engines.signals.aave_trend_signal import AaveTrendSignalEngine
from hf.engines.signals.bnb_trend_signal import BnbTrendSignalEngine
from hf.engines.signals.eth_trend_signal import EthTrendSignalEngine
from hf.engines.signals.xrp_trend_signal import XrpTrendSignalEngine
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
def _safe_rvol20(df, idx: int, volume_col: str = "volume", window: int = 20) -> float:
    try:
        if idx < 0 or volume_col not in df.columns:
            return 0.0
        start = max(0, int(idx) - int(window) + 1)
        vol = df.iloc[start:int(idx)+1][volume_col].astype(float)
        if vol.empty:
            return 0.0
        cur = float(vol.iloc[-1] or 0.0)
        avg = float(vol.mean() or 0.0)
        if avg <= 0.0:
            return 0.0
        return float(cur / avg)
    except Exception:
        return 0.0



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



def _load_selector_time_registry(registry_path: Optional[str]) -> dict:
    if not registry_path:
        return {}
    p = Path(str(registry_path))
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _load_strategy_registry_rows(strategy_registry_path: str) -> list[dict]:
    p = Path(strategy_registry_path)
    if not p.exists():
        return [
            {"strategy_id": "btc_trend", "symbol": LEGACY_SYMBOLS["BTC"], "enabled": True},
            {"strategy_id": "sol_bbrsi", "symbol": LEGACY_SYMBOLS["SOL"], "enabled": True},
        ]

    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Strategy registry must be a list")
    return [dict(x or {}) for x in payload]


def _extract_universe_symbols(strategy_registry_rows: list[dict]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in strategy_registry_rows:
        if not bool((row or {}).get("enabled", True)):
            continue
        sym = str((row or {}).get("symbol", "") or "").strip()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out or [LEGACY_SYMBOLS["BTC"], LEGACY_SYMBOLS["SOL"]]


def _build_symbol_cluster_metadata(strategy_registry_rows: list[dict]) -> tuple[dict[str, str], dict[str, float]]:
    symbol_cluster_map: dict[str, str] = {}
    cluster_cap_map: dict[str, float] = {}

    for row in strategy_registry_rows:
        if not bool((row or {}).get("enabled", True)):
            continue
        sym = str((row or {}).get("symbol", "") or "").strip()
        if not sym:
            continue

        cluster_id = str((row or {}).get("cluster_id", "") or "").strip()
        if cluster_id:
            symbol_cluster_map[sym] = cluster_id

        if cluster_id:
            cap_val = (row or {}).get("cluster_cap", None)
            if cap_val is not None and str(cap_val).strip() != "":
                try:
                    cluster_cap_map[cluster_id] = float(cap_val)
                except Exception:
                    pass

    return symbol_cluster_map, cluster_cap_map


def _apply_symbol_top_n_and_cluster_caps(
    weights: dict[str, float],
    *,
    symbol_cluster_map: dict[str, str] | None = None,
    cluster_cap_map: dict[str, float] | None = None,
    top_n_symbols: int | None = None,
) -> dict[str, float]:
    out = {str(k): float(v or 0.0) for k, v in (weights or {}).items()}

    if top_n_symbols is not None and int(top_n_symbols) > 0:
        ranked = sorted(
            [sym for sym, w in out.items() if abs(float(w or 0.0)) > 0.0],
            key=lambda sym: (abs(float(out.get(sym, 0.0) or 0.0)), str(sym)),
            reverse=True,
        )
        keep = set(ranked[: int(top_n_symbols)])
        for sym in list(out.keys()):
            if sym not in keep:
                out[sym] = 0.0

    symbol_cluster_map = dict(symbol_cluster_map or {})
    cluster_cap_map = dict(cluster_cap_map or {})

    if symbol_cluster_map and cluster_cap_map:
        cluster_abs_sum: dict[str, float] = {}
        for sym, w in out.items():
            cid = symbol_cluster_map.get(sym)
            if not cid:
                continue
            cluster_abs_sum[cid] = float(cluster_abs_sum.get(cid, 0.0) + abs(float(w or 0.0)))

        for cid, total_abs in cluster_abs_sum.items():
            cap = cluster_cap_map.get(cid, None)
            if cap is None:
                continue
            cap = float(cap)
            if cap <= 0.0 or total_abs <= cap:
                continue

            scale = float(cap / total_abs)
            for sym in list(out.keys()):
                if symbol_cluster_map.get(sym) == cid:
                    out[sym] = float(out[sym] * scale)

    return out


def _fetch_symbol_ohlcv_map(
    *,
    symbols: list[str],
    start_ms: int,
    end_ms: int | None,
    exchange: str,
    cache_dir: str,
    refresh_cache: bool,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = fetch_ohlcv_ccxt(
            sym,
            "1h",
            start_ms,
            end_ms,
            exchange_id=exchange,
            cache_dir=cache_dir,
            use_cache=True,
            refresh_if_no_end=refresh_cache,
        )
        if df is None or df.empty:
            raise SystemExit(f"OHLCV empty for {sym}. Check cache_dir/exchange/symbol/timeframe.")
        out[sym] = df.set_index("timestamp").sort_index()
    return out


def _compute_common_ts(data_by_symbol: dict[str, pd.DataFrame]) -> pd.Index:
    symbols = list(data_by_symbol.keys())
    if not symbols:
        raise SystemExit("No symbols loaded.")

    common_ts = data_by_symbol[symbols[0]].index
    for sym in symbols[1:]:
        common_ts = common_ts.intersection(data_by_symbol[sym].index)

    if len(common_ts) < 10:
        raise SystemExit(f"Not enough overlapping candles across universe: {len(common_ts)}")
    return common_ts


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
    portfolio_regime_defensive_scale: float = 1.0,
    portfolio_regime_defensive_conviction_k: float = 0.0,
    portfolio_regime_aggressive_scale: float = 1.0,
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
    ml_edge_score_ref: float = 0.20,
    ml_edge_mult_min: float = 0.75,
    ml_edge_mult_max: float = 1.25,
    ml_size_scale: float = 1.0,
    ml_size_min: float = 0.0,
    ml_size_max: float = 1.0,
    ml_size_mode: str = "linear_edge",
    ml_size_base: float = 0.25,
    ml_size_pwin_threshold: float = 0.55,
    ml_size_artifact_path: str = "artifacts/ml_position_size_map_v1.json",
    strategy_registry_path: str = "artifacts/strategy_registry.json",
    disabled_strategy_sides: Optional[str] = None,
    strategy_side_post_ml_weight_rules: Optional[str] = None,
    selector_time_filter_enabled: bool = False,
    selector_time_registry_path: Optional[str] = None,
    selector_time_pwin_min: float = 0.52,
    selector_time_apply_only_in_defensive: bool = True,
    opportunity_selection_mode: str = "best_per_symbol",
    allocation_engine_mode: str = "regime",
    strategy_score_power: float = 1.0,
    strategy_symbol_score_agg: str = "sum",
    allocator_blend_alpha: float = 0.40,
    allocator_rebalance_deadband: float = 0.0,
    allocator_symbol_cap: float = 1.0,
    allocator_target_exposure: float = 0.0,
    execution_symbol_cap: float = 0.25,
    research_partial_tp_enabled: bool = True,
    research_partial_tp1_fraction: float = 0.5,
    research_partial_tp1_ret_threshold: float = 0.01,
    portfolio_regime_detection: bool = True,
    portfolio_regime_breadth_aggressive: int = 4,
    portfolio_regime_breadth_defensive: int = 2,
    portfolio_regime_breadth_high_risk: int = 5,
    portfolio_regime_pwin_aggressive: float = 0.58,
    portfolio_regime_pwin_defensive: float = 0.52,
    portfolio_regime_pwin_high_risk: float = 0.56,
    portfolio_regime_atrp_defensive: float = 0.030,
    allocator_max_step_per_bar: float = 1.0,
    portfolio_riskoff_filter: bool = False,
    portfolio_riskoff_primary_adx_min: float = 18.0,
    portfolio_riskoff_primary_slope_min: float = 1.5,
    portfolio_riskoff_secondary_atrp_max: float = 0.035,
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

    strategy_registry_rows = _load_strategy_registry_rows(str(strategy_registry_path))
    _selector_time_registry = _load_selector_time_registry(selector_time_registry_path)
    _selector_time_models = {}
    if bool(selector_time_filter_enabled) and _selector_time_registry:
        try:
            for _period, _meta in dict((_selector_time_registry.get("periods") or {})).items():
                _p_path = (_meta or {}).get("pwin_model_path")
                _r_path = (_meta or {}).get("retnet_model_path")
                if _p_path and _r_path and Path(_p_path).exists() and Path(_r_path).exists():
                    with open(_p_path, "rb") as _f:
                        _p_model = pickle.load(_f)
                    with open(_r_path, "rb") as _f:
                        _r_model = pickle.load(_f)
                    _selector_time_models[str(_period)] = {
                        "pwin": _p_model,
                        "retnet": _r_model,
                    }
        except Exception:
            _selector_time_models = {}
    universe_symbols = _extract_universe_symbols(strategy_registry_rows)
    symbol_cluster_map, cluster_cap_map = _build_symbol_cluster_metadata(strategy_registry_rows)

    allocator_top_n_symbols = int(os.environ.get("HF_ALLOCATOR_TOP_N_SYMBOLS", "0") or 0)
    allocator_apply_cluster_caps = str(os.environ.get("HF_ALLOCATOR_APPLY_CLUSTER_CAPS", "0")).strip().lower() in {"1", "true", "yes", "on"}

    data_by_symbol = _fetch_symbol_ohlcv_map(
        symbols=universe_symbols,
        start_ms=start_ms,
        end_ms=end_ms,
        exchange=exchange,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
    )
    common_ts = _compute_common_ts(data_by_symbol)

    primary_feature_symbol = btc_sym if btc_sym in data_by_symbol else universe_symbols[0]
    secondary_feature_symbol = sol_sym if sol_sym in data_by_symbol else (
        universe_symbols[1] if len(universe_symbols) > 1 else universe_symbols[0]
    )

    btc = data_by_symbol[primary_feature_symbol]
    sol = data_by_symbol[secondary_feature_symbol]


    # --- feature calc (for Regime3Engine) ---
    close_by_symbol = {
        sym: data_by_symbol[sym]["close"].astype(float)
        for sym in universe_symbols
        if sym in data_by_symbol
    }

    # BTC
    btc_close = close_by_symbol[primary_feature_symbol]
    btc_atr = _atr(btc, 14)
    btc_adx = _adx(btc, 14)
    btc_ema_fast = _ema(btc_close, 20)
    btc_ema_slow = _ema(btc_close, 200)

    # SOL
    sol_close = close_by_symbol[secondary_feature_symbol]

    trend_feature_series_by_symbol = {}
    for sym in universe_symbols:
        _df = data_by_symbol[sym]
        _close = close_by_symbol[sym]
        _atr_series = _atr(_df, 14)
        _atrp_series = _atr_series / _close.replace(0.0, np.nan)
        trend_feature_series_by_symbol[sym] = {
            "adx": _adx(_df, 14),
            "atr": _atr_series,
            "atrp": _atrp_series,
            "ema_fast": _ema(_close, 20),
            "ema_slow": _ema(_close, 200),
        }
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
                "link_trend_signal": lambda cfg: LinkTrendSignalEngine(
                    **dict(cfg.get("params", {}) or {})
                ),
                "aave_trend_signal": lambda cfg: AaveTrendSignalEngine(
                    **dict(cfg.get("params", {}) or {})
                ),
                "bnb_trend_signal": lambda cfg: BnbTrendSignalEngine(
                    **dict(cfg.get("params", {}) or {})
                ),
                "eth_trend_signal": lambda cfg: EthTrendSignalEngine(
                    **dict(cfg.get("params", {}) or {})
                ),
                "xrp_trend_signal": lambda cfg: XrpTrendSignalEngine(
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
        target_exposure=float(allocator_target_exposure),
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
    candles_by_symbol = {sym: [] for sym in universe_symbols}
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
    _disabled_strategy_side_pairs = set()
    if disabled_strategy_sides:
        for _raw in str(disabled_strategy_sides).split(","):
            _raw = str(_raw).strip()
            if not _raw or "|" not in _raw:
                continue
            _sid, _side = _raw.split("|", 1)
            _disabled_strategy_side_pairs.add((str(_sid).strip().lower(), str(_side).strip().lower()))

    _strategy_side_post_ml_weight_rules = {}
    if strategy_side_post_ml_weight_rules:
        for _raw in str(strategy_side_post_ml_weight_rules).split(","):
            _raw = str(_raw).strip()
            if not _raw or _raw.count("|") != 4:
                continue
            _sid, _side, _ref, _min_mult, _max_mult = _raw.split("|", 4)
            try:
                _strategy_side_post_ml_weight_rules[
                    (str(_sid).strip().lower(), str(_side).strip().lower())
                ] = (
                    float(_ref),
                    float(_min_mult),
                    float(_max_mult),
                )
            except Exception:
                pass
    # --- end signal diagnostics ---


    for ts in common_ts:
        selected_opps_for_alloc = []
        feature_map_by_symbol: Dict[str, dict[str, pd.Series]] = {
            sym: dict(trend_feature_series_by_symbol.get(sym, {}))
            for sym in universe_symbols
        }
        if sol_sym in universe_symbols:
            feature_map_by_symbol[sol_sym].update({
                "adx": sol_adx,
                "atr": sol_atr,
                "atrp": sol_atrp,
                "rsi": sol_rsi,
                "bb_mid": sol_bb_mid,
                "bb_up": sol_bb_up,
                "bb_low": sol_bb_low,
                "bb_width": sol_bb_width,
                "ema_fast": sol_ema_fast,
                "ema_slow": sol_ema_slow,
                "donchian_high": sol_vol_dc_high,
                "donchian_low": sol_vol_dc_low,
                "range_expansion": sol_range_expansion,
            })

        candles: Dict[str, Candle] = {}
        for sym in universe_symbols:
            row_df = data_by_symbol[sym]
            if ts not in row_df.index:
                continue

            sym_features = {}
            for feat_name, feat_series in feature_map_by_symbol.get(sym, {}).items():
                sym_features[feat_name] = float(feat_series.loc[ts]) if ts in feat_series.index else float("nan")

            candles[sym] = _row_to_candle(ts, row_df.loc[ts], features=sym_features)

        for sym, candle in candles.items():
            candles_by_symbol[sym].append(candle)

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
                    try:
                        _sym_df = df_map.get(_sym)
                        _rvol_idx = -1
                        if _sym_df is not None and ts in _sym_df.index:
                            try:
                                _rvol_idx = int(_sym_df.index.get_loc(ts))
                            except Exception:
                                _rvol_idx = -1
                        _meta["rvol20"] = float(_safe_rvol20(_sym_df, _rvol_idx)) if (_sym_df is not None and _rvol_idx >= 0) else 0.0
                    except Exception:
                        _meta["rvol20"] = 0.0
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
                            _base_ml_mult = float(ml_score_position_sizer.size_from_pwin(_p_win))
                            _post_ml_score = float(_meta.get("competitive_score", 0.0) or 0.0) * float(_p_win)
                            _edge_ref = max(float(ml_edge_score_ref), 1e-9)
                            _edge_mult = float(_post_ml_score) / float(_edge_ref)
                            _edge_mult = max(float(ml_edge_mult_min), min(float(ml_edge_mult_max), _edge_mult))
                            _final_ml_mult = float(_base_ml_mult) * float(_edge_mult)
                            _final_ml_mult = max(0.0, min(1.0, _final_ml_mult))

                            _meta["ml_position_size_mult"] = float(_final_ml_mult)
                            _meta["ml_position_size_scale"] = float(ml_score_position_sizer.scale)
                            _meta["ml_position_size_mult_base"] = float(_base_ml_mult)
                            _meta["ml_position_size_edge_mult"] = float(_edge_mult)
                            _meta["post_ml_score"] = float(_post_ml_score)
                        else:
                            _meta["ml_position_size_mult"] = 1.0
                            _meta["post_ml_score"] = float(_meta.get("competitive_score", 0.0) or 0.0) * float(_p_win)

                    _opp.meta = _meta

            _selected_opps = select_opportunities(_last_opps, mode=str(opportunity_selection_mode)) if _last_opps else []
            selected_opps_for_alloc = list(_selected_opps)

            try:
                from hf_core.selection_engine import compute_enhanced_score, apply_cross_sectional_ranking

                if selected_opps_for_alloc:
                    _sel_rows = []
                    for _opp in list(selected_opps_for_alloc):
                        _meta = dict(getattr(_opp, "meta", {}) or {})
                        _sel_rows.append({
                            "symbol": str(getattr(_opp, "symbol", "")),
                            "strategy_id": str(getattr(_opp, "strategy_id", "")),
                            "side": str(getattr(_opp, "side", "flat")),
                            "strength": float(getattr(_opp, "strength", 0.0) or 0.0),
                            "p_win": float(_meta.get("p_win", 0.0) or 0.0),
                            "base_weight": float(_meta.get("base_weight", 0.0) or 0.0),
                            "competitive_score": float(_meta.get("competitive_score", 0.0) or 0.0),
                            "post_ml_score": float(_meta.get("post_ml_score", 0.0) or 0.0),
                        })

                    _sel_df = pd.DataFrame(_sel_rows)
                    if not _sel_df.empty:
                        _sel_df = compute_enhanced_score(_sel_df)
                        _sel_df = apply_cross_sectional_ranking(_sel_df, top_pct=0.20)

                        _keep_keys = {
                            (
                                str(r["symbol"]),
                                str(r["strategy_id"]),
                                str(r["side"]),
                                round(float(r["strength"]), 10),
                            )
                            for _, r in _sel_df[_sel_df["accept_ranked"] == True].iterrows()
                        }

                        selected_opps_for_alloc = [
                            _opp for _opp in list(selected_opps_for_alloc)
                            if (
                                str(getattr(_opp, "symbol", "")),
                                str(getattr(_opp, "strategy_id", "")),
                                str(getattr(_opp, "side", "flat")),
                                round(float(getattr(_opp, "strength", 0.0) or 0.0), 10),
                            ) in _keep_keys
                        ]
            except Exception as e:
                print("WARNING selection_engine:", e)

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

        if _disabled_strategy_side_pairs:
            for _sym, _sig in list((signals or {}).items()):
                _side_now = str(getattr(_sig, "side", "flat") or "flat").lower()
                _meta_now = dict(getattr(_sig, "meta", {}) or {})
                _sid_now = str(_meta_now.get("strategy_id", "") or "").lower()
                if (_sid_now, _side_now) in _disabled_strategy_side_pairs:
                    signals[_sym] = Signal(
                        symbol=str(_sym),
                        side="flat",
                        strength=0.0,
                        meta={
                            **_meta_now,
                            "reason": "disabled_strategy_side",
                            "disabled_strategy_side": f"{_sid_now}|{_side_now}",
                        },
                    )

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

        # --- selector_time filter on candidate opportunities (UPSTREAM) ---
        if bool(selector_time_filter_enabled) and _selector_time_models and selected_opps_for_alloc:
            try:
                _apply_selector_now = True
                if bool(selector_time_apply_only_in_defensive):
                    _apply_selector_now = str(_portfolio_regime).lower() == "defensive"

                if _apply_selector_now:
                    _selector_filtered_opps = []
                    _selector_debug_by_symbol = {}

                    _num_cols = list((_selector_time_registry.get("feature_numeric") or []))
                    _cat_cols = list((_selector_time_registry.get("feature_categorical") or []))

                    _w_2024, _w_2025, _w_2026 = 0.15, 0.35, 0.50
                    _w_sum = _w_2024 + _w_2025 + _w_2026

                    for _opp in list(selected_opps_for_alloc):
                        _sym = str(getattr(_opp, "symbol", "") or "")
                        _side = str(getattr(_opp, "side", "flat") or "flat").lower()
                        _meta = dict(getattr(_opp, "meta", {}) or {})

                        _feat = {}
                        for _c in _num_cols:
                            if _c == "abs_weight":
                                _feat[_c] = float(abs(_meta.get("base_weight", 0.0) or 0.0))
                            elif _c == "signed_weight":
                                _base_w = float(_meta.get("base_weight", 0.0) or 0.0)
                                _feat[_c] = float(_base_w if _side == "long" else -_base_w)
                            elif _c == "portfolio_breadth":
                                _feat[_c] = float(_portfolio_breadth)
                            elif _c == "portfolio_avg_pwin":
                                _feat[_c] = float(_portfolio_avg_pwin)
                            elif _c == "portfolio_avg_atrp":
                                _feat[_c] = float(_portfolio_avg_atrp)
                            elif _c == "portfolio_avg_strength":
                                _feat[_c] = float(_portfolio_avg_strength)
                            elif _c == "portfolio_conviction":
                                _feat[_c] = float(_portfolio_conviction)
                            elif _c == "portfolio_regime_scale_applied":
                                _feat[_c] = float(_scale)
                            else:
                                _feat[_c] = float(_meta.get(_c, 0.0) or 0.0)

                        for _c in _cat_cols:
                            if _c == "symbol":
                                _feat[_c] = str(_sym).replace("/USDT:USDT", "")
                            elif _c == "strategy_id":
                                _feat[_c] = str(_meta.get("strategy_id", "") or "")
                            elif _c == "side":
                                _feat[_c] = str(_side)
                            elif _c == "portfolio_regime":
                                _feat[_c] = str(_portfolio_regime)
                            else:
                                _feat[_c] = str(_meta.get(_c, "missing") or "missing")

                        _x = pd.DataFrame([_feat])

                        _p_final = 0.0
                        _r_final = 0.0

                        if "2024" in _selector_time_models:
                            _p = float(_selector_time_models["2024"]["pwin"].predict_proba(_x)[0, 1])
                            _r = float(_selector_time_models["2024"]["retnet"].predict(_x)[0])
                            _p_final += (_w_2024 / _w_sum) * _p
                            _r_final += (_w_2024 / _w_sum) * _r
                        if "2025" in _selector_time_models:
                            _p = float(_selector_time_models["2025"]["pwin"].predict_proba(_x)[0, 1])
                            _r = float(_selector_time_models["2025"]["retnet"].predict(_x)[0])
                            _p_final += (_w_2025 / _w_sum) * _p
                            _r_final += (_w_2025 / _w_sum) * _r
                        if "2026_ytd" in _selector_time_models:
                            _p = float(_selector_time_models["2026_ytd"]["pwin"].predict_proba(_x)[0, 1])
                            _r = float(_selector_time_models["2026_ytd"]["retnet"].predict(_x)[0])
                            _p_final += (_w_2026 / _w_sum) * _p
                            _r_final += (_w_2026 / _w_sum) * _r

                        _score = float(_p_final * max(_r_final, 0.0))

                        _accept = True
                        _mult = 1.0

                        # --- V4 policy: harder LONG filter in defensive / high-risk ---
                        if str(_portfolio_regime).lower() == "defensive":
                            if _side == "long":
                                _high_risk = (
                                    float(_portfolio_breadth) >= float(portfolio_regime_breadth_high_risk)
                                    and float(_portfolio_avg_pwin) <= float(portfolio_regime_pwin_high_risk)
                                )

                                _p_thr = 0.58 if _high_risk else 0.55

                                _accept = (float(_r_final) > 0.0) and (float(_p_final) >= float(_p_thr))
                                _mult = 1.0 if _accept else 0.0

                                _base_w = float(_meta.get("base_weight", 0.0) or 0.0)
                                _meta["base_weight"] = float(_base_w) * float(_mult)
                            else:
                                _accept = True
                                _mult = 1.0
                        else:
                            _accept = True
                            _mult = 1.0

                        _selector_debug_by_symbol[_sym] = {
                            "side": str(_side),
                            "strategy_id": str(_meta.get("strategy_id", "") or ""),
                            "pwin": float(_p_final),
                            "retnet": float(_r_final),
                            "score": float(_score),
                            "mult": float(_mult),
                            "accept": bool(_accept),
                        }

                        if _accept:
                            _meta["selector_time_pwin"] = float(_p_final)
                            _meta["selector_time_retnet"] = float(_r_final)
                            _meta["selector_time_score"] = float(_score)
                            _meta["selector_time_mult"] = float(_mult)
                            _meta["selector_time_accept"] = True
                            try:
                                _opp.meta = _meta
                            except Exception:
                                pass
                            _selector_filtered_opps.append(_opp)

                    selected_opps_for_alloc = list(_selector_filtered_opps)
            except Exception:
                pass

        if bool(strategy_regime_gating) and selected_opps_for_alloc:
            regime_on_by_symbol = {
                sym: bool(getattr(regimes.get(sym), "on", False))
                for sym in universe_symbols
            }
            _btc_regime_on = bool(regime_on_by_symbol.get(primary_feature_symbol, False))
            _sol_regime_on = bool(regime_on_by_symbol.get(secondary_feature_symbol, False))

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

        # Build ATRP snapshot from the real feature series at current timestamp.
        signal_meta_by_symbol = {
            sym: dict(getattr(signals.get(sym), "meta", {}) or {})
            for sym in universe_symbols
        }

        atrp_now_by_symbol = {}
        for sym in universe_symbols:
            _sym_atrp = 0.0
            try:
                _feat_map = feature_map_by_symbol.get(sym, {}) or {}
                _atrp_series = _feat_map.get("atrp")
                _ts_dt = pd.to_datetime(ts, unit="ms", utc=True)
                _ts_int = int(ts)

                if _atrp_series is not None:
                    _atrp_val = None

                    # 1) direct datetime lookup
                    try:
                        _atrp_val = _atrp_series.loc[_ts_dt]
                    except Exception:
                        _atrp_val = None

                    # 2) direct int-ms lookup
                    if _atrp_val is None:
                        try:
                            _atrp_val = _atrp_series.loc[_ts_int]
                        except Exception:
                            _atrp_val = None

                    # 3) tolerant reindex/ffill lookup on datetime index
                    if _atrp_val is None:
                        try:
                            _s_dt = _atrp_series.copy()
                            _s_dt.index = pd.to_datetime(_s_dt.index, utc=True, errors="coerce")
                            _s_dt = _s_dt[~_s_dt.index.isna()]
                            if len(_s_dt):
                                _atrp_val = _s_dt.sort_index().reindex(
                                    _s_dt.index.union([_ts_dt])
                                ).sort_index().ffill().loc[_ts_dt]
                        except Exception:
                            _atrp_val = None

                    # 4) tolerant reindex/ffill lookup on int-ms index
                    if _atrp_val is None:
                        try:
                            _s_int = _atrp_series.copy()
                            _s_int.index = pd.to_numeric(_s_int.index, errors="coerce")
                            _s_int = _s_int[~pd.isna(_s_int.index)]
                            if len(_s_int):
                                _s_int = _s_int.sort_index()
                                _new_index = sorted(set(list(_s_int.index) + [_ts_int]))
                                _atrp_val = _s_int.reindex(_new_index).ffill().loc[_ts_int]
                        except Exception:
                            _atrp_val = None

                    if _atrp_val is not None and pd.notna(_atrp_val):
                        _sym_atrp = float(_atrp_val)

                if _sym_atrp == 0.0:
                    _sym_sig = signals.get(sym)
                    _sym_meta = dict(getattr(_sym_sig, "meta", {}) or {}) if _sym_sig is not None else {}
                    _sym_atrp = float(_sym_meta.get("atrp", 0.0) or 0.0)
            except Exception:
                _sym_atrp = 0.0
            atrp_now_by_symbol[sym] = float(_sym_atrp)

        # Portfolio-level risk-off filter:
        if bool(portfolio_riskoff_filter):
            _sol_atrp_now = float(atrp_now_by_symbol.get(secondary_feature_symbol, 0.0) or 0.0)

            _riskoff = _sol_atrp_now > float(portfolio_riskoff_secondary_atrp_max)

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
            _sol_candle = candles.get(secondary_feature_symbol)
            _sol_sig = signals.get(secondary_feature_symbol)

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
                    "portfolio_risk_scale_secondary_atrp": float(_sol_atrp_now),
                    "portfolio_risk_scale_by_symbol": {
                        sym: {
                            "atrp": float(atrp_now_by_symbol.get(sym, 0.0) or 0.0),
                            "adx": float((signal_meta_by_symbol.get(sym, {}) or {}).get("adx", 0.0) or 0.0),
                            "bb_width": float((signal_meta_by_symbol.get(sym, {}) or {}).get("bb_width", 0.0) or 0.0),
                        }
                        for sym in universe_symbols
                    },
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
        alloc_pre_cluster_controls = dict(alloc.weights)

        alloc = Allocation(
            weights=_apply_symbol_top_n_and_cluster_caps(
                dict(alloc.weights),
                symbol_cluster_map=symbol_cluster_map if bool(allocator_apply_cluster_caps) else {},
                cluster_cap_map=cluster_cap_map if bool(allocator_apply_cluster_caps) else {},
                top_n_symbols=(allocator_top_n_symbols if int(allocator_top_n_symbols) > 0 else None),
            ),
            meta={
                **dict(getattr(alloc, "meta", {}) or {}),
                "allocator_top_n_symbols": int(allocator_top_n_symbols or 0),
                "allocator_apply_cluster_caps": bool(allocator_apply_cluster_caps),
                "symbol_cluster_map": dict(symbol_cluster_map),
                "cluster_cap_map": dict(cluster_cap_map),
                "alloc_pre_cluster_controls": dict(alloc_pre_cluster_controls),
            },
        )

        alloc_after_signal_gating_weights = dict(alloc.weights)

        symbol_meta = {
            sym: dict(getattr(signals.get(sym), "meta", {}) or {})
            for sym in universe_symbols
        }
        planner_by_symbol = {
            primary_feature_symbol: btc_subpos_planner,
            secondary_feature_symbol: sol_subpos_planner,
        }
        custom_subpos_weights_by_symbol = {
            primary_feature_symbol: btc_subpos_weights,
            secondary_feature_symbol: sol_subpos_weights,
        }

        symbol_subpos_info = {}
        symbol_clusters = {}
        symbol_cluster_risks = {}
        symbol_clusters_for_execution = {}
        symbol_execution_plans = {}
        raw_execution_fills = []

        for sym in universe_symbols:
            sym_meta = dict(symbol_meta.get(sym, {}) or {})
            sym_signal = signals.get(sym)
            sym_side = str(getattr(sym_signal, "side", "flat"))
            sym_planner = planner_by_symbol.get(sym, btc_subpos_planner)

            sym_subpos_info = _plan_subpositions_for_symbol(
                planner=sym_planner,
                symbol=sym,
                strategy_id=str(sym_meta.get("strategy_id", "") or ""),
                side=sym_side,
                total_target_weight=float(alloc.weights.get(sym, 0.0) or 0.0),
                raw_custom_weights=custom_subpos_weights_by_symbol.get(sym),
                meta=sym_meta,
            )
            symbol_subpos_info[sym] = sym_subpos_info

            sym_cluster = cluster_builder.build_from_weights(
                cluster_id=f"{int(ts)}::{sym}::{str(sym_meta.get('strategy_id', '') or '')}",
                symbol=sym,
                strategy_id=str(sym_meta.get("strategy_id", "") or ""),
                side=sym_side,
                target_weight=float(alloc.weights.get(sym, 0.0) or 0.0),
                weights=[float(x) for x in getattr(sym_subpos_info["plan"], "slices", [])],
                meta={
                    **sym_meta,
                    "ts": int(ts),
                },
            )
            symbol_clusters[sym] = sym_cluster

            sym_cluster_risk = cluster_risk_engine.evaluate(sym_cluster)
            symbol_cluster_risks[sym] = sym_cluster_risk

            sym_cluster_for_execution = sym_cluster.__class__(
                cluster_id=str(sym_cluster.cluster_id),
                symbol=str(sym_cluster.symbol),
                strategy_id=str(sym_cluster.strategy_id),
                side=str(sym_cluster.side),
                target_weight=float(sym_cluster_risk.adjusted_target_weight),
                subpositions=tuple(sym_cluster_risk.adjusted_subpositions),
                entry_schedule=tuple(getattr(sym_cluster, "entry_schedule", ()) or ()),
                exit_schedule=tuple(getattr(sym_cluster, "exit_schedule", ()) or ()),
                risk_limits=dict(getattr(sym_cluster, "risk_limits", {}) or {}),
                meta={
                    **dict(getattr(sym_cluster, "meta", {}) or {}),
                    "risk_adjusted": True,
                },
            )
            symbol_clusters_for_execution[sym] = sym_cluster_for_execution

            sym_execution_plan = execution_planner.build_plan(
                cluster=sym_cluster_for_execution,
                execution_mode=str(execution_mode),
                default_order_type="market",
                time_in_force="GTC",
                ladder_limit_offsets=parsed_execution_ladder_offsets,
                time_sliced_offsets=parsed_execution_time_offsets,
                meta={"ts": int(ts), "risk_approved": bool(sym_cluster_risk.approved)},
            )
            symbol_execution_plans[sym] = sym_execution_plan

            if sym not in candles:
                continue

            sym_fills = order_sim.simulate_plan(
                plan=sym_execution_plan,
                bar_index=int(len(allocs)),
                open_price=float(candles[sym].open),
                high_price=float(candles[sym].high),
                low_price=float(candles[sym].low),
                close_price=float(candles[sym].close),
            )
            raw_execution_fills.extend(list(sym_fills))

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

        # --- portfolio regime detection (logging only) ---
        _portfolio_regime = "normal"
        _portfolio_breadth = 0
        _portfolio_avg_strength = 0.0
        _portfolio_avg_pwin = 0.0
        _portfolio_avg_atrp = 0.0

        if portfolio_regime_detection:
            _active_syms = []
            _strengths = []
            _pwins = []
            _atrps = []

            for _sym, _sig in (signals or {}).items():
                _side = str(getattr(_sig, "side", "flat") or "flat").lower()
                if _side != "flat":
                    _active_syms.append(_sym)

                    try:
                        _strengths.append(float(getattr(_sig, "strength", 0.0) or 0.0))
                    except Exception:
                        pass

                    _meta_sig = dict(getattr(_sig, "meta", {}) or {})
                    try:
                        _pwins.append(float(_meta_sig.get("post_ml_score", _meta_sig.get("p_win", 0.0)) or 0.0))
                    except Exception:
                        pass

                    try:
                        _atrps.append(float(atrp_now_by_symbol.get(_sym, 0.0) or 0.0))
                    except Exception:
                        pass

            _portfolio_breadth = int(len(_active_syms))
            _portfolio_avg_strength = float(sum(_strengths) / len(_strengths)) if _strengths else 0.0
            _portfolio_avg_pwin = float(sum(_pwins) / len(_pwins)) if _pwins else 0.0
            _portfolio_avg_atrp = float(sum(_atrps) / len(_atrps)) if _atrps else 0.0

            if (
                _portfolio_breadth >= int(portfolio_regime_breadth_aggressive)
                and _portfolio_avg_pwin >= float(portfolio_regime_pwin_aggressive)
                and _portfolio_avg_atrp <= float(portfolio_regime_atrp_defensive)
            ):
                _portfolio_regime = "aggressive"
            elif (
                _portfolio_breadth >= int(portfolio_regime_breadth_high_risk)
                and _portfolio_avg_pwin <= float(portfolio_regime_pwin_high_risk)
            ):
                _portfolio_regime = "defensive"
            elif (
                _portfolio_breadth <= int(portfolio_regime_breadth_defensive)
                or _portfolio_avg_pwin <= float(portfolio_regime_pwin_defensive)
                or _portfolio_avg_atrp >= float(portfolio_regime_atrp_defensive)
            ):
                _portfolio_regime = "defensive"

        # --- portfolio conviction index ---
        _portfolio_conviction = 0.0
        try:
            _breadth_score = min(1.0, float(_portfolio_breadth) / 5.0)
            _pwin_score = max(0.0, min(1.0, float(_portfolio_avg_pwin)))
            _strength_score = max(0.0, min(1.0, float(_portfolio_avg_strength)))
            _portfolio_conviction = float(_breadth_score * _pwin_score * _strength_score)
        except Exception:
            _portfolio_conviction = 0.0

        # soft portfolio-level scaling from conviction + explicit regime scaling
        _regime_symbol_cap_mult = 1.0
        _scale = 0.95 + 0.40 * float(_portfolio_conviction)

        if _portfolio_regime == "defensive":
            _def_base = float(portfolio_regime_defensive_scale)
            _def_k = float(portfolio_regime_defensive_conviction_k)
            _conv = max(0.0, min(1.0, float(_portfolio_conviction)))

            # anchored dynamic modulation around the fixed defensive winner
            _dynamic_mult = 1.0 + _def_k * (_conv - 0.5)
            _dynamic_mult = max(0.70, min(1.30, _dynamic_mult))

            _scale *= float(max(0.0, _def_base * _dynamic_mult))
        elif _portfolio_regime == "aggressive":
            _scale *= float(portfolio_regime_aggressive_scale)

        _weights_regime = {}
        for _sym, _w in dict(alloc.weights or {}).items():
            _w2 = float(_w or 0.0) * float(_scale)

            if _strategy_side_post_ml_weight_rules and abs(_w2) > 1e-12:
                _shape_meta = dict(symbol_meta.get(_sym, {}) or {})
                _shape_signal = signals.get(_sym)
                _shape_side = str(
                    _shape_meta.get(
                        "side",
                        getattr(_shape_signal, "side", "flat") if _shape_signal is not None else "flat",
                    ) or "flat"
                ).lower()
                _shape_sid = str(_shape_meta.get("strategy_id", "") or "").lower()
                _rule = _strategy_side_post_ml_weight_rules.get((_shape_sid, _shape_side))
                if _rule is not None:
                    _ref, _min_mult, _max_mult = _rule
                    _post_ml = float(_shape_meta.get("post_ml_score", 0.0) or 0.0)
                    if float(_ref) > 0.0:
                        _shape_mult = float(_post_ml) / float(_ref)
                        _shape_mult = max(float(_min_mult), min(float(_max_mult), float(_shape_mult)))
                        _w2 = float(_w2) * float(_shape_mult)

            _weights_regime[_sym] = _w2

        alloc = Allocation(
            weights=dict(_weights_regime),
            meta={
                **dict(getattr(alloc, "meta", {}) or {}),
                "subposition_plan_by_symbol": {
                    sym: {
                        "count": int(symbol_subpos_info[sym]["count"]),
                        "weights": [float(x) for x in getattr(symbol_subpos_info[sym]["plan"], "slices", [])],
                    }
                    for sym in universe_symbols
                },
                "position_cluster_by_symbol": {
                    sym: {
                        "cluster_id": str(symbol_clusters[sym].cluster_id),
                        "strategy_id": str(symbol_clusters[sym].strategy_id),
                        "side": str(symbol_clusters[sym].side),
                        "target_weight": float(symbol_clusters[sym].target_weight),
                        "planned_weight": float(symbol_clusters[sym].planned_weight),
                        "subposition_count": int(symbol_clusters[sym].subposition_count),
                    }
                    for sym in universe_symbols
                },
                "cluster_risk_by_symbol": {
                    sym: {
                        "approved": bool(symbol_cluster_risks[sym].approved),
                        "adjusted_target_weight": float(symbol_cluster_risks[sym].adjusted_target_weight),
                        "adjusted_subposition_count": int(len(symbol_cluster_risks[sym].adjusted_subpositions)),
                        "reasons": list(symbol_cluster_risks[sym].reasons),
                    }
                    for sym in universe_symbols
                },
                "portfolio_regime": str(_portfolio_regime),
                "portfolio_breadth": int(_portfolio_breadth),
                "portfolio_avg_strength": float(_portfolio_avg_strength),
                "portfolio_avg_pwin": float(_portfolio_avg_pwin),
                "portfolio_avg_atrp": float(_portfolio_avg_atrp),
                "portfolio_conviction": float(_portfolio_conviction),
                "portfolio_regime_symbol_cap_mult": float(_regime_symbol_cap_mult),
                "portfolio_regime_scale_applied": float(_scale),
                "selector_time_by_symbol": dict(locals().get("_selector_debug_by_symbol", {}) or {}),
                "execution_plan_by_symbol": {
                    sym: {
                        "cluster_id": str(symbol_execution_plans[sym].cluster_id),
                        "slice_count": int(symbol_execution_plans[sym].slice_count),
                        "planned_weight": float(
                            max(-float(execution_symbol_cap), min(float(execution_symbol_cap), float(symbol_execution_plans[sym].planned_weight)))
                        ),
                        "target_weight": float(
                            max(-float(execution_symbol_cap), min(float(execution_symbol_cap), float(symbol_execution_plans[sym].total_target_weight)))
                        ),
                        "execution_mode": str((symbol_execution_plans[sym].meta or {}).get("execution_mode", str(execution_mode))),
                        "order_type": str(symbol_execution_plans[sym].slices[0].order_type if symbol_execution_plans[sym].slices else ""),
                        "time_in_force": str(symbol_execution_plans[sym].slices[0].time_in_force if symbol_execution_plans[sym].slices else "GTC"),
                        "time_offsets": "|".join(str(int(x.time_offset_bars)) for x in symbol_execution_plans[sym].slices),
                        "ladder_offsets": "|".join(str(float((x.meta or {}).get("limit_offset_pct", 0.0))) for x in symbol_execution_plans[sym].slices),
                    }
                    for sym in universe_symbols
                },
                "execution_fill_count": int(execution_fill_count),
                "avg_execution_cost_bps": float(avg_execution_cost_bps_ts),
                "avg_execution_cost_pct": float(avg_execution_cost_pct_ts),
            },
        )

        _execution_meta_by_symbol = ((alloc.meta or {}).get("execution_plan_by_symbol", {}) or {})

        _perf_weights = {}
        for sym in universe_symbols:
            _execution_meta = (_execution_meta_by_symbol.get(sym, {}) or {})
            _exec_target_weight = float(_execution_meta.get("target_weight", 0.0) or 0.0)
            _exec_target_weight = max(-float(execution_symbol_cap), min(float(execution_symbol_cap), _exec_target_weight))
            _perf_weights[sym] = float(_exec_target_weight)

        alloc_for_perf = Allocation(
            weights=dict(_perf_weights),
            meta=dict(getattr(alloc, "meta", {}) or {}),
        )

        _prev_alloc_for_rows = prev_alloc
        prev_alloc = alloc_for_perf
        allocs.append(alloc_for_perf)

        def _sym_field_key(sym: str) -> str:
            return str(sym).lower().replace("/", "_").replace(":", "_").replace("-", "_")

        for sym in universe_symbols:
            sym_meta = dict(symbol_meta.get(sym, {}) or {})
            sym_signal = signals.get(sym)
            final_selected_rows.append({
                "ts": int(ts),
                "ts_utc": pd.to_datetime(int(ts), unit="ms", utc=True).isoformat(),
                "symbol": str(sym),
                "strategy_id": sym_meta.get("strategy_id"),
                "engine": sym_meta.get("engine"),
                "registry_symbol": sym_meta.get("registry_symbol"),
                "side": str(getattr(sym_signal, "side", "flat")),
                "strength": float(getattr(sym_signal, "strength", 0.0) or 0.0),
                "base_weight": float(sym_meta.get("base_weight", 1.0) or 1.0),
                "p_win": float(sym_meta.get("p_win", 0.0) or 0.0),
                "ml_position_size_mult": float(sym_meta.get("ml_position_size_mult", 0.0) or 0.0),
                "competitive_score": float(sym_meta.get("competitive_score", 0.0) or 0.0),
                "post_ml_score": float((sym_meta.get("competitive_score", 0.0) or 0.0) * (sym_meta.get("p_win", 0.0) or 0.0)),
                "is_active": bool(getattr(sym_signal, "is_active")()) if hasattr(sym_signal, "is_active") else False,
            })

        row = {
            "ts": int(ts),
            "ts_utc": pd.to_datetime(int(ts), unit="ms", utc=True).isoformat(),
            "pipeline_weight_order": "raw->ml->smoothing->signal_gating",
            "allocation_case": str((alloc.meta or {}).get("case", "")),
            "portfolio_regime": str((alloc.meta or {}).get("portfolio_regime", "normal")),
            "portfolio_breadth": int(((alloc.meta or {}).get("portfolio_breadth")) or 0),
            "portfolio_avg_strength": float(((alloc.meta or {}).get("portfolio_avg_strength")) or 0.0),
            "portfolio_avg_pwin": float(((alloc.meta or {}).get("portfolio_avg_pwin")) or 0.0),
            "portfolio_avg_atrp": float(((alloc.meta or {}).get("portfolio_avg_atrp")) or 0.0),
            "portfolio_conviction": float(((alloc.meta or {}).get("portfolio_conviction")) or 0.0),
            "portfolio_regime_symbol_cap_mult": float(((alloc.meta or {}).get("portfolio_regime_symbol_cap_mult")) or 1.0),
            "portfolio_regime_scale_applied": float(((alloc.meta or {}).get("portfolio_regime_scale_applied")) or 1.0),
            "selector_time_by_symbol": str(((alloc.meta or {}).get("selector_time_by_symbol", {}) or {})),
            "execution_fill_count": int(((alloc.meta or {}).get("execution_fill_count", 0) or 0)),
            "execution_slippage_bps": float(((alloc.meta or {}).get("execution_slippage_bps", execution_slippage_bps) or execution_slippage_bps)),
            "execution_size_slippage_factor": float(((alloc.meta or {}).get("execution_size_slippage_factor", execution_size_slippage_factor) or execution_size_slippage_factor)),
            "case": (alloc.meta or {}).get("case", ""),
            "regime_on_by_symbol": str({
                sym: bool(getattr(regimes.get(sym), "on", False))
                for sym in universe_symbols
            }),
            "portfolio_risk_scale_by_symbol": str(((alloc.meta or {}).get("portfolio_risk_scale_by_symbol", {}) or {})),
        }

        _subpos_meta_by_symbol = ((alloc.meta or {}).get("subposition_plan_by_symbol", {}) or {})
        _cluster_meta_by_symbol = ((alloc.meta or {}).get("position_cluster_by_symbol", {}) or {})
        _cluster_risk_meta_by_symbol = ((alloc.meta or {}).get("cluster_risk_by_symbol", {}) or {})
        _execution_meta_by_symbol = ((alloc.meta or {}).get("execution_plan_by_symbol", {}) or {})

        for sym in universe_symbols:
            sym_key = _sym_field_key(sym)
            sym_meta = dict(symbol_meta.get(sym, {}) or {})
            sym_signal = signals.get(sym)

            _prev_w = float(_prev_alloc_for_rows.weights.get(sym, 0.0)) if _prev_alloc_for_rows is not None else 0.0
            _curr_w = float(alloc.weights.get(sym, 0.0) or 0.0)
            _subpos_info = symbol_subpos_info.get(sym, {"count": 0, "weights": [], "plan": None})
            _cluster_meta = (_cluster_meta_by_symbol.get(sym, {}) or {})
            _cluster_risk_meta = (_cluster_risk_meta_by_symbol.get(sym, {}) or {})
            _execution_meta = (_execution_meta_by_symbol.get(sym, {}) or {})

            row[f"w_{sym_key}"] = _curr_w
            row[f"prev_w_{sym_key}"] = _prev_w
            row[f"dw_{sym_key}"] = abs(_curr_w - _prev_w)

            row[f"{sym_key}_w_raw_allocator"] = float(alloc_raw_weights.get(sym, 0.0) or 0.0)
            row[f"{sym_key}_w_after_ml_position_sizing"] = float(alloc_after_ml_weights.get(sym, 0.0) or 0.0)
            row[f"{sym_key}_w_after_smoothing"] = float(alloc_after_smoothing_weights.get(sym, 0.0) or 0.0)
            row[f"{sym_key}_w_after_signal_gating"] = float(alloc_after_signal_gating_weights.get(sym, 0.0) or 0.0)

            row[f"{sym_key}_side"] = str(getattr(sym_signal, "side", "flat"))
            row[f"{sym_key}_strength"] = float(getattr(sym_signal, "strength", 0.0) or 0.0)
            row[f"{sym_key}_p_win"] = float(sym_meta.get("p_win", 0.0) or 0.0)
            row[f"{sym_key}_post_ml_score"] = float((sym_meta.get("competitive_score", 0.0) or 0.0) * (sym_meta.get("p_win", 0.0) or 0.0))
            row[f"{sym_key}_ml_size_mult"] = float(sym_meta.get("ml_position_size_mult", 0.0) or 0.0)

            row[f"{sym_key}_subpos_count"] = int(_subpos_info.get("count", 0) or 0)
            row[f"{sym_key}_subpos_weights"] = str(_subpos_info.get("weights", []))

            row[f"{sym_key}_cluster_id"] = str(_cluster_meta.get("cluster_id", ""))
            row[f"{sym_key}_cluster_strategy_id"] = str(_cluster_meta.get("strategy_id", ""))
            row[f"{sym_key}_cluster_side"] = str(_cluster_meta.get("side", ""))
            row[f"{sym_key}_cluster_target_weight"] = float(_cluster_meta.get("target_weight", 0.0) or 0.0)
            row[f"{sym_key}_cluster_planned_weight"] = float(_cluster_meta.get("planned_weight", 0.0) or 0.0)
            row[f"{sym_key}_cluster_subposition_count"] = int(_cluster_meta.get("subposition_count", 0) or 0)

            row[f"{sym_key}_cluster_risk_approved"] = int(bool(_cluster_risk_meta.get("approved", False)))
            row[f"{sym_key}_cluster_risk_target_weight"] = float(_cluster_risk_meta.get("adjusted_target_weight", 0.0) or 0.0)
            row[f"{sym_key}_cluster_risk_subposition_count"] = int(_cluster_risk_meta.get("adjusted_subposition_count", 0) or 0)
            row[f"{sym_key}_cluster_risk_reasons"] = "|".join(_cluster_risk_meta.get("reasons", []) or [])

            row[f"{sym_key}_execution_slice_count"] = int(_execution_meta.get("slice_count", 0) or 0)
            _exec_planned_weight = float(_execution_meta.get("planned_weight", 0.0) or 0.0)
            _exec_target_weight = float(_execution_meta.get("target_weight", 0.0) or 0.0)

            _exec_planned_weight = max(-float(execution_symbol_cap), min(float(execution_symbol_cap), _exec_planned_weight))
            _exec_target_weight = max(-float(execution_symbol_cap), min(float(execution_symbol_cap), _exec_target_weight))

            row[f"{sym_key}_execution_planned_weight"] = float(_exec_planned_weight)
            row[f"{sym_key}_execution_target_weight"] = float(_exec_target_weight)
            row[f"{sym_key}_execution_mode"] = str(_execution_meta.get("execution_mode", "") or "")
            row[f"{sym_key}_execution_order_type"] = str(_execution_meta.get("order_type", "") or "")
            row[f"{sym_key}_execution_time_offsets"] = str(_execution_meta.get("time_offsets", "") or "")
            row[f"{sym_key}_execution_ladder_offsets"] = str(_execution_meta.get("ladder_offsets", "") or "")

            row[f"{sym_key}_strategy_id"] = sym_meta.get("strategy_id")
            row[f"{sym_key}_engine"] = sym_meta.get("engine")
            row[f"{sym_key}_reason"] = ((getattr(sym_signal, "meta", {}) or {}).get("reason") or (getattr(sym_signal, "meta", {}) or {}).get("skip"))
            row[f"{sym_key}_registry_symbol"] = sym_meta.get("registry_symbol")
            row[f"{sym_key}_base_weight"] = float(sym_meta.get("base_weight", 1.0) or 1.0)
            row[f"{sym_key}_competitive_score"] = float(sym_meta.get("competitive_score", 0.0) or 0.0)

        rows.append(row)

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
                sym: data_by_symbol[sym]["close"].astype(float)
                for sym in universe_symbols
                if sym in data_by_symbol
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
    perf = pe.run(candles_by_symbol=candles_by_symbol, allocations=allocs, symbols=tuple(universe_symbols))
    perf_out = perf.reset_index().copy()
    perf_out["gross_port_ret"] = pd.to_numeric(perf_out["port_ret"], errors="coerce").fillna(0.0)
    perf_out["gross_equity"] = pd.to_numeric(perf_out["equity"], errors="coerce").fillna(0.0)
    perf_out["gross_drawdown_pct"] = pd.to_numeric(perf_out["drawdown_pct"], errors="coerce").fillna(0.0)
    perf_out["execution_turnover"] = 0.0
    perf_out["execution_cost_rate"] = 0.0
    perf_out["execution_cost_drag_pct"] = 0.0

    if bool(research_partial_tp_enabled):
        _tp_frac = float(research_partial_tp1_fraction)
        _tp_frac = max(0.0, min(1.0, _tp_frac))
        _tp_thr = float(research_partial_tp1_ret_threshold)

        _adj_port_ret = pd.Series(0.0, index=perf_out.index, dtype="float64")

        for sym in universe_symbols:
            _ret_col = f"ret_{sym}"
            _w_col = f"w_{sym}"

            if _ret_col not in perf_out.columns or _w_col not in perf_out.columns:
                continue

            _ret = pd.to_numeric(perf_out[_ret_col], errors="coerce").fillna(0.0)
            _w = pd.to_numeric(perf_out[_w_col], errors="coerce").fillna(0.0)

            # signed pnl contribution direction
            _signed_ret = _w * _ret

            # favorable move for current position
            _favorable = (_signed_ret > 0.0) & (_ret.abs() > _tp_thr)

            # if favorable move exceeds threshold, assume TP1 realizes on fraction of size
            _ret_adj = _ret.copy()
            _ret_adj.loc[_favorable & (_w > 0)] = (
                (1.0 - _tp_frac) * _ret.loc[_favorable & (_w > 0)]
                + _tp_frac * _tp_thr
            )
            _ret_adj.loc[_favorable & (_w < 0)] = (
                (1.0 - _tp_frac) * _ret.loc[_favorable & (_w < 0)]
                - _tp_frac * _tp_thr
            )

            _adj_port_ret = _adj_port_ret.add(_w * _ret_adj, fill_value=0.0)

        perf_out["port_ret"] = _adj_port_ret.values

        # recompute equity after research partial TP adjustment
        _equity = perf_out["equity"].copy()
        _equity.iloc[0] = 1000.0
        for i in range(1, len(_equity)):
            _equity.iloc[i] = _equity.iloc[i-1] * (1.0 + perf_out["port_ret"].iloc[i])
        perf_out["equity"] = _equity

        _peak = _equity.cummax()
        perf_out["drawdown"] = (_equity / _peak) - 1.0
        perf_out["drawdown_pct"] = perf_out["drawdown"] * 100


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

    # Enriquecer el allocations CSV con métricas reales de performance del backtest.
    # Aquí evitamos merge por ts porque allocations suele llevar ts epoch-ms (int)
    # mientras perf_out puede traer datetime tz-aware. Como ambos dataframes salen del
    # mismo run y misma grilla temporal, preferimos alineación posicional validada.
    _perf_cols = [
        "gross_port_ret",
        "port_ret",
        "gross_equity",
        "equity",
        "gross_drawdown_pct",
        "drawdown_pct",
        "execution_turnover",
        "execution_cost_rate",
        "execution_cost_drag_pct",
    ]
    _perf_cols = [c for c in _perf_cols if c in perf_out.columns]
    _perf_enrich = perf_out[_perf_cols].copy().reset_index(drop=True)

    if "gross_equity" in _perf_enrich.columns:
        _gross_eq = pd.to_numeric(_perf_enrich["gross_equity"], errors="coerce")
        _perf_enrich["gross_pnl"] = _gross_eq.diff().fillna(0.0)
        _perf_enrich["gross_pnl_cum"] = _gross_eq - float(_gross_eq.iloc[0])

    if "equity" in _perf_enrich.columns:
        _net_eq = pd.to_numeric(_perf_enrich["equity"], errors="coerce")
        _perf_enrich["pnl"] = _net_eq.diff().fillna(0.0)
        _perf_enrich["pnl_cum"] = _net_eq - float(_net_eq.iloc[0])

    if len(df) != len(_perf_enrich):
        raise ValueError(
            f"Cannot enrich allocations with perf columns: len(df)={len(df)} != len(perf_out)={len(_perf_enrich)}"
        )

    df = pd.concat([df.reset_index(drop=True), _perf_enrich], axis=1)
    df.to_csv(f"results/pipeline_allocations_{name}.csv", index=False)

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
    ap.add_argument("--portfolio-regime-defensive-scale", type=float, default=1.0)
    ap.add_argument("--portfolio-regime-defensive-conviction-k", type=float, default=0.0)
    ap.add_argument("--portfolio-regime-aggressive-scale", type=float, default=1.0)
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
    ap.add_argument("--disabled-strategy-sides", default=None, help="Comma-separated strategy_id|side blocks, e.g. aave_trend|short,xrp_trend|short")
    ap.add_argument("--strategy-side-post-ml-weight-rules", default=None, help="Comma-separated rules strategy_id|side|ref|min_mult|max_mult, e.g. trx_trend|short|0.50|0.80|1.00")
    ap.add_argument("--selector-time-filter-enabled", action="store_true")
    ap.add_argument("--selector-time-registry-path", default=None)
    ap.add_argument("--selector-time-pwin-min", type=float, default=0.52)
    ap.add_argument("--selector-time-apply-only-in-defensive", action="store_true")
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
        "--allocator-target-exposure",
        type=float,
        default=0.0,
        help="Minimum gross exposure target for MultiStrategyAllocator after deadband and smoothing.",
    )
    ap.add_argument(
        "--allocator-max-step-per-bar",
        type=float,
        default=1.0,
        help="Maximum absolute weight change allowed per symbol in a single bar after smoothing.",
    )
    ap.add_argument("--portfolio-regime-breadth-high-risk", type=int, default=5)
    ap.add_argument("--portfolio-regime-pwin-high-risk", type=float, default=0.56)
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
        portfolio_regime_defensive_scale=float(args.portfolio_regime_defensive_scale),
        portfolio_regime_defensive_conviction_k=float(args.portfolio_regime_defensive_conviction_k),
        portfolio_regime_aggressive_scale=float(args.portfolio_regime_aggressive_scale),
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
        disabled_strategy_sides=args.disabled_strategy_sides,
        strategy_side_post_ml_weight_rules=args.strategy_side_post_ml_weight_rules,
        selector_time_filter_enabled=bool(args.selector_time_filter_enabled),
        selector_time_registry_path=args.selector_time_registry_path,
        selector_time_pwin_min=float(args.selector_time_pwin_min),
        selector_time_apply_only_in_defensive=bool(args.selector_time_apply_only_in_defensive),
        opportunity_selection_mode=str(args.opportunity_selection_mode),
        allocation_engine_mode=str(args.allocation_engine_mode),
        strategy_score_power=float(args.strategy_score_power),
        strategy_symbol_score_agg=str(args.strategy_symbol_score_agg),
        allocator_blend_alpha=float(args.allocator_blend_alpha),
        allocator_rebalance_deadband=float(args.allocator_rebalance_deadband),
        allocator_symbol_cap=float(args.allocator_symbol_cap),
        allocator_target_exposure=float(args.allocator_target_exposure),
        allocator_max_step_per_bar=float(args.allocator_max_step_per_bar),
        portfolio_regime_breadth_high_risk=int(args.portfolio_regime_breadth_high_risk),
        portfolio_regime_pwin_high_risk=float(args.portfolio_regime_pwin_high_risk),
        portfolio_riskoff_filter=bool(args.portfolio_riskoff_filter),
        portfolio_riskoff_primary_adx_min=float(
            getattr(args, "portfolio_riskoff_primary_adx_min",
                    getattr(args, "portfolio_riskoff_btc_adx_min", 18.0))
        ),
        portfolio_riskoff_primary_slope_min=float(
            getattr(args, "portfolio_riskoff_primary_slope_min",
                    getattr(args, "portfolio_riskoff_btc_slope_min", 1.5))
        ),
        portfolio_riskoff_secondary_atrp_max=float(
            getattr(args, "portfolio_riskoff_secondary_atrp_max",
                    getattr(args, "portfolio_riskoff_sol_atrp_max", 0.035))
        ),
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
