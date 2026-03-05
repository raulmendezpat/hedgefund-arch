from __future__ import annotations
import json

import argparse
from dataclasses import asdict
from typing import Dict, Optional

import pandas as pd
import numpy as np

from hf.core.types import Candle, Allocation
from hf.engines.alloc_regime import RegimeAllocator
from hf.engines.portfolio_engine import SimplePortfolioEngine
from hf.engines.portfolio_metrics import PortfolioMetricsEngine
from hf.engines.report_engine import ReportEngine
from hf.engines.execution_simulator import ExecutionCostModel, ExecutionSimulator
from hf.engines.regime_regime3 import Regime3Engine

from hf.engines.signals import FlatSignalEngine, BtcTrendSignalEngine, SolBbrsiSignalEngine
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

    prev_alloc: Optional[Allocation] = None
    rows = []

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
                'rsi': float('nan'),
                'bb_mid': float('nan'),
                'bb_up': float('nan'),
                'bb_low': float('nan'),
                'bb_width': float('nan'),
            }),
        }

        # collect candles for PortfolioEngine
        candles_by_symbol[btc_sym].append(candles[btc_sym])
        candles_by_symbol[sol_sym].append(candles[sol_sym])

        signals = sig_engine.generate(candles)
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

        rows.append({
            "ts": int(ts),
            "ts_utc": pd.to_datetime(int(ts), unit="ms", utc=True).isoformat(),
            "w_btc": float(alloc.weights.get(btc_sym, 0.0)),
            "w_sol": float(alloc.weights.get(sol_sym, 0.0)),
            "case": (alloc.meta or {}).get("case", ""),
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"results/pipeline_allocations_{name}.csv", index=False)

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
            'btc_adx_min': float(btc_adx_min),
            'btc_slope_min': float(btc_slope_min),
            'signal_engine': str(signal_engine),
            'signal_gate_applied': bool(signal_gate_applied_any),
            'signal_gated_counts': dict(signal_gated_counts),
            'signal_side_counts': dict(signal_side_counts),
            'signal_skip_counts': dict(signal_skip_counts),
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
    ap.add_argument("--btc-adx-min", type=float, default=18.0)
    ap.add_argument("--btc-slope-min", type=float, default=1.5)
    ap.add_argument(
        "--signal-engine",
        choices=["flat", "btc_trend", "sol_bbrsi"],
        default="flat",
        help="Signal engine to use (default: flat placeholder).",
    )

    ap.add_argument("--sticky-flat", action="store_true", help="If set: keep previous weights when flat (no active trades)")
    args = ap.parse_args()

    df = run(
        args.name, args.start, args.end, args.exchange, args.cache_dir, args.refresh_cache,
        args.trades_csv,
        args.sol_atrp_min, args.sol_adx_max, args.btc_adx_min, args.btc_slope_min, args.signal_engine,
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


    )
    print(f"Saved -> results/pipeline_allocations_{args.name}.csv (rows={len(df)})")

if __name__ == "__main__":
    main()
