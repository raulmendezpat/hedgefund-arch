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

from hf.engines.signals import FlatSignalEngine, BtcTrendSignalEngine
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
    fallback_sol_weight: float = 0.0,  fee_bps: float = 0.0, slippage_bps: float = 0.0) -> pd.DataFrame:
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
    # ---------------------------------------

    # SignalEngine layer (default: flat placeholder)
    if signal_engine == "btc_trend":
        sig_engine = BtcTrendSignalEngine(adx_min=float(btc_adx_min))
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
            }),
        }

        # collect candles for PortfolioEngine
        candles_by_symbol[btc_sym].append(candles[btc_sym])
        candles_by_symbol[sol_sym].append(candles[sol_sym])

        signals = sig_engine.generate(candles)
        regimes = reg_engine.evaluate(candles, signals)
        alloc = allocator.allocate(candles, signals, regimes, prev_alloc)
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
    ap.add_argument("--btc-adx-min", type=float, default=18.0)
    ap.add_argument("--btc-slope-min", type=float, default=1.5)
    ap.add_argument(
        "--signal-engine",
        choices=["flat", "btc_trend"],
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
    )
    print(f"Saved -> results/pipeline_allocations_{args.name}.csv (rows={len(df)})")

if __name__ == "__main__":
    main()
