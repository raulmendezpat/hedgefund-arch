from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from hf.data.ohlcv import fetch_ohlcv_ccxt, dt_to_ms_utc
from hf.pipeline.run_portfolio import _adx, _atr, _ema
from hf_core.trade_lifecycle import TradeLifecycleEngine
from hf_core.ml.feature_expansion import build_bb_rsi_feature_frame


def _f(x, default=0.0):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def load_exit_registry(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_exit_policy(exit_cfg: dict, strategy_id: str) -> dict:
    strategy_id = str(strategy_id or "")
    by_strategy = dict(exit_cfg.get("strategies", {}) or {})
    if strategy_id in by_strategy:
        return dict(by_strategy[strategy_id] or {})
    return dict(exit_cfg.get("default", {}) or {})


def extract_selected_candidates(trace_path: str) -> pd.DataFrame:
    rows = []
    p = Path(trace_path)

    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)
        ts_raw = obj.get("ts")
        ts = pd.Timestamp(ts_raw, tz="UTC") if isinstance(ts_raw, str) else pd.to_datetime(ts_raw, utc=True)

        trace_stage_weights = dict(obj.get("legacy_stage_weights", {}) or {})
        trace_final_weights = dict(obj.get("weights", {}) or {})

        stage_final = dict(trace_stage_weights.get("after_step_guardrail", {}) or {})
        weight_source = stage_final if stage_final else trace_final_weights
        weight_source_name = "legacy_stage_weights.after_step_guardrail" if stage_final else "weights"

        for c in obj.get("selected_candidates", []) or []:
            sm = dict(c.get("signal_meta", {}) or {})
            symbol = str(c.get("symbol", ""))

            trace_target_weight = _f(weight_source.get(symbol), 0.0)

            rows.append(
                {
                    "ts": ts,
                    "symbol": symbol,
                    "strategy_id": str(c.get("strategy_id", "")),
                    "side": str(c.get("side", "")).lower(),
                    "signal_strength": _f(c.get("signal_strength"), 0.0),
                    "base_weight": _f(sm.get("base_weight"), 1.0),
                    "p_win": _f(sm.get("p_win"), 0.0),
                    "expected_return": _f(sm.get("expected_return"), 0.0),
                    "policy_score": _f(sm.get("policy_score"), 0.0),
                    "policy_size_mult": _f(sm.get("policy_size_mult"), 0.0),
                    "post_ml_score": _f(sm.get("post_ml_score"), 0.0),
                    "post_ml_competitive_score": _f(sm.get("post_ml_competitive_score"), 0.0),
                    "competitive_score": _f(sm.get("competitive_score"), 0.0),
                    "meta_post_ml_score": _f(sm.get("meta_post_ml_score"), 0.0),
                    "meta_post_ml_competitive_score": _f(sm.get("meta_post_ml_competitive_score"), 0.0),
                    "meta_competitive_score": _f(sm.get("meta_competitive_score"), 0.0),
                    "ml_position_size_mult": _f(sm.get("ml_position_size_mult"), 1.0),
                    "trace_target_weight": float(trace_target_weight),
                    "trace_weight_source": weight_source_name,
                    "signal_meta": sm,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["ts", "symbol", "strategy_id"]).reset_index(drop=True)
    return df


def load_symbol_df(symbol: str, start: str, end: str, exchange: str, cache_dir: str) -> pd.DataFrame:
    end_ms = dt_to_ms_utc(end) if end else None
    df = fetch_ohlcv_ccxt(
        symbol=symbol,
        timeframe="1h",
        start_ms=dt_to_ms_utc(start),
        end_ms=end_ms,
        exchange_id=exchange,
        cache_dir=cache_dir,
        use_cache=True,
        refresh_if_no_end=False,
    ).copy()

    df["ts"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).copy()
    df = df[(df["ts"] >= pd.Timestamp(start, tz="UTC")) & (df["ts"] <= pd.Timestamp(end, tz="UTC"))].copy()
    df = df.sort_values("ts").reset_index(drop=True)

    close = df["close"].astype(float)
    df["atr"] = _atr(df, 14)
    df["adx"] = _adx(df, 14)
    df["ema_fast"] = _ema(close, 20)
    df["ema_slow"] = _ema(close, 200)

    _bb_rsi = build_bb_rsi_feature_frame(df)
    for _col in ["rsi", "bb_mid", "bb_up", "bb_low", "bb_width"]:
        if _col in _bb_rsi.columns:
            df[_col] = _bb_rsi[_col]

    return df


def build_symbol_frames(symbols: list[str], start: str, end: str, exchange: str, cache_dir: str) -> dict[str, pd.DataFrame]:
    out = {}
    for sym in symbols:
        out[sym] = load_symbol_df(sym, start, end, exchange, cache_dir)
    return out


def _resolve_post_ml_score(sm: dict) -> float:
    try:
        return float(
            sm.get(
                "post_ml_competitive_score",
                sm.get(
                    "meta_post_ml_competitive_score",
                    sm.get(
                        "post_ml_score",
                        sm.get("meta_post_ml_score", sm.get("competitive_score", sm.get("meta_competitive_score", 0.0))),
                    ),
                ),
            ) or 0.0
        )
    except Exception:
        return 0.0


def _resolve_exit_context_for_symbol(rows: list[dict], symbol: str) -> dict:
    matches = [r for r in (rows or []) if str(r.get("symbol", "")) == str(symbol)]
    if not matches:
        return {
            "target_weight": 0.0,
            "signal_side": "",
            "regime_on": True,
            "context_source": "no_selected_candidate_for_symbol",
        }

    best = None
    best_score = float("-inf")
    for row in matches:
        sm = dict(row.get("signal_meta", {}) or {})
        score = _resolve_post_ml_score(sm)
        if score > best_score:
            best = row
            best_score = score

    row = dict(best or {})
    sm = dict(row.get("signal_meta", {}) or {})
    side = str(row.get("side", "")).lower()

    trace_stage_weights = dict(row.get("trace_legacy_stage_weights", {}) or {})
    trace_final_weights = dict(row.get("trace_weights", {}) or {})

    target_weight = None

    stage_final = dict(trace_stage_weights.get("after_step_guardrail", {}) or {})
    if symbol in stage_final:
        target_weight = _f(stage_final.get(symbol), 0.0)
    elif symbol in trace_final_weights:
        target_weight = _f(trace_final_weights.get(symbol), 0.0)

    if target_weight is None:
        magnitude = max(
            _f(row.get("policy_size_mult"), 0.0),
            _f(row.get("base_weight"), 0.0),
            _f(row.get("post_ml_competitive_score"), _f(row.get("post_ml_score"), 0.0)),
        )

        target_weight = 0.0
        if side == "long":
            target_weight = float(magnitude)
        elif side == "short":
            target_weight = -float(magnitude)

    regime_on = bool(sm.get("regime_on", sm.get("strategy_regime_on", True)))

    return {
        "target_weight": float(target_weight),
        "signal_side": str(side),
        "regime_on": bool(regime_on),
        "context_source": "selected_candidate_for_symbol",
        "context_strategy_id": str(row.get("strategy_id", "")),
        "context_score": float(best_score if best_score != float("-inf") else 0.0),
    }


def simulate_lifecycle(
    *,
    selected_df: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    exit_cfg: dict,
    initial_balance: float,
    maker_fee: float,
    taker_fee: float,
    cooldown_after_close_bars: int,
):
    engine = TradeLifecycleEngine(
        maker_fee=maker_fee,
        taker_fee=taker_fee,
        cooldown_after_close_bars=cooldown_after_close_bars,
    )

    timeline_ts: set[pd.Timestamp] = set()

    if not selected_df.empty:
        timeline_ts.update(pd.to_datetime(selected_df["ts"], utc=True).tolist())

    for _sym, _df in frames.items():
        if _df is None or _df.empty or "ts" not in _df.columns:
            continue
        timeline_ts.update(pd.to_datetime(_df["ts"], utc=True).tolist())

    all_ts = sorted(timeline_ts)
    equity_rows = []
    open_rows = []
    event_rows = []

    balance = float(initial_balance)

    selected_map: dict[pd.Timestamp, list[dict]] = {}
    if not selected_df.empty:
        _selected_df = selected_df.copy()
        _selected_df["ts"] = pd.to_datetime(_selected_df["ts"], utc=True)
        for ts, g in _selected_df.groupby("ts", sort=True):
            selected_map[ts] = g.to_dict("records")

    for ts in all_ts:
        engine.decrement_cooldowns()

        # exits first
        for symbol, df in frames.items():
            pos = engine.get_open_position(symbol)
            if pos is None:
                continue

            hit = df[df["ts"] == ts]
            if hit.empty:
                continue
            i = int(hit.index[0])
            if i <= 0:
                continue

            prev = df.iloc[i - 1]
            cur = df.iloc[i]

            exit_context = _resolve_exit_context_for_symbol(selected_map.get(ts, []), symbol)

            strat_cfg = resolve_exit_policy(exit_cfg, pos.strategy_id)
            decision = engine.evaluate_exit(
                symbol=symbol,
                prev_bar=prev.to_dict(),
                current_bar=cur.to_dict(),
                exit_policy_cfg=strat_cfg,
                context={
                    "symbol": symbol,
                    "strategy_id": pos.strategy_id,
                    "target_weight": float(exit_context.get("target_weight", 0.0) or 0.0),
                    "signal_side": str(exit_context.get("signal_side", "") or ""),
                    "regime_on": bool(exit_context.get("regime_on", True)),
                    "context_source": str(exit_context.get("context_source", "") or ""),
                },
                exit_ts=int(cur["timestamp"]),
            )

            if str(decision.action).lower() == "close":
                rec = engine.trade_log[-1]
                balance += float(rec.pnl)
                event_rows.append(
                    {
                        "ts": ts.isoformat(),
                        "symbol": symbol,
                        "strategy_id": pos.strategy_id,
                        "event": "exit",
                        "side": pos.side,
                        "exit_reason": rec.exit_reason,
                        "entry_px": rec.entry_px,
                        "exit_px": rec.exit_px,
                        "qty": rec.qty,
                        "pnl": rec.pnl,
                        "bars_held": rec.bars_held,
                    }
                )

        # entries second: driven by allocator target weight, one open per symbol max
        opened_symbols = set()

        for row in selected_map.get(ts, []):
            symbol = str(row["symbol"])
            strategy_id = str(row["strategy_id"])

            if symbol in opened_symbols:
                continue
            if not engine.can_open(symbol):
                continue

            tw = row.get("trace_target_weight", row.get("target_weight", 0.0))
            target_weight = _f(tw, 0.0)
            if abs(target_weight) <= 1e-12:
                continue

            side = "long" if target_weight > 0 else "short"

            df = frames.get(symbol)
            if df is None:
                continue
            hit = df[df["ts"] == ts]
            if hit.empty:
                continue
            i = int(hit.index[0])
            if i <= 0:
                continue

            prev = df.iloc[i - 1]
            cur = df.iloc[i]

            entry_px = _f(prev.get("close"), 0.0)
            atr = _f(prev.get("atr"), 0.0)
            adx = _f(prev.get("adx"), float("nan"))
            atrp = atr / max(abs(entry_px), 1e-12) if entry_px != 0.0 else float("nan")

            if entry_px <= 0.0 or atr <= 0.0:
                continue

            notional_frac = min(max(abs(target_weight), 0.01), 0.30)
            notional = balance * notional_frac
            qty = notional / entry_px
            fee = notional * maker_fee
            balance -= fee

            meta = {
                "signal_meta": dict(row.get("signal_meta", {}) or {}),
                "cooldown_after_close_bars": int(cooldown_after_close_bars),
                "trace_target_weight": float(target_weight),
                "entry_notional_frac": float(notional_frac),
            }

            engine.open_position(
                symbol=symbol,
                strategy_id=strategy_id,
                side=side,
                entry_ts=int(cur["timestamp"]),
                entry_px=float(entry_px),
                qty=float(qty),
                entry_atr=float(atr),
                entry_adx=float(adx),
                entry_atrp=float(atrp),
                entry_strength=_f(row.get("signal_strength"), 0.0),
                entry_reason="allocator_target_weight",
                entry_meta=meta,
            )

            opened_symbols.add(symbol)

            event_rows.append(
                {
                    "ts": ts.isoformat(),
                    "symbol": symbol,
                    "strategy_id": strategy_id,
                    "event": "entry",
                    "side": side,
                    "entry_px": float(entry_px),
                    "qty": float(qty),
                    "target_weight": float(target_weight),
                    "notional_frac": float(notional_frac),
                }
            )

        mtm = 0.0
        for symbol, pos in list(engine.open_positions.items()):
            df = frames.get(symbol)
            if df is None:
                continue
            hit = df[df["ts"] == ts]
            if hit.empty:
                continue
            px = _f(hit.iloc[0].get("close"), pos.entry_px)
            if str(pos.side).lower() == "long":
                mtm += (px - pos.entry_px) * pos.qty
            elif str(pos.side).lower() == "short":
                mtm += (pos.entry_px - px) * pos.qty

            open_rows.append(
                {
                    "ts": ts.isoformat(),
                    "symbol": pos.symbol,
                    "strategy_id": pos.strategy_id,
                    "side": pos.side,
                    "entry_ts": int(pos.entry_ts),
                    "entry_px": float(pos.entry_px),
                    "qty": float(pos.qty),
                    "bars_held": int(pos.bars_held),
                    "trail_stop": None if pos.trail_stop is None else float(pos.trail_stop),
                    "breakeven_armed": bool(pos.breakeven_armed),
                }
            )

        equity_rows.append(
            {
                "ts": ts.isoformat(),
                "equity": float(balance + mtm),
                "balance": float(balance),
                "open_positions": int(len(engine.open_positions)),
            }
        )

    trades_df = pd.DataFrame([{
        "symbol": t.symbol,
        "strategy_id": t.strategy_id,
        "side": t.side,
        "entry_ts": t.entry_ts,
        "exit_ts": t.exit_ts,
        "entry_px": t.entry_px,
        "exit_px": t.exit_px,
        "qty": t.qty,
        "pnl": t.pnl,
        "exit_reason": t.exit_reason,
        "fee": t.fee,
        "bars_held": t.bars_held,
        "entry_adx": t.entry_adx,
        "entry_atrp": t.entry_atrp,
        "entry_atr": t.entry_atr,
    } for t in engine.trade_log])

    equity_df = pd.DataFrame(equity_rows)
    open_df = pd.DataFrame(open_rows)
    events_df = pd.DataFrame(event_rows)

    return trades_df, equity_df, open_df, events_df




def _annualized_sharpe(returns: pd.Series, periods_per_year: int = 24 * 365) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return 0.0
    mu = float(r.mean())
    sd = float(r.std(ddof=0))
    if sd <= 0.0:
        return 0.0
    return float((mu / sd) * np.sqrt(periods_per_year))


def _max_drawdown_pct(equity: pd.Series) -> float:
    x = pd.to_numeric(equity, errors="coerce").ffill().dropna()
    if x.empty:
        return 0.0
    peak = x.cummax()
    dd = (x / peak.replace(0.0, np.nan)) - 1.0
    dd = dd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float(dd.min() * 100.0)


def summarize(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict:
    equity = pd.to_numeric(equity_df.get("equity", pd.Series(dtype=float)), errors="coerce").dropna()

    if equity.empty:
        equity_start = 0.0
        equity_final = 0.0
        total_return_pct = 0.0
        sharpe_annual = 0.0
        max_drawdown_pct = 0.0
        vol_annual = 0.0
    else:
        equity_start = float(equity.iloc[0])
        equity_final = float(equity.iloc[-1])
        total_return_pct = ((equity_final / equity_start) - 1.0) * 100.0 if equity_start != 0.0 else 0.0

        rets = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        sharpe_annual = _annualized_sharpe(rets) if len(rets) else 0.0
        max_drawdown_pct = _max_drawdown_pct(equity)
        vol_annual = float(rets.std(ddof=0) * np.sqrt(24 * 365)) if len(rets) else 0.0

    trade_count = int(len(trades_df))
    if trade_count > 0 and "pnl" in trades_df.columns:
        pnl = pd.to_numeric(trades_df["pnl"], errors="coerce").fillna(0.0)
        win_rate_pct = float((pnl.gt(0).sum() / max(1, len(pnl))) * 100.0)
    else:
        win_rate_pct = 0.0

    return {
        "equity_start": float(equity_start),
        "equity_final": float(equity_final),
        "total_return_pct": float(total_return_pct),
        "sharpe_annual": float(sharpe_annual),
        "max_drawdown_pct": float(max_drawdown_pct),
        "vol_annual": float(vol_annual),
        "trade_count": int(trade_count),
        "win_rate_pct": float(win_rate_pct),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--allocation-inputs-trace", required=True)
    ap.add_argument("--exit-policy-registry", default="artifacts/exit_policy_registry.json")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--exchange", default="binanceusdm")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--initial-balance", type=float, default=1000.0)
    ap.add_argument("--maker-fee", type=float, default=0.0002)
    ap.add_argument("--taker-fee", type=float, default=0.0006)
    ap.add_argument("--cooldown-after-close-bars", type=int, default=1)
    args = ap.parse_args()

    selected_df = extract_selected_candidates(args.allocation_inputs_trace)
    if selected_df.empty:
        raise SystemExit("No selected candidates found in allocation trace")

    symbols = sorted(set(selected_df["symbol"].tolist()))
    exit_cfg = load_exit_registry(args.exit_policy_registry)
    frames = build_symbol_frames(symbols, args.start, args.end, args.exchange, args.cache_dir)

    trades_df, equity_df, open_df, event_df = simulate_lifecycle(
        selected_df=selected_df,
        frames=frames,
        exit_cfg=exit_cfg,
        initial_balance=float(args.initial_balance),
        maker_fee=float(args.maker_fee),
        taker_fee=float(args.taker_fee),
        cooldown_after_close_bars=int(args.cooldown_after_close_bars),
    )

    metrics = summarize(trades_df, equity_df)

    out_trades = Path(f"results/trade_lifecycle_trades_{args.name}.csv")
    out_equity = Path(f"results/trade_lifecycle_equity_{args.name}.csv")
    out_open = Path(f"results/trade_lifecycle_open_positions_{args.name}.csv")
    out_events = Path(f"results/trade_lifecycle_events_{args.name}.csv")
    out_metrics = Path(f"results/trade_lifecycle_metrics_{args.name}.json")

    trades_df.to_csv(out_trades, index=False)
    equity_df.to_csv(out_equity, index=False)
    open_df.to_csv(out_open, index=False)
    event_df.to_csv(out_events, index=False)
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"saved: {out_trades}")
    print(f"saved: {out_equity}")
    print(f"saved: {out_open}")
    print(f"saved: {out_events}")
    print(f"saved: {out_metrics}")

    print("\n=== METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if not trades_df.empty:
        print("\n=== EXIT REASONS ===")
        print(trades_df["exit_reason"].value_counts(dropna=False).to_string())

        print("\n=== BY STRATEGY ===")
        g = (
            trades_df.groupby(["strategy_id", "side"], dropna=False)
            .agg(
                trades=("symbol", "size"),
                pnl_sum=("pnl", "sum"),
                pnl_mean=("pnl", "mean"),
                bars_held_mean=("bars_held", "mean"),
            )
            .reset_index()
            .sort_values(["pnl_sum", "trades"], ascending=[False, False])
        )
        print(g.to_string(index=False))


if __name__ == "__main__":
    main()
