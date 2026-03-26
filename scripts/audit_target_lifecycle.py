from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from hf.data.ohlcv import fetch_ohlcv_ccxt, dt_to_ms_utc


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _safe_str(x: Any, default: str = "") -> str:
    if x is None:
        return str(default)
    return str(x)


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _annualized_sharpe(returns: pd.Series, periods_per_year: int = 24 * 365) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return 0.0
    mu = float(r.mean())
    sd = float(r.std(ddof=0))
    if sd <= 0.0:
        return 0.0
    return float((mu / sd) * math.sqrt(periods_per_year))


def _max_drawdown_pct(equity: pd.Series) -> float:
    x = pd.to_numeric(equity, errors="coerce").ffill().fillna(0.0)
    if x.empty:
        return 0.0
    peak = x.cummax()
    dd = (x / peak.replace(0.0, np.nan)) - 1.0
    dd = dd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float(dd.min() * 100.0)


def _normalize_dt_for_ccxt(x: Any) -> str:
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class LifecycleAction:
    action: str
    symbol: str
    strategy_id: str
    family: str
    side: str
    reason: str
    prev_weight: float
    target_weight: float
    delta_weight: float
    close_weight: float
    open_weight: float
    meta: dict[str, Any]


@dataclass
class ExitDecision:
    action: str
    exit_reason: str
    exit_price: Optional[float]
    tp_price: Optional[float]
    sl_price: Optional[float]
    trail_stop: Optional[float]
    breakeven_armed: bool
    meta: dict[str, Any]


@dataclass
class PositionState:
    symbol: str
    strategy_id: str
    family: str
    side: str
    entry_ts: int
    entry_price: float
    entry_atr: float
    entry_weight: float
    qty: float
    bars_held: int = 0
    last_tp: Optional[float] = None
    last_sl: Optional[float] = None
    trail_stop: Optional[float] = None
    breakeven_armed: bool = False


def _ema(s: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").ewm(span=int(span), adjust=False).mean()


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1.0 / float(n), adjust=False).mean()


def _bb_mid(close: pd.Series, n: int = 20) -> pd.Series:
    return pd.to_numeric(close, errors="coerce").rolling(int(n), min_periods=int(n)).mean()


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    rows = sorted(rows, key=lambda r: str(r.get("ts", "")))
    return rows


def _extract_stage_weights(row: dict[str, Any]) -> dict[str, float]:
    legacy_stage_weights = dict(row.get("legacy_stage_weights", {}) or {})

    for k in [
        "after_step_guardrail",
        "after_postprocessor",
        "after_strategy_side_overlay",
        "allocator_core",
    ]:
        if k in legacy_stage_weights and isinstance(legacy_stage_weights.get(k), dict):
            return {str(sym): float(w or 0.0) for sym, w in dict(legacy_stage_weights[k]).items()}

    for k in ["weights", "target_weights", "allocation_weights"]:
        if isinstance(row.get(k), dict):
            return {str(sym): float(w or 0.0) for sym, w in dict(row.get(k)).items()}

    return {}


def _family_from_strategy_id(strategy_id: str) -> str:
    s = str(strategy_id or "").lower()
    if "bbrsi" in s or "mr" in s or "mean" in s:
        return "mean_reversion"
    if "breakout" in s or "compression" in s or "expansion" in s:
        return "breakout"
    if "trend" in s:
        return "trend"
    return "generic"


def _resolve_post_ml_score(meta: dict[str, Any]) -> float:
    try:
        return float(
            meta.get(
                "post_ml_competitive_score",
                meta.get(
                    "meta_post_ml_competitive_score",
                    meta.get(
                        "post_ml_score",
                        meta.get(
                            "meta_post_ml_score",
                            meta.get("competitive_score", meta.get("meta_competitive_score", 0.0)),
                        ),
                    ),
                ),
            ) or 0.0
        )
    except Exception:
        return 0.0


def _pick_symbol_candidate(
    selected_candidates: list[dict[str, Any]],
    symbol: str,
    signed_weight: float,
) -> Optional[dict[str, Any]]:
    if not selected_candidates:
        return None

    want_side = "long" if signed_weight > 0 else "short"

    best: Optional[dict[str, Any]] = None
    best_score = float("-inf")

    for c in selected_candidates:
        if str(c.get("symbol", "")) != str(symbol):
            continue
        if str(c.get("side", "")).lower() != want_side:
            continue

        sm = dict(c.get("signal_meta", {}) or {})
        score = _resolve_post_ml_score(sm)
        if score > best_score:
            best_score = score
            best = c

    return best


def _extract_open_decisions(
    row: dict[str, Any],
    target_weights: dict[str, float],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    selected = list(row.get("selected_candidates", []) or [])

    for symbol, w in target_weights.items():
        if abs(float(w)) <= 1e-12:
            continue

        chosen = _pick_symbol_candidate(selected, symbol, float(w))
        if chosen is None:
            side = "long" if float(w) > 0 else "short"
            out[str(symbol)] = {
                "symbol": str(symbol),
                "strategy_id": "",
                "family": "generic",
                "side": side,
                "target_weight": float(w),
                "signal_meta": {},
            }
            continue

        sm = dict(chosen.get("signal_meta", {}) or {})
        strategy_id = str(chosen.get("strategy_id", sm.get("strategy_id", "")) or "")
        side = str(chosen.get("side", "long" if float(w) > 0 else "short")).lower()

        out[str(symbol)] = {
            "symbol": str(symbol),
            "strategy_id": strategy_id,
            "family": _family_from_strategy_id(strategy_id),
            "side": side,
            "target_weight": float(w),
            "signal_meta": sm,
        }

    return out


def load_symbol_df(
    symbol: str,
    start: str,
    end: str,
    *,
    exchange_id: str,
    cache_dir: str,
    timeframe: str = "1h",
) -> pd.DataFrame:
    start_norm = _normalize_dt_for_ccxt(start)
    end_norm = _normalize_dt_for_ccxt(end) if end else None
    end_ms = dt_to_ms_utc(end_norm) if end_norm else None

    df = fetch_ohlcv_ccxt(
        symbol=symbol,
        timeframe=timeframe,
        start_ms=dt_to_ms_utc(start_norm),
        end_ms=end_ms,
        exchange_id=exchange_id,
        cache_dir=cache_dir,
        use_cache=True,
        refresh_if_no_end=False,
    ).copy()

    if df.empty:
        return pd.DataFrame()

    df["ts"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True, errors="coerce")
    df = df[df["ts"].notna()].copy()

    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")

    end_ts = pd.Timestamp(end)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")

    df = df[(df["ts"] >= start_ts) & (df["ts"] <= end_ts)].copy()
    df = df.set_index("ts").sort_index()

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["atr_14"] = _atr(df, 14)
    df["ema_fast_20"] = _ema(df["close"], 20)
    df["ema_slow_200"] = _ema(df["close"], 200)
    df["bb_mid_20"] = _bb_mid(df["close"], 20)

    return df


def build_market_data(
    symbols: list[str],
    *,
    start: str,
    end: str,
    exchange_id: str,
    cache_dir: str,
    timeframe: str = "1h",
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = load_symbol_df(
                sym,
                start,
                end,
                exchange_id=exchange_id,
                cache_dir=cache_dir,
                timeframe=timeframe,
            )
            if not df.empty:
                out[str(sym)] = df
        except Exception as e:
            print(f"[WARN] no pude cargar OHLC para {sym}: {e}")
    return out


class TrendAtrDynamicExitPolicy:
    def __init__(
        self,
        *,
        tp_atr_mult: float = 2.0,
        stop_atr_mult: float = 1.5,
        breakeven_activate_atr: float = 0.9,
        breakeven_offset_atr: float = 0.1,
        trail_activate_atr: float = 1.0,
        trail_stop_atr_mult: float = 1.2,
        time_stop_bars: int = 96,
    ):
        self.tp_atr_mult = float(tp_atr_mult)
        self.stop_atr_mult = float(stop_atr_mult)
        self.breakeven_activate_atr = float(breakeven_activate_atr)
        self.breakeven_offset_atr = float(breakeven_offset_atr)
        self.trail_activate_atr = float(trail_activate_atr)
        self.trail_stop_atr_mult = float(trail_stop_atr_mult)
        self.time_stop_bars = int(time_stop_bars)

    def decide(self, pos: PositionState, prev_bar: pd.Series, bar: pd.Series) -> ExitDecision:
        atr_now = _safe_float(prev_bar.get("atr_14"), 0.0)
        if atr_now <= 0.0:
            return ExitDecision("hold", "", None, None, None, pos.trail_stop, pos.breakeven_armed, {})

        close_now = _safe_float(prev_bar.get("close"), 0.0)
        ema_fast = _safe_float(prev_bar.get("ema_fast_20"), np.nan)
        ema_slow = _safe_float(prev_bar.get("ema_slow_200"), np.nan)

        h = _safe_float(bar.get("high"), np.nan)
        l = _safe_float(bar.get("low"), np.nan)

        side = str(pos.side).lower()
        tp = None
        sl = None
        trail_stop = pos.trail_stop
        breakeven_armed = bool(pos.breakeven_armed)

        if side == "long":
            tp = float(pos.entry_price + self.tp_atr_mult * atr_now)
            sl = float(pos.entry_price - self.stop_atr_mult * atr_now)
            move_atr = (close_now - pos.entry_price) / atr_now if atr_now > 0 else 0.0

            if move_atr >= self.breakeven_activate_atr:
                sl = max(sl, pos.entry_price + self.breakeven_offset_atr * atr_now)
                breakeven_armed = True

            if move_atr >= self.trail_activate_atr:
                trail_stop = float(close_now - self.trail_stop_atr_mult * atr_now)
                sl = max(sl, trail_stop)

            if l <= sl <= h:
                return ExitDecision("close", "sl", float(sl), float(tp), float(sl), trail_stop, breakeven_armed, {})
            if l <= tp <= h:
                return ExitDecision("close", "tp", float(tp), float(tp), float(sl), trail_stop, breakeven_armed, {})
            if not np.isnan(ema_fast) and not np.isnan(ema_slow) and ema_fast < ema_slow:
                px = _safe_float(bar.get("open"), close_now)
                return ExitDecision("close", "trend_break", float(px), float(tp), float(sl), trail_stop, breakeven_armed, {})
        else:
            tp = float(pos.entry_price - self.tp_atr_mult * atr_now)
            sl = float(pos.entry_price + self.stop_atr_mult * atr_now)
            move_atr = (pos.entry_price - close_now) / atr_now if atr_now > 0 else 0.0

            if move_atr >= self.breakeven_activate_atr:
                sl = min(sl, pos.entry_price - self.breakeven_offset_atr * atr_now)
                breakeven_armed = True

            if move_atr >= self.trail_activate_atr:
                trail_stop = float(close_now + self.trail_stop_atr_mult * atr_now)
                sl = min(sl, trail_stop)

            if l <= sl <= h:
                return ExitDecision("close", "sl", float(sl), float(tp), float(sl), trail_stop, breakeven_armed, {})
            if l <= tp <= h:
                return ExitDecision("close", "tp", float(tp), float(tp), float(sl), trail_stop, breakeven_armed, {})
            if not np.isnan(ema_fast) and not np.isnan(ema_slow) and ema_fast > ema_slow:
                px = _safe_float(bar.get("open"), close_now)
                return ExitDecision("close", "trend_break", float(px), float(tp), float(sl), trail_stop, breakeven_armed, {})

        if pos.bars_held >= self.time_stop_bars:
            px = _safe_float(bar.get("open"), close_now)
            return ExitDecision("close", "time_stop", float(px), float(tp), float(sl), trail_stop, breakeven_armed, {})

        return ExitDecision("hold", "", None, tp, sl, trail_stop, breakeven_armed, {})


class MeanReversionExitPolicy:
    def __init__(
        self,
        *,
        stop_atr_mult: float = 1.5,
        time_stop_bars: int = 48,
    ):
        self.stop_atr_mult = float(stop_atr_mult)
        self.time_stop_bars = int(time_stop_bars)

    def decide(self, pos: PositionState, prev_bar: pd.Series, bar: pd.Series) -> ExitDecision:
        atr_now = _safe_float(prev_bar.get("atr_14"), 0.0)
        basis = _safe_float(prev_bar.get("bb_mid_20"), np.nan)
        if atr_now <= 0.0 or np.isnan(basis):
            return ExitDecision("hold", "", None, None, None, pos.trail_stop, pos.breakeven_armed, {})

        h = _safe_float(bar.get("high"), np.nan)
        l = _safe_float(bar.get("low"), np.nan)
        close_now = _safe_float(prev_bar.get("close"), 0.0)

        side = str(pos.side).lower()
        if side == "long":
            tp = float(basis)
            sl = float(pos.entry_price - self.stop_atr_mult * atr_now)

            if l <= sl <= h:
                return ExitDecision("close", "sl", float(sl), float(tp), float(sl), None, False, {})
            if l <= tp <= h:
                return ExitDecision("close", "tp", float(tp), float(tp), float(sl), None, False, {})
        else:
            tp = float(basis)
            sl = float(pos.entry_price + self.stop_atr_mult * atr_now)

            if l <= sl <= h:
                return ExitDecision("close", "sl", float(sl), float(tp), float(sl), None, False, {})
            if l <= tp <= h:
                return ExitDecision("close", "tp", float(tp), float(tp), float(sl), None, False, {})

        if pos.bars_held >= self.time_stop_bars:
            px = _safe_float(bar.get("open"), close_now)
            return ExitDecision("close", "time_stop", float(px), float(tp), float(sl), None, False, {})

        return ExitDecision("hold", "", None, tp, sl, None, False, {})


def build_exit_policy(family: str) -> Any:
    fam = str(family or "").lower()
    if fam == "mean_reversion":
        return MeanReversionExitPolicy()
    return TrendAtrDynamicExitPolicy()


def build_actions_for_ts(
    *,
    prev_weights: dict[str, float],
    curr_targets: dict[str, dict[str, Any]],
) -> list[LifecycleAction]:
    actions: list[LifecycleAction] = []
    symbols = sorted(set(prev_weights.keys()) | set(curr_targets.keys()))

    for sym in symbols:
        prev_w = float(prev_weights.get(sym, 0.0) or 0.0)
        curr = curr_targets.get(sym)
        target_w = float((curr or {}).get("target_weight", 0.0) or 0.0)

        prev_sign = _sign(prev_w)
        target_sign = _sign(target_w)

        strategy_id = _safe_str((curr or {}).get("strategy_id", ""), "")
        family = _safe_str((curr or {}).get("family", "generic"), "generic")
        side = _safe_str((curr or {}).get("side", "flat"), "flat")

        if abs(prev_w) <= 1e-12 and abs(target_w) <= 1e-12:
            continue

        if abs(prev_w) <= 1e-12 and abs(target_w) > 1e-12:
            actions.append(
                LifecycleAction(
                    action="open",
                    symbol=sym,
                    strategy_id=strategy_id,
                    family=family,
                    side=side,
                    reason="target_open",
                    prev_weight=prev_w,
                    target_weight=target_w,
                    delta_weight=target_w - prev_w,
                    close_weight=0.0,
                    open_weight=abs(target_w),
                    meta=dict(curr or {}),
                )
            )
            continue

        if abs(prev_w) > 1e-12 and abs(target_w) <= 1e-12:
            actions.append(
                LifecycleAction(
                    action="close",
                    symbol=sym,
                    strategy_id=strategy_id,
                    family=family,
                    side="long" if prev_w > 0 else "short",
                    reason="target_flatten",
                    prev_weight=prev_w,
                    target_weight=target_w,
                    delta_weight=target_w - prev_w,
                    close_weight=abs(prev_w),
                    open_weight=0.0,
                    meta={},
                )
            )
            continue

        if prev_sign != 0 and target_sign != 0 and prev_sign != target_sign:
            actions.append(
                LifecycleAction(
                    action="close",
                    symbol=sym,
                    strategy_id=strategy_id,
                    family=family,
                    side="long" if prev_w > 0 else "short",
                    reason="target_reverse",
                    prev_weight=prev_w,
                    target_weight=target_w,
                    delta_weight=target_w - prev_w,
                    close_weight=abs(prev_w),
                    open_weight=0.0,
                    meta={},
                )
            )
            actions.append(
                LifecycleAction(
                    action="open",
                    symbol=sym,
                    strategy_id=strategy_id,
                    family=family,
                    side=side,
                    reason="target_reverse",
                    prev_weight=prev_w,
                    target_weight=target_w,
                    delta_weight=target_w - prev_w,
                    close_weight=0.0,
                    open_weight=abs(target_w),
                    meta=dict(curr or {}),
                )
            )
            continue

        if abs(target_w) > 1e-12:
            actions.append(
                LifecycleAction(
                    action="hold_target",
                    symbol=sym,
                    strategy_id=strategy_id,
                    family=family,
                    side=side,
                    reason="target_hold_same_side",
                    prev_weight=prev_w,
                    target_weight=target_w,
                    delta_weight=target_w - prev_w,
                    close_weight=0.0,
                    open_weight=0.0,
                    meta=dict(curr or {}),
                )
            )

    return actions


def run_audit(
    *,
    name: str,
    allocation_inputs_trace: str,
    initial_equity: float,
    maker_fee_bps: float,
    taker_fee_bps: float,
    cooldown_after_close_bars: int,
    exchange_id: str,
    cache_dir: str,
    timeframe: str,
) -> None:
    rows = load_trace(allocation_inputs_trace)
    if not rows:
        raise RuntimeError("allocation_inputs_trace vacío")

    ts_list = [pd.Timestamp(str(r["ts"])) for r in rows]
    ts_list = sorted([x for x in ts_list if x == x])
    if not ts_list:
        raise RuntimeError("No se encontraron timestamps válidos en el trace")

    start_ts = ts_list[0] - pd.Timedelta(hours=300)
    end_ts = ts_list[-1] + pd.Timedelta(hours=2)

    all_symbols: set[str] = set()
    for r in rows:
        stage_weights = _extract_stage_weights(r)
        all_symbols.update(stage_weights.keys())
        for c in list(r.get("selected_candidates", []) or []):
            sym = str(c.get("symbol", "") or "")
            if sym:
                all_symbols.add(sym)

    symbols = sorted(s for s in all_symbols if s)
    if not symbols:
        raise RuntimeError("No se pudieron inferir símbolos del trace")

    market = build_market_data(
        symbols,
        start=str(start_ts),
        end=str(end_ts),
        exchange_id=exchange_id,
        cache_dir=cache_dir,
        timeframe=timeframe,
    )

    if not market:
        raise RuntimeError("No se pudo cargar market data real para ningún símbolo")

    maker_fee = float(maker_fee_bps) / 10000.0
    taker_fee = float(taker_fee_bps) / 10000.0

    open_positions: dict[str, PositionState] = {}
    cooldown_left_by_symbol: dict[str, int] = {}
    prev_target_weights: dict[str, float] = {}

    actions_rows: list[dict[str, Any]] = []
    trades_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    cash = float(initial_equity)
    blocked_open_no_price = 0
    opened_positions = 0

    for row in rows:
        ts = pd.Timestamp(str(row["ts"]))
        ts_ms = int(ts.value // 10**6)

        mtm = 0.0
        for sym, pos in open_positions.items():
            df = market.get(sym)
            if df is None or ts not in df.index:
                continue
            close_px = _safe_float(df.loc[ts, "close"], np.nan)
            if np.isnan(close_px):
                continue
            if pos.side == "long":
                mtm += (close_px - pos.entry_price) * pos.qty
            else:
                mtm += (pos.entry_price - close_px) * pos.qty

        equity = cash + mtm
        equity_rows.append(
            {
                "ts": ts.isoformat(),
                "timestamp": ts_ms,
                "cash": float(cash),
                "mtm": float(mtm),
                "equity": float(equity),
                "open_positions": int(len(open_positions)),
            }
        )

        for sym in list(open_positions.keys()):
            pos = open_positions.get(sym)
            if pos is None:
                continue

            df = market.get(sym)
            if df is None or ts not in df.index:
                pos.bars_held += 1
                continue

            iloc = df.index.get_loc(ts)
            if isinstance(iloc, slice) or isinstance(iloc, np.ndarray):
                pos.bars_held += 1
                continue
            if int(iloc) <= 0:
                pos.bars_held += 1
                continue

            prev_bar = df.iloc[int(iloc) - 1]
            bar = df.iloc[int(iloc)]

            policy = build_exit_policy(pos.family)
            decision = policy.decide(pos, prev_bar, bar)

            pos.last_tp = decision.tp_price
            pos.last_sl = decision.sl_price
            pos.trail_stop = decision.trail_stop
            pos.breakeven_armed = decision.breakeven_armed

            if decision.action != "close" or decision.exit_price is None:
                pos.bars_held += 1
                continue

            exit_px = float(decision.exit_price)
            fee = abs(exit_px * pos.qty) * taker_fee

            if pos.side == "long":
                pnl = (exit_px - pos.entry_price) * pos.qty - fee
            else:
                pnl = (pos.entry_price - exit_px) * pos.qty - fee

            cash += pnl

            trades_rows.append(
                {
                    "symbol": sym,
                    "strategy_id": pos.strategy_id,
                    "family": pos.family,
                    "side": pos.side,
                    "entry_ts": pd.to_datetime(pos.entry_ts, unit="ms", utc=True).isoformat(),
                    "exit_ts": ts.isoformat(),
                    "entry_price": float(pos.entry_price),
                    "exit_price": float(exit_px),
                    "qty": float(pos.qty),
                    "entry_weight": float(pos.entry_weight),
                    "bars_held": int(pos.bars_held),
                    "exit_reason": str(decision.exit_reason),
                    "pnl": float(pnl),
                    "tp_price": decision.tp_price,
                    "sl_price": decision.sl_price,
                    "trail_stop": decision.trail_stop,
                    "breakeven_armed": bool(decision.breakeven_armed),
                }
            )

            cooldown_left_by_symbol[sym] = int(cooldown_after_close_bars)
            open_positions.pop(sym, None)

        curr_target_weights = _extract_stage_weights(row)
        open_decisions = _extract_open_decisions(row, curr_target_weights)
        actions = build_actions_for_ts(prev_weights=prev_target_weights, curr_targets=open_decisions)

        for action in actions:
            actions_rows.append(
                {
                    "ts": ts.isoformat(),
                    "timestamp": ts_ms,
                    **asdict(action),
                }
            )

            sym = action.symbol

            if action.action == "close":
                pos = open_positions.get(sym)
                if pos is None:
                    continue

                df = market.get(sym)
                if df is None or ts not in df.index:
                    continue

                open_px = _safe_float(df.loc[ts, "open"], np.nan)
                if np.isnan(open_px):
                    open_px = _safe_float(df.loc[ts, "close"], np.nan)
                if np.isnan(open_px):
                    continue

                fee = abs(open_px * pos.qty) * taker_fee
                if pos.side == "long":
                    pnl = (open_px - pos.entry_price) * pos.qty - fee
                else:
                    pnl = (pos.entry_price - open_px) * pos.qty - fee

                cash += pnl

                trades_rows.append(
                    {
                        "symbol": sym,
                        "strategy_id": pos.strategy_id,
                        "family": pos.family,
                        "side": pos.side,
                        "entry_ts": pd.to_datetime(pos.entry_ts, unit="ms", utc=True).isoformat(),
                        "exit_ts": ts.isoformat(),
                        "entry_price": float(pos.entry_price),
                        "exit_price": float(open_px),
                        "qty": float(pos.qty),
                        "entry_weight": float(pos.entry_weight),
                        "bars_held": int(pos.bars_held),
                        "exit_reason": str(action.reason),
                        "pnl": float(pnl),
                        "tp_price": pos.last_tp,
                        "sl_price": pos.last_sl,
                        "trail_stop": pos.trail_stop,
                        "breakeven_armed": bool(pos.breakeven_armed),
                    }
                )

                cooldown_left_by_symbol[sym] = int(cooldown_after_close_bars)
                open_positions.pop(sym, None)

            elif action.action == "open":
                if sym in open_positions:
                    continue

                if int(cooldown_left_by_symbol.get(sym, 0) or 0) > 0:
                    continue

                df = market.get(sym)
                if df is None or ts not in df.index:
                    blocked_open_no_price += 1
                    continue

                candle = df.loc[ts]
                entry_px = _safe_float(candle.get("open"), np.nan)
                if np.isnan(entry_px):
                    entry_px = _safe_float(candle.get("close"), np.nan)
                if np.isnan(entry_px) or entry_px <= 0.0:
                    blocked_open_no_price += 1
                    continue

                sig_meta = dict(action.meta.get("signal_meta", {}) or {})
                entry_atr = _safe_float(candle.get("atr_14"), sig_meta.get("atr", 0.0))
                if entry_atr <= 0.0:
                    entry_atr = max(1e-9, entry_px * 0.01)

                current_equity = cash
                notional = max(0.0, float(current_equity) * abs(float(action.open_weight)))
                if notional <= 0.0:
                    continue

                qty = notional / entry_px
                fee = abs(notional) * maker_fee
                cash -= fee

                open_positions[sym] = PositionState(
                    symbol=sym,
                    strategy_id=str(action.strategy_id),
                    family=str(action.family),
                    side=str(action.side),
                    entry_ts=ts_ms,
                    entry_price=float(entry_px),
                    entry_atr=float(entry_atr),
                    entry_weight=float(action.open_weight),
                    qty=float(qty),
                )
                opened_positions += 1

        for sym in list(cooldown_left_by_symbol.keys()):
            if int(cooldown_left_by_symbol[sym]) > 0:
                cooldown_left_by_symbol[sym] = int(cooldown_left_by_symbol[sym]) - 1

        prev_target_weights = {str(k): float(v or 0.0) for k, v in curr_target_weights.items()}

    eq = pd.DataFrame(equity_rows)
    if eq.empty:
        raise RuntimeError("No se generó equity")

    eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
    eq["ret"] = eq["equity"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    trades = pd.DataFrame(trades_rows)
    actions_df = pd.DataFrame(actions_rows)

    equity_start = float(eq["equity"].iloc[0])
    equity_final = float(eq["equity"].iloc[-1])

    metrics = {
        "equity_start": float(equity_start),
        "equity_final": float(equity_final),
        "total_return_pct": float((equity_final / equity_start - 1.0) * 100.0) if equity_start != 0 else 0.0,
        "sharpe_annual": float(_annualized_sharpe(eq["ret"])),
        "max_drawdown_pct": float(_max_drawdown_pct(eq["equity"])),
        "trade_count": int(len(trades)),
        "win_rate_pct": float((trades["pnl"] > 0).mean() * 100.0) if not trades.empty else 0.0,
        "opened_positions": int(opened_positions),
        "blocked_open_no_price": int(blocked_open_no_price),
    }

    out_metrics = Path(f"results/trade_lifecycle_metrics_{name}.json")
    out_trades = Path(f"results/trade_lifecycle_trades_{name}.csv")
    out_equity = Path(f"results/trade_lifecycle_equity_{name}.csv")
    out_actions = Path(f"results/trade_lifecycle_actions_{name}.csv")

    out_metrics.parent.mkdir(parents=True, exist_ok=True)

    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    trades.to_csv(out_trades, index=False)
    eq.to_csv(out_equity, index=False)
    actions_df.to_csv(out_actions, index=False)

    print(f"saved: {out_metrics}")
    print(f"saved: {out_trades}")
    print(f"saved: {out_equity}")
    print(f"saved: {out_actions}")

    print("\n=== LIFECYCLE METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if not trades.empty:
        print("\n=== EXIT REASONS ===")
        print(trades["exit_reason"].value_counts(dropna=False).to_string())

        print("\n=== BY STRATEGY ===")
        g = (
            trades.groupby(["strategy_id", "side"], dropna=False)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--allocation-inputs-trace", required=True)
    ap.add_argument("--initial-equity", type=float, default=1000.0)
    ap.add_argument("--maker-fee-bps", type=float, default=0.0)
    ap.add_argument("--taker-fee-bps", type=float, default=0.0)
    ap.add_argument("--cooldown-after-close-bars", type=int, default=1)
    ap.add_argument("--exchange-id", default="binanceusdm")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--timeframe", default="1h")
    args = ap.parse_args()

    run_audit(
        name=str(args.name),
        allocation_inputs_trace=str(args.allocation_inputs_trace),
        initial_equity=float(args.initial_equity),
        maker_fee_bps=float(args.maker_fee_bps),
        taker_fee_bps=float(args.taker_fee_bps),
        cooldown_after_close_bars=int(args.cooldown_after_close_bars),
        exchange_id=str(args.exchange_id),
        cache_dir=str(args.cache_dir),
        timeframe=str(args.timeframe),
    )


if __name__ == "__main__":
    main()
