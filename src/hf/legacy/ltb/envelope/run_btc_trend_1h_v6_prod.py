# run_btc_trend_1h_v6_prod.py
# BTC Trend 1H v6.0 PROD — incremental improvements (no arch changes)
#
# Based on: run_btc_trend_1h_v5_prod.py
# Adds:
# - cooldown after position closes (anti-whipsaw)
# - ATR% filter (avoid too low/high vol)
# - trend strength filter: EMA separation in ATR
# - EMA slope filter
# - exits: break-even activation + trailing (keeps bracket + refresh exits)
#
# Wrapper methods used:
#   fetch_recent_ohlcv, fetch_open_positions, fetch_balance
#   fetch_open_trigger_orders, cancel_trigger_order
#   place_trigger_limit_order, place_trigger_market_order

import os
import sys
import json
import fcntl
from datetime import datetime

import ta
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from hf.legacy.ltb.utilities.bitget_futures import BitgetFutures

params = {
    "symbol": "BTC/USDT:USDT",
    "timeframe": "1h",
    "margin_mode": "isolated",
    "balance_fraction": 1,
    "leverage": 2,

    "position_size_percentage": 20,
    "use_longs": True,
    "use_shorts": True,

    "ema_fast": 20,
    "ema_slow": 150,          # adjusted for 200-candle API return
    "adx_period": 14,
    "adx_enter_min": 27,
    "adx_exit_min": 22,

    "atr_period": 14,
    "stop_atr_mult": 1.4,
    "tp_atr_mult": 2.6,

    "entry_buffer_atr_mult": 0.20,

    "trail_activate_atr": 1.0,
    "trail_stop_atr_mult": 1.2,

    # v6: break-even
    "breakeven_activate_atr": 0.9,
    "breakeven_offset_atr": 0.10,

    "trigger_price_delta": 0.0015,

    # bracket
    "preplace_bracket": True,
    "preplace_ttl_runs": 3,

    # v6: volatility filter (ATR% = ATR/close)
    "atrp_min": 0.0025,   # 0.25%
    "atrp_max": 0.0200,   # 2.00%

    # v6: trend quality filters
    "min_ema_sep_atr": 0.30,   # require |ema_fast-ema_slow| >= X*ATR
    "ema_slope_lookback": 6,   # candles back to measure EMA fast slope
    "min_ema_slope_atr": 0.05, # require slope magnitude >= X*ATR

    # v6: cooldown after a position closes
    "cooldown_after_close_runs": 2,
}

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
KEY_PATH = os.path.join(ROOT_DIR, "secret.json")
KEY_NAME = "envelope"

TRACKER_FILE = os.path.join(os.path.dirname(__file__), "tracker_btc_trend_1h_v6_prod.json")
LOCK_FILE = "/tmp/btc_trend_1h.lock"


def _now():
    return datetime.now().strftime("%H:%M:%S")


def acquire_lock():
    f = open(LOCK_FILE, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except BlockingIOError:
        return None


def load_tracker():
    if os.path.exists(TRACKER_FILE):
        try:
            with open(TRACKER_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_tracker(state):
    try:
        with open(TRACKER_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"{_now()}: tracker save error: {e}")


def get_bitget():
    with open(KEY_PATH, "r") as f:
        api_setup = json.load(f)[KEY_NAME]
    return BitgetFutures(api_setup)


def fetch_ohlcv_df(bitget: BitgetFutures, symbol: str, timeframe: str, limit: int = 350) -> pd.DataFrame:
    df = bitget.fetch_recent_ohlcv(symbol, timeframe, limit)
    # Drop the currently forming candle
    if isinstance(df, pd.DataFrame) and len(df) > 1:
        return df.iloc[:-1].reset_index(drop=False)
    return df


def get_open_position(bitget: BitgetFutures, symbol: str):
    positions = bitget.fetch_open_positions(symbol)
    return positions[0] if positions else None


def wallet_usdt(bitget: BitgetFutures) -> float:
    bal = bitget.fetch_balance()
    usdt = bal.get("USDT", {})
    return float(usdt.get("total", usdt.get("free", 0.0)))


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=params["ema_fast"]).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=params["ema_slow"]).ema_indicator()

    adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=params["adx_period"])
    df["adx"] = adx_ind.adx()

    atr_ind = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=params["atr_period"])
    df["atr"] = atr_ind.average_true_range()
    return df


def _extract_oid(o):
    if not isinstance(o, dict):
        return None
    return o.get("id") or o.get("orderId") or o.get("info", {}).get("orderId")


def _is_reduce_only(o):
    if not isinstance(o, dict):
        return False
    info = o.get("info", {}) if isinstance(o.get("info", {}), dict) else {}
    return bool(o.get("reduceOnly") or info.get("reduceOnly") or info.get("reduce") or o.get("reduce"))


def cancel_trigger_ids(bitget, symbol, ids):
    for oid in ids:
        if not oid:
            continue
        try:
            bitget.cancel_trigger_order(oid, symbol)
        except Exception as e:
            print(f"{_now()}: cancel_trigger_order failed for {oid}: {e}")


def cancel_all_reduce_only_triggers(bitget, symbol):
    """Best-effort: cancel all open reduce-only trigger orders for this symbol."""
    try:
        orders = bitget.fetch_open_trigger_orders(symbol) or []
    except Exception:
        return
    for o in orders:
        if _is_reduce_only(o):
            oid = _extract_oid(o)
            if oid:
                try:
                    bitget.cancel_trigger_order(oid, symbol)
                except Exception:
                    pass


def candle_epoch_seconds(sig_row: pd.Series, df: pd.DataFrame):
    if "timestamp" in df.columns:
        try:
            return int(pd.Timestamp(sig_row["timestamp"]).timestamp())
        except Exception:
            pass
    try:
        return int(pd.Timestamp(sig_row.name).timestamp())
    except Exception:
        return None


def preplace_bracket(bitget, symbol, entry_side, amount, expected_entry, atr, tracker):
    if atr != atr or atr <= 0:
        print(f"{_now()}: ATR invalid -> skip preplace bracket")
        return

    if entry_side == "buy":
        close_side = "sell"
        tp = expected_entry + params["tp_atr_mult"] * atr
        sl = expected_entry - params["stop_atr_mult"] * atr
    else:
        close_side = "buy"
        tp = expected_entry - params["tp_atr_mult"] * atr
        sl = expected_entry + params["stop_atr_mult"] * atr

    tp_res = bitget.place_trigger_market_order(
        symbol=symbol, side=close_side, amount=amount,
        trigger_price=tp, reduce=True, print_error=True
    )
    sl_res = bitget.place_trigger_market_order(
        symbol=symbol, side=close_side, amount=amount,
        trigger_price=sl, reduce=True, print_error=True
    )

    tracker["pre_bracket"] = {
        "runs_left": int(params.get("preplace_ttl_runs", 3)),
        "entry_side": entry_side,
        "expected_entry": float(expected_entry),
        "tp": float(tp),
        "sl": float(sl),
        "tp_id": _extract_oid(tp_res),
        "sl_id": _extract_oid(sl_res),
    }
    save_tracker(tracker)
    print(f"{_now()}: preplaced TP/SL for pending entry. TP={tp:.2f} SL={sl:.2f}")


def _update_cooldown(tracker: dict, has_position_now: bool):
    # If position existed last run and now it doesn't -> start cooldown
    had_prev = bool(tracker.get("had_position_last_run", False))
    if had_prev and (not has_position_now):
        cd = int(params.get("cooldown_after_close_runs", 0))
        if cd > 0:
            tracker["cooldown_runs_left"] = cd
            tracker["last_event"] = "position_closed"
            save_tracker(tracker)
            print(f"{_now()}: position closed detected -> cooldown {cd} runs")

    # Decrement if active
    cd_left = int(tracker.get("cooldown_runs_left", 0))
    if cd_left > 0:
        tracker["cooldown_runs_left"] = cd_left - 1
        save_tracker(tracker)
        print(f"{_now()}: cooldown active -> {tracker['cooldown_runs_left']} runs left")


def main():
    lock = acquire_lock()
    if not lock:
        print(f"{_now()}: another run is active -> skip")
        return

    print(f"\n{_now()}: >>> starting execution for {params['symbol']} (TREND v6.0 PROD)")
    bitget = get_bitget()

    data = fetch_ohlcv_df(bitget, params["symbol"], params["timeframe"], limit=350)
    if data is None or len(data) < 180:
        print(f"{_now()}: not enough data")
        return

    data = compute_indicators(data)
    sig = data.iloc[-1]
    sig_ts = candle_epoch_seconds(sig, data)

    position = get_open_position(bitget, params["symbol"])

    tracker = load_tracker()

    # cooldown detection based on previous run
    has_pos_now = bool(position and position.get("contracts", 0) > 0)
    _update_cooldown(tracker, has_pos_now)
    tracker["had_position_last_run"] = has_pos_now
    save_tracker(tracker)

    # =========================
    # If position exists: exits
    # =========================
    if has_pos_now:
        # Clean pre-bracket (entry is now filled)
        pre = tracker.get("pre_bracket", {})
        cancel_trigger_ids(bitget, params["symbol"], [pre.get("tp_id"), pre.get("sl_id")])
        tracker.pop("pre_bracket", None)
        save_tracker(tracker)

        # Refresh exits: cancel old reduce-only triggers before re-placing
        cancel_all_reduce_only_triggers(bitget, params["symbol"])

        side = position["side"]  # "long" or "short"
        entry = float(position["info"]["openPriceAvg"])
        amount = position["contracts"] * position["contractSize"]

        atr = float(sig["atr"])
        close_now = float(sig["close"])
        if atr != atr:
            print(f"{_now()}: ATR NaN, skip exits")
            return

        if side == "long":
            close_side = "sell"
            tp = entry + params["tp_atr_mult"] * atr
            sl = entry - params["stop_atr_mult"] * atr

            move_atr = (close_now - entry) / atr if atr > 0 else 0.0

            # v6: break-even
            if move_atr >= params.get("breakeven_activate_atr", 0.9):
                be_sl = entry + params.get("breakeven_offset_atr", 0.1) * atr
                sl = max(sl, be_sl)

            # trailing
            if move_atr >= params.get("trail_activate_atr", 1.0):
                trail_sl = close_now - params.get("trail_stop_atr_mult", 1.2) * atr
                sl = max(sl, trail_sl)

        else:
            close_side = "buy"
            tp = entry - params["tp_atr_mult"] * atr
            sl = entry + params["stop_atr_mult"] * atr

            move_atr = (entry - close_now) / atr if atr > 0 else 0.0

            # v6: break-even
            if move_atr >= params.get("breakeven_activate_atr", 0.9):
                be_sl = entry - params.get("breakeven_offset_atr", 0.1) * atr
                sl = min(sl, be_sl)

            # trailing
            if move_atr >= params.get("trail_activate_atr", 1.0):
                trail_sl = close_now + params.get("trail_stop_atr_mult", 1.2) * atr
                sl = min(sl, trail_sl)

        tracker["trail_sl"] = float(sl)
        save_tracker(tracker)

        bitget.place_trigger_market_order(
            symbol=params["symbol"], side=close_side, amount=amount,
            trigger_price=tp, reduce=True, print_error=True
        )
        bitget.place_trigger_market_order(
            symbol=params["symbol"], side=close_side, amount=amount,
            trigger_price=sl, reduce=True, print_error=True
        )

        print(f"{_now()}: pos {side} entry={entry:.2f} TP={tp:.2f} SL={sl:.2f} amount={amount}")
        return

    # ==============================
    # No position: anti-dup + bracket
    # ==============================

    # If cooldown active -> skip entries
    if int(tracker.get("cooldown_runs_left", 0)) > 0:
        print(f"{_now()}: cooldown -> skip new entries")
        return

    # If we pre-placed exits but entry not filled: decrement TTL and cancel on expiry
    pre = tracker.get("pre_bracket")
    if pre:
        pre["runs_left"] = int(pre.get("runs_left", 1)) - 1
        tracker["pre_bracket"] = pre
        save_tracker(tracker)
        if pre["runs_left"] <= 0:
            cancel_trigger_ids(bitget, params["symbol"], [pre.get("tp_id"), pre.get("sl_id")])
            tracker.pop("pre_bracket", None)
            save_tracker(tracker)
            print(f"{_now()}: canceled stale preplaced TP/SL (entry not filled)")

    # Anti-dup within same candle
    if sig_ts is not None and tracker.get("last_entry_candle_ts") == sig_ts:
        print(f"{_now()}: already placed/attempted entry this candle -> skip")
        return

    close_v = float(sig["close"])
    ema_fast = float(sig["ema_fast"])
    ema_slow = float(sig["ema_slow"])
    adx_v = float(sig["adx"])
    atr_v = float(sig["atr"])

    if any(x != x for x in [close_v, ema_fast, ema_slow, adx_v, atr_v]):
        print(f"{_now()}: indicators not ready")
        return

    # v6: ATR% filter
    atrp = (atr_v / close_v) if close_v > 0 else 0.0
    if atrp < params.get("atrp_min", 0.0) or atrp > params.get("atrp_max", 999.0):
        print(f"{_now()}: ATR%={atrp:.4f} out of [{params.get('atrp_min')},{params.get('atrp_max')}], skip")
        return

    # ADX hysteresis regime
    tracker.setdefault("trend_regime", False)
    in_regime = bool(tracker.get("trend_regime", False))

    if (not in_regime) and adx_v < params["adx_enter_min"]:
        print(f"{_now()}: ADX={adx_v:.1f} < {params['adx_enter_min']} (enter), skip")
        return

    if in_regime and adx_v < params["adx_exit_min"]:
        tracker["trend_regime"] = False
        save_tracker(tracker)
        print(f"{_now()}: ADX={adx_v:.1f} < {params['adx_exit_min']} (exit), disable trend regime")
        return

    if (not in_regime) and adx_v >= params["adx_enter_min"]:
        tracker["trend_regime"] = True
        save_tracker(tracker)

    # v6: trend quality filters (EMA separation + slope)
    sep = abs(ema_fast - ema_slow)
    if sep < params.get("min_ema_sep_atr", 0.0) * atr_v:
        print(f"{_now()}: EMA sep={sep:.2f} < {params.get('min_ema_sep_atr',0.0)}*ATR, skip")
        return

    lookback = int(params.get("ema_slope_lookback", 6))
    if lookback > 0 and len(data) > lookback:
        ema_fast_prev = float(data.iloc[-1 - lookback]["ema_fast"])
        if ema_fast_prev == ema_fast_prev:
            slope = ema_fast - ema_fast_prev
            min_slope = params.get("min_ema_slope_atr", 0.0) * atr_v
        else:
            slope = 0.0
            min_slope = 0.0
    else:
        slope = 0.0
        min_slope = 0.0

    w = wallet_usdt(bitget) * float(params.get("balance_fraction", 1))
    notional = (w * (params["position_size_percentage"] / 100.0)) * params["leverage"]
    buf = params.get("entry_buffer_atr_mult", 0.2) * atr_v

    placed = False

    # Long entry
    if params["use_longs"] and ema_fast > ema_slow and close_v <= (ema_fast - buf):
        # v6: slope check
        if slope < min_slope:
            print(f"{_now()}: EMA slope={slope:.2f} < {min_slope:.2f} (long), skip")
            return

        entry_price = ema_fast
        qty = notional / entry_price
        trigger_price = entry_price * (1 + params["trigger_price_delta"])

        bitget.place_trigger_limit_order(
            symbol=params["symbol"], side="buy", amount=qty,
            trigger_price=trigger_price, price=entry_price, print_error=True
        )
        placed = True
        if params.get("preplace_bracket", True):
            preplace_bracket(bitget, params["symbol"], "buy", qty, entry_price, atr_v, tracker)

        print(f"{_now()}: placed LONG @EMA{params['ema_fast']}={entry_price:.2f} trigger={trigger_price:.2f} ADX={adx_v:.1f} ATR%={atrp:.4f} qty={qty:.6f}")

    # Short entry
    elif params["use_shorts"] and ema_fast < ema_slow and close_v >= (ema_fast + buf):
        # v6: slope check
        if (-slope) < min_slope:
            print(f"{_now()}: EMA slope={slope:.2f} > -{min_slope:.2f} (short), skip")
            return

        entry_price = ema_fast
        qty = notional / entry_price
        trigger_price = entry_price * (1 - params["trigger_price_delta"])

        bitget.place_trigger_limit_order(
            symbol=params["symbol"], side="sell", amount=qty,
            trigger_price=trigger_price, price=entry_price, print_error=True
        )
        placed = True
        if params.get("preplace_bracket", True):
            preplace_bracket(bitget, params["symbol"], "sell", qty, entry_price, atr_v, tracker)

        print(f"{_now()}: placed SHORT @EMA{params['ema_fast']}={entry_price:.2f} trigger={trigger_price:.2f} ADX={adx_v:.1f} ATR%={atrp:.4f} qty={qty:.6f}")

    if placed and sig_ts is not None:
        tracker["last_entry_candle_ts"] = sig_ts
        tracker["last_action"] = "placed_entry"
        save_tracker(tracker)


if __name__ == "__main__":
    main()
