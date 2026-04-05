import time
import json
import os
from pathlib import Path

import pandas as pd
import ta

from hf.legacy.ltb.utilities.bitget_futures import BitgetFutures
from hf.execution.protective_orders import (
    classify_reduce_orders as tested_classify_reduce_orders,
    extract_trigger_amount,
    extract_trigger_price,
    should_keep_partial_tps,
)

APP = Path(__file__).resolve().parents[1]
SECRET = json.loads((APP / "secret.json").read_text())["envelope"]

LIVE_TRADING = os.getenv("LIVE_TRADING", "0") == "1"

LIVE_EXECUTION_CONFIG_PATH = APP / "deploy" / "live_execution_config.json"
LIVE_EXECUTION_CONFIG = json.loads(LIVE_EXECUTION_CONFIG_PATH.read_text(encoding="utf-8"))

DEFAULT_SYMBOL_CONFIG = dict(LIVE_EXECUTION_CONFIG.get("default_symbol_config", {}) or {})
SYMBOL_OVERRIDES = dict(LIVE_EXECUTION_CONFIG.get("symbol_overrides", {}) or {})

def symbol_to_prefix(symbol: str) -> str:
    return (
        str(symbol)
        .replace("/USDT:USDT", "")
        .replace("/USDT", "")
        .replace(":USDT", "")
        .replace("/", "_")
        .lower()
    )

def build_runtime_symbols(df: pd.DataFrame) -> dict:
    out = {}

    symbol_cols = sorted({
        c.replace("_execution_target_weight", "")
        for c in df.columns if c.endswith("_execution_target_weight")
    } | {
        c.replace("_cluster_target_weight", "")
        for c in df.columns if c.endswith("_cluster_target_weight")
    } | {
        c.replace("_w_after_signal_gating", "")
        for c in df.columns if c.endswith("_w_after_signal_gating")
    } | {
        c.replace("_w_after_smoothing", "")
        for c in df.columns if c.endswith("_w_after_smoothing")
    } | {
        c.replace("_w_after_ml_position_sizing", "")
        for c in df.columns if c.endswith("_w_after_ml_position_sizing")
    } | {
        c.replace("_w_raw_allocator", "")
        for c in df.columns if c.endswith("_w_raw_allocator")
    } | {
        c[2:]
        for c in df.columns if c.startswith("w_")
    })

    for base in symbol_cols:
        sym = base.replace("_usdt_usdt", "").upper() + "/USDT:USDT"
        prefix = symbol_to_prefix(sym)

        cfg = dict(DEFAULT_SYMBOL_CONFIG)
        cfg.update(SYMBOL_OVERRIDES.get(sym, {}))
        cfg["symbol"] = sym
        out[prefix] = cfg

    return out

def signed_weight(side: str, w: float) -> float:
    side = str(side or "flat").lower()
    w = float(w or 0.0)
    if side == "long":
        return abs(w)
    if side == "short":
        return -abs(w)
    return 0.0


def current_signed_qty(pos_list):
    if not pos_list:
        return 0.0
    p = pos_list[0]
    contracts = float(p.get("contracts") or 0.0)
    side = str(p.get("side") or "").lower()
    if side == "long":
        return contracts
    if side == "short":
        return -contracts
    info = p.get("info", {}) if isinstance(p.get("info"), dict) else {}
    hold_side = str(info.get("holdSide") or "").lower()
    if hold_side == "long":
        return contracts
    if hold_side == "short":
        return -contracts
    return contracts


def position_side_from_qty(qty: float) -> str:
    if qty > 0:
        return "long"
    if qty < 0:
        return "short"
    return "flat"


def refresh_current_qty(bitget, symbol: str, attempts: int = 3, sleep_seconds: float = 1.0) -> float:
    for attempt in range(1, attempts + 1):
        try:
            pos_list = bitget.fetch_open_positions(symbol) or []
            qty = current_signed_qty(pos_list)
            if abs(qty) > 1e-12:
                if attempt > 1:
                    print(f"POSITION_REFRESH_OK -> symbol={symbol} qty={qty} attempt={attempt}")
                return float(qty)
        except Exception as e:
            print(f"POSITION_REFRESH_WARN -> symbol={symbol} attempt={attempt} error={e}")
        if attempt < attempts and LIVE_TRADING:
            time.sleep(sleep_seconds)
    return 0.0


def place_market(bitget, symbol: str, side: str, qty: float, reduce: bool = False):
    qty = float(qty)
    if qty <= 0:
        return None
    print(f"ORDER -> symbol={symbol} side={side} qty={qty} reduceOnly={reduce} live={LIVE_TRADING}")
    if not LIVE_TRADING:
        return None
    try:
        return bitget.place_market_order(symbol=symbol, side=side, amount=qty, reduce=reduce)
    except Exception as e:
        msg = str(e)
        if reduce and ("22002" in msg or "No position to close" in msg):
            print(f"ORDER_WARN -> benign reduce-only rejection for {symbol}: {msg}")
            return None
        raise


def fetch_last_price(bitget, symbol: str) -> float:
    ticker = bitget.fetch_ticker(symbol)
    return float(ticker.get("last") or ticker.get("close") or 0.0)


def fetch_atr(bitget, symbol: str, timeframe: str, limit: int, period: int) -> float:
    df = bitget.fetch_recent_ohlcv(symbol, timeframe, limit)
    if len(df) > 1:
        df = df.iloc[:-1].copy()
    df = df.reset_index(drop=False)
    atr = ta.volatility.AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=period,
    ).average_true_range()
    v = float(atr.iloc[-1])
    if not (v > 0):
        raise RuntimeError(f"Invalid ATR for {symbol}")
    return v


def desired_stop_price(side: str, ref_price: float, atr: float, sl_atr_mult: float) -> float:
    if side == "long":
        return ref_price - sl_atr_mult * atr
    if side == "short":
        return ref_price + sl_atr_mult * atr
    return 0.0


def desired_take_profit_price(
    side: str,
    ref_price: float,
    atr: float,
    tp_atr_mult_min: float,
    tp_atr_mult_max: float,
    tp_atr_pct_pivot: float,
) -> float:
    atr_pct = atr / ref_price if ref_price > 0 else 0.0
    pivot = max(float(tp_atr_pct_pivot), 1e-9)

    raw_mult = float(tp_atr_mult_min) + (atr_pct / pivot) * (
        float(tp_atr_mult_max) - float(tp_atr_mult_min)
    )
    tp_mult = max(float(tp_atr_mult_min), min(float(tp_atr_mult_max), raw_mult))

    if side == "long":
        return ref_price + tp_mult * atr
    if side == "short":
        return ref_price - tp_mult * atr
    return 0.0


def desired_take_profit_price_fixed_mult(side: str, ref_price: float, atr: float, tp_atr_mult: float) -> float:
    if side == "long":
        return ref_price + float(tp_atr_mult) * atr
    if side == "short":
        return ref_price - float(tp_atr_mult) * atr
    return 0.0


def extract_trigger_order_id(o):
    if not isinstance(o, dict):
        return None
    return o.get("id") or o.get("orderId") or (o.get("info", {}) or {}).get("orderId")


def extract_trigger_price(o):
    if not isinstance(o, dict):
        return None
    info = o.get("info", {}) if isinstance(o.get("info"), dict) else {}
    for k in ["triggerPrice", "trigger_price", "stopPrice", "planPrice", "executePrice"]:
        v = info.get(k, None)
        if v not in (None, ""):
            try:
                return float(v)
            except Exception:
                pass
    for k in ["triggerPrice", "stopPrice", "price"]:
        v = o.get(k, None)
        if v not in (None, ""):
            try:
                return float(v)
            except Exception:
                pass
    return None


def is_reduce_only(o):
    if not isinstance(o, dict):
        return False
    info = o.get("info", {}) if isinstance(o.get("info"), dict) else {}
    trade_side = str(info.get("tradeSide") or "").lower()
    plan_status = str(info.get("planStatus") or "").lower()
    return bool(
        o.get("reduceOnly")
        or o.get("reduce")
        or info.get("reduceOnly")
        or info.get("reduce")
        or trade_side == "close"
        or plan_status == "live" and trade_side == "close"
    )


def side_matches_reduce_order(pos_side: str, order_obj) -> bool:
    raw_side = str(order_obj.get("side") or (order_obj.get("info", {}) or {}).get("side") or "").lower()
    if pos_side == "long":
        return raw_side == "sell"
    if pos_side == "short":
        return raw_side == "buy"
    return False


def fetch_reduce_only_trigger_orders(bitget, symbol: str):
    try:
        orders = bitget.fetch_open_trigger_orders(symbol) or []
    except Exception:
        return []
    return [o for o in orders if is_reduce_only(o)]


def cancel_trigger_order_safe(bitget, symbol: str, oid):
    if not oid:
        return
    print(f"CANCEL_TRIGGER -> symbol={symbol} id={oid} live={LIVE_TRADING}")
    if LIVE_TRADING:
        bitget.cancel_trigger_order(oid, symbol)


def classify_reduce_orders(open_reduce, pos_side: str, ref_price: float, pos_qty: float | None = None):
    return tested_classify_reduce_orders(open_reduce, pos_side, ref_price, pos_qty)


def log_trigger_snapshot(label: str, symbol: str, orders) -> None:
    orders = orders or []
    print(f"{label} -> symbol={symbol} count={len(orders)}")
    for o in orders:
        info = o.get("info") or {}
        print(
            "TRIGGER_SNAPSHOT -> "
            f"symbol={symbol} "
            f"id={o.get('id')} "
            f"amount={o.get('amount')} "
            f"trigger={o.get('triggerPrice')} "
            f"trigger_type={info.get('triggerType')} "
            f"plan_status={info.get('planStatus')} "
            f"side={o.get('side')} "
            f"trade_side={info.get('tradeSide')}"
        )

def maintain_single_trigger(bitget, symbol: str, desired_price: float, pos_side: str, qty: float, kind: str, rel_tol: float, ref_price: float):
    reduce_side = "sell" if pos_side == "long" else "buy"
    open_reduce = fetch_reduce_only_trigger_orders(bitget, symbol)
    log_trigger_snapshot("PROTECTIVE_BEFORE", symbol, open_reduce)
    stop_like, tp_like = classify_reduce_orders(open_reduce, pos_side, ref_price, qty)
    current_bucket = stop_like if kind == "sl" else tp_like
    other_bucket = tp_like if kind == "sl" else stop_like

    keep_existing = False
    if len(current_bucket) == 1:
        existing_price = extract_trigger_price(current_bucket[0])
        if existing_price and desired_price > 0:
            rel_diff = abs(existing_price - desired_price) / max(abs(desired_price), 1e-9)
            if rel_diff <= rel_tol:
                keep_existing = True
                print(f"{kind}_action: keep existing (existing={existing_price}, desired={desired_price}, rel_diff={rel_diff:.6f})")

    if keep_existing:
        extras = [o for o in current_bucket[1:]]
        for o in extras:
            cancel_trigger_order_safe(bitget, symbol, extract_trigger_order_id(o))
        return

    for o in current_bucket:
        cancel_trigger_order_safe(bitget, symbol, extract_trigger_order_id(o))

    print(f"PLACE_{kind.upper()} -> symbol={symbol} side={reduce_side} qty={qty} trigger={desired_price} live={LIVE_TRADING}")
    if LIVE_TRADING:
        bitget.place_trigger_market_order(
            symbol=symbol,
            side=reduce_side,
            amount=qty,
            trigger_price=desired_price,
            reduce=True,
            trigger_type="mark_price",
        )



def fetch_open_profit_loss_orders(bitget, symbol: str):
    try:
        return bitget.fetch_open_tpsl_orders(symbol) or []
    except Exception:
        return []

def get_plan_type(o):
    info = o.get("info") or {}
    return str(o.get("planType") or info.get("planType") or "").lower()

def get_plan_order_id(o):
    info = o.get("info") or {}
    return o.get("orderId") or info.get("orderId") or o.get("id") or info.get("id")

def get_plan_client_oid(o):
    info = o.get("info") or {}
    return o.get("clientOid") or info.get("clientOid")

def get_plan_size(o):
    info = o.get("info") or {}
    v = o.get("size") or info.get("size") or o.get("amount") or info.get("amount") or 0
    try:
        return float(v)
    except Exception:
        return 0.0

def get_plan_trigger_price(o):
    info = o.get("info") or {}
    for k in ["triggerPrice", "stopSurplusTriggerPrice", "stopLossTriggerPrice"]:
        v = o.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except Exception:
                pass
        v = info.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except Exception:
                pass
    return 0.0

def get_plan_trigger_type(o):
    info = o.get("info") or {}
    return str(o.get("triggerType") or info.get("triggerType") or "").lower()

def log_plan_snapshot(label: str, symbol: str, orders) -> None:
    orders = orders or []
    print(f"{label} -> symbol={symbol} count={len(orders)}")
    for o in orders:
        print(
            "PLAN_SNAPSHOT -> "
            f"symbol={symbol} "
            f"order_id={get_plan_order_id(o)} "
            f"client_oid={get_plan_client_oid(o)} "
            f"plan_type={get_plan_type(o)} "
            f"size={get_plan_size(o)} "
            f"trigger={get_plan_trigger_price(o)} "
            f"trigger_type={get_plan_trigger_type(o)}"
        )

def cancel_plan_order_safe(bitget, symbol: str, order_id: str, plan_type: str):
    if not order_id:
        return
    print(f"CANCEL_PLAN -> symbol={symbol} order_id={order_id} plan_type={plan_type} live={LIVE_TRADING}")
    if LIVE_TRADING:
        bitget.cancel_plan_order(symbol, order_id, plan_type)

def _rel_close(a: float, b: float, rel_tol: float = 1e-6) -> bool:
    if a == b:
        return True
    denom = max(abs(b), 1e-9)
    return abs(a - b) / denom <= rel_tol

def _find_matching_plan(orders, plan_type: str, trigger_price: float, size: float, trigger_type: str, price_tol: float):
    for o in orders:
        if get_plan_type(o) != plan_type:
            continue
        if get_plan_trigger_type(o) != str(trigger_type).lower():
            continue
        if not _rel_close(get_plan_trigger_price(o), float(trigger_price), price_tol):
            continue
        if not _rel_close(get_plan_size(o), float(size), 1e-9):
            continue
        return o
    return None


def ensure_protective_orders(bitget, symbol: str, pos_qty: float, ref_price: float, atr: float, cfg: dict):
    pos_side = position_side_from_qty(pos_qty)
    hold_side = "long" if pos_side == "long" else "short"
    qty = abs(pos_qty)

    # migrated model:
    # - SL and TP live as profit/loss plan orders
    # - stale trigger orders should be removed
    stale_triggers = fetch_reduce_only_trigger_orders(bitget, symbol)
    if stale_triggers:
        log_trigger_snapshot("STALE_TRIGGER_BEFORE", symbol, stale_triggers)
        for o in stale_triggers:
            cancel_trigger_order_safe(bitget, symbol, extract_trigger_order_id(o))

    plan_orders = fetch_open_profit_loss_orders(bitget, symbol)
    log_plan_snapshot("PROTECTIVE_PLAN_BEFORE", symbol, plan_orders)

    if pos_side == "flat" or qty <= 0:
        for o in plan_orders:
            cancel_plan_order_safe(bitget, symbol, get_plan_order_id(o), get_plan_type(o))
        print(f"protective_action: no position for {symbol} -> cancelled stale plan orders (count={len(plan_orders)})")
        return

    sl_price = desired_stop_price(pos_side, ref_price, atr, cfg["sl_atr_mult"])
    tp2_price = desired_take_profit_price(
        pos_side,
        ref_price,
        atr,
        cfg["tp_atr_mult_min"],
        cfg["tp_atr_mult_max"],
        cfg["tp_atr_pct_pivot"],
    )

    partial_tp_enabled = bool(cfg.get("partial_tp_enabled", True))
    partial_tp1_atr_mult = float(cfg.get("partial_tp1_atr_mult", 1.0) or 1.0)
    partial_tp1_fraction = float(cfg.get("partial_tp1_fraction", 0.5) or 0.5)

    sl_price = float(bitget.price_to_precision(symbol, sl_price))
    tp2_price = float(bitget.price_to_precision(symbol, tp2_price))
    tp1_price = desired_take_profit_price_fixed_mult(
        pos_side,
        ref_price,
        atr,
        partial_tp1_atr_mult,
    )
    tp1_price = float(bitget.price_to_precision(symbol, tp1_price))

    qty1 = 0.0
    qty2 = float(bitget.amount_to_precision(symbol, qty)) if qty > 0 else 0.0

    if partial_tp_enabled:
        raw_qty1 = max(0.0, float(qty) * partial_tp1_fraction)
        try:
            qty1 = float(bitget.amount_to_precision(symbol, raw_qty1)) if raw_qty1 > 0 else 0.0
            qty2 = max(0.0, float(qty) - float(qty1))
            qty2 = float(bitget.amount_to_precision(symbol, qty2)) if qty2 > 0 else 0.0

            if qty1 <= 0.0 or qty2 <= 0.0:
                print(
                    f"TP_SPLIT_COLLAPSED -> symbol={symbol} qty={qty} "
                    f"raw_qty1={raw_qty1} qty1={qty1} qty2={qty2}"
                )
                qty1 = 0.0
                qty2 = float(bitget.amount_to_precision(symbol, qty)) if qty > 0 else 0.0
        except Exception as e:
            print(
                f"TP_SPLIT_FALLBACK -> symbol={symbol} qty={qty} "
                f"raw_qty1={raw_qty1} error={e!r}"
            )
            qty1 = 0.0
            qty2 = float(bitget.amount_to_precision(symbol, qty)) if qty > 0 else 0.0

    price_tol = float(cfg.get("tp_refresh_rel_tol", 0.001) or 0.001)

    loss_plans = [o for o in plan_orders if get_plan_type(o) == "loss_plan"]
    profit_plans = [o for o in plan_orders if get_plan_type(o) == "profit_plan"]

    # ---- ensure SL exactly matches current position size ----
    sl_match = _find_matching_plan(loss_plans, "loss_plan", sl_price, qty, "mark_price", float(cfg.get("stop_refresh_rel_tol", 0.001) or 0.001))
    if sl_match:
        print(f"sl_action: keep existing loss_plan (qty={get_plan_size(sl_match)}, trigger={get_plan_trigger_price(sl_match)})")
        for o in loss_plans:
            if get_plan_order_id(o) != get_plan_order_id(sl_match):
                cancel_plan_order_safe(bitget, symbol, get_plan_order_id(o), "loss_plan")
    else:
        for o in loss_plans:
            cancel_plan_order_safe(bitget, symbol, get_plan_order_id(o), "loss_plan")
        print(f"PLACE_SL_PLAN -> symbol={symbol} hold_side={hold_side} qty={qty} trigger={sl_price} live={LIVE_TRADING}")
        if LIVE_TRADING:
            bitget.place_pos_stop_loss(
                symbol=symbol,
                hold_side=hold_side,
                trigger_price=sl_price,
                size=qty,
                client_oid=f"sl-{symbol.replace('/', '').replace(':', '')}-{int(time.time()*1000)}",
                trigger_type="mark_price",
            )

    if not partial_tp_enabled:
        profit_match = _find_matching_plan(profit_plans, "profit_plan", tp2_price, qty, "fill_price", price_tol)
        if profit_match:
            print(f"tp_action: keep single TP2 plan (qty={get_plan_size(profit_match)}, trigger={get_plan_trigger_price(profit_match)})")
            for o in profit_plans:
                if get_plan_order_id(o) != get_plan_order_id(profit_match):
                    cancel_plan_order_safe(bitget, symbol, get_plan_order_id(o), "profit_plan")
        else:
            for o in profit_plans:
                cancel_plan_order_safe(bitget, symbol, get_plan_order_id(o), "profit_plan")
            print(f"PLACE_TP2_PLAN -> symbol={symbol} hold_side={hold_side} qty={qty} trigger={tp2_price} live={LIVE_TRADING}")
            if LIVE_TRADING:
                bitget.place_pos_take_profit(
                    symbol=symbol,
                    hold_side=hold_side,
                    trigger_price=tp2_price,
                    size=qty,
                    client_oid=f"tp2-{symbol.replace('/', '').replace(':', '')}-{int(time.time()*1000)}",
                    trigger_type="fill_price",
                )
        final_plan_orders = fetch_open_profit_loss_orders(bitget, symbol)
        log_plan_snapshot("PROTECTIVE_PLAN_AFTER", symbol, final_plan_orders)
        return

    # ---- partial TP model ----
    # Case A: one TP remains.
    # If it matches TP2, assume TP1 already executed and keep only the final TP.
    if len(profit_plans) == 1:
        lone = profit_plans[0]

        lone_trigger = get_plan_trigger_price(lone)
        lone_size = get_plan_size(lone)

        price_match = False
        qty_match = False

        if lone_trigger is not None and tp2_price > 0:
            rel_diff = abs(float(lone_trigger) - float(tp2_price)) / max(abs(float(tp2_price)), 1e-9)
            price_match = rel_diff <= price_tol

        if lone_size is None:
            # Bitget sometimes omits size on open TP/SL plan responses.
            # In that case, if the remaining single TP matches TP2 by price, keep it.
            qty_match = True
        else:
            qty_tol = max(1e-12, abs(float(qty)) * 0.05)
            qty_match = abs(float(lone_size) - abs(float(qty))) <= qty_tol

        if price_match and qty_match:
            print(
                f"tp_action: keep remaining TP2 after partial execution "
                f"(plan_trigger={lone_trigger}, expected_tp2={tp2_price}, plan_size={lone_size}, qty={qty})"
            )
            final_plan_orders = fetch_open_profit_loss_orders(bitget, symbol)
            log_plan_snapshot("PROTECTIVE_PLAN_AFTER", symbol, final_plan_orders)
            return

        # If there is a single TP but it does not clearly match the desired final TP,
        # refresh to a single final TP only. Do not recreate TP1.
        oid = get_plan_order_id(lone)
        if oid:
            cancel_plan_order_safe(bitget, symbol, oid, "profit_plan")

        print(
            f"tp_action: replace ambiguous single TP with final-only TP "
            f"(plan_trigger={lone_trigger}, expected_tp2={tp2_price}, plan_size={lone_size}, qty={qty})"
        )

        print(f"PLACE_TP2_PLAN -> symbol={symbol} hold_side={hold_side} qty={qty} trigger={tp2_price} live={LIVE_TRADING}")
        if LIVE_TRADING:
            bitget.place_pos_take_profit(
                symbol=symbol,
                hold_side=hold_side,
                trigger_price=tp2_price,
                size=qty,
                client_oid=f"tp2-{symbol.replace('/', '').replace(':', '')}-{int(time.time()*1000)}",
                trigger_type="fill_price",
            )

        final_plan_orders = fetch_open_profit_loss_orders(bitget, symbol)
        log_plan_snapshot("PROTECTIVE_PLAN_AFTER", symbol, final_plan_orders)
        return

    # Case B: full split should exist on a fresh/full position
    tp1_match = _find_matching_plan(profit_plans, "profit_plan", tp1_price, qty1, "fill_price", price_tol) if qty1 > 0 else None
    tp2_match = _find_matching_plan(profit_plans, "profit_plan", tp2_price, qty2, "fill_price", price_tol) if qty2 > 0 else None

    if qty1 > 0 and qty2 > 0 and tp1_match and tp2_match and get_plan_order_id(tp1_match) != get_plan_order_id(tp2_match):
        print(f"tp_action: keep existing partial TP plans (count={len(profit_plans)})")
        for o in profit_plans:
            oid = get_plan_order_id(o)
            if oid not in {get_plan_order_id(tp1_match), get_plan_order_id(tp2_match)}:
                cancel_plan_order_safe(bitget, symbol, oid, "profit_plan")
        final_plan_orders = fetch_open_profit_loss_orders(bitget, symbol)
        log_plan_snapshot("PROTECTIVE_PLAN_AFTER", symbol, final_plan_orders)
        return

    for o in profit_plans:
        cancel_plan_order_safe(bitget, symbol, get_plan_order_id(o), "profit_plan")

    if qty1 > 0 and qty2 > 0:
        print(f"PLACE_TP1_PLAN -> symbol={symbol} hold_side={hold_side} qty={qty1} trigger={tp1_price} live={LIVE_TRADING}")
        if LIVE_TRADING:
            bitget.place_pos_take_profit(
                symbol=symbol,
                hold_side=hold_side,
                trigger_price=tp1_price,
                size=qty1,
                client_oid=f"tp1-{symbol.replace('/', '').replace(':', '')}-{int(time.time()*1000)}",
                trigger_type="fill_price",
            )

        print(f"PLACE_TP2_PLAN -> symbol={symbol} hold_side={hold_side} qty={qty2} trigger={tp2_price} live={LIVE_TRADING}")
        if LIVE_TRADING:
            bitget.place_pos_take_profit(
                symbol=symbol,
                hold_side=hold_side,
                trigger_price=tp2_price,
                size=qty2,
                client_oid=f"tp2-{symbol.replace('/', '').replace(':', '')}-{int(time.time()*1000)}",
                trigger_type="fill_price",
            )
    else:
        print(f"PLACE_TP2_PLAN -> symbol={symbol} hold_side={hold_side} qty={qty} trigger={tp2_price} live={LIVE_TRADING}")
        if LIVE_TRADING:
            bitget.place_pos_take_profit(
                symbol=symbol,
                hold_side=hold_side,
                trigger_price=tp2_price,
                size=qty,
                client_oid=f"tp2-{symbol.replace('/', '').replace(':', '')}-{int(time.time()*1000)}",
                trigger_type="fill_price",
            )

    final_plan_orders = fetch_open_profit_loss_orders(bitget, symbol)
    log_plan_snapshot("PROTECTIVE_PLAN_AFTER", symbol, final_plan_orders)


bitget = BitgetFutures(SECRET)


def ensure_leverage(bitget, symbol: str, leverage: int = 2):
    active_pos = []
    active_orders = []
    active_trigger_orders = []
    active_tpsl_orders = []

    try:
        active_pos = bitget.fetch_open_positions(symbol) or []
    except Exception as e:
        print(f"MARGIN_MODE_PRECHECK_WARN -> symbol={symbol} source=open_positions error={e!r}")

    try:
        active_orders = bitget.fetch_open_orders(symbol) or []
    except Exception as e:
        print(f"MARGIN_MODE_PRECHECK_WARN -> symbol={symbol} source=open_orders error={e!r}")

    try:
        active_trigger_orders = bitget.fetch_open_trigger_orders(symbol) or []
    except Exception as e:
        print(f"MARGIN_MODE_PRECHECK_WARN -> symbol={symbol} source=open_trigger_orders error={e!r}")

    try:
        active_tpsl_orders = bitget.fetch_open_tpsl_orders(symbol) or []
    except Exception as e:
        print(f"MARGIN_MODE_PRECHECK_WARN -> symbol={symbol} source=open_tpsl_orders error={e!r}")

    if active_pos or active_orders or active_trigger_orders or active_tpsl_orders:
        print(
            f"MARGIN_MODE_SKIP -> symbol={symbol} margin_mode=isolated "
            f"reason=active_state positions={len(active_pos)} orders={len(active_orders)} "
            f"trigger_orders={len(active_trigger_orders)} tpsl_orders={len(active_tpsl_orders)}"
        )
    else:
        try:
            bitget.set_margin_mode(symbol, "isolated")
            print(f"MARGIN_MODE_OK -> symbol={symbol} margin_mode=isolated")
        except Exception as e:
            print(f"MARGIN_MODE_ERROR -> symbol={symbol} margin_mode=isolated error={e!r}")

    try:
        bitget.set_leverage(symbol, "isolated", leverage)
        print(f"LEVERAGE_OK -> symbol={symbol} margin_mode=isolated leverage={leverage}")
    except Exception as e:
        print(f"LEVERAGE_ERROR -> symbol={symbol} margin_mode=isolated leverage={leverage} error={e!r}")

df = pd.read_csv(APP / "results/research_runtime_prod_v2_live_candidate.csv", low_memory=False)

SYMBOLS = build_runtime_symbols(df)
print("runtime_symbols:", list(SYMBOLS.keys()))
row = df.iloc[-1]

bal = bitget.fetch_balance()
usdt_total = float((bal.get("USDT", {}) or {}).get("total", (bal.get("USDT", {}) or {}).get("free", 0.0)))

print("=== LIVE RECONCILIATION ===")
print("live_trading:", LIVE_TRADING)

execution_snapshot = {}

print("usdt_total:", usdt_total)
print()

for prefix, cfg in SYMBOLS.items():
    symbol = cfg["symbol"]
    csv_prefix = f"{prefix}_usdt_usdt"
    ensure_leverage(bitget, symbol, 2)
    last = fetch_last_price(bitget, symbol)

    side = str(row.get(f"{csv_prefix}_side", "flat"))
    exec_weight = float(row.get(f"{csv_prefix}_execution_target_weight", 0.0) or 0.0)
    cluster_weight = float(row.get(f"{csv_prefix}_cluster_target_weight", 0.0) or 0.0)
    signal_gating_weight = float(row.get(f"{csv_prefix}_w_after_signal_gating", 0.0) or 0.0)
    smoothing_weight = float(row.get(f"{csv_prefix}_w_after_smoothing", 0.0) or 0.0)
    ml_sizing_weight = float(row.get(f"{csv_prefix}_w_after_ml_position_sizing", 0.0) or 0.0)
    raw_allocator_weight = float(row.get(f"{csv_prefix}_w_raw_allocator", 0.0) or 0.0)
    raw_weight = float(row.get(f"w_{csv_prefix}", 0.0) or 0.0)

    weight = exec_weight
    if abs(weight) <= 1e-12:
        weight = cluster_weight
    if abs(weight) <= 1e-12:
        weight = signal_gating_weight
    if abs(weight) <= 1e-12:
        weight = smoothing_weight
    if abs(weight) <= 1e-12:
        weight = ml_sizing_weight
    if abs(weight) <= 1e-12:
        weight = raw_allocator_weight
    if abs(weight) <= 1e-12:
        weight = raw_weight

    exec_side = str(row.get(f"{csv_prefix}_execution_side", "") or "").lower()
    cluster_side = str(row.get(f"{csv_prefix}_cluster_side", "") or "").lower()
    signal_side = str(row.get(f"{csv_prefix}_side", "") or "").lower()

    if abs(weight) > 1e-12:
        if weight > 0:
            side = "long"
        elif weight < 0:
            side = "short"
        else:
            side = "flat"
    elif exec_side in {"long", "short"}:
        side = exec_side
    elif cluster_side in {"long", "short"}:
        side = cluster_side
    elif signal_side in {"long", "short"}:
        side = signal_side
    else:
        side = "flat"

    target_weight = float(weight)

    max_target_weight = float(cfg.get("max_target_weight", 0.25) or 0.25)
    uncapped_target_weight = float(target_weight)
    target_weight = max(-max_target_weight, min(max_target_weight, float(target_weight)))

    if abs(target_weight - uncapped_target_weight) > 1e-12:
        print(
            "execution_cap_action:",
            f"capped target_weight from {uncapped_target_weight} to {target_weight}",
            f"(max_target_weight={max_target_weight})",
        )

    target_qty = abs(usdt_total * target_weight) / last if last > 0 else 0.0
    target_qty = float(bitget.amount_to_precision(symbol, target_qty)) if target_qty > 0 else 0.0
    if target_weight < 0:
        target_qty = -target_qty

    pos = bitget.fetch_open_positions(symbol)
    current_qty = current_signed_qty(pos)
    delta_qty = target_qty - current_qty
    delta_notional = abs(delta_qty) * last

    print(f"=== {symbol} ===")
    print("side:", side)
    print("target_weight:", target_weight)
    print("last:", last)
    print("current_qty:", current_qty)
    print("target_qty:", target_qty)
    print("delta_qty:", delta_qty)
    print("delta_notional:", delta_notional)

    _action = "unknown"
    _blocked_reason = ""

    if abs(delta_notional) < cfg["min_delta_notional"]:
        _action = "skip_below_min_delta_notional"
        print("action: skip rebalance (below min_delta_notional)")
        effective_qty = current_qty
        try:
            atr = fetch_atr(bitget, symbol, cfg["timeframe"], cfg["ohlcv_limit"], cfg["atr_period"])
            ensure_protective_orders(bitget, symbol, effective_qty, last, atr, cfg)
        except Exception as e:
            print(f"protective_action: skipped due to ATR/order error: {e}")
        execution_snapshot[symbol] = {
            "side": str(side),
            "target_weight": float(target_weight),
            "last": float(last),
            "current_qty": float(current_qty),
            "target_qty": float(target_qty),
            "delta_qty": float(delta_qty),
            "delta_notional": float(delta_notional),
            "action": str(_action),
            "blocked_reason": str(_blocked_reason),
        }
        print()
        continue

    reducing_existing_position = False
    current_side = position_side_from_qty(current_qty)
    target_side = position_side_from_qty(target_qty)

    if abs(current_qty) > 1e-12:
        if target_side == "flat":
            reducing_existing_position = True
        elif current_side == target_side and abs(target_qty) < abs(current_qty):
            reducing_existing_position = True

    if abs(delta_notional) > cfg["max_delta_notional"] and not reducing_existing_position:
        _action = "blocked_above_max_delta_notional"
        _blocked_reason = "above_max_delta_notional"
        print("action: BLOCKED (above max_delta_notional)")
        execution_snapshot[symbol] = {
            "side": str(side),
            "target_weight": float(target_weight),
            "last": float(last),
            "current_qty": float(current_qty),
            "target_qty": float(target_qty),
            "delta_qty": float(delta_qty),
            "delta_notional": float(delta_notional),
            "action": str(_action),
            "blocked_reason": str(_blocked_reason),
        }
        print()
        continue

    if abs(delta_notional) > cfg["max_delta_notional"] and reducing_existing_position:
        print("action: ALLOW reduce-only rebalance (above max_delta_notional but reducing risk)")

    if current_qty > 0 and target_qty == 0:
        _effective_qty_override = None
        _action = "reduce_long"
        active_profit_plans = [o for o in fetch_open_profit_loss_orders(bitget, symbol) if get_plan_type(o) == "profit_plan"]
        if active_profit_plans:
            _action = "skip_due_to_active_tp"
            _blocked_reason = "active_profit_plans"
            _effective_qty_override = current_qty
            print(
                f"CLOSE_POSITION_SKIPPED -> symbol={symbol} side=long "
                f"reason=active_profit_plans count={len(active_profit_plans)} "
                f"current_qty={current_qty} target_qty={target_qty}"
            )
        else:
            print(f"CLOSE_POSITION -> symbol={symbol} side=long live={LIVE_TRADING}")
            if LIVE_TRADING:
                bitget.flash_close_position(symbol, side="long")
                current_qty = refresh_current_qty(bitget, symbol)
                print(f"POSITION_AFTER_CLOSE -> symbol={symbol} qty={current_qty}")

    elif current_qty < 0 and target_qty == 0:
        _effective_qty_override = None
        _action = "reduce_short"
        active_profit_plans = [o for o in fetch_open_profit_loss_orders(bitget, symbol) if get_plan_type(o) == "profit_plan"]
        if active_profit_plans:
            _action = "skip_due_to_active_tp"
            _blocked_reason = "active_profit_plans"
            _effective_qty_override = current_qty
            print(
                f"CLOSE_POSITION_SKIPPED -> symbol={symbol} side=short "
                f"reason=active_profit_plans count={len(active_profit_plans)} "
                f"current_qty={current_qty} target_qty={target_qty}"
            )
        else:
            print(f"CLOSE_POSITION -> symbol={symbol} side=short live={LIVE_TRADING}")
            if LIVE_TRADING:
                bitget.flash_close_position(symbol, side="short")
                current_qty = refresh_current_qty(bitget, symbol)
                print(f"POSITION_AFTER_CLOSE -> symbol={symbol} qty={current_qty}")

    elif current_qty == 0 or (current_qty > 0 and target_qty > 0) or (current_qty < 0 and target_qty < 0):
        _effective_qty_override = None

        if delta_qty > 0:
            if current_qty > 0 and target_qty > 0:
                active_profit_plans = [
                    o for o in fetch_open_profit_loss_orders(bitget, symbol)
                    if get_plan_type(o) == "profit_plan"
                ]
                if active_profit_plans:
                    _action = "skip_due_to_active_tp"
                    _blocked_reason = "active_profit_plans"
                    _effective_qty_override = current_qty
                    print(
                        f"BLOCK_INCREASE_DUE_TO_TP -> symbol={symbol} side=long "
                        f"current_qty={current_qty} target_qty={target_qty} delta_qty={delta_qty} "
                        f"active_profit_plans={len(active_profit_plans)}"
                    )
                    print("action: skip rebalance (active profit_plan after TP1/TP ladder)")
                else:
                    _action = "open_long"
                    place_market(bitget, symbol, "buy", abs(delta_qty), reduce=False)
            else:
                _action = "open_long"
                place_market(bitget, symbol, "buy", abs(delta_qty), reduce=False)
        elif delta_qty < 0:
            if current_qty > 0 and target_qty > 0:
                _action = "reduce_long"
                place_market(bitget, symbol, "sell", abs(delta_qty), reduce=True)
            elif current_qty < 0 and target_qty < 0:
                if abs(target_qty) > abs(current_qty):
                    _action = "open_short"
                    place_market(bitget, symbol, "sell", abs(delta_qty), reduce=False)
                else:
                    _action = "reduce_short"
                    place_market(bitget, symbol, "buy", abs(delta_qty), reduce=True)
            elif current_qty == 0 and target_qty < 0:
                _action = "open_short"
                place_market(bitget, symbol, "sell", abs(delta_qty), reduce=False)
            else:
                _action = "none"
                print("action: none")
        else:
            print("action: none")

        effective_qty = current_qty if _effective_qty_override is not None else target_qty
        try:
            atr = fetch_atr(bitget, symbol, cfg["timeframe"], cfg["ohlcv_limit"], cfg["atr_period"])
            if _blocked_reason == "active_profit_plans" and abs(current_qty) > 0:
                print(
                    f"protective_action: preserve current protection due to active TP "
                    f"(symbol={symbol}, current_qty={current_qty}, target_qty={target_qty})"
                )
                ensure_protective_orders(bitget, symbol, current_qty, last, atr, cfg)
            else:
                effective_qty_for_protection = effective_qty
                if _action in {"open_long", "open_short"} and LIVE_TRADING:
                    refreshed_qty = refresh_current_qty(bitget, symbol, attempts=3, sleep_seconds=1.0)
                    if abs(refreshed_qty) > 1e-12:
                        effective_qty_for_protection = refreshed_qty
                    else:
                        effective_qty_for_protection = 0.0
                        print(
                            f"PROTECTIVE_WAIT_SKIP -> symbol={symbol} action={_action} "
                            f"target_qty={target_qty} reason=position_not_visible_after_retry"
                        )
                ensure_protective_orders(bitget, symbol, effective_qty_for_protection, last, atr, cfg)
        except Exception as e:
            print(f"protective_action: skipped due to ATR/order error: {e}")

        execution_snapshot[symbol] = {
            "side": str(side),
            "target_weight": float(target_weight),
            "last": float(last),
            "current_qty": float(current_qty),
            "target_qty": float(target_qty),
            "delta_qty": float(delta_qty),
            "delta_notional": float(delta_notional),
            "action": str(_action),
            "blocked_reason": str(_blocked_reason),
        }
        print()
        continue

    _action = "flip_position"
    print("flip detected: closing current position first")
    if current_qty > 0:
        place_market(bitget, symbol, "sell", abs(current_qty), reduce=True)
    elif current_qty < 0:
        place_market(bitget, symbol, "buy", abs(current_qty), reduce=True)

    if target_qty > 0:
        place_market(bitget, symbol, "buy", abs(target_qty), reduce=False)
    elif target_qty < 0:
        place_market(bitget, symbol, "sell", abs(target_qty), reduce=False)

    effective_qty = target_qty
    try:
        atr = fetch_atr(bitget, symbol, cfg["timeframe"], cfg["ohlcv_limit"], cfg["atr_period"])
        ensure_protective_orders(bitget, symbol, effective_qty, last, atr, cfg)
    except Exception as e:
        print(f"protective_action: skipped due to ATR/order error: {e}")

    if abs(current_qty) > 1e-12:
        refreshed_pos_side = position_side_from_qty(current_qty)
        if refreshed_pos_side in {"long", "short"}:
            try:
                protective_orders_now = fetch_open_profit_loss_orders(bitget, symbol) or []
            except Exception as e:
                protective_orders_now = []
                print(f"PROTECTIVE_FETCH_WARN -> symbol={symbol} error={e}")

            print(
                f"PROTECTIVE_RECHECK -> symbol={symbol} "
                f"pos_side={refreshed_pos_side} qty={current_qty} count={len(protective_orders_now)}"
            )

            if len(protective_orders_now) == 0:
                print(f"PROTECTIVE_MISSING -> symbol={symbol} recreating protection for live position")
                try:
                    maintain_partial_tp_and_sl(
                        bitget=bitget,
                        symbol=symbol,
                        pos_side=refreshed_pos_side,
                        qty=abs(current_qty),
                        ref_price=last,
                        atr=atr,
                        sl_atr_mult=cfg["sl_atr_mult"],
                        tp1_atr_mult=cfg["tp1_atr_mult"],
                        tp2_atr_mult=cfg["tp2_atr_mult"],
                        tp1_fraction=cfg["tp1_fraction"],
                        sl_rel_tol=cfg["sl_rel_tol"],
                        tp_rel_tol=cfg["tp_rel_tol"],
                    )
                except Exception as e:
                    print(f"PROTECTIVE_RECREATE_ERROR -> symbol={symbol} error={e}")

    execution_snapshot[symbol] = {
        "side": str(side),
        "target_weight": float(target_weight),
        "last": float(last),
        "current_qty": float(current_qty),
        "target_qty": float(target_qty),
        "delta_qty": float(delta_qty),
        "delta_notional": float(delta_notional),
        "action": str(_action),
        "blocked_reason": str(_blocked_reason),
    }
    print()


print("=== EXECUTION SNAPSHOT ===")
import json
print(json.dumps(execution_snapshot, indent=2))
