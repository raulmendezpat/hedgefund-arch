import json
import os
from pathlib import Path

import pandas as pd
import ta

from hf.legacy.ltb.utilities.bitget_futures import BitgetFutures

APP = Path("/home/ubuntu/hedgefund-arch")
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
    weight_cols = [c for c in df.columns if c.startswith("w_")]
    for col in weight_cols:
        sym = col[2:].replace("_usdt_usdt", "").upper() + "/USDT:USDT"
        prefix = symbol_to_prefix(sym)

        side_col = f"{prefix}_side"
        cluster_side_col = f"{prefix}_cluster_side"

        latest_w = float(df[col].fillna(0.0).iloc[-1]) if col in df.columns else 0.0
        latest_side = "flat"
        if cluster_side_col in df.columns:
            latest_side = str(df[cluster_side_col].fillna("flat").iloc[-1]).lower()
        elif side_col in df.columns:
            latest_side = str(df[side_col].fillna("flat").iloc[-1]).lower()


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


def place_market(bitget, symbol: str, side: str, qty: float, reduce: bool = False):
    if qty <= 0:
        return None
    print(f"ORDER -> symbol={symbol} side={side} qty={qty} reduceOnly={reduce} live={LIVE_TRADING}")
    if not LIVE_TRADING:
        return None
    return bitget.place_market_order(symbol=symbol, side=side, amount=qty, reduce=reduce)


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

def extract_trigger_amount(o):
    if not isinstance(o, dict):
        return 0.0
    info = o.get("info", {}) if isinstance(o.get("info"), dict) else {}
    for k in ("amount", "size"):
        v = o.get(k, None)
        if v not in (None, ""):
            try:
                return float(v)
            except Exception:
                pass
        v = info.get(k, None)
        if v not in (None, ""):
            try:
                return float(v)
            except Exception:
                pass
    return 0.0

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
    matched_side = [o for o in open_reduce if side_matches_reduce_order(pos_side, o)]

    qty = abs(float(pos_qty or 0.0))
    if qty > 0:
        qty_tol = max(1.0, qty * 0.10)
        full_like = []
        partial_like = []

        for o in matched_side:
            amt = float(extract_trigger_amount(o) or 0.0)
            px = extract_trigger_price(o)
            if px is None:
                continue
            if abs(amt - qty) <= qty_tol or amt >= qty * 0.90:
                full_like.append(o)
            else:
                partial_like.append(o)

        # Si encontramos parciales, asumimos esquema: 1 SL full-size + N TPs parciales
        if partial_like:
            stop_like = []
            for o in full_like:
                px = extract_trigger_price(o)
                if px is None:
                    continue
                if pos_side == "long":
                    if px < ref_price:
                        stop_like.append(o)
                elif pos_side == "short":
                    if px > ref_price:
                        stop_like.append(o)

            if not stop_like:
                stop_like = list(full_like)

            tp_like = [o for o in partial_like if extract_trigger_price(o) is not None]
            return stop_like, tp_like

    stop_like = []
    tp_like = []
    for o in matched_side:
        px = extract_trigger_price(o)
        if px is None:
            continue
        if pos_side == "long":
            if px < ref_price:
                stop_like.append(o)
            else:
                tp_like.append(o)
        elif pos_side == "short":
            if px > ref_price:
                stop_like.append(o)
            else:
                tp_like.append(o)
    return stop_like, tp_like

def maintain_single_trigger(bitget, symbol: str, desired_price: float, pos_side: str, qty: float, kind: str, rel_tol: float, ref_price: float):
    reduce_side = "sell" if pos_side == "long" else "buy"
    open_reduce = fetch_reduce_only_trigger_orders(bitget, symbol)
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
        )


def ensure_protective_orders(bitget, symbol: str, pos_qty: float, ref_price: float, atr: float, cfg: dict):
    pos_side = position_side_from_qty(pos_qty)
    qty = abs(pos_qty)

    if pos_side == "flat" or qty <= 0:
        for o in fetch_reduce_only_trigger_orders(bitget, symbol):
            cancel_trigger_order_safe(bitget, symbol, extract_trigger_order_id(o))
        print("protective_action: no position -> stale SL/TP cancelled if any")
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

    maintain_single_trigger(
        bitget=bitget,
        symbol=symbol,
        desired_price=sl_price,
        pos_side=pos_side,
        qty=qty,
        kind="sl",
        rel_tol=float(cfg["stop_refresh_rel_tol"]),
        ref_price=ref_price,
    )

    if not partial_tp_enabled:
        maintain_single_trigger(
            bitget=bitget,
            symbol=symbol,
            desired_price=tp2_price,
            pos_side=pos_side,
            qty=qty,
            kind="tp",
            rel_tol=float(cfg["tp_refresh_rel_tol"]),
            ref_price=ref_price,
        )
        return

    tp1_price = desired_take_profit_price_fixed_mult(
        pos_side,
        ref_price,
        atr,
        partial_tp1_atr_mult,
    )
    tp1_price = float(bitget.price_to_precision(symbol, tp1_price))

    qty1 = max(0.0, float(qty) * partial_tp1_fraction)
    qty1 = float(bitget.amount_to_precision(symbol, qty1)) if qty1 > 0 else 0.0
    qty2 = max(0.0, float(qty) - float(qty1))
    qty2 = float(bitget.amount_to_precision(symbol, qty2)) if qty2 > 0 else 0.0

    if qty1 <= 0 or qty2 <= 0:
        maintain_single_trigger(
            bitget=bitget,
            symbol=symbol,
            desired_price=tp2_price,
            pos_side=pos_side,
            qty=qty,
            kind="tp",
            rel_tol=float(cfg["tp_refresh_rel_tol"]),
            ref_price=ref_price,
        )
        return

    reduce_side = "sell" if pos_side == "long" else "buy"
    open_reduce = fetch_reduce_only_trigger_orders(bitget, symbol)
    stop_like, tp_like = classify_reduce_orders(open_reduce, pos_side, ref_price, qty)

    # Keep existing TP orders while the position remains open.
    # TP should be fixed from the moment the position is opened.
    if len(tp_like) >= 2:
        print(f"tp_action: keep existing partial TPs (count={len(tp_like)})")
        extras = tp_like[2:]
        for o in extras:
            cancel_trigger_order_safe(bitget, symbol, extract_trigger_order_id(o))
        return

    # If there is 1 TP or 0 TP, rebuild TP set cleanly.
    for o in tp_like:
        cancel_trigger_order_safe(bitget, symbol, extract_trigger_order_id(o))

    print(f"PLACE_TP1 -> symbol={symbol} side={reduce_side} qty={qty1} trigger={tp1_price} live={LIVE_TRADING}")
    if LIVE_TRADING:
        bitget.place_trigger_market_order(
            symbol=symbol,
            side=reduce_side,
            amount=qty1,
            trigger_price=tp1_price,
            reduce=True,
        )

    print(f"PLACE_TP2 -> symbol={symbol} side={reduce_side} qty={qty2} trigger={tp2_price} live={LIVE_TRADING}")
    if LIVE_TRADING:
        bitget.place_trigger_market_order(
            symbol=symbol,
            side=reduce_side,
            amount=qty2,
            trigger_price=tp2_price,
            reduce=True,
        )


bitget = BitgetFutures(SECRET)


def ensure_leverage(bitget, symbol: str, leverage: int = 2):
    try:
        bitget.set_margin_mode(symbol, "isolated")
    except Exception:
        pass
    try:
        bitget.set_leverage(symbol, "isolated", leverage)
    except Exception:
        pass

df = pd.read_csv(APP / "results/pipeline_allocations_prod_candidate_live.csv", low_memory=False)
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
    raw_weight = float(row.get(f"w_{csv_prefix}", 0.0) or 0.0)

    weight = exec_weight
    if abs(weight) <= 1e-12:
        weight = cluster_weight
    if abs(weight) <= 1e-12:
        weight = raw_weight

    exec_side = str(row.get(f"{csv_prefix}_execution_side", "") or "").lower()
    cluster_side = str(row.get(f"{csv_prefix}_cluster_side", "") or "").lower()
    signal_side = str(row.get(f"{csv_prefix}_side", "") or "").lower()

    if exec_side in {"long", "short"}:
        side = exec_side
    elif cluster_side in {"long", "short"}:
        side = cluster_side
    elif signal_side in {"long", "short"}:
        side = signal_side
    else:
        side = "flat"

    if abs(weight) > 1e-12 and side == "flat":
        # fallback: if we have non-zero target weight but no valid side, infer long by sign convention
        side = "long"

    target_weight = signed_weight(side, weight)

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

    if abs(delta_notional) > cfg["max_delta_notional"]:
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

    if current_qty == 0 or (current_qty > 0 and target_qty >= 0) or (current_qty < 0 and target_qty <= 0):
        if delta_qty > 0:
            _action = "open_long"
            place_market(bitget, symbol, "buy", abs(delta_qty), reduce=False)
        elif delta_qty < 0:
            if current_qty > 0 and target_qty >= 0:
                _action = "reduce_long"
                place_market(bitget, symbol, "sell", abs(delta_qty), reduce=True)
            elif current_qty < 0 and target_qty <= 0:
                _action = "reduce_short"
                place_market(bitget, symbol, "buy", abs(delta_qty), reduce=True)
            else:
                _action = "none"
                _action = "none"
            print("action: none")
        else:
            print("action: none")

        effective_qty = target_qty
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
