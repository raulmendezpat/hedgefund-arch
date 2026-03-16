from __future__ import annotations

from typing import Any, Dict, List, Tuple


def extract_trigger_price(o: Dict[str, Any]) -> float | None:
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


def extract_trigger_amount(o: Dict[str, Any]) -> float:
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


def side_matches_reduce_order(pos_side: str, order_obj: Dict[str, Any]) -> bool:
    raw_side = str(order_obj.get("side") or (order_obj.get("info", {}) or {}).get("side") or "").lower()
    if pos_side == "long":
        return raw_side == "sell"
    if pos_side == "short":
        return raw_side == "buy"
    return False


def classify_reduce_orders(
    open_reduce: List[Dict[str, Any]],
    pos_side: str,
    ref_price: float,
    pos_qty: float | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    matched_side = [o for o in open_reduce if side_matches_reduce_order(pos_side, o)]

    qty = abs(float(pos_qty or 0.0))
    if qty > 0:
        qty_tol = max(1.0, qty * 0.10)
        full_like: List[Dict[str, Any]] = []
        partial_like: List[Dict[str, Any]] = []

        for o in matched_side:
            amt = float(extract_trigger_amount(o) or 0.0)
            px = extract_trigger_price(o)
            if px is None:
                continue
            if abs(amt - qty) <= qty_tol or amt >= qty * 0.90:
                full_like.append(o)
            else:
                partial_like.append(o)

        if partial_like:
            stop_like: List[Dict[str, Any]] = []
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

    stop_like: List[Dict[str, Any]] = []
    tp_like: List[Dict[str, Any]] = []
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


def should_keep_partial_tps(
    tp_like: List[Dict[str, Any]],
    pos_side: str,
    pos_qty: float,
) -> bool:
    qty = abs(float(pos_qty or 0.0))
    if qty <= 0:
        return False
    if len(tp_like) != 2:
        return False

    tp_sorted = sorted(
        tp_like,
        key=lambda o: float(extract_trigger_price(o) or 0.0),
        reverse=(pos_side == "short"),
    )
    prices = [float(extract_trigger_price(o) or 0.0) for o in tp_sorted]
    amts = [float(extract_trigger_amount(o) or 0.0) for o in tp_sorted]

    if not all(px > 0 for px in prices):
        return False

    qty_tol = max(1.0, qty * 0.02)
    return abs(sum(amts) - qty) <= qty_tol
