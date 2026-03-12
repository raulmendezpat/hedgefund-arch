import json
from pathlib import Path

import pandas as pd

from hf.legacy.ltb.utilities.bitget_futures import BitgetFutures

APP = Path("/home/ubuntu/hedgefund-arch")
SECRET = json.loads((APP / "secret.json").read_text())["envelope"]

SYMBOLS = {
    "btc": "BTC/USDT:USDT",
    "sol": "SOL/USDT:USDT",
}

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

bitget = BitgetFutures(SECRET)
df = pd.read_csv(APP / "results/pipeline_allocations_prod_candidate_live.csv")
row = df.iloc[-1]

bal = bitget.fetch_balance()
usdt_total = float((bal.get("USDT", {}) or {}).get("total", (bal.get("USDT", {}) or {}).get("free", 0.0)))

print("=== DRY RUN RECONCILIATION ===")
print("usdt_total:", usdt_total)
print()

for prefix, symbol in SYMBOLS.items():
    ticker = bitget.fetch_ticker(symbol)
    last = float(ticker.get("last") or ticker.get("close") or 0.0)

    side = str(row.get(f"{prefix}_side", "flat"))
    weight = float(row.get(f"{prefix}_execution_target_weight", row.get(f"w_{prefix}", 0.0)) or 0.0)
    target_weight = signed_weight(side, weight)

    target_qty = abs(usdt_total * target_weight) / last if last > 0 else 0.0
    target_qty = float(bitget.amount_to_precision(symbol, target_qty)) if target_qty > 0 else 0.0
    if target_weight < 0:
        target_qty = -target_qty

    pos = bitget.fetch_open_positions(symbol)
    current_qty = current_signed_qty(pos)
    delta_qty = target_qty - current_qty

    print(f"=== {symbol} ===")
    print("side:", side)
    print("target_weight:", target_weight)
    print("last:", last)
    print("current_qty:", current_qty)
    print("target_qty:", target_qty)
    print("delta_qty:", delta_qty)

    if abs(delta_qty) == 0:
        print("action: none")
    elif delta_qty > 0:
        print("action: BUY market", abs(delta_qty))
    else:
        print("action: SELL market", abs(delta_qty))
    print()
