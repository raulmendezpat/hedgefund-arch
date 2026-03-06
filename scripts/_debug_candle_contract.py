from pprint import pprint

from hf.data.ohlcv import load_ohlcv
from hf.core.types import Candle

print("Candle annotations:")
try:
    pprint(Candle.__annotations__)
except Exception as e:
    print("no __annotations__", e)

print("\nCandle dir sample:")
print([x for x in dir(Candle) if not x.startswith("_")][:80])

btc = load_ohlcv("BTC/USDT:USDT", timeframe="1h")
sol = load_ohlcv("SOL/USDT:USDT", timeframe="1h")

print("\nBTC type:", type(btc))
print("SOL type:", type(sol))

def inspect_obj(name, obj):
    print(f"\n--- {name} ---")
    print("type:", type(obj))
    if hasattr(obj, "__dict__"):
        print("__dict__ keys:", sorted(obj.__dict__.keys()))
    else:
        print("no __dict__")

    for attr in [
        "symbol", "timestamp", "ts", "open", "high", "low", "close", "volume",
        "features", "history", "candles", "rows", "df", "dataframe"
    ]:
        try:
            v = getattr(obj, attr)
            t = type(v)
            if attr in {"history", "candles", "rows"} and v is not None:
                try:
                    n = len(v)
                except Exception:
                    n = "?"
                print(f"{attr}: type={t} len={n}")
            else:
                print(f"{attr}: type={t} value={repr(v)[:300]}")
        except Exception as e:
            print(f"{attr}: <ERR {e}>")

if isinstance(btc, dict):
    print("\nBTC keys:", list(btc.keys())[:20])
    for k, v in list(btc.items())[:2]:
        inspect_obj(f"btc[{k}]", v)
else:
    inspect_obj("btc", btc)

if isinstance(sol, dict):
    print("\nSOL keys:", list(sol.keys())[:20])
    for k, v in list(sol.items())[:2]:
        inspect_obj(f"sol[{k}]", v)
else:
    inspect_obj("sol", sol)
