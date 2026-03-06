import inspect
import hf.data.ohlcv as ohlcv

print("\n=== module path ===")
print(ohlcv.__file__)

print("\n=== exported names ===")
names = [n for n in dir(ohlcv) if not n.startswith("_")]
for n in names:
    print(n)

print("\n=== callables ===")
for n in names:
    obj = getattr(ohlcv, n)
    if callable(obj):
        try:
            sig = inspect.signature(obj)
        except Exception:
            sig = "(signature unavailable)"
        print(f"{n}{sig}")
