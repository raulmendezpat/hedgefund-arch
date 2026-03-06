import inspect
from pprint import pprint

from hf.core.types import Candle, Signal
import hf.pipeline.run_portfolio as rp

print("\n=== Candle annotations ===")
try:
    pprint(Candle.__annotations__)
except Exception as e:
    print("annotations error:", e)

print("\n=== Candle fields/attrs ===")
for name in [n for n in dir(Candle) if not n.startswith("_")][:200]:
    print(name)

print("\n=== Signal annotations ===")
try:
    pprint(Signal.__annotations__)
except Exception as e:
    print("annotations error:", e)

print("\n=== run_portfolio path ===")
print(rp.__file__)

print("\n=== functions in run_portfolio ===")
for name in [n for n in dir(rp) if not n.startswith("_")]:
    obj = getattr(rp, name)
    if callable(obj):
        try:
            sig = inspect.signature(obj)
        except Exception:
            sig = "(signature unavailable)"
        print(f"{name}{sig}")

print("\n=== source: run() ===")
try:
    print(inspect.getsource(rp.run))
except Exception as e:
    print("could not read run():", e)

if hasattr(rp, "main"):
    print("\n=== source: main() ===")
    try:
        print(inspect.getsource(rp.main))
    except Exception as e:
        print("could not read main():", e)
