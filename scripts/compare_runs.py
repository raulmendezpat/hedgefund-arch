import json
from pathlib import Path
import sys

def load(p):
    return json.loads(Path(p).read_text())

def pct_diff(a, b):
    if a == 0:
        return None
    return (b - a) / abs(a)

def main(base_path, new_path):
    base = load(base_path)
    new = load(new_path)

    keys = [
        "total_return_pct",
        "sharpe_annual",
        "max_drawdown_pct",
        "win_rate_pct",
        "vol_annual",
        "avg_n_opps",
        "avg_n_accepts",
        "avg_active_symbols",
        "avg_gross_weight",
        "total_seconds",
    ]

    print("\n=== COMPARISON ===\n")

    for k in keys:
        b = base.get(k)
        n = new.get(k)

        diff = None
        if isinstance(b, (int, float)) and isinstance(n, (int, float)):
            diff = pct_diff(b, n)

        print(f"{k}:")
        print(f"  base: {b}")
        print(f"  new : {n}")
        print(f"  diff: {diff}")
        print("")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: compare_runs.py base.json new.json")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
