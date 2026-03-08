
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--quantile", type=float, default=0.4)
    return ap.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input)

    if "strategy_id" not in df.columns or "p_win" not in df.columns:
        raise SystemExit("dataset no contiene columnas necesarias")

    df = df[df["side_raw"] != "flat"].copy()

    thresholds = {}

    for strategy, g in df.groupby("strategy_id"):
        q = g["p_win"].quantile(args.quantile)
        thresholds[str(strategy)] = float(q)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        json.dump(
            {
                "quantile": args.quantile,
                "thresholds": thresholds,
            },
            f,
            indent=2,
        )

    print("thresholds:")
    for k, v in thresholds.items():
        print(f"{k:25s} {v:.4f}")

    print("\nSaved ->", out)


if __name__ == "__main__":
    main()
