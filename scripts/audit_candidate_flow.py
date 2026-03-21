import argparse
import pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
args = ap.parse_args()

p = Path(f"results/research_runtime_candidates_{args.name}.csv")
df = pd.read_csv(p)

print("\n=== CANDIDATE FLOW OVERVIEW ===")
print("rows:", len(df))

print("\n=== ACCEPT RATE ===")
print(df["accept"].value_counts(dropna=False))
print(df["accept"].value_counts(normalize=True, dropna=False))

print("\n=== BY STRATEGY ===")
g = df.groupby("strategy_id").agg(
    rows=("strategy_id", "size"),
    accept_rate=("accept", "mean"),
    avg_p_win=("p_win", "mean"),
    avg_expected_return=("expected_return", "mean"),
    avg_score=("score", "mean"),
    avg_size=("size_mult", "mean"),
)
print(g.sort_values("accept_rate"))

print("\n=== BY SIDE ===")
g2 = df.groupby("side").agg(
    rows=("side", "size"),
    accept_rate=("accept", "mean"),
    avg_p_win=("p_win", "mean"),
    avg_expected_return=("expected_return", "mean"),
    avg_score=("score", "mean"),
    avg_size=("size_mult", "mean"),
)
print(g2.sort_values("accept_rate"))

print("\n=== BANDS ===")
print(df["band"].value_counts(dropna=False))

print("\n=== REASONS ===")
print(df["reason"].value_counts(dropna=False).head(20))

print("\n=== SCORE DESCRIBE ===")
print(df[["p_win", "expected_return", "score", "size_mult"]].describe())

flag_cols = [c for c in ["adx_below_min","ema_gap_below_min","atrp_low","adx_low","range_expansion_low"] if c in df.columns]
if flag_cols:
    print("\n=== REGIME FLAGS RATE ===")
    for c in flag_cols:
        s = df[c].fillna(False).astype(bool)
        print(c, float(s.mean()))
