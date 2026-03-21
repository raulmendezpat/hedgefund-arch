import argparse
import pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
args = ap.parse_args()

p = Path(f"results/research_runtime_{args.name}.csv")
df = pd.read_csv(p)

print("\n=== FLOW OVERVIEW ===")
print("rows:", len(df))

for col in ["n_opps", "n_features", "n_scores", "n_accepts", "gross_weight", "active_symbols", "port_ret"]:
    if col in df.columns:
        print(f"\n--- {col} ---")
        print(df[col].describe())

print("\n=== ACCEPT / EXPOSURE CHECKS ===")
if "n_opps" in df.columns and "n_accepts" in df.columns:
    print("rows with opps > 0:", int((df["n_opps"] > 0).sum()))
    print("rows with accepts > 0:", int((df["n_accepts"] > 0).sum()))
    print("rows with opps == accepts:", int((df["n_opps"] == df["n_accepts"]).sum()))
    print("rows with opps > accepts:", int((df["n_opps"] > df["n_accepts"]).sum()))

if "gross_weight" in df.columns:
    print("rows gross_weight == 0:", int((df["gross_weight"] == 0).sum()))
    print("rows gross_weight > 0:", int((df["gross_weight"] > 0).sum()))
    print("rows gross_weight >= 0.05:", int((df["gross_weight"] >= 0.05).sum()))
    print("rows gross_weight >= 0.07:", int((df["gross_weight"] >= 0.07).sum()))

print("\n=== SAMPLE ===")
cols = [c for c in ["ts", "n_opps", "n_features", "n_scores", "n_accepts", "gross_weight", "active_symbols", "port_ret", "equity"] if c in df.columns]
print(df[cols].head(20).to_string(index=False))
