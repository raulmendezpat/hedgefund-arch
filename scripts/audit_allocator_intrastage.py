import pandas as pd

df = pd.read_csv("results/pipeline_stage_detail_diag_6m_contextual_policy_v1.csv")

print("\n=== OVERALL ===")
print("rows:", len(df))
print("accept_rate:", float(df["accept"].mean()))
print("alloc_input_present_rate:", float(df["alloc_input_present"].mean()))
print("nonzero_final_weight_rate:", float((df["final_symbol_weight"].abs() > 1e-12).mean()))

print("\n=== COLLAPSE BY SYMBOL ===")
g = df.groupby("symbol").agg(
    rows=("symbol", "size"),
    strategies=("strategy_id", "nunique"),
    accepted=("accept", "sum"),
    alloc_inputs=("alloc_input_present", "sum"),
    nonzero_weight_rate=("final_symbol_weight", lambda s: float((s.abs() > 1e-12).mean())),
    avg_final_weight=("final_symbol_weight", "mean"),
    min_final_weight=("final_symbol_weight", "min"),
    max_final_weight=("final_symbol_weight", "max"),
)
print(g.to_string())

print("\n=== COLLAPSE BY SYMBOL + STRATEGY ===")
g2 = df.groupby(["symbol", "strategy_id"]).agg(
    rows=("strategy_id", "size"),
    accept_rate=("accept", "mean"),
    alloc_input_rate=("alloc_input_present", "mean"),
    nonzero_weight_rate=("final_symbol_weight", lambda s: float((s.abs() > 1e-12).mean())),
    avg_final_weight=("final_symbol_weight", "mean"),
).reset_index()
print(g2.to_string(index=False))

print("\n=== SAME-SYMBOL STRATEGY CONFLICT CHECK ===")
pivot = (
    g2.pivot(index="symbol", columns="strategy_id", values="avg_final_weight")
    if not g2.empty else pd.DataFrame()
)
if not pivot.empty:
    print(pivot.to_string())

print("\n=== WEIGHTS DESCRIBE ===")
print(df["final_symbol_weight"].describe())

print("\n=== ZERO-WEIGHT ACCEPTED ROWS SAMPLE ===")
z = df[(df["accept"] == True) & (df["final_symbol_weight"].abs() <= 1e-12)].copy()
cols = [c for c in [
    "ts","symbol","strategy_id","side","p_win","expected_return","score",
    "size_mult","band","reason","ctx_backdrop","ctx_side_backdrop_alignment",
    "ctx_expected_holding_bars","ctx_exit_profile"
] if c in z.columns]
print(z[cols].head(100).to_string(index=False))
