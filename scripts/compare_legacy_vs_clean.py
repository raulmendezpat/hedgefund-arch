import json
from pathlib import Path

legacy = json.loads(Path("results/pipeline_metrics_diag_6m_base.json").read_text())
clean = json.loads(Path("results/research_runtime_metrics_diag_6m_policy_factory_v1.json").read_text())

pairs = [
    ("total_return_pct", "total_return_pct"),
    ("sharpe_annual", "sharpe_annual"),
    ("max_drawdown_pct", "max_drawdown_pct"),
    ("win_rate_pct", "win_rate_pct"),
]

print("\n=== LEGACY PRE-ARCHITECTURE ===")
for lk, _ in pairs:
    print(f"{lk}: {legacy.get(lk)}")

print("\n=== CLEAN ARCHITECTURE ===")
for _, ck in pairs:
    print(f"{ck}: {clean.get(ck)}")

print("\n=== DIFF CLEAN - LEGACY ===")
for lk, ck in pairs:
    lv = legacy.get(lk)
    cv = clean.get(ck)
    if isinstance(lv, (int, float)) and isinstance(cv, (int, float)):
        print(f"{ck}: {cv - lv}")
    else:
        print(f"{ck}: n/a")

print("\n=== RUNTIME / FLOW (clean only) ===")
for k in [
    "total_seconds",
    "avg_n_opps",
    "avg_n_accepts",
    "avg_active_symbols",
    "avg_gross_weight",
]:
    print(f"{k}: {clean.get(k)}")
